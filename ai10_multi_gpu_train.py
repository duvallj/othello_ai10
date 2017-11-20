from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf
import ai10
import ai10_input

parser = ai10.parser

parser.add_argument('--train_dir', type=str, default='/tmp/cifar10_train',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--max_steps', type=int, default=1000000,
                    help='Number of batches to run.')

parser.add_argument('--num_gpus', type=int, default=1,
                    help='How many GPUs to use.')

parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

def tower_loss(scope, boards, scores):
    """Calculate total loss on a single tower.

    Args:
        scope: unique prefix string identifying the tower.
        boards: Boards. 4D tensor of size [batch_size, height, width, 1]
        scores: Scores. 1D tensor of shape [batch_size]

    Returns:
        Tensor of shape [] containing total loss for a batch of data
    """

    # Build inference Graph
    logits = ai10.inference(boards)

    # Build loss function. Total loss will be calculated using other function
    _ = ai10.loss(logits, scores)

    # Get loss for current tower only
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')

    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % ai10.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. Outer list
            is over individual gradients. Inner list is over gradient calculation
            for each tower

    Returns:
        List of (gradient, variable) tuples where gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # Add a tower dimension to average over
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        # Average over that dimension
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Return pointer to Variable b/c reasons
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train():
    """Train AI10 for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count train() calls.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate learning rate schedule
        num_batches_per_epoch = (ai10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * ai10.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(ai10.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        ai10.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)

        boards, scores = ai10.inputs(False)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [boards, scores], capacity=2 * FLAGS.num_gpus)
        # Calculate gradients for each model tower
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (ai10.TOWER_NAME, i)) as scope:
                        # Get batch
                        board_batch, score_batch = batch_queue.dequeue()

                        # Calculate loss for the tower
                        loss = tower_loss(scope, board_batch, score_batch)

                        # Reuse variables for the next tower
                        tf.get_variable_scope().reuse_variables()

                        # Retain summaries from the final tower
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate gradients for this tower
                        grads = opt.compute_gradients(loss)

                        # Add gradients to list
                        tower_grads.append(grads)

        # Synchronize gradients
        grads = average_gradients(tower_grads)

        # Track learning rate
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
                ai10.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                            'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()
