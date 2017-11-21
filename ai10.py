# Most of this borrowed from TensorFlow's CIFAR-10
# tutorial, check it out at https://www.tensorflow.org/tutorials/deep_cnn

import tensorflow as tf
import argparse
import os
import sys
import re

import ai10_input

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1024,
                    help='Number of boards to process in a batch.')

parser.add_argument('--data_dir', type=str, default='./train_files',
                    help='Path to the othello data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

FLAGS = parser.parse_args()

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

TOWER_NAME = 'tower'

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                         tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns:
    Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inputs(eval_data):
    """Construct input for Othello evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    boards, scores = ai10_input.inputs(eval_data=eval_data,
                                         data_dir=FLAGS.data_dir,
                                         batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        boards = tf.cast(boards, tf.float16)
        scores = tf.cast(scores, tf.float16)
    return boards, scores
    
def inference(boards):
    """Build the AI10 model

    Args:
        boards: Boards returned from inputs()

    Returns:
     Logits.
    """

    # Network layout:
    # conv4x4
    #    |    
    #    |      full8x8board
    #    |          |
    #    +----------+  
    #          |
    #   +------+
    #   |
    # relu3 (200)
    #   |
    # relu4 (100)
    #   |
    # logi5 (1)
    with tf.variable_scope('conv4x4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[4, 4, 1, 128],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(boards, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4x4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv4x4)


    with tf.variable_scope('relu3') as scope:
        reshape_a = tf.reshape(conv4x4, [FLAGS.batch_size, -1])
        reshape_b = tf.reshape(boards, [FLAGS.batch_size, -1])
        concat = tf.concat([reshape_a, reshape_b], 1)
        dim = concat.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 200],
                                                stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [200], tf.constant_initializer(0.1))
        relu3 = tf.nn.relu(tf.matmul(concat, weights) + biases, name=scope.name)
        _activation_summary(relu3)

    with tf.variable_scope('relu4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[200, 100],
                                                stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [100], tf.constant_initializer(0.1))
        relu4 = tf.nn.relu(tf.matmul(relu3, weights) + biases, name=scope.name)
        _activation_summary(relu4)

    with tf.variable_scope('logi5') as scope:
        weights = _variable_with_weight_decay('weights', [100, 1],
                                                stddev=1/100.0, wd=None)
        biases = _variable_on_cpu('biases', [1],
                                    tf.constant_initializer(0.0))

        # logi5 = tf.multiply(
        #    tf.subtract(
        #        tf.nn.sigmoid(
        #            tf.divide(
        #                tf.matmul(relu4, weights) + biases,
        #                32
        #            )
        #        ),
        #        0.5
        #    ),
        #    128, name=scope.name
        #)
        # ^^^ is what you do to get actual score
        # this code used signmoid_cross_entropy_with_logits for better results
        logi5 = tf.matmul(relu4, weights) + biases
        
        _activation_summary(logi5)

    return logi5

def loss(logits, scores):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from nputs(). 1-D tensor
            of shape [batch_size]

    Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=scores,
        logits=tf.reshape(logits, [-1]),
        name='cross_entropy_per_example'
    )
    print(cross_entropy)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    print(cross_entropy_mean)
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    """Add summaries for losses in AI10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step):
    """Train AI10 model.

    Create optimizer, moving averages, yadda yadda

    Args:
        total_loss: Total loss from loss()
        global_step: Integer counting the number of training steps
            processed.
    Returns:
        train_op: op for training
    """
    num_batches_per_epoch = ai10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradients_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
