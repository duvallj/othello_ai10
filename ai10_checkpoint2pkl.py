import pickle
import argparse

import tensorflow as tf

import ai10

parser = argparse.ArgumentParser(parents=[ai10.parser])

parser.add_argument('--checkpoint_dir', type=str, default='./ai10_train',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--output_file', type=str, default='./networkA.pkl',
                    help='File to write the collected variables from.')

layers = ['relu2', 'relu3', 'relu4', 'logi5']
subvars = ['weights', 'biases']

def recall_variable(sess, scopename, varname, vardict):
    with tf.variable_scope(layers[scopename], reuse=True) as scope:
        var = tf.get_variable(subvars[varname])
        val = sess.run(var)
        print(scopename, varname, val.shape, val.dtype)
        vardict[scopename][varname] = val

def main(argv=None):
    vardict = [[None for x in subvars] for y in layers]
    with tf.Graph().as_default() as g:
        boards, scores = ai10.inputs(True)

        logits = ai10.inference(boards)
        avg_error = tf.abs(logits - tf.reshape(scores, [-1, 1]))
        
        variable_averages = tf.train.ExponentialMovingAverage(
            ai10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            for layer in range(len(layers)):
                recall_variable(sess, layer, 0, vardict) # Weights
                recall_variable(sess, layer, 1, vardict) # Biases

    vardict = tuple(tuple(x) for x in vardict)
    varfile = open(FLAGS.output_file, 'wb')
    pickle.dump(vardict, varfile)
    varfile.close()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()
