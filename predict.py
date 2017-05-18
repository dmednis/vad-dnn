import tensorflow as tf
import numpy as np

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./model/model")
    # tf.set_
    # predictions = sess.run(pred, )
