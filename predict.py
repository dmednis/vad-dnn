import tensorflow as tf
import numpy as np

data = np.load('data/audio4.npy')
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./model/model")
    # tf.set_
    softmax_tensor = sess.graph.get_tensor_by_name('final_tensor:0')
    predictions = sess.run(softmax_tensor, {x: data})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score=%.5f)' % (human_string, score))