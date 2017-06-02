import tensorflow as tf
import numpy as np
from random import randint
from network import nn_layer


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels), dtype=complex)
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def train():
    features = np.load('data/audio-3.npy')
    labels = one_hot_encode(np.load('data/vad-3.npy'))
    test_features = np.load('data/audio4.npy')
    test_labels = one_hot_encode(np.load('data/vad4.npy'))

    # Parameters
    learning_rate = 0.005
    training_epochs = 200
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_hidden_1 = 512 # 1st layer number of features
    n_hidden_2 = 256 # 2nd layer number of features
    n_hidden_3 = 128 # 2nd layer number of features
    n_input = features.shape[1]
    sd = 1 / np.sqrt(n_input)
    n_classes = 2

    print("Input shape: ", features.shape)
    print("Input size: ", n_input)
    print("H1 size: ", n_hidden_1)
    print("H2 size: ", n_hidden_2)
    print("H3 size: ", n_hidden_3)
    print("Output size: ", n_classes)

    sess = tf.InteractiveSession()

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y_ = tf.placeholder("float", [None, n_classes])

    hidden1 = nn_layer(x, n_input, n_hidden_1, 'layer1', act=tf.nn.tanh, sd=sd)
    hidden2 = nn_layer(hidden1, n_hidden_1, n_hidden_2, 'layer2', act=tf.nn.sigmoid, sd=sd)
    hidden3 = nn_layer(hidden2, n_hidden_2, n_hidden_3, 'layer3', sd=sd)

    # with tf.name_scope('dropout'):
    #     keep_prob = tf.placeholder(tf.float32)
    #     tf.summary.scalar('dropout_keep_probability', keep_prob)
    #     dropped = tf.nn.dropout(hidden3, keep_prob)

    # y = nn_layer(dropped, n_hidden_3, n_classes, 'layer4', act=tf.identity, sd=sd)
    y = nn_layer(hidden3, n_hidden_3, n_classes, 'layer4', act=tf.nn.softmax, sd=sd, name='final_tensor')


    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./log/train', sess.graph)
    test_writer = tf.summary.FileWriter('./log/test')

    # Initializing the variables
    tf.global_variables_initializer().run()

    def feed_dict(_train):
        if _train:
            rnd = randint(0, len(features) - batch_size)
            batch_x = features[rnd:rnd + batch_size]
            batch_y = labels[rnd:rnd + batch_size]
            k = 0.9
        else:
            batch_x = test_features
            batch_y = test_labels
            k = 1.0
        # return {x: batch_x, y_: batch_y, keep_prob: k}
        return {x: batch_x, y_: batch_y}

    # Training cycle
    for epoch in range(training_epochs):
        if epoch % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, epoch)
            print('Accuracy at step %s: %s' % (epoch, acc))
        else:  # Record train set summaries, and train
            if epoch % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
                train_writer.add_summary(summary, epoch)
                print('Adding run metadata for', epoch)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, epoch)
    train_writer.close()
    test_writer.close()

    saver = tf.train.Saver()
    saver.save(sess, "./model/model")


if tf.gfile.Exists('./log'):
    tf.gfile.DeleteRecursively('./log')
tf.gfile.MakeDirs('./log')
train()

