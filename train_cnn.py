from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from glob import glob
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 16, 1024, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=12,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=24,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=48,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1, 1], strides=2)

    # Dense Layer
    pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 48])
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def generate_one_hot(count, true):
    if true == 1:
        oh = 1
    else:
        oh = 0

    ohs = []
    for i in range(count):
        ohs.append(oh)

    return ohs



def normalize(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    normalised = min_max_scaler.fit_transform(data)
    return normalised


def load_data():
    noise = glob('data/noise/*')
    voice = glob('data/voice/*')
    x = []
    y = []
    for idx in range(200):
        noise_data = np.load(noise[idx])
        voice_data = np.load(voice[idx])
        x.extend(voice_data)
        y.extend(generate_one_hot(len(voice_data), 1))
        x.extend(noise_data)
        y.extend(generate_one_hot(len(noise_data), 0))
        # for i in range(voice_data.shape[0] // 16):
        #     x.append(voice_data[i*16:i*16+16])
        #     y.append(1)
        # for j in range(voice_data.shape[0] // 16):
        #     x.append(noise_data[j*16:j*16+16])
        #     y.append(0)

    x = np.array(x)
    # x = normalize(x)
    y = np.array(y)

    return x, y


def main(unused_argv):

    # Load training and eval data
    x, y = load_data()

    x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


    # Create the Estimator
    vad = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="model")

    # Set up logging for predictions
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    vad.fit(
        x=x_,
        y=y_,
        batch_size=80,
        steps=1300,
        monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    eval_results = vad.evaluate(
        x=x_test[:80], y=y_test[:80], metrics=metrics)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
