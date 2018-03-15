"""
    Copyright 2017-2022 Department of Electrical and Computer Engineering
    University of Houston, TX/USA

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    Please contact Sho Nakagome for more info about licensing snakagome@uh.edu
    via github issues section.

    **********************************************************************************
    Author:     Sho Nakagome
    Date:       3/15/18
    File:       TF04_main
    Comments:   This is the main file to run a ResNet model on classifing plant seedlings.
    Reference:  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/layers/cnn_mnist.py
                https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py
    ToDo:       * Quantify the results
                * Explore more and provide more insights in the data
    **********************************************************************************
"""
import tensorflow as tf
import tflearn
import numpy as np
import pandas as pd
from subprocess import check_output
# For Pycharm IDE, avoiding a certain warning when using GPU computing
import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# Specify data path
LOG_FOLDER = '../tensorflow_logs/TF04'
# Specify training path
TRAIN_PATH = '../data/plant_seedlings_classification/train'
# Specify test path
TEST_PATH = '../data/plant_seedlings_classification/test'
# Specify model directory
MODEL_PATH = '../models/TF04'

# ===== Define global variables =====
# Whether to delete previous model or not
# If you want to train from the beginning, just make this True
DELETE_MODEL = True

# Image related (resizing image dimensions)
IMG_WIDTH = 32
IMG_HEIGHT = 32

# Number of classes
NUM_CLASS = 12

# Optimization related
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Visualization related
DISPLAY_FREQ = BATCH_SIZE

# Model related
# Number of channels
NUM_CHANNELS = 3
# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
N = 5
# ===================================


# tf.logging.set_verbosity(tf.logging.INFO)


# Helper functions (Importing preprocessing data)
def input_parser(img_path, label):
    # convert the label to one-hot encoding
    # one_hot = tf.one_hot(label, NUM_CLASS)

    # read the image from file
    img_file = tf.read_file(img_path)

    # there's other methods: decode_png, decode_jpeg etc. but this one is more robust.
    # one downside is that you need the next line to assign shape
    img_decoded = tf.image.decode_image(img_file, channels=NUM_CHANNELS)

    # you need this line if you use decode_image because it won't assign shape for you
    img_decoded.set_shape([None, None, None])

    # reshape to your intended image size
    img_resized = tf.image.resize_images(img_decoded, [IMG_HEIGHT, IMG_WIDTH])

    return img_resized, label

def train_preprocess(image, label):
    # randomly flip an image horizontally
    image = tf.image.random_flip_left_right(image)

    # adjust the brightness by a random factor
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)

    # adjust the saturation of an RGB image by a random factor
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # linearly scales image to have zero mean and unit norm
    image = tf.image.per_image_standardization(image)

    # clip tensor values to a specified min and max
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

# Model function
# using Estimator, you define the following 5 elements in the model function
# 1) Model itself
# 2) Predicting
# 3) Loss function
# 4) Training
# 5) Evaluating
def resnet_model_fn(features, labels, mode):
    # Define: Input layer
    # based on your specified size, reshape the input to 4-D tensor:
    # [batch_size, image_width, image_height, channels]
    input_layer = tf.reshape(features["x"], [-1, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])

    # ResNet using tflearn
    net = tflearn.conv_2d(input_layer, 16, 3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.residual_block(net, N, 16)
    net = tflearn.residual_block(net, 1, 32, downsample=True)
    net = tflearn.residual_block(net, N - 1, 32)
    net = tflearn.residual_block(net, 1, 64, downsample=True)
    net = tflearn.residual_block(net, N - 1, 64)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, NUM_CLASS]
    logits = tf.layers.dense(inputs=dropout, units=NUM_CLASS)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    # This is necessary for model function. Every model function has to get "mode"
    # as an argument to switch beteern TRAIN, EVAL, PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):

    # # Show the current version of tensorflow
    print("Tensorflow version: ", tf.__version__)

    # # 1) Import and sort data
    print("\nLoading data info...\n")

    # Get subfolder names and create labels and inverse label dictionaries
    classes = check_output(["ls", TRAIN_PATH]).decode("utf8").strip().split("\n")
    labels = {classes[i]: i for i in range(0, len(classes))}
    inv_labels = {val: key for key, val in labels.items()}

    # Create a list of filenames per class in subfolder
    train_list = []
    for c in classes:
        files = check_output(["ls", TRAIN_PATH + '/%s' % c]).decode("utf8").strip().split("\n")
        train_list.append(files)

    filenames = []
    filelabels = []
    for c, files in zip(classes, train_list):
        for file in files:
            filelabels.append(labels[c])
            filenames.append(TRAIN_PATH + '/' + c + '/' + file)

    print("\nFinished loading data info.\n")

    # Clean the log folder (used to log results in a folder for later tensorboard usage)
    if tf.gfile.Exists(LOG_FOLDER):
        tf.gfile.DeleteRecursively(LOG_FOLDER)
    tf.gfile.MakeDirs(LOG_FOLDER)

    # Clean the model folder
    if DELETE_MODEL:
        if tf.gfile.Exists(MODEL_PATH):
            tf.gfile.DeleteRecursively(MODEL_PATH)
        tf.gfile.MakeDirs(MODEL_PATH)

    # Make sure that you never call this before creating a model
    # since it requires that you have defined a graph already.
    init = tf.global_variables_initializer()

    # merge all the summaries for tensorboard
    merged = tf.summary.merge_all()

    with tf.Session() as sess:

        print("-------------------------------")
        print("Initializing all the variables.")
        sess.run(init)
        writer = tf.summary.FileWriter(LOG_FOLDER + "/train", tf.Graph())

        # Create Dataset object in tensorflow
        dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
        dataset = dataset.shuffle(len(filenames))
        dataset = dataset.map(input_parser, num_parallel_calls=32)
        dataset = dataset.map(train_preprocess, num_parallel_calls=32)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(1)

        # Create tensorflow iterator object
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        next_element = iterator.get_next()

        # create two initialization ops to switch between the datasets
        training_init_op = iterator.make_initializer(dataset)

        sess.run(training_init_op)

        # Run the optimization with tensorboard summary
        x_batch, y_batch = sess.run(next_element)

        assert not np.any(np.isnan(x_batch))

        # Create the Estimator
        plant_classifier = tf.estimator.Estimator(
            model_fn=resnet_model_fn, model_dir=MODEL_PATH)

        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x_batch},
            y=y_batch,
            batch_size=BATCH_SIZE,
            num_epochs=None,
            shuffle=True)
        plant_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])

        a=1

    #
    # # # 2) Define a graph
    # graph = tf.Graph()
    # with graph.as_default():
    #
    #
    #     # # 3) Running tensorflow session
    #     # Make sure that you never call this before creating a model
    #     # since it requires that you have defined a graph already.
    #     init = tf.global_variables_initializer()
    #
    #     # merge all the summaries for tensorboard
    #     merged = tf.summary.merge_all()
    #
    #     # Running a tensorflow session
    #     with tf.Session() as sess:
    #         print("-------------------------------")
    #         print("Initializing all the variables.")
    #         sess.run(init)
    #         writer = tf.summary.FileWriter(LOG_FOLDER + "/train", graph)
    #         writer_val = tf.summary.FileWriter(LOG_FOLDER + "/validation", graph)
    #         # Calculate the number of iterations needed based on your batch size
    #         num_iteration = int(len(filenames) / BATCH_SIZE)
    #         # Define global step: a way to keep track of your trained samples over multiple epochs
    #         global_step = 0
    #
    #         # Create Dataset object in tensorflow
    #         dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
    #         dataset = dataset.shuffle(len(filenames))
    #         dataset = dataset.map(input_parser, num_parallel_calls=32)
    #         dataset = dataset.map(train_preprocess, num_parallel_calls=32)
    #         dataset = dataset.batch(BATCH_SIZE)
    #         dataset = dataset.prefetch(1)
    #
    #         # Create tensorflow iterator object
    #         iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    #         next_element = iterator.get_next()
    #
    #         # create two initialization ops to switch between the datasets
    #         training_init_op = iterator.make_initializer(dataset)
    #
    #         print("Start training.")
    #         for epoch in range(NUM_EPOCHS):
    #             print("Training epochs: {}".format(epoch))
    #             for i in range(num_iteration):
    #                 # Get a batch of training samples. Every time this is called within a loop,
    #                 # it gets the next batch.
    #                 global_step += 1
    #
    #                 sess.run(training_init_op)
    #
    #                 # Run the optimization with tensorboard summary
    #                 x_batch, y_batch = sess.run(next_element)
    #                 feed_dict_train = {X_image: x_batch, Y: y_batch}
    #                 _, train_summary = sess.run([optimizer, merged], feed_dict=feed_dict_train)
    #
    #                 # Show loss and accuracy with a certain display frequency
    #                 if i % DISPLAY_FREQ == 0:
    #                     train_batch_loss, train_batch_acc = sess.run([loss, accuracy], feed_dict=feed_dict_train)
    #                     print("iter {0:3d}:\t Loss={1:.2f},\tTraining accuracy=\t{2:.01%}".format(i,
    #                                                                                               train_batch_loss,
    #                                                                                               train_batch_acc))
    #                 # log results
    #                 writer.add_summary(train_summary, global_step)
    #
    #             # Just for better visualization on logs
    #             print("------------------------------")
    #
    #         print("Training finished.")
    #         print("------------------")
    #
    #         # Close tensorflow session
    #         sess.close()
    #
    # print("\nFinished training.\n")


# Just to make sure you start running from the main function
if __name__ == '__main__':
    # main()
    tf.app.run()