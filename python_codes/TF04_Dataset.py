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
    Date:       3/11/18
    File:       TF04_main
    Comments:   This is the main file to run a ResNet model on classifing plant seedlings.
    ToDo:       * Quantify the results
                * Explore more and provide more insights in the data
    **********************************************************************************
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from subprocess import check_output
from cnn_ops import *
# For Pycharm IDE, avoiding a certain warning when using GPU computing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Specify data path
LOG_FOLDER = '../tensorflow_logs/TF04'
# Specify training path
TRAIN_PATH = '../data/plant_seedlings_classification/train'
# Specify test path
TEST_PATH = '../data/plant_seedlings_classification/test'

# ===== Define global variables =====
# Image related (resizing image dimensions)
IMG_HEIGHT = 32
IMG_WIDTH = 32

# Number of classes
NUM_CLASS = 12

# Optimization related
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Visualization related
DISPLAY_FREQ = BATCH_SIZE

# CNN related
# Number of channels
NUM_CHANNELS = 3  # Since we are using gray-scale
# 1st CNN layer info
FILTER_SIZE1 = 5
NUM_FILTERS1 = 20
# 2nd CNN layer info
FILTER_SIZE2 = 10
NUM_FILTERS2 = 20
# ===================================

def input_parser(img_path, label):
    # convert the label to one-hot encoding
    one_hot = tf.one_hot(label, NUM_CLASS)

    # read the image from file
    img_file = tf.read_file(img_path)

    img_decoded = tf.image.decode_png(img_file, channels=NUM_CHANNELS)

    img_resized = tf.image.resize_images(img_decoded, [IMG_HEIGHT, IMG_WIDTH])

    return img_resized, one_hot

def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def main():

    # # Show the current version of tensorflow
    print("Tensorflow version: ", tf.__version__)

    # # 1) Import and sort data
    print("\nLoading data info...\n")

    # Get subfolder names and create labels and inverse label dictionaries
    classes = check_output(["ls", TRAIN_PATH]).decode("utf8").strip().split("\n")
    labels = {classes[i]: i for i in range(0, len(classes))}
    inv_labels = {val: key for key, val in labels.items()}

    # Create a list of filenames per classes in subfolder
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


    # # 2) Define a graph
    graph = tf.Graph()
    with graph.as_default():
        # Define place holders. These are where your input and output goes when actually computing.
        with tf.name_scope('Inputs'):
            X_image = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS], name="X")
            Y = tf.placeholder(tf.float32, [None, NUM_CLASS], name="Y")
            Y_true_class = tf.argmax(Y, axis=1, name="Y_true_class")
            # Pass images to tensorboard for visualization (only 3 images)
            tf.summary.image('input_images', X_image, 3)

        # Defining a simple CNN model using functions defined before
        with tf.name_scope('Model'):
            # shape of the first layer's filter
            shape1 = [FILTER_SIZE1, FILTER_SIZE1, NUM_CHANNELS, NUM_FILTERS1]
            # make the first layer
            conv1 = new_conv(X_image, shape1, 'conv1')
            # shape of the second layer's filter
            shape2 = [FILTER_SIZE2, FILTER_SIZE2, NUM_FILTERS1, NUM_FILTERS2]
            # make the second layer
            conv2 = new_conv(conv1, shape2, 'conv2')
            # add to fully-connected layer
            model = new_fc(conv2, 'output_layer', NUM_CLASS)
            # Use softmax layer to normalize the output and then get the highest number to determine the class
            Y_pred = tf.nn.softmax(model)
            Y_pred_class = tf.argmax(Y_pred, axis=1)

        # Define loss function
        with tf.name_scope('Loss'):
            # Using cross entropy to calculate the loss
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y), name="cross_entropy")
            # Log loss in tensorboard
            tf.summary.scalar('loss', loss)

        with tf.name_scope('Optimizer'):
            # Here we are using the Adam optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='Adam_optimizer').minimize(loss)

        with tf.name_scope('Accuracy'):
            # Calculating the accuracy by comparing with the true labels
            correct_prediction = tf.equal(Y_pred_class, Y_true_class)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # Log accuracy in tensorboard
            tf.summary.scalar('accuracy', accuracy)

        # # 3) Running tensorflow session
        # Make sure that you never call this before creating a model
        # since it requires that you have defined a graph already.
        init = tf.global_variables_initializer()

        # merge all the summaries for tensorboard
        merged = tf.summary.merge_all()

        # Running a tensorflow session
        with tf.Session() as sess:
            print("-------------------------------")
            print("Initializing all the variables.")
            sess.run(init)
            writer = tf.summary.FileWriter(LOG_FOLDER + "/train", graph)
            writer_val = tf.summary.FileWriter(LOG_FOLDER + "/validation", graph)
            # Calculate the number of iterations needed based on your batch size
            num_iteration = int(len(filenames) / BATCH_SIZE)
            # Define global step: a way to keep track of your trained samples over multiple epochs
            global_step = 0

            # Create Dataset object in tensorflow
            dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
            dataset = dataset.shuffle(len(filenames))
            dataset = dataset.map(input_parser, num_parallel_calls=4)
            dataset = dataset.map(train_preprocess, num_parallel_calls=4)
            dataset = dataset.batch(BATCH_SIZE)
            dataset = dataset.prefetch(1)

            # Create tensorflow iterator object
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            next_element = iterator.get_next()

            # create two initialization ops to switch between the datasets
            training_init_op = iterator.make_initializer(dataset)

            print("Start training.")
            for epoch in range(NUM_EPOCHS):
                print("Training epochs: {}".format(epoch))
                for i in range(num_iteration):
                    # Get a batch of training samples. Every time this is called within a loop,
                    # it gets the next batch.
                    global_step += 1

                    sess.run(training_init_op)

                    # Run the optimization with tensorboard summary
                    x_batch, y_batch = sess.run(next_element)
                    feed_dict_train = {X_image: x_batch, Y: y_batch}
                    _, train_summary = sess.run([optimizer, merged], feed_dict=feed_dict_train)

                    # Show loss and accuracy with a certain display frequency
                    if i % DISPLAY_FREQ == 0:
                        train_batch_loss, train_batch_acc = sess.run([loss, accuracy], feed_dict=feed_dict_train)
                        print("iter {0:3d}:\t Loss={1:.2f},\tTraining accuracy=\t{2:.01%}".format(i,
                                                                                                  train_batch_loss,
                                                                                                  train_batch_acc))
                    # log results
                    writer.add_summary(train_summary, global_step)

                # Just for better visualization on logs
                print("------------------------------")

            print("Training finished.")
            print("------------------")

            # Close tensorflow session
            sess.close()

    print("\nFinished training.\n")


# Just to make sure you start running from the main function
if __name__ == '__main__':
    main()