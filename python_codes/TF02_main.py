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
    Date:       2/15/18
    File:       TF02_main
    Comments:   This is the main file to run a simple CNN model based classification on
                fashion MNIST: https://github.com/zalandoresearch/fashion-mnist
    ToDo:       * Implement Tensorboard to visualize learning process results
    **********************************************************************************
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from visualization import plt_image_labels
from cnn_ops import new_weights, new_conv, new_fc
# For Pycharm IDE, avoiding a certain warning when using GPU computing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Specify data path
LOG_FOLDER = '../tensorflow_logs/TF02'

# ===== Define global variables =====
# Image related
IMG_HEIGHT = 28
IMG_WIDTH = 28
# total number of pixels in an image
IMG_TOT = IMG_HEIGHT * IMG_WIDTH
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH)
# number of classes
NUM_CLASSES = 10
# labels corresponding to the numbers
LABELS = ['t_shirt_top',  # 0
          'trouser',      # 1
          'pullover',     # 2
          'dress',        # 3
          'coat',         # 4
          'sandal',       # 5
          'shirt',        # 6
          'sneaker',      # 7
          'bag',          # 8
          'ankle_boots']  # 9

# Optimization related
LEARNING_RATE = 1e-3
BATCH_SIZE = 64  # Better to have a batch size 2^n
NUM_EPOCHS = 10  # How many times you go through the entire dataset
# Visualization related
DISPLAY_FREQ = BATCH_SIZE
# CNN related
# Number of channels
NUM_CHANNELS = 1  # Since we are using gray-scale
# 1st CNN layer info
FILTER_SIZE1 = 5
NUM_FILTERS1 = 20
# 2nd CNN layer info
FILTER_SIZE2 = 10
NUM_FILTERS2 = 20
# ===================================



def main():

    # # Show the current version of tensorflow
    print("Tensorflow version: ", tf.__version__)

    # # 1) Import and sort data
    print("\nImporting data...\n")
    from tensorflow.examples.tutorials.mnist import input_data

    data = input_data.read_data_sets('../data/fashion',
                                     source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
                                     one_hot=True)
    # Define another class labels where the true class is indicated by the integer values
    data.train.cls = np.array([label.argmax() for label in data.train.labels])
    data.validation.cls = np.array([label.argmax() for label in data.validation.labels])
    data.test.cls = np.array([label.argmax() for label in data.test.labels])

    images = data.train.images[0:16]
    class_true = data.train.cls[0:16]

    # Plot example images with labels using the function you made above
    plt_image_labels(images=images,
                     image_shape=IMG_SHAPE,
                     num_row=4,
                     num_col=4,
                     class_labels=LABELS,
                     class_true=class_true)

    # Clean the log folder (used to log results in a folder for later tensorboard usage)
    if tf.gfile.Exists(LOG_FOLDER):
        tf.gfile.DeleteRecursively(LOG_FOLDER)
    tf.gfile.MakeDirs(LOG_FOLDER)


    # # 2) Define a graph
    graph = tf.Graph()
    with graph.as_default():
        # Define place holders. These are where your input and output goes when actually computing.
        with tf.name_scope('Inputs'):
            X = tf.placeholder(tf.float32, shape=[None, IMG_TOT], name="X")
            X_image = tf.reshape(X, [-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])
            Y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="Y")
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
            model = new_fc(conv2, 'output_layer', NUM_CLASSES)
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
            num_iteration = int(len(data.train.labels) / BATCH_SIZE)
            # Define global step: a way to keep track of your trained samples over multiple epochs
            global_step = 0

            print("Start training.")
            for epoch in range(NUM_EPOCHS):
                print("Training epochs: {}".format(epoch))
                for i in range(num_iteration):
                    # Get a batch of training samples. Every time this is called within a loop,
                    # it gets the next batch.
                    x_batch, y_batch = data.train.next_batch(BATCH_SIZE)
                    global_step += 1

                    # Create a dictionary with batched samples. This will go into the optimizer to train the model.
                    feed_dict_train = {X: x_batch, Y: y_batch}

                    # Run the optimization with tensorboard summary
                    _, train_summary = sess.run([optimizer, merged], feed_dict=feed_dict_train)

                    # Show loss and accuracy with a certain display frequency
                    if i % DISPLAY_FREQ == 0:
                        train_batch_loss, train_batch_acc = sess.run([loss, accuracy], feed_dict=feed_dict_train)
                        print("iter {0:3d}:\t Loss={1:.2f},\tTraining accuracy=\t{2:.01%}".format(i,
                                                                                                train_batch_loss,
                                                                                                train_batch_acc))
                    # log results
                    writer.add_summary(train_summary, global_step)

                    # Also run validation
                    x_batch_val, y_batch_val = data.validation.next_batch(BATCH_SIZE)
                    feed_dict_val = {X: x_batch_val, Y: y_batch_val}
                    val_batch_loss, val_batch_acc, val_summary = sess.run([loss, accuracy, merged], feed_dict=feed_dict_val)
                    writer_val.add_summary(val_summary, global_step)
                    if i % DISPLAY_FREQ == 0:
                        print("iter {0:3d}:\t Loss={1:.2f},\tValidation accuracy=\t{2:.01%}".format(i,
                                                                                                val_batch_loss,
                                                                                                val_batch_acc))
                # Just for better visualization on logs
                print("------------------------------")

            print("Training finished.")
            print("------------------")
            print("\nStart testing.")
            # We are going to feed all the test samples
            feed_dict_test = {X: data.test.images,
                              Y: data.test.labels,
                              Y_true_class: data.test.cls}

            # Calculate the accuracy
            acc, true_label, pred_label = sess.run([accuracy, correct_prediction, Y_pred_class],
                                                   feed_dict=feed_dict_test)

            # Close tensorflow session
            sess.close()

        # Print the accuracy
        print("----------------------------------")
        print("Test accuracy: {0:.1%}".format(acc))
        print("----------------------------------")


if __name__ == '__main__':
    main()