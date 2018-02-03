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
    Date:       2/02/18
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
# For Pycharm IDE, avoiding a certain warning when using GPU computing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Specify data path
LOG_FOLDER = '../TF02/tensorflow_logs'

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
NUM_EPOCHS = BATCH_SIZE * 10
# CNN related
# Number of channels
NUM_CHANNELS = 1  # Since we are using gray-scale
# 1st CNN layer info
FILTER_SIZE1 = 5
NUM_FILTERS1 = 20
# 2nd CNN layer info
FILTER_SIZE2 = 10
NUM_FILTERS2 = 20
# Last fully-connected layer info
FC_SIZE = 128
# ===================================


# Some helper functions to construct a simple CNN
def new_weights(name, shape):
    """
    Defining a new weights based on the name and the shape.
    Initialized with Xavier initializer

    :param name: name of the new defined weights
    :param shape: shape of the new defined filter
    """

    W = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    return W

def simpleCNN(X):
    """
    Plot images with true labels. If specified, predicted class too.

    :param images: the image data itself, containing pixel data.
    :param num_row: how many number of rows you want to plot.
    :param num_col: how many number of columns you want to plot.
    :param class_true: the true class labels for the images you give as the first input.
    :param class_pred: the predicted class labels. If not specified, it does not plot the labels.
    """

    # Get new weights
    W1 = new_weights("W1", [FILTER_SIZE1, FILTER_SIZE1, NUM_CHANNELS, NUM_FILTERS1])
    W2 = new_weights("W2", [FILTER_SIZE2, FILTER_SIZE2, NUM_FILTERS1, NUM_FILTERS2])

    # CONV2D: Define your first CNN layer
    Z1 = tf.nn.conv2d(input=X,
                      filter=W1,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # CONV2D: 2nd CNN layer
    Z2 = tf.nn.conv2d(input=A1,
                      filter=W2,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # FLATTEN (from a convolutional volume, we are flatting it out to a single vector like layer)
    P = tf.contrib.layers.flatten(A2)
    # FULLY-CONNECTED layer
    # We have 10 neurons in the output layer because we have 10 classes to classify.
    final_output = tf.contrib.layers.fully_connected(P, 10, activation_fn=None)

    return final_output



def main():

    # # Show the current version of tensorflow
    print(tf.__version__)

    # # 1) Import and sort data
    from tensorflow.examples.tutorials.mnist import input_data

    data = input_data.read_data_sets('data/fashion',
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
        X = tf.placeholder(tf.float32, shape=[None, IMG_TOT], name="X")
        X_image = tf.reshape(X, [-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])
        Y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="Y")
        Y_true_class = tf.argmax(Y, axis=1, name="Y_true_class")

        # Defining a simple CNN model using the function defined before
        model = simpleCNN(X_image)

        # Use softmax layer to normalize the output and then get the highest number to determine the class
        Y_pred = tf.nn.softmax(model)
        Y_pred_class = tf.argmax(Y_pred, axis=1)

        # Define loss function. Least squared error
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y),
                              name="cross_entropy")

        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        global_step = tf.train.get_or_create_global_step(graph)
        training_operation = optimizer.minimize(cost,
                                                global_step=global_step,
                                                name='minimizer')

        correct_prediction = tf.equal(Y_pred_class, Y_true_class)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Attach summary writers (for tensorboard):
        with tf.name_scope("train") as scope:
            loss_train = tf.summary.scalar('train_loss', cost)
            acc_summary_train = tf.summary.scalar('train_acc', accuracy)
        with tf.name_scope("validate") as scope:
            loss_val = tf.summary.scalar('validation_loss', cost)
            acc_summary_val = tf.summary.scalar('validation_acc', accuracy)
        writer = tf.summary.FileWriter(LOG_FOLDER, graph)

        # # 3) Running tensorflow session
        # Make sure that you never call this before creating a model
        # since it requires that you have defined a graph already.
        init = tf.global_variables_initializer()

        # Running a tensorflow session
        with tf.Session() as sess:
            print("Initializing all the variables.")
            sess.run(init)

            print("Start training.")
            for i in range(NUM_EPOCHS):
                # Get a batch of training samples. Every time this is called within a loop,
                # it gets the next batch.
                x_batch, y_batch = data.train.next_batch(BATCH_SIZE)
                x_batch_val, y_batch_val = data.validation.next_batch(BATCH_SIZE)

                # Create a dict with batched samples. This will go into the optimizer to train the model.
                feed_dict_train = {X: x_batch, Y: y_batch}
                feed_dict_validation = {X: x_batch_val, Y: y_batch_val}

                # Run the optimization and calculate the loss
                result_l, train_loss_summary, _ = \
                    sess.run([cost, loss_train, training_operation], feed_dict=feed_dict_train)
                mean_loss = np.mean(result_l)

                # Also calculate the training accuracy
                y_batch_class = np.array([label.argmax() for label in y_batch])
                feed_dict_train_no_one_hot = {X: x_batch, Y_true_class: y_batch_class}
                acc_train, train_acc_summary = \
                    sess.run([accuracy, acc_summary_train], feed_dict=feed_dict_train_no_one_hot)

                # log results
                writer.add_summary(train_loss_summary, i)
                writer.add_summary(train_acc_summary, i)

                # Do the same for the validation set
                result_l_val, val_loss_summary, _ = \
                    sess.run([cost, loss_val, training_operation], feed_dict=feed_dict_validation)
                mean_loss_val = np.mean(result_l_val)

                y_batch_class_val = np.array([label.argmax() for label in y_batch_val])
                feed_dict_val_no_one_hot = {X: x_batch_val, Y_true_class: y_batch_class_val}
                acc_train, val_acc_summary = \
                    sess.run([accuracy, acc_summary_val], feed_dict=feed_dict_val_no_one_hot)

                writer.add_summary(val_loss_summary, i)
                writer.add_summary(val_acc_summary, i)

                # Print the loss and the training accuracy every 10 iteration
                # Detailed logs will be displayed in tensorboard
                if i % BATCH_SIZE == 0:
                    print("Loss function: %.4f" % mean_loss)
                    print("Training accuracy: {0:>6.1%}".format(acc_train))

            print("Training finished.")

            print("\nStart testing.")
            # We are going to feed all the test samples
            feed_dict_test = {X: data.test.images,
                              Y: data.test.labels,
                              Y_true_class: data.test.cls}

            # Calculate the accuracy
            acc, true_label, pred_label = sess.run([accuracy, correct_prediction, Y_pred_class],
                                                   feed_dict=feed_dict_test)

        # Print the accuracy
        print("Test accuracy: {0:.1%}".format(acc))


if __name__ == '__main__':
    main()