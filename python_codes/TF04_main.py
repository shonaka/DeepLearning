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
    Comments:   This is the main file to run a CNN model on classifing plant seedlings.
                Main intension is to explain how to use tf.data.Dataset and tf.estimator APIs.
    Reference:  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/layers/cnn_mnist.py
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
IMG_WIDTH = 64
IMG_HEIGHT = 64

# Number of classes
NUM_CLASS = 12

# Optimization related
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 500

# Visualization related
DISPLAY_FREQ = BATCH_SIZE

# Model related
# Number of channels
NUM_CHANNELS = 3
# Which optimizer to use
OPTIMIZER = "sgd"
# ===================================

# Helper functions (Importing preprocessing data)
def input_parser(img_path, label):
    # convert the label to one-hot encoding
    one_hot = tf.one_hot(label, NUM_CLASS)

    # read the image from file
    img_file = tf.read_file(img_path)

    # there's other methods: decode_png, decode_jpeg etc. but this one is more robust.
    # one downside is that you need the next line to assign shape
    img_decoded = tf.image.decode_image(img_file, channels=NUM_CHANNELS)

    # you need this line if you use decode_image because it won't assign shape for you
    img_decoded.set_shape([None, None, None])

    # reshape to your intended image size
    img_resized = tf.image.resize_images(img_decoded, [IMG_HEIGHT, IMG_WIDTH])

    return img_resized, one_hot


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


def imgs_input_fn(filenames, filelabels, batch_size, do_shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
    # only for training, not for testing
    if do_shuffle:
        dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(input_parser, num_parallel_calls=32)
    dataset = dataset.map(train_preprocess, num_parallel_calls=32)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset


def custom_conv(net, num_filter, kernel_size):
    net = tflearn.conv_2d(net, num_filter, kernel_size)
    net = tflearn.batch_normalization(net)

    return net


def simpleCNN(optimizer):
    net = tflearn.input_data(shape=[None, IMG_WIDTH, IMG_HEIGHT, 3])
    net = custom_conv(net, 32, 3)
    net = tflearn.max_pool_2d(net, 3, strides=2)
    net = custom_conv(net, 64, 3)
    net = custom_conv(net, 64, 3)
    net = tflearn.max_pool_2d(net, 3, strides=2)
    net = custom_conv(net, 128, 3)
    net = custom_conv(net, 128, 3)
    net = tflearn.max_pool_2d(net, 3, strides=2)
    net = custom_conv(net, 256, 3)
    net = custom_conv(net, 256, 3)
    net = custom_conv(net, 256, 3)
    net = tflearn.fully_connected(net, 1024, activation="relu")

    # Regression
    net = tflearn.fully_connected(net, NUM_CLASS, activation='softmax')
    net = tflearn.regression(net, optimizer=optimizer,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=3,
                        tensorboard_dir=LOG_FOLDER)

    return model


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

    # Run through the directory and subfolders to get all the images and labels
    filenames = []
    filelabels = []
    for c, files in zip(classes, train_list):
        for file in files:
            filelabels.append(labels[c])
            filenames.append(TRAIN_PATH + '/' + c + '/' + file)

    # Get images and labels from test set too
    test_list = os.listdir(TEST_PATH)
    test_images = []
    names = []
    for image in test_list:
        names.append(image)
        test_images.append(TEST_PATH + '/' + image)

    # just give dammy labels for test set now
    test_labels = np.zeros([len(test_images)]).astype(int)

    print("\nFinished loading data info.\n")
    print("-------------------------------")

    # Clean the log folder (used to log results in a folder for later tensorboard usage)
    if tf.gfile.Exists(LOG_FOLDER):
        print("Cleaning the log folder")
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


    with tf.Session() as sess:
        print("-------------------------------")
        print("Initializing all the variables.")
        sess.run(init)

        # get next element training data
        dataset = imgs_input_fn(filenames, filelabels, BATCH_SIZE)

        # Create tensorflow iterator object
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        next_iterator = iterator.get_next()

        # need to create initializer for both training and testing
        training_init_op = iterator.make_initializer(dataset)

        # get actual batch of images and labels
        sess.run(training_init_op)
        x_batch, y_batch = sess.run(next_iterator)

        # Building Simple CNN Network
        model = simpleCNN(OPTIMIZER)

        # Training
        print("Start training...")
        model.fit(x_batch, y_batch, n_epoch=NUM_EPOCHS, validation_set=0.1,
                  snapshot_epoch=False, snapshot_step=500,
                  show_metric=True, batch_size=BATCH_SIZE, shuffle=True,
                  run_id='cnn_plant')
        print("\nFinished training.\n")

        # same for test data
        test_dataset = \
            imgs_input_fn(test_images, test_labels, len(names), do_shuffle=False)
        test_init_op = iterator.make_initializer(test_dataset)

        # do the same for test set, but there's no labels
        # and want to get all the data for testing
        sess.run(test_init_op)
        testX, _ = sess.run(next_iterator)

        # Testing
        print("Start testing...")
        prediction = model.predict(testX)
        test_len = prediction.shape[0]
        prediction_ind = np.argmax(prediction, axis=-1)
        predicted_labels = [inv_labels[prediction_ind[i]] for i in range(test_len)]

        df = pd.DataFrame(data={'file': names, 'species': predicted_labels})
        df.to_csv('../Results/plant_seedlings_results.csv', index=False)


# Just to make sure you start running from the main function
if __name__ == '__main__':
    # main()
    tf.app.run()
