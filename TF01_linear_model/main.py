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
    Date:       1/10/18
    File:       main
    Comments:   This is the main file to run linear model based classification on
                fashion MNIST: https://github.com/zalandoresearch/fashion-mnist
    ToDo:       * Implement Tensorboard to visualize learning process results
    **********************************************************************************
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from visualization import plt_image_labels
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
BATCH_SIZE = 50
NUM_EPOCHS = 500
# ===================================

# Import data
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

# Plot using the function you made above
plt_image_labels(images=images, num_row=4, num_col=4, class_true=class_true)

hello = tf.constant("WTF")
sess = tf.Session()
print(sess.run(hello))