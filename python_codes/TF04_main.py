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
import tflearn
from tflearn.data_utils import image_preloader
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# For Pycharm IDE, avoiding a certain warning when using GPU computing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Specify data path
LOG_FOLDER = '../tensorflow_logs/TF03'
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

# Labels
LABELS = {
    0: 'Black-glass',
    1: 'Charlock',
    2: 'Cleavers',
    3: 'Common Chickwead',
    4: 'Common wheat',
    5: 'Fat Hen',
    6: 'Loose Silky-bent',
    7: 'Maize',
    8: 'Scentless Mayweed',
    9: 'Shepherds Purse',
    10: 'Small=flowered Cranesbill',
    11: 'Sugar beet'}

# Optimization related
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 10

# ResNet related
# 32 layers: N=5, 56 layers: N=9, 110 layers: N=18
N = 5
# ===================================

def main():

    # # Show the current version of tensorflow
    print("Tensorflow version: ", tf.__version__)

    # # 1) Import and sort data
    print("\nImporting data...\n")

    trainX, trainY = image_preloader(TRAIN_PATH,
                                     mode='folder',
                                     image_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                     categorical_labels=True,
                                     normalize=True,
                                     files_extension='.png',
                                     grayscale=True
                                     )

    imageList = os.listdir(TEST_PATH)
    test_images = []
    names = []
    for image in imageList:
        if image[-3:] != 'png':
            continue
        names.append(image.split('.')[0])
        new_image = Image.open(TEST_PATH+'/'+image)
        test_images.append(new_image)

    # Clean the log folder (used to log results in a folder for later tensorboard usage)
    if tf.gfile.Exists(LOG_FOLDER):
        tf.gfile.DeleteRecursively(LOG_FOLDER)
    tf.gfile.MakeDirs(LOG_FOLDER)



# Just to make sure you start running from the main function
if __name__ == '__main__':
    main()