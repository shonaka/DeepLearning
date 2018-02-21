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
    Date:       2/20/18
    File:       TF03_main
    Comments:   This is the main file to run a simple RNN model on predicting
                bitcoin price in USD.
    ToDo:       * Quantify the results
                * Explore more and provide more insights in the data
    **********************************************************************************
"""
import tensorflow as tf
import numpy as np
import tflearn
import pandas as pd
import math
import matplotlib.pyplot as plt
# For Pycharm IDE, avoiding a certain warning when using GPU computing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Specify data path
LOG_FOLDER = '../tensorflow_logs/TF03'

# ===== Define global variables =====
# RNN related
NUM_HIDDEN = 32
FUTURE_STEP = 1  # How many samples ahead you want to predict

# Optimization related
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 100
# ===================================

# A helper function to organize the data for easier RNN training
def chunking(x, future_step):
    # Initialize the sequence and the next value
    seq, next_val= [], []
    # Based on the batch size and the future step size,
    # run a for loop to create chunks.
    # Here, it's BATCH_SIZE - 1 because we are trying to predict
    # one sample ahead. You could change this to your own way
    # e.g. want to predict 5 samples ahead, then - 5
    for i in range(0, len(x) - BATCH_SIZE - future_step, future_step):
        seq.append(x[i: i + BATCH_SIZE])
        next_val.append(x[i + BATCH_SIZE + future_step - 1])

    # So now the data is [Samples, Batch size, One step prediction]
    seq = np.reshape(seq, [-1, BATCH_SIZE, 1])
    next_val = np.reshape(next_val, [-1, 1])

    X = np.array(seq)
    Y = np.array(next_val)

    return X, Y

# Some other helper functions to quantify the goodness of the fit
def rmse(y_target, y_predicted):
    return np.sqrt(((y_target[FUTURE_STEP:] - y_predicted[:-FUTURE_STEP]) ** 2).mean())

def rmsle(y_target, y_predicted):
    return np.sqrt(np.square(np.log(y_target[FUTURE_STEP:] + 1) - np.log(y_predicted[:-FUTURE_STEP] + 1)).mean())

def mae(y_target, y_predicted):
    return np.mean(np.abs((y_target[FUTURE_STEP:] - y_predicted[:-FUTURE_STEP])))

def mape(y_target, y_predicted):
    return np.mean(np.abs((y_target[FUTURE_STEP:] - y_predicted[:-FUTURE_STEP]) / y_target[FUTURE_STEP:])) * 100


# Defining the RNN function
def RNN(activator, optimizer):
    # Building network
    with tf.name_scope('Input'):
        net = tflearn.input_data(shape=[None, BATCH_SIZE, 1])
    # Here, I'm just using the most basic rnn cell type
    # You could also replace the following with tflearn.lstm or tflearn.gru
    with tf.name_scope('Model'):
        net = tflearn.layers.recurrent.simple_rnn(net, NUM_HIDDEN, dropout=0.9, bias=True)
        net = tflearn.fully_connected(net, 1, activation=activator)

    with tf.name_scope('Optimizer'):
        net = tflearn.regression(net, optimizer=optimizer, loss='mean_square', learning_rate=LEARNING_RATE)

    # You could pass the directory path to the following code to create tensorboard logs
    model = tflearn.DNN(net, tensorboard_dir=LOG_FOLDER, tensorboard_verbose=3)
    # model = tflearn.DNN(net, tensorboard_verbose=3)

    return model


def main():

    # # Show the current version of tensorflow
    print("Tensorflow version: ", tf.__version__)

    # # 1) Import and sort data
    print("\nImporting data...\n")
    # In this code, we are directly fetching the interested data (Price in USD).
    # If you want to see the exploration of the data, please check the jupyter notebook
    # on the same repository.
    df = pd.read_csv('../data/.coinbaseUSD.csv')
    df.columns = ['TimeStamp', 'PriceUSD', 'Volume']

    # Encode the date and replace the index by the date
    df.TimeStamp = pd.to_datetime(df['TimeStamp'], unit='s')

    # Change the index with the encoded date
    df.index = df.TimeStamp

    # Group by day
    df_day = df.resample('D').mean()

    # Removing the rows with Nans
    df_day = df_day.dropna()

    # Plot just for checking
    print("Showing some plots\n")
    df_day.plot()
    plt.show()

    # The data we are going to use for prediction
    data = df_day[(df_day.index > '2016-01-01')]

    # Check the shape of the data that we are using
    print('Data shape: ', data.shape)

    # Normalize the data (using min-max scaling)
    data_max = data.PriceUSD.max()
    data_min = data.PriceUSD.min()
    norm_data = (data.PriceUSD - data_min) / (data_max - data_min)

    # Divide the data into training and testing sets
    train_len = 0.9
    tot_len = data.shape[0]
    train_data = norm_data[:np.int(tot_len * train_len)]
    test_data = norm_data[np.int(tot_len * train_len):]
    train_target = norm_data[FUTURE_STEP:np.int(tot_len * train_len) + FUTURE_STEP]

    # Just for checking the dimensions
    print('Train data shape:', train_data.shape)
    print('Test data shape:', test_data.shape)
    print('Train target shape:', train_target.shape)

    # Use the helper function to prepare the data
    # Note that we are assigning 1 as the 2nd argument since
    # we are predicting one step ahead.
    trainX, trainY = chunking(train_data, FUTURE_STEP)
    testX, testY = chunking(test_data, FUTURE_STEP)
    print('Data shape that we are actually going to use: ', trainX.shape)

    # Clean the log folder (used to log results in a folder for later tensorboard usage)
    if tf.gfile.Exists(LOG_FOLDER):
        tf.gfile.DeleteRecursively(LOG_FOLDER)
    tf.gfile.MakeDirs(LOG_FOLDER)


    # # 2) Define a graph
    graph = tf.Graph()
    with graph.as_default():
        # Training
        print("----------------------------------")
        print("Starting to train...")
        print("----------------------------------")
        model = RNN('sigmoid', 'adam')
        model.fit(trainX, trainY, n_epoch=NUM_EPOCHS, validation_set=0.1, batch_size=BATCH_SIZE)

        # Testing
        print("----------------------------------")
        print("Training finished.")
        print("----------------------------------")
        print("Starting to test...")
        print("----------------------------------")

        # testing and predicting
        prediction = model.predict(testX)

        # Scale both the prediction and the target between 0 and 1
        max_predict = prediction.max()
        min_predict = prediction.min()
        scaled_pred = (prediction - min_predict) / (max_predict - min_predict)
        max_target = testY.max()
        min_target = testY.min()
        scaled_target = (testY - min_target) / (max_target - min_target)

        # Plot to check
        # Shifting FUTURE_STEP because we wanted to predict FUTURE_STEP sample
        # ahead so to align the graph, we are shifting by that many samples
        plt.plot(scaled_pred[FUTURE_STEP:], label='Predicted')
        plt.plot(scaled_target[:-FUTURE_STEP], label='Actual')
        plt.legend()
        plt.show()

        # Let's also check the goodness of the fit
        # There's many options out there
        #   e.g. RMSE   = Root Mean Squared Error
        #        RMSLE  = Root Mean Squared Logarithmic Error
        #        MAE    = Mean Absolute Error
        #        MAPE   = Mean Absolute Percentage Error
        #
        # These metrics will be useful when comparing the performance
        # across different models or optimizer for example.
        print(scaled_target[FUTURE_STEP:])
        print('-----')
        print(scaled_pred[:-FUTURE_STEP])

        test_rmse = rmse(scaled_target, scaled_pred)
        print("RMSE: {0:.2f}".format(test_rmse))
        test_rmsle = rmsle(scaled_target, scaled_pred)
        print("RMSLE: {0:.2f}".format(test_rmsle))
        test_mae = mae(scaled_target, scaled_pred)
        print("MAE: {0:.2f}".format(test_mae))
        test_mape = mape(scaled_target, scaled_pred)
        print("MAPE: {0:.2f}".format(test_mape))


# Just to make sure you start running from the main function
if __name__ == '__main__':
    main()