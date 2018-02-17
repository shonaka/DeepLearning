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
    File:       cnn_ops
    Comments:   Helper functions for cnn
    ToDo:       * Make a class and your own library
    **********************************************************************************
"""
import tensorflow as tf

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

def new_conv(X, shape, name, use_relu=True):
    """
    Defining a convolutional layer with specific shape
    :param X: input from previous layer
    :param shape: shape of the new defined filter
    :param name: name of the new conv layer
    :param use_relu: boolean to use relu or not
    :return: the newly created layer in an array
    """
    # For tensorboard, defining variable scope
    with tf.variable_scope(name):
        # First get the new weights
        W = new_weights(name, shape)

        # Add tensorboard summary to visualize the weights
        tf.summary.histogram('W', W)

        # Create the conv layer
        layer = tf.nn.conv2d(input=X,
                         filter=W,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

        # If using relu, compute the output
        if use_relu:
            layer = tf.nn.relu(layer)

        # Return the output
        return layer

def new_fc(X, name, num_class):
    """
    Defining a fully-connected layer
    :param X: input from previous layer
    :param name: name of the fc_layer
    :param num_class: number of classes
    :return: fully-connected layer with number of images x number of classes
    """
    with tf.variable_scope(name):
        # Get the previous layer and flatten it
        P = tf.contrib.layers.flatten(X)

        # Make a fully-connected layer with the number of classes specified
        fc_layer = tf.contrib.layers.fully_connected(P, num_class, activation_fn=None)

        # Return the output
        return fc_layer