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
    Date:       12/09/17
    File:       main
    Comments:   This is the main file to run linear model based classification on
                fashion MNIST: https://github.com/zalandoresearch/fashion-mnist
    ToDo:       * Implement Tensorboard to visualize learning process results
    **********************************************************************************
"""
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
hello = tf.constant("WTF")
sess = tf.Session()
print(sess.run(hello))