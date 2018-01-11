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
    File:       visualization
    Comments:   Functions to visualize images for checking purposes
    ToDo:       * Make a class and your own library
    **********************************************************************************
"""
import matplotlib.pyplot as plt

def plt_image_labels(images, image_shape, num_row, num_col, class_labels, class_true, class_pred=None):
    """
    Plot images with true labels. If specified, predicted class too.

    :param images: the image data itself, containing pixel data.
    :param image_shape: shape of the images (height, width).
    :param num_row: how many number of rows you want to plot.
    :param num_col: how many number of columns you want to plot.
    :param class_labels: class labels in string corresponding to the input numbers.
    :param class_true: the true class labels for the images you give as the first input.
    :param class_pred: the predicted class labels. If not specified, it does not plot the labels.
    """
    # Check if the specified num_row and num_col matches the total number of images
    assert len(images) == len(class_true) == num_row * num_col

    # Create a subplot of num_row x num_col
    fig, axes = plt.subplots(num_row, num_col)
    fig.subplots_adjust(hspace=0.1*num_row, wspace=0.1*num_col)

    # for each image, plot the image and the true label. If there's predicted, plot that too.
    for i, ax in enumerate(axes.flat):
        # Plot the image
        ax.imshow(images[i].reshape(image_shape), cmap='binary')

        # Show labels
        if class_pred is None:
            xlabel = "True: {0}".format(class_labels[class_true[i]])
        else:
            xlabel = "True: {0},  Pred: {1}".format(class_labels[class_true[i]],
                                                    class_labels[class_pred[i]])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])