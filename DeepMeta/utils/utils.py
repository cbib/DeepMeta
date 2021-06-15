#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import skimage.measure as measure
import tensorflow as tf


def list_files_path(path):
    """
    List files from a path.

    :param path: Folder path
    :type path: str
    :return: A list containing all files in the folder
    :rtype: List
    """
    return [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def plot_learning_curves(history, name, metric, path="plots/"):
    """
    Plot training curves.

    :param history: Result of model.fit
    :param name: Name of the plot (saving name)
    :type name: str
    :param metric: Metric to monitor
    :type metric: str
    :param path: Saving path
    :type path: str

    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history[metric]
    val_acc = history.history["val_" + metric]
    epochs = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if metric != "loss":
        ax1.plot(epochs, acc, label="Entraînement")
        ax1.plot(epochs, val_acc, label="Validation")
        ax1.set_title("Précision - Données entraînement vs. validation.")
        ax1.set_ylabel("Précision (" + metric + ")")
        ax1.set_xlabel("Epoch")
        ax1.legend()

    ax2.plot(epochs, loss, label="Entraînement")
    ax2.plot(epochs, val_loss, label="Validation")
    ax2.set_title("Perte - Données entraînement vs. validation.")
    ax2.set_ylabel("Perte")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.savefig(path + name + ".png")


def print_red(skk):
    """
    Print in red.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[91m{}\033[00m".format(skk))


def print_gre(skk):
    """
    Print in green.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[92m{}\033[00m".format(skk))


def sorted_alphanumeric(data):
    """
    Sort function.

    :param data: str list
    :type data: List
    :return: Sorted list
    :rtype: List
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa
    return sorted(data, key=alphanum_key)


def border_detected(dataset, k, seg, path_result, name_folder):
    """
    Draw mask borders on image and save it.

    :param dataset: Image you want to draw on
    :type dataset: np.array
    :param k: Index of the image
    :type k: int
    :param seg: Mask
    :type seg: np.array
    :param path_result: path where you want to save images
    :type path_result: str
    :param name_folder: Folder in which you want to save images.
    :type name_folder: str
    """
    cell_contours = measure.find_contours(seg[k], 0.8)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    for contour in cell_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="red")
    plt.xlim((0, 128))
    plt.ylim((128, 0))
    plt.imshow(dataset[k], cmap="gray")
    plt.savefig(path_result + str(name_folder) + "/" + str(k) + ".png")
    plt.close(fig)


def get_args():
    """
    Argument parser.

    :return: Object containing all the parameters needed to train a model
    :rtype: Dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=300, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument(
        "--model_name",
        type=str,
        default="small++",
        help="Name of the model you want to train",
        choices=["small++", "unet"],
    )
    parser.add_argument(
        "--meta", dest="meta", action="store_true", help="If flag, segment metas"
    )
    parser.set_defaults(meta=False)
    parser.add_argument(
        "--weighted",
        dest="weighted",
        action="store_true",
        help="If flag, use weighted crossentropy",
    )
    parser.set_defaults(weighted=False)
    parser.add_argument(
        "--size", type=int, default=128, help="Size of the image, one number"
    )
    parser.add_argument(
        "--drop_r", type=float, default=0.2, help="Size of the image, one number"
    )
    parser.add_argument(
        "--filters", type=int, default=16, help="Size of the image, one number"
    )
    parser.add_argument("--w1", type=int, default=2, help="weight inside")
    parser.add_argument("--w2", type=int, default=4, help="Weight border")
    parser.add_argument(
        "--patience", type=int, default=10, help="Set patience value for early stopper"
    )
    args = parser.parse_args()
    print_red(args)
    return args


class PrintLR(tf.keras.callbacks.Callback):
    """
    Callback used to print learning rate at each epoch.
    """

    def on_epoch_begin(self, epoch, logs=None):
        """
        At the beginning of each epoch, print the current learning rate.

        """
        print_gre(
            "\nLearning rate for epoch {} is {}".format(
                epoch + 1, self.model.optimizer.lr.numpy()
            )
        )


class LRDecay(tf.keras.callbacks.Callback):
    """
    Callback used to linearly reduce the learning rate.

    :param epoch_decay: Number of epoch to do before reduce learning rate
    :type epoch_decay: int
    :param coef: Reduce coefficient
    :type coef: int
    """

    def __init__(self, epoch_decay, coef=10):
        super().__init__()
        self.epoch_decay = epoch_decay
        self.coef = coef

    def on_epoch_begin(self, epoch, logs=None):
        """
        At the beginning of each epoch, if enough epochs are done, reduce the learning rate by coef.
        """
        if (epoch + 1) % self.epoch_decay == 0:
            self.model.optimizer.lr = self.model.optimizer.lr / self.coef
            print_gre("\nLearning rate is {}".format(self.model.optimizer.lr.numpy()))


class CosLRDecay(tf.keras.callbacks.Callback):
    """
    Callback used to perform cosine learning rate decay.

    .. note::
       Idea come from : https://openreview.net/forum?id=Skq89Scxx&noteId=Skq89Scxx

    :param nb_epochs: Number total of epoch to run.
    :type nb_epochs: int
    """

    def __init__(self, nb_epochs, lr):
        super().__init__()
        # self.f_lr = self.model.optimizer.lr
        self.nb_epochs = nb_epochs

    def on_epoch_begin(self, epoch, logs=None):
        """
        At the beginning of each epoch, process a new learning rate.
        """
        self.model.optimizer.lr = (
            0.5
            * (1 + math.cos(epoch * math.pi / self.nb_epochs))
            * self.model.optimizer.lr
        )
        if self.model.optimizer.lr == 0.0:
            self.model.optimizer.lr = 1e-10
        print_gre("\nLearning rate is {}".format(self.model.optimizer.lr.numpy()))
