#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import math
import os
import re

import matplotlib.pyplot as plt
import skimage.measure as measure
import tensorflow as tf


def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def plot_learning_curves(history, name, metric, path="plots/"):
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
    print("\033[91m{}\033[00m".format(skk))


def print_gre(skk):
    print("\033[92m{}\033[00m".format(skk))


def sorted_aphanumeric(data):
    """
    :param data: list d'element alphanumerique.
    :return: list triee dans l'ordre croissant alphanumerique.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa
    return sorted(data, key=alphanum_key)


def calcul_numSouris(path_souris):
    """
    :param path_souris: path vers le dossier contenant des images de souris .tif
    :return: une liste contenant le numéro de chaque souris
    """
    list_souris = sorted_aphanumeric(os.listdir(path_souris))
    num_souris = []
    for k in range(len(list_souris)):
        num_souris.append(int(re.findall(r"\d+", list_souris[k])[0]))
    return num_souris


def border_detected(dataset, k, seg, path_result, name_folder, prefix="/p_"):
    cell_contours = measure.find_contours(seg[k], 0.8)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    for contour in cell_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="red")
    plt.xlim((0, 128))
    plt.ylim((128, 0))
    plt.imshow(dataset[k], cmap="gray")
    plt.savefig(path_result + str(name_folder) + prefix + str(k) + ".png")
    plt.close(fig)


class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print_gre(
            "\nLearning rate for epoch {} is {}".format(
                epoch + 1, self.model.optimizer.lr.numpy()
            )
        )


class LRDecay(tf.keras.callbacks.Callback):
    def __init__(self, epoch_decay, coef=10):
        super().__init__()
        self.epoch_decay = epoch_decay
        self.coef = coef

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_decay == 0:
            self.model.optimizer.lr = self.model.optimizer.lr / self.coef
            print_gre("\nLearning rate is {}".format(self.model.optimizer.lr.numpy()))


class CosLRDecay(tf.keras.callbacks.Callback):
    def __init__(self, nb_epochs, lr):
        super().__init__()
        self.nb_epochs = nb_epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.lr = (
            0.5
            * (1 + math.cos(epoch * math.pi / self.nb_epochs))
            * self.model.optimizer.lr
        )
        if self.model.optimizer.lr == 0.0:
            self.model.optimizer.lr = 1e-10
        print_gre("\nLearning rate is {}".format(self.model.optimizer.lr.numpy()))
