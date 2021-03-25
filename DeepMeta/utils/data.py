#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage.exposure as exposure
import skimage.io as io
import skimage.measure as measure
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.preprocessing.image import load_img

import DeepMeta.models.utils_model as utils_model

from . import global_vars as gv
from . import utils


def shuffle_lists(lista, listb, seed=42):
    random.seed(seed)
    random.shuffle(lista)
    random.seed(seed)
    random.shuffle(listb)
    return lista, listb


def get_label(file_path):
    data_dir = pathlib.Path(gv.path_classif_metas)
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = tf.cast(parts[-2] == class_names, dtype=tf.uint8)
    # one_hot = tf.argmax(one_hot)
    return one_hot


def decode_img(img):
    img = tf.cast(tf.image.decode_jpeg(img, channels=1), dtype=tf.float32) / 255
    return tf.image.resize(img, [128, 128])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def configure_for_performance(ds, batch_size):
    ds = ds.shuffle(buffer_size=50000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    return ds


def save_model_name(opt, path_save):
    if opt["meta"]:
        res = "Metastases/" + str(opt["size"]) + "model_" + opt["model_name"]
    else:
        res = "Poumons/" + str(opt["size"]) + "model_" + opt["model_name"]
    if opt["weighted"]:
        res += "_weighted" + str(opt["w1"]) + str(opt["w2"])
    res += ".h5"
    return os.path.join(path_save, res)


def weight_map(label, a, b, size=128):
    """
    Weight map creation.
    Outside pixels have no weight, their value is one by default.

    :param label: Mask array
    :param a: Inside pixel weight
    :param b: Border pixel weight
    :param size: size of the image
    :return: Weight map array
    """
    weight = np.zeros((label.shape[0], size, size))

    for k, lab in enumerate(label):
        if len(np.shape(lab)) > 2:
            lab = lab.reshape(size, size)
        contour = measure.find_contours(lab, 0.8)
        indx_mask = np.where(lab == 1)[0]
        indy_mask = np.where(lab == 1)[1]

        w = np.ones((size, size))
        w[indx_mask, indy_mask] = a

        for i in range(len(contour)):
            indx_cont = np.array(contour[i][:, 0], dtype="int")
            indy_cont = np.array(contour[i][:, 1], dtype="int")
            w[indx_cont, indy_cont] = b

        # w = w ** 2
        weight[k] = w

    return weight


def get_label_weights(dataset, label, n_sample, w1, w2, size=128):
    weight_2D = weight_map(label, w1, w2, size)
    dataset = dataset.reshape(-1, size, size, 1)[n_sample]  # 1 ici si pas de concat
    label = label.reshape(-1, size, size, 1)[n_sample]
    weight_2D = weight_2D.reshape(-1, size, size, 1)[n_sample]
    y = np.zeros((dataset.shape[0], size, size, 2))
    y[:, :, :, 0] = label[:, :, :, 0]
    y[:, :, :, 1] = weight_2D[:, :, :, 0]
    return dataset, y


def get_dataset(path_data, path_label, opt):
    data_files = utils.sorted_alphanumeric(utils.list_files_path(path_data))
    label_files = utils.sorted_alphanumeric(utils.list_files_path(path_label))
    n_val = int(0.8 * len(data_files))
    dataset = Dataset(
        opt["batch_size"],
        opt["size"],
        data_files[:n_val],
        label_files[:n_val],
        opt["weighted"],
        opt["w1"],
        opt["w2"],
    )
    dataset_val = Dataset(
        opt["batch_size"],
        opt["size"],
        data_files[n_val:],
        label_files[n_val:],
        opt["weighted"],
        opt["w1"],
        opt["w2"],
    )
    return dataset, dataset_val


def weighted_bin_acc(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(
        tf.reshape(y_true[:, :, :, 0], (-1, 128, 128, 1)), y_pred
    )


def mcc(y_true, y_pred):
    """
    Matthews Correlation Coefficient
    """
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


def mcc_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0) * 1e2
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0) / 1e2

    up = tp * tn - fp * fn
    down = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    mcc = up / (down + K.epsilon())
    mcc = tf.where(tf.math.is_nan(mcc), tf.zeros_like(mcc), mcc)

    return 1 - K.mean(mcc)


def prepare_for_training(path_data, path_label, file_path, opt):
    dataset, dataset_val = get_dataset(path_data, path_label, opt)
    utils.print_gre("Prepare for Training...")
    input_shape = (opt["size"], opt["size"], 1)
    utils.print_gre("Getting model...")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model_seg = gv.model_list[opt["model_name"]](
            input_shape, filters=opt["filters"], drop_r=opt["drop_r"]
        )
        metric = "loss"
        # metric_fn = weighted_auc
        optim = tf.keras.optimizers.Adam(lr=opt["lr"])
        checkpoint = callbacks.ModelCheckpoint(
            file_path,
            monitor="val_" + metric,
            verbose=1,
            save_best_only=True,
            mode="min",
        )
        if opt["weighted"]:
            loss_fn = utils_model.weighted_cross_entropy
        else:
            loss_fn = "binary_crossentropy"
        model_seg.compile(
            loss=loss_fn,
            optimizer=optim,
            # metrics=[metric_fn],
        )
    utils.print_gre("Done!")
    utils.print_gre("Prepared !")
    return dataset, dataset_val, model_seg, checkpoint, metric


def contrast_and_reshape(souris, size=128):
    """
    :param souris: Ensemble d'image, verification si ensemble ou image
    unique avec la condition if.
    :return: Ensemble d'image avec contraste ameliore et shape modifie
    pour entrer dans le reseaux.
    """
    if len(souris.shape) > 2:
        data = []
        for i in np.arange(souris.shape[0]):
            img_adapteq = exposure.equalize_adapthist(
                souris[i], clip_limit=0.03
            )  # clip_limit=0.03 de base
            data.append(img_adapteq)
        data = np.array(data).reshape(-1, size, size, 1)
        return data
    else:
        img_adapteq = exposure.equalize_adapthist(souris, clip_limit=0.03)
        img = np.array(img_adapteq).reshape(size, size, 1)
        return img


def get_predict_dataset(path_souris, contrast=True):
    mouse = io.imread(path_souris, plugin="tifffile").astype(np.uint8)
    mouse = np.array(mouse) / np.amax(mouse)
    if contrast:
        mouse = contrast_and_reshape(mouse)
    else:
        mouse = np.array(mouse).reshape(-1, 128, 128, 1)
    return mouse


class Dataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(
        self,
        batch_size,
        img_size,
        input_img_paths,
        target_img_paths,
        weighted=False,
        w1=None,
        w2=None,
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.weighted = weighted
        self.w1 = w1
        self.w2 = w2
        assert len(input_img_paths) == len(target_img_paths)

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros(
            (self.batch_size, self.img_size, self.img_size, 1), dtype="float32"
        )
        for j, path in enumerate(batch_input_img_paths):
            img = np.array(load_img(path, color_mode="grayscale")) / 255
            x[j] = np.expand_dims(img, 2)
        y = np.zeros(
            (self.batch_size, self.img_size, self.img_size, 1), dtype=np.float32
        )
        for j, path in enumerate(batch_target_img_paths):
            label = (
                np.array(load_img(path, color_mode="grayscale"), dtype=np.float32) / 255
            )
            y[j] = np.expand_dims(label, 2)
        if self.weighted:
            n = range(self.batch_size)
            n_sample = random.sample(list(n), len(n))
            x, y = get_label_weights(x, y, n_sample, self.w1, self.w2)
        return x, y


def plot_iou(result, label, title, save=False):
    fig = plt.figure(1, figsize=(12, 12))
    for i in range(len(label)):
        plt.plot(np.arange(len(result)) + 29, result.values[:, i], label=label[i])
    plt.ylim(0.2, 1)
    plt.xlabel("Slices", fontsize=18)
    plt.ylabel("IoU", fontsize=18)
    plt.legend()
    plt.title(title)
    if save:
        fig.savefig(gv.PATH_RES + "stats/plot_iou_" + title + ".png")
    plt.close("all")


def box_plot(result, label, title, save=False):
    fig = plt.figure(1, figsize=(15, 15))
    plt.boxplot([result.values[:, i] for i in range(len(label))])
    plt.ylim(0, 1)
    plt.xlabel("Model", fontsize=18)
    plt.ylabel("IoU", fontsize=18)
    plt.gca().xaxis.set_ticklabels(label)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=18)
    if save:
        fig.savefig(gv.PATH_RES + "stats/boxplot_" + title + ".png")
    plt.close("all")


def heat_map(result, title, slice_begin=30, slice_end=106, save=False):
    fig = plt.figure(1, figsize=(12, 12))
    sns.heatmap(result[slice_begin:slice_end])
    plt.title(title)
    if save:
        fig.savefig(gv.PATH_RES + "stats/heat_map_" + title + ".png")
    plt.close("all")
