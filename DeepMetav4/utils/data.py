#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os
import pathlib
import random

import cv2
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
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from tensorflow.keras.preprocessing.image import load_img

import DeepMetav4.models.utils_model as utils_model

from . import global_vars as gv
from . import utils


def shuffle_lists(lista, listb, seed=42):
    random.seed(seed)
    random.shuffle(lista)
    random.seed(seed)
    random.shuffle(listb)
    return lista, listb


def rotate_img(img):
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    scale = 1.0

    M = cv2.getRotationMatrix2D(center, 90, scale)
    r90 = cv2.warpAffine(img, M, (h, w))
    M = cv2.getRotationMatrix2D(center, 180, scale)
    r180 = cv2.warpAffine(img, M, (h, w))
    M = cv2.getRotationMatrix2D(center, 270, scale)
    r270 = cv2.warpAffine(img, M, (h, w))
    return r90, r180, r270


def elastic_transform(image, alpha=60, sigma=4, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        )
        * alpha
    )

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


def concat_and_normalize(l0, l1):
    random.shuffle(l0)
    random.shuffle(l1)
    inv = False
    if len(l1) < len(l0):
        # better -> concat_and_normalize(l1, l0, inv=True) -> tried but no working
        sm_l = l1
        bg_l = l0
        inv = True
    else:
        sm_l = l0
        bg_l = l1
    list_size = len(sm_l)
    bg_l = bg_l[0:list_size]
    data_detec = sm_l + bg_l
    if inv:
        label_detec = np.concatenate(
            (
                np.ones(
                    list_size,
                ),
                np.zeros(
                    list_size,
                ),
            )
        )
    else:
        label_detec = np.concatenate(
            (
                np.zeros(
                    list_size,
                ),
                np.ones(
                    list_size,
                ),
            )
        )
    assert len(sm_l) == len(bg_l)
    assert len(data_detec) == len(label_detec)
    return shuffle_lists(data_detec, label_detec)


def get_images_detect(tab, path_img):
    data_detec_0 = []
    data_detec_1 = []
    for i in range(len(tab)):
        try:
            im = io.imread(
                path_img + "img_" + str(tab[i, 0]) + ".tif", plugin="tifffile"
            )
            im = im / np.amax(im)
            im90, im180, im270 = rotate_img(im)
            # elastic = elastic_transform(im)
            if tab[i, 3] == 1:
                data_detec_1 += [im, im90, im180, im270]
            else:
                data_detec_0 += [im, im90, im180, im270]
        except Exception as e:
            utils.print_red("IMG {} not found".format(tab[i, 0]))
            utils.print_red("\t" + str(e))
    return data_detec_0, data_detec_1


def process_da(im):
    l_im = [im]
    flipped = cv2.flip(im, 1)
    l_im.append(flipped)
    im90, im180, im270 = rotate_img(im)
    l_im += [im90, im180, im270]
    fim90, fim180, fim270 = rotate_img(flipped)
    l_im += [fim90, fim180, fim270]
    l_el = []
    for img in l_im:
        l_el.append(elastic_transform(img))
    return l_im + l_el


def get_images_detect_meta(tab, path_img, split=11136):
    old0 = []
    new0 = []
    old1 = []
    new1 = []
    for i in range(len(tab))[:100]:
        if tab[i, 4] == 1:
            try:
                im = io.imread(
                    path_img + "img_" + str(tab[i, 0]) + ".tif", plugin="tifffile"
                )
                im = im / np.amax(im)
                l_im = process_da(im)
                if tab[i, 3] == 1:
                    # if os.path.isfile(gv.new_meta_path_img + str(tab[i, 0]) + ".tif"):
                    if i > split:
                        new1 += l_im
                    else:
                        old1 += l_im
                else:
                    if i > split:
                        new0 += l_im
                    else:
                        old0 += l_im
            except Exception as e:
                utils.print_red("IMG {} not found".format(tab[i, 0]))
                utils.print_red("\t" + str(e))
    # uncomment if we want the same number of img per batch
    # data_detec_0, _ = concat_and_normalize(old0, new0)
    # data_detec_1, _ = concat_and_normalize(old1, new1)
    data_detec_0 = old0 + new0
    data_detec_1 = old1 + new1
    utils.print_gre("Images 0 : {}".format(len(data_detec_0)))
    utils.print_gre("Images 1 : {}".format(len(data_detec_1)))
    return data_detec_0, data_detec_1


def create_dataset_detect(path_img, tab, size, meta=False):
    """
    :param path_img: ensemble des images de souris où les poumons ont été annotés.
    :param tab: tableau résumant les identifiants et annotations pour les souris.

    :return:
        - data_detec : ensemble des slices de souris où les poumons ont été annotés.
        - label_detec : label de chaque slices 1 présentant des poumons, 0 sinon.
    """
    utils.print_gre("Creating dataset...")
    if meta:
        data_detec_0, data_detec_1 = get_images_detect_meta(tab, path_img)
    else:
        data_detec_0, data_detec_1 = get_images_detect(tab, path_img)
    data_detec, label_detec = concat_and_normalize(data_detec_0, data_detec_1)
    data_detec = np.array(data_detec)
    no = range(len(data_detec))
    no_sample = random.sample(list(no), len(no))
    data_detec = data_detec.reshape(-1, size, size, 1)[no_sample].astype("float32")
    label_detec = keras.utils.to_categorical(label_detec[no_sample])
    utils.print_gre("Created !")
    utils.print_gre("Nb of images : {}".format(len(data_detec)))
    return data_detec, label_detec


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


def new_dataset_detect(data_path, opt):
    data_dir = pathlib.Path(data_path)
    image_count = len(list(data_dir.glob("*/*.jpg")))
    list_ds = tf.data.Dataset.list_files(str(data_dir / "*/*"), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=True)
    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)
    train_ds = train_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # for image, label in train_ds.take(10):
    #     print("Image min: ", np.amin(image.numpy()))
    #     print("Image max: ", np.amax(image.numpy()))
    #     print("Label :", label.numpy())
    train_ds = configure_for_performance(train_ds, opt["batch_size"])
    val_ds = configure_for_performance(val_ds, opt["batch_size"])
    utils.print_gre("NB of images : {}".format(image_count))
    return train_ds, val_ds


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
    Création du carte de poids définissant une valeur d'importance pour chaque pixel
    Les pixels n'appartenant pas au masque ont une valeur de poids
    définit à 1 par défaut
    :param label: ensemble de x masque label 128x128
    :param a: valeur du poids pour pixel appartenant au masque
    :param b: valeur du poids pour pixel appartenant au contour du masque
    :param size: size of the image
    :return: ensemble de y weight map 128x128
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
    weight_2D = weight_map(
        label, w1, w2, size
    )  # 2, 4 -> weight, background = 1 par default, inside 2, border 4
    dataset = dataset.reshape(-1, size, size, 1)[n_sample]  # 1 ici si pas de concat
    label = label.reshape(-1, size, size, 1)[n_sample]
    weight_2D = weight_2D.reshape(-1, size, size, 1)[n_sample]
    y = np.zeros((dataset.shape[0], size, size, 2))
    y[:, :, :, 0] = label[:, :, :, 0]
    y[:, :, :, 1] = weight_2D[:, :, :, 0]
    return dataset, y


def get_dataset(path_data, path_label, opt):
    data_files = utils.list_files_path(path_data)
    label_files = utils.list_files_path(path_label)
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


def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


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
        metric = "MatthewsCorrelationCoefficient"
        metric_fn = matthews_correlation_coefficient
        optim = tf.keras.optimizers.Adam(lr=opt["lr"])
        checkpoint = callbacks.ModelCheckpoint(
            file_path,
            monitor="val_" + metric,
            verbose=1,
            save_best_only=True,
            mode="max",
        )
        if opt["weighted"]:
            loss_fn = utils_model.weighted_cross_entropy
        else:
            loss_fn = "binary_crossentropy"
        model_seg.compile(
            loss=loss_fn,
            optimizer=optim,
            metrics=[metric_fn],
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
