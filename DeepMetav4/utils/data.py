#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os
import random

import cv2
import numpy as np
import skimage.exposure as exposure
import skimage.io as io
import skimage.measure as measure
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as callbacks

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


def create_dataset_detect(path_img, tab, numSouris, size):
    """
    :param path_img: ensemble des images de souris où les poumons ont été annotés.
    :param tab: tableau résumant les identifiants et annotations pour les souris.

    :return:
        - data_detec : ensemble des slices de souris où les poumons ont été annotés.
        - label_detec : label de chaque slices 1 présentant des poumons, 0 sinon.
    """
    utils.print_gre("Creating dataset...")
    data_detec_0 = []
    data_detec_1 = []
    for i in range(len(tab)):
        try:
            im = io.imread(path_img + "img_" + str(i) + ".tif", plugin="tifffile")
            if tab[i, 3] == 1:
                im = im / np.amax(im)
                im90, im180, im270 = rotate_img(im)
                data_detec_1.append(im)
                data_detec_1.append(im90)
                data_detec_1.append(im180)
                data_detec_1.append(im270)
            else:
                im = im / np.amax(im)
                im90, im180, im270 = rotate_img(im)
                data_detec_0.append(im)
                data_detec_0.append(im90)
                data_detec_0.append(im180)
                data_detec_0.append(im270)
        except Exception:
            utils.print_red("IMG {} not found".format(i))
    list_size = len(data_detec_0)
    random.shuffle(data_detec_1)
    data_detec_1 = data_detec_1[0:list_size]
    data_detec = data_detec_0 + data_detec_1
    label_detec = np.concatenate(
        (np.zeros((len(data_detec_0),)), np.ones((len(data_detec_1),)))
    )
    data_detec, label_detec = shuffle_lists(data_detec, label_detec)
    data_detec = np.array(data_detec)
    no = range(len(data_detec))
    no_sample = random.sample(list(no), len(no))
    data_detec = data_detec.reshape(-1, size, size, 1)[no_sample].astype("float32")
    label_detec = keras.utils.to_categorical(label_detec[no_sample])
    utils.print_gre("Created !")
    utils.print_gre("label 0 : {}".format(len(data_detec_0)))
    utils.print_gre("label 1 : {}".format(len(data_detec_1)))
    return data_detec, label_detec


def concat_with_mask(im, i, path_mask, pref, png=True):
    if png:
        im_mask = io.imread(path_mask + pref + str(i) + ".png")  # , plugin='tifffile')
    else:
        im_mask = io.imread(path_mask + pref + str(i), plugin="tifffile")
    return np.concatenate([im[:, :, np.newaxis], im_mask[:, :, np.newaxis]], 2)


def apply_mask(img, path_img, path_mask):
    im = io.imread(path_img + img)
    im_mask = io.imread(path_mask + img) / 255  # -> donne img normales
    return im * im_mask


def create_dataset_detect_meta(path_img, path_mask, tab, size):  # todo : refactor
    utils.print_gre("Creating dataset...")
    prefix = ["", "90_", "180_", "270_", "t_", "t_90_", "t_180_", "t_270_"]
    data_detec_0 = []
    label_detec_0 = []
    data_detec_1 = []
    label_detec_1 = []
    for i in range(len(tab)):
        # if tab[i, 5] == 1: # since we use an optimized tab for metas
        if tab[i, 7] == 0:
            # sampled_prefix = random.sample(prefix, 4)
            for pref in prefix:
                try:
                    # im = io.imread(
                    #     path_img + pref + str(int(tab[i, 0])) + ".tif",
                    # plugin='tifffile')
                    # .tif if not generated
                    im = apply_mask(
                        pref + str(int(tab[i, 0])) + ".png", path_img, path_mask
                    )
                    # im = concat_with_mask(im, i, path_mask, pref)
                    data_detec_0.append(im / 255)
                    label_detec_0.append(0)
                except Exception:
                    utils.print_red(
                        "Image "
                        + pref
                        + "{} not found for label 0".format(int(tab[i, 0]))
                    )
        else:
            for pref in prefix:
                try:
                    # im = io.imread(
                    #     path_img + pref + str(int(tab[i, 0])) + '.tif',
                    # plugin='tifffile')  # .tif if not generated
                    # im = concat_with_mask(im, i, path_mask, pref)
                    im = apply_mask(
                        pref + str(int(tab[i, 0])) + ".png", path_img, path_mask
                    )
                    data_detec_1.append(im / 255)
                    label_detec_1.append(1)
                except Exception:
                    utils.print_red(
                        "Image "
                        + pref
                        + "{} not found for label 1".format(int(tab[i, 0]))
                    )
    list_size = len(data_detec_1)
    data_detec_0 = data_detec_0[0:list_size]
    label_detec_0 = label_detec_0[0:list_size]
    data_detec = data_detec_0 + data_detec_1
    label_detec = label_detec_0 + label_detec_1
    data_detec, label_detec = shuffle_lists(data_detec, label_detec, seed=42)
    data_detec = np.array(data_detec)
    label_detec = np.array(label_detec)
    no = range(len(data_detec))
    no_sample = random.sample(list(no), len(no))
    data_detec = data_detec.reshape(-1, size, size, 1)[no_sample].astype(
        "float32"
    )  # ici le deux est un 1 si on ne concat pas)
    label_detec = keras.utils.to_categorical(label_detec[no_sample])
    utils.print_gre("Created !")
    return data_detec, label_detec


def create_dataset(path_img, path_label, size):
    utils.print_gre("Creating dataset...")
    dataset = []
    label_list = []
    file_list = utils.list_files(path_img)
    for file in file_list:
        try:
            img = io.imread(path_img + file, plugin="tifffile")
            label = io.imread(path_label + file, plugin="tifffile")
            dataset.append((np.array(img) / 255).reshape(-1, size, size, 1))
            label_list.append(label)
        except Exception:
            utils.print_red("Image {} not found.".format(file))
    utils.print_gre("Created !")
    utils.print_gre("Nb of images: {}".format(len(dataset)))
    return np.array(dataset), np.array(label_list, dtype=np.bool)


def create_dataset_concat(path_img, path_label, path_mask, opt):
    utils.print_gre("Creating dataset...")
    dataset = []
    label_list = []
    file_list = utils.list_files(path_img)
    for file in file_list:
        try:
            img = io.imread(path_img + file, plugin="tifffile")
            label = io.imread(path_label + file, plugin="tifffile")
            img_mask = concat_with_mask(img, "", path_mask + file, "", png=False)
            dataset.append(
                (np.array(img_mask) / 255).reshape(-1, opt.size, opt.size, 2)
            )
            label_list.append(label)
        except Exception:
            print(file)
    utils.print_gre("Created !")
    return np.array(dataset), np.array(label_list, dtype=np.bool)


def save_model_name(opt, path_save):
    if opt.meta:
        res = "Metastases/" + str(opt.size) + "model_" + opt.model_name
    else:
        res = "Poumons/" + str(opt.size) + "model_" + opt.model_name
    if opt.weighted:
        res += "_weighted" + str(opt.w1) + str(opt.w2)
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

    for k in range(label.shape[0]):
        lab = label[k]
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


# function used to get data and model ready for training | todo: refactor
def prepare_for_training(path_data, path_label, file_path, opt):
    dataset, label = create_dataset(
        path_img=path_data, path_label=path_label, size=opt.size
    )
    utils.print_gre("Prepare for Trainning...")
    n = range(np.shape(dataset)[0])
    n_sample = random.sample(list(n), len(n))
    input_shape = (opt.size, opt.size, 1)
    utils.print_gre("Getting model...")
    strategy = tf.distribute.MirroredStrategy()
    if opt.weighted:
        with strategy.scope():
            dataset, label = get_label_weights(
                dataset, label, n_sample, opt.w1, opt.w2, size=opt.size
            )
            model_seg = gv.model_list[opt.model_name](input_shape)
            checkpoint = callbacks.ModelCheckpoint(
                file_path,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min",
            )
            metric = "loss"
            model_seg.compile(
                loss=utils_model.weighted_cross_entropy,
                optimizer=tf.keras.optimizers.Adam(lr=opt.lr),
            )
    else:
        with strategy.scope():
            dataset = dataset.reshape(-1, opt.size, opt.size, 1)[n_sample]
            label = label.reshape(-1, opt.size, opt.size, 1)[n_sample]
            model_seg = gv.model_list[opt.model_name](input_shape)
            checkpoint = callbacks.ModelCheckpoint(
                file_path,
                monitor="val_accuracy",
                verbose=1,
                save_best_only=True,
                mode="max",
            )
            metric = "accuracy"
            model_seg.compile(
                optimizer=tf.keras.optimizers.Adam(lr=opt.lr),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )  # todo : fix mean iou
    utils.print_gre("Done!")
    utils.print_gre("Prepared !")
    return dataset, label, model_seg, checkpoint, metric


def contraste_and_reshape(souris, size=128):
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
