#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os

import numpy as np
import skimage.io as io
import tensorflow as tf

import DeepMetav4.predict_seg as p_seg
import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.utils as utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def inverse_binary_mask(msk):
    """
    :param msk: masque binaire 128x128
    :return: masque avec binarisation inversée 128x128
    """
    new_mask = np.ones((128, 128)) - msk
    return new_mask


def stats_pixelbased(y_true, y_pred):  # todo: trop de calcul à chaque appel pour rien
    """Calculates pixel-based statistics
    (Dice, Jaccard, Precision, Recall, F-measure)
    Takes in raw_data prediction and truth data in order to calculate accuracy
    metrics for pixel based classfication. Statistics were chosen according
    to the guidelines presented in Caicedo et al. (2018) Evaluation of Deep
    Learning Strategies for Nucleus Segmentation in Fluorescence Images.
    BioRxiv 335216.
    Args:
        y_true (3D np.array): Binary ground truth annotations for a single
            feature, (batch,x,y)
        y_pred (3D np.array): Binary predictions for a single feature,
            (batch,x,y)
    Returns:
        dictionary: Containing a set of calculated statistics
    Raises:
        ValueError: Shapes of `y_true` and `y_pred` do not match.
    Warning:
        Comparing labeled to unlabeled data will produce low accuracy scores.
        Make sure to input the same type of data for `y_true` and `y_pred`
    """
    y_pred = y_pred.reshape(128, 128)
    if y_pred.shape != y_true.shape:
        raise ValueError(
            "Shape of inputs need to match. Shape of prediction "
            "is: {}.  Shape of y_true is: {}".format(y_pred.shape, y_true.shape)
        )
    pred = y_pred
    truth = y_true
    if truth.sum() == 0:
        pred = inverse_binary_mask(pred)
        truth = inverse_binary_mask(truth)
    # Calculations for IOU
    intersection = np.logical_and(pred, truth)
    union = np.logical_or(pred, truth)
    # Sum gets count of positive pixels
    # dice = (2 * intersection.sum() / (pred.sum() + truth.sum()))
    jaccard = intersection.sum() / union.sum()
    # precision = intersection.sum() / pred.sum()
    # recall = intersection.sum() / truth.sum()
    # Fmeasure = (2 * precision * recall) / (precision + recall)
    return {
        # 'Dice': dice,
        "IoU": jaccard,
        # 'precision': precision,
        # 'recall': recall,
        # 'Fmeasure': Fmeasure
    }


def select_slices(dataset, gt):
    res = []
    for i, detect in enumerate(gt):
        if detect == 1:
            res.append(dataset[i])
    return np.array(res).reshape(len(res), 128, 128, 1)


def process_acc(pred, gt):
    metric_list = []
    for i, m_pred in enumerate(pred):
        metric = tf.keras.metrics.binary_accuracy(gt[i].flatten(), m_pred.flatten())
        metric_list.append(metric)
    return metric_list


def get_label_masks(mouse_path, name, folder="lungs"):
    res = []
    folder_path = (
        "/".join(mouse_path.split("/")[:-1])
        + "/"
        + name.split(".")[0]
        + "/"
        + folder
        + "/"
    )
    file_list = utils.list_files_path(folder_path)
    for file in file_list:
        im = io.imread(file, plugin="tifffile")
        im = (im / np.amax(im)).astype(np.uint8)
        res.append(im)
    return res


if __name__ == "__main__":
    path_model_seg_lungs = os.path.join(
        gv.PATH_SAVE, "Poumons/best_small++_weighted_24.h5"
    )
    path_model_seg_metas = os.path.join(gv.PATH_SAVE, "Metastases/128model_small++.h5")

    merged_list = zip(gv.slist, gv.nlist, gv.label_list, gv.meta_label_list)
    for (souris, name, label, label_meta) in merged_list:
        utils.print_red(name)
        dataset = data.get_predict_dataset(souris[0], souris[1])
        # LUNGS
        slices = select_slices(dataset, label)
        res = p_seg.predict_seg(slices, path_model_seg_lungs)
        res = res.reshape(len(res), 128, 128)
        label_masks = get_label_masks(souris[0], name)
        assert len(res) == len(label_masks), "len res : {}; len labels : {}".format(
            len(res), len(label_masks)
        )
        acc = process_acc(res, label_masks)
        utils.print_gre("acc lungs : {}%".format(np.mean(acc) * 100))
        # METAS  -> if masked data, slices * label_lungs
        slices_metas = select_slices(dataset, label_meta)
        if len(slices_metas) > 0:
            res_metas = p_seg.predict_seg(slices_metas, path_model_seg_metas, tresh=0.2)
            res_metas = res_metas.reshape(len(res_metas), 128, 128)
            label_masks_metas = get_label_masks(souris[0], name, folder="metas")
            assert len(res_metas) == len(
                label_masks_metas
            ), "Meta : len res : {}; len labels : {}".format(
                len(res_metas), len(label_masks_metas)
            )
            acc = process_acc(res_metas, label_masks_metas)
            utils.print_gre("acc metas : {}%".format(np.mean(acc) * 100))
        else:
            utils.print_gre("no metas")
