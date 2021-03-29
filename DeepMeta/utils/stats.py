#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import math

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def count_pixels(mask_list):
    """
    Process a mask list to count pixels divided in two categories (0 and non 0)
    :param mask_list: Mask list
    :type mask_list: np.array
    :return: zero count, non zero count
    :rtype: (int, int)
    """
    count0 = 0
    count1_pred = 0
    for j in tqdm(range(0, len(mask_list))):
        count1_pred += np.count_nonzero(mask_list[j])
        count0 += 16384 - np.count_nonzero(mask_list[j])
    return count0, count1_pred


def process_intersections(mask_pred_list, mask_gt_list):
    """
    Process confusion matrix between masks.

    :param mask_pred_list: Prediction list
    :type mask_pred_list: np.array
    :param mask_gt_list: Ground truth list
    :type mask_gt_list: np.array
    :return: TN, TP, FN, FP
    :rtype: (int, int, int, int)
    """
    intersection0_tab = []
    intersection1_tab = []

    erreur_pred0_tab = []
    erreur_pred1_tab = []
    for i, mask in tqdm(enumerate(mask_pred_list)):
        mask_gt = mask_gt_list[i]
        intersection0 = 0
        intersection1 = 0
        erreur_pred0 = 0
        erreur_pred1 = 0
        for lx in range(0, len(mask)):
            for m in range(0, len(mask[0])):
                if mask[lx][m] == 0 and mask_gt[lx][m] == 0:
                    intersection0 += 1
                elif mask[lx][m] == 0 and mask_gt[lx][m] == 1:
                    erreur_pred1 += 1
                elif mask[lx][m] == 1 and mask_gt[lx][m] == 0:
                    erreur_pred0 += 1
                elif mask[lx][m] == 1 and mask_gt[lx][m] == 1:
                    intersection1 += 1
                else:
                    print(
                        "something weird happened : "
                        + str(mask_pred_list[lx][m])
                        + " : "
                        + str(mask_gt_list[lx][m])
                        + "\n"
                    )

        intersection0_tab.append(intersection0)
        intersection1_tab.append(intersection1)
        erreur_pred0_tab.append(erreur_pred0)
        erreur_pred1_tab.append(erreur_pred1)
    return intersection0_tab, intersection1_tab, erreur_pred0_tab, erreur_pred1_tab


def inverse_binary_mask(msk):
    """
    Invert bytes of mask.

    :param msk: Binary mask (128x128)
    :type msk: np.array
    :return: Inverted mask
    :rtype: np.array
    """
    new_mask = np.ones((128, 128)) - msk
    return new_mask


def stats_pixelbased(y_true, y_pred):
    """
    Process IoU between pred and gt.

    :param y_true: Ground truth
    :type y_true: np.array
    :param y_pred: Prediction
    :type y_pred: np.array
    :return: IoU coefficient
    :rtype: float
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
    if union.sum() == 0:
        jaccard = -2
    else:
        jaccard = intersection.sum() / union.sum()
    # precision = intersection.sum() / pred.sum()
    # recall = intersection.sum() / truth.sum()
    # Fmeasure = (2 * precision * recall) / (precision + recall)
    return jaccard


def process_iou(mask_pred, mask_gt):
    """
    Process IoU over lists.

    :param mask_pred: Prediction list
    :type mask_pred: np.array
    :param mask_gt: Ground truth list
    :type mask_gt: np.array
    :return: Mean IoU over the lists
    :rtype: float
    """
    tmp = []
    for i, pred in enumerate(mask_pred):
        gt = mask_gt[i]
        iou = stats_pixelbased(gt, pred)
        tmp.append(iou)
    return np.mean(tmp)


def process_mcc(
    intersection0_tab, intersection1_tab, erreur_pred1_tab, erreur_pred0_tab
):
    """
    Process MCC over confusion matrix.

    :param intersection0_tab: TN
    :type intersection0_tab: int
    :param intersection1_tab: TP
    :type intersection1_tab: int
    :param erreur_pred1_tab: FP
    :type erreur_pred1_tab: int
    :param erreur_pred0_tab: FN
    :type erreur_pred0_tab: int
    :return: Return mean MCC
    :rtype: float
    """
    TP = sum(intersection1_tab)
    TN = sum(intersection0_tab)
    FP = sum(erreur_pred1_tab)
    FN = sum(erreur_pred0_tab)

    numerateur = TP * TN - FP * FN
    denominateur = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if math.sqrt(denominateur) == 0:
        res = -2
    else:
        res = numerateur / math.sqrt(denominateur)
    return res


def process_auc(gt_list, pred_list):
    """
    Process AUC (AUROC) over lists.

    :param gt_list: Ground truth list
    :type gt_list: np.array
    :param pred_list: Predictions list
    :type pred_list: np.array
    :return: Mean AUC over the lists
    :rtype: float
    """
    res = []
    for i, gt in enumerate(gt_list):
        if np.amax(gt) != 0:
            pred = pred_list[i].flatten()
            gt = gt.flatten()
            res.append(roc_auc_score(gt, pred))
    return np.mean(res)


def do_stats(mask_pred_list, mask_gt_list, save_path):
    """
    Run stat metrics over lists and write a report in a file.

    :param mask_pred_list: Prediction list
    :type mask_pred_list: np.array
    :param mask_gt_list: Ground truth list
    :type mask_gt_list: np.array
    :param save_path: Path for the report file
    :type save_path: str
    :return:
    :rtype:
    """
    f = open(save_path + "finestat.txt", "w")
    f.write("STAT PREDICTION" + "\n\n")

    count0, count1_pred = count_pixels(mask_pred_list)
    f.write("PRED : \n")
    f.write("pixel count0 : " + str(count0) + "\n")
    f.write("pixel count1 : " + str(count1_pred) + "\n")

    # GT
    count0, count1_GT = count_pixels(mask_gt_list)
    f.write("GT : \n")
    f.write("pixel count0 : " + str(count0) + "\n")
    f.write("pixel count1 : " + str(count1_GT) + "\n\n")

    (
        intersection0_tab,
        intersection1_tab,
        erreur_pred0_tab,
        erreur_pred1_tab,
    ) = process_intersections(mask_pred_list, mask_gt_list)

    f.write("Moyenne de pixels blanc bien classifiés (moyenne sur toutes les images)\n")
    moyenne1 = sum(intersection1_tab) / len(mask_pred_list)
    f.write(str(moyenne1) + "\n")

    f.write("Moyenne de pixels blanc images GT (moyenne sur toutes les images)\n")
    moyenne2 = count1_GT / len(mask_pred_list)
    f.write(str(moyenne2) + "\n")

    f.write("Moyenne de pixels blanc images predites (moyenne sur toutes les images)\n")
    moyenne3 = count1_pred / len(mask_pred_list)
    f.write(str(moyenne3) + "\n")

    f.write("Pourcentage de pixels blanc bien classifiés\n")
    # c'est intersection1 sur la somme des pixels blanc (intersection1 + erreur_pred1)
    if sum(intersection1_tab) + sum(erreur_pred1_tab) == 0:
        pourcentage = -2
    else:
        pourcentage = sum(intersection1_tab) / (
            sum(intersection1_tab) + sum(erreur_pred1_tab)
        )
    f.write(str(pourcentage) + "\n\n")

    mcc = process_mcc(
        intersection0_tab, intersection1_tab, erreur_pred1_tab, erreur_pred0_tab
    )

    f.write("MCC : \n")
    f.write(str(mcc) + "\n\n")

    iou = process_iou(mask_pred_list, mask_gt_list)

    f.write("IOU : \n")
    f.write(str(iou) + "\n\n")

    auc = process_auc(mask_gt_list, mask_pred_list)

    f.write("AUC : \n")
    f.write(str(auc))

    f.close()
