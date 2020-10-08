import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import DeepMetav4.predict as predict
import seaborn as sns
import skimage.io as io

import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed


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


def qualite_model(num, path_model_seg):
    """
    :param num: numéro de la souris auquel on veut appliquer le modèle
    (8, 28 ou 56 pour nos souris test)
    :param path_model_seg: le path du modèle de segmentation que l'on veut appliquer
    :param time: si time == 0 : réseau classique sans lstm / time == 'la valeur du time'
    pour un modèle LstmCnn
    :param wei: utilisation ou non d'un réseau avec pixels pondérés
    :return: liste des valeurs des IoU pour l'ensemble des slices de la souris test
    """
    # todo: PATH !!!!!????? Voir pour gerer les metas ! Enorme dégat !!!

    path_model_detect = os.path.join(gv.PATH_SAVE, "Poumons/model_detection.h5")
    path_test = gv.PATH_DATA + "Souris_Test/"

    path_souris = os.path.join(path_test, "souris_" + str(num) + ".tif")
    path_souris_annotee = os.path.join(
        path_test, "Masque_Poumons/masque_" + str(num) + "/"
    )
    name_folder = "souris_" + str(num)

    # Detection label
    tab2 = gv.tab[np.where(gv.tab[:, 1] == num)]
    detect_annot = tab2[:, 2][np.where(tab2[:, 3] == 1)]
    n = len(detect_annot)  # -1 pour Souris 56 => surement erreur dans annotation...
    if num == 56:
        n -= 1
    # Segmentation label
    list_msk = utils.sorted_aphanumeric(os.listdir(path_souris_annotee))
    y = [
        io.imread(path_souris_annotee + list_msk[i], plugin="tifffile")
        for i in range(len(list_msk))
    ]
    seg_true = np.array(y, dtype="bool")
    # Segmentation prédite
    detect, seg = predict.methode_detect_seg(
        path_souris, path_model_detect, path_model_seg, gv.PATH_RES, name_folder
    )
    seg_pred = seg[detect_annot]
    # Calcul IoU
    IoU = [stats_pixelbased(seg_true[j], seg_pred[j]).get("IoU") for j in range(n)]
    return IoU


def qualite_model_meta(num, path_seg, path_detect, detect_l, seg_l, size=128):
    path_souris = os.path.join(
        gv.PATH_DATA + "Souris_Test/", "souris_" + str(num) + ".tif"
    )
    name = "..."  # idem
    path_msk = os.path.join(
        gv.PATH_DATA + "Souris_Test/", "Masque_Metas/Meta_" + str(num) + "/"
    )
    list_msk = utils.sorted_aphanumeric(os.listdir(path_msk))
    seg_true = np.zeros((128, size, size))
    for i in np.arange(len(list_msk)):
        seg_true[i] = io.imread(path_msk + list_msk[i], plugin="tifffile")
    # todo : ne prendre que les bons masques

    detect, seg = predict.methode_detect_seg(
        path_souris,
        path_detect,
        path_seg,
        gv.PATH_RES,
        name,
        meta=True,
        detect_l=detect_l,
        seg_l=seg_l,
        stat=True,
    )
    iou = [
        stats_pixelbased(seg_true[j], seg[j].reshape(128, 128)).get("IoU")
        for j in range(128)
    ]
    return iou


def csv_qualite(list_result, list_label, num, name_tab):
    """
    :param list_result: liste contenant les listes résultats IoU de différents
                        modèles
    :param list_label: liste contenant le nom des méthodes utilisées
    :param num: le numéro de la souris test
    :param name_tab: titre du tableau csv
    :return: Tableau csv comportant les résultats IoU de différentes méthodes
            sur une souris test
    """
    result = np.zeros((len(list_result[0]), len(list_result)))
    for i in range(len(list_result)):
        result[:, i] = list_result[i]
    df = pd.DataFrame(result, columns=list_label)
    df.to_csv(
        os.path.join(gv.PATH_RES, "csv/") + name_tab + "souris_" + str(num) + ".csv",
        index=None,
        header=True,
    )


def get_values_from_csv(path_csv, label_names):
    result = pd.read_csv(path_csv)
    ind = ["slice_" + str(i) for i in range(len(result))]
    result = result.set_index(pd.Index(ind))
    label = list(result)
    m = [np.median(result.values[:, i]) for i in range(len(label))]  # m and m_2 ????
    m_2 = np.zeros(len(label_names))
    for i in range(len(label_names)):
        m_2[i] = np.where(m == np.sort(m)[len(label_names) - 1 - i])[0][0]
    result = result.iloc[:, m_2]
    label = [label_names[int(m_2[i])] for i in range(len(label_names))]
    return result, label


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
    plt.xlabel("Modele", fontsize=18)
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


def get_name_from_path(path):
    res = path.split("/")[-1]
    return res.split(".")[0]


def create_csv(
    list_num, list_path, model_detect, model_detect_lungs=None, model_seg_lungs=None
):
    utils.print_red("Making csv....")
    meta = "Metastases" in list_path[0]
    for num in list_num:
        utils.print_gre("\tSouris : " + str(num))
        list_result = []
        list_label = []
        for model in list_path:
            if meta:
                iou = qualite_model_meta(
                    num, model, model_detect, model_detect_lungs, model_seg_lungs
                )
            else:
                iou = qualite_model(num, model)
            list_result.append(iou)
            list_label.append(get_name_from_path(model))
        csv_qualite(list_result, list_label, num, "csv_iou_")
    utils.print_red("Created !")


if __name__ == "__main__":

    model_detect_lungs = os.path.join(gv.PATH_SAVE, "Poumons/model_detection.h5")
    model_seg_lungs = os.path.join(gv.PATH_SAVE, "Poumons/best_small++_weighted24.h5")

    model_detect = os.path.join(gv.PATH_SAVE, "Metastases/best_resnet50.h5")

    # Path
    small_unet = os.path.join(gv.PATH_SAVE, "Metastases/128model_unet_weighted1015.h5")
    # small_unet_wei = os.path.join(gv.PATH_SAVE,
    # "Metastases/128model_small++_weighted1525.h5")
    # unet = os.path.join(gv.PATH_SAVE, "Metastases/128model_small++_weighted1520.h5")
    # unet2 = os.path.join(gv.PATH_SAVE, "Metastases/128model_small++_weighted1020.h5")
    # unet3 = os.path.join(gv.PATH_SAVE, "Metastases/128model_small++_weighted1520.h5")
    # unet4 = os.path.join(gv.PATH_SAVE, "Metastases/128model_unet_weighted1015.h5")
    # unet5 = os.path.join(gv.PATH_SAVE, "Metastases/128model_small++_weighted1015.h5")

    list_num = [8, 56]
    list_path = [small_unet]  # , small_unet_wei, unet, unet2] #, unet3]

    # enregistrer csv

    create_csv(
        list_num,
        list_path,
        model_detect,
        model_detect_lungs=model_detect_lungs,
        model_seg_lungs=model_seg_lungs,
    )

    # use csv
    for num in list_num:
        utils.print_red("souris " + str(num))
        result, label = get_values_from_csv(
            gv.PATH_RES + "csv/csv_iou_souris_" + str(num) + ".csv", ["10-15"]
        )  # , "15-25", "15-20", "10-20"])
        # print(result)
        print(np.mean(result))
        plot_iou(result, label, str(num), save=True)
        box_plot(result, label, str(num), save=True)
        heat_map(result, str(num), save=True)
