#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os

import DeepMetav4.models.utils_model as utils_model
import numpy as np
import DeepMetav4.postprocessing.post_process_and_count as postprocess
import skimage.io as io
import tensorflow.keras as keras

import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_seg_dataset(full_souris, path_souris):
    if full_souris:
        mouse = io.imread(path_souris, plugin="tifffile")
    else:
        slices_list = utils.sorted_aphanumeric(os.listdir(path_souris))
        s = np.zeros((len(slices_list), 128, 128))
        for i in range(len(slices_list)):
            s[i] = io.imread(path_souris + slices_list[i])
        mouse = np.array(s)
    return data.contraste_and_reshape(mouse)


def save_mask(dataset, seg, k, mask_path, path_result, name_folder):
    io.imsave(
        "/home/elefevre/Projects/DeepMetav3/Dataset/Images/" + str(k + 128) + ".png",
        dataset[k] * 255,
    )
    io.imsave(mask_path + str(k + 128) + ".png", seg[k] * 255)
    im = ((dataset[k] * 255).reshape(128, 128) * seg[k].reshape(128, 128)).astype(
        np.uint8
    )
    im.reshape(128, 128, 1)
    io.imsave(path_result + str(name_folder) + "/" + str(k) + ".png", im)


def save_res(dataset, seg, path_result, name_folder, mask=False, mask_path=None):
    if not os.path.exists(path_result + str(name_folder)):
        os.makedirs(path_result + str(name_folder))
    seg = seg.reshape(len(seg), 128, 128)
    for k in range(len(dataset)):
        if mask:
            save_mask(dataset, seg, k, mask_path, path_result, name_folder)
        utils.border_detected(dataset, k, seg, path_result, name_folder)


def predict_detect(dataset, path_model_detect):
    model_detect = keras.models.load_model(path_model_detect)
    detect = model_detect.predict(dataset)
    return detect.argmax(axis=-1)


def predict_seg(dataset, path_model_seg):
    if "weighted" not in path_model_seg:
        model_seg = keras.models.load_model(
            path_model_seg, custom_objects={"mean_iou": utils_model.mean_iou}
        )
    else:
        model_seg = keras.models.load_model(
            path_model_seg,
            custom_objects={
                "weighted_cross_entropy": utils_model.weighted_cross_entropy
            },
        )
    return (
        (model_seg.predict(dataset) > 0.5)
        .astype(np.uint8)
        .reshape(len(dataset), 128, 128, 1)
    )


def select_slices(dataset, detect):
    res = []
    for i, elt in enumerate(detect):
        if elt == 1:
            res.append(dataset[i])
    return np.array(res)


def apply_mask(dataset, seg):
    res = []
    for i, elt in enumerate(seg):
        res.append(dataset[i] * elt)
    return np.array(res)


def create_vector_to_stat(detect_l, detect_m, seg):
    res = np.zeros((128, 128, 128))
    i = np.where(detect_l == 1)[0][0]
    k = 0
    for j, elt in enumerate(detect_m):
        if elt == 1:
            res[i + j] = seg[k].reshape(128, 128)
            k += 1
    return res


def methode_detect_seg(
    path_souris,
    path_model_detect,
    path_model_seg,
    path_result,
    name_folder,
    full_souris=True,
    save=False,
    stat=False,
    meta=False,
    detect_l=None,
    seg_l=None,
):
    """
    Méthode 2D qui permet de segmenter slice par slice.
    """
    dataset = get_seg_dataset(full_souris, path_souris)
    dataset_og = get_seg_dataset(full_souris, path_souris)
    if meta:
        detect_lungs, seg_lungs = methode_detect_seg(
            path_souris,
            detect_l,
            seg_l,
            "...",
            "...",
            full_souris,
            save=False,
            meta=False,
        )
        dataset = apply_mask(select_slices(dataset, detect_lungs), seg_lungs)
    detect = predict_detect(dataset, path_model_detect)
    dataset = select_slices(dataset_og, detect)
    seg = predict_seg(dataset, path_model_seg)
    if save:
        dataset = dataset.reshape(len(dataset), 128, 128)
        save_res(
            dataset,
            seg,
            path_result,
            name_folder,
            mask=True,
            mask_path="data/results/",
        )
    if stat:
        seg = create_vector_to_stat(detect_lungs, detect, seg)
    return detect, seg


def postprocess_loop(seg):
    res = []
    for elt in seg:
        blobed = postprocess.remove_blobs(elt)
        eroded = postprocess.dilate_and_erode(blobed)
        res.append(eroded / 255)
    return res


if __name__ == "__main__":
    # Souris Test #
    souris_8 = os.path.join(gv.PATH_DATA, "Souris_Test/souris_8.tif")
    souris_28 = os.path.join(gv.PATH_DATA, "Souris_Test/souris_28.tif")
    souris_56 = os.path.join(gv.PATH_DATA, "Souris_Test/souris_56.tif")

    # Modèle de détection #

    path_model_detect = os.path.join(gv.PATH_SAVE, "Poumons/model_detection.h5")

    # Modèle de détection meta #

    path_model_detect_meta = os.path.join(gv.PATH_SAVE, "Metastases/best_resnet50.h5")

    # Modèle de segmentation #

    path_model_seg = os.path.join(gv.PATH_SAVE, "Poumons/best_small++_weighted24.h5")

    # Modèle seg meta #

    path_model_seg_meta = os.path.join(
        gv.PATH_SAVE, "Metastases/128model_small++_weighted1015.h5"
    )

    slist = [souris_8, souris_28, souris_56]
    nlist = [
        "souris_8",
        "souris_28",
        "souris_56",
    ]  # 28 saine, 8 petites meta, 56 grosses meta

    for i, souris in enumerate(slist):
        utils.print_red(nlist[i])
        detect, seg = methode_detect_seg(
            souris,
            path_model_detect,
            path_model_seg,
            gv.PATH_RES,
            nlist[i],
            save=True,
            meta=False,
            detect_l=path_model_detect,
            seg_l=path_model_seg,
        )
        print(np.sum(detect))

        """
        ########################################################################################################################
        # TEST
        #
        gt_list = ["60-66", "0", "70-72"]
        dataset = get_seg_dataset(True, souris)
        res = predict_detect(dataset, path_model_detect)
        seg = predict_seg(dataset, path_model_seg)
        # seg = postprocess_loop(seg)
        # save_res(dataset, res, seg, gv.PATH_RES, nlist[i], True)
        dataset_lungs = []
        test = []
        for j, elt in enumerate(res):
            if elt == 1:
                # im_concat = tf.concat([dataset[j],
                tf.expand_dims(seg[j].astype(np.double), -1)], 2)
                # dataset_lungs.append(np.array(im_concat))  # * seg[j])
                dataset_lungs.append(dataset[j] * seg[j].reshape(128, 128, 1))
                test.append(j)
        dataset_lungs = np.array(dataset_lungs)  # .reshape(-1, 128, 128, 2)
        # print(np.amax(dataset_lungs))
        res_meta = predict_detect(dataset_lungs, path_model_detect_meta)
        print(res_meta)
        print(str(np.sum(res_meta)) + "/" + gt_list[i])
        print(len(res_meta))
        # for j, elt in enumerate(res_meta):
        #     print(str(test[j]) + " : " + str(res_meta[j]))
        ########################################################################################################################
        """
