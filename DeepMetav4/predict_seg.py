#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os

import numpy as np
import skimage.io as io
import tensorflow.keras as keras

import DeepMetav4.models.utils_model as utils_model
import DeepMetav4.postprocessing.post_process_and_count as postprocess
import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.utils as utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def predict_seg(dataset, path_model_seg, tresh=0.5):
    if "weighted" not in path_model_seg:
        model_seg = keras.models.load_model(
            path_model_seg,
        )
    else:
        model_seg = keras.models.load_model(
            path_model_seg,
            custom_objects={
                "weighted_cross_entropy": utils_model.weighted_cross_entropy,
                "weighted_bin_acc": data.weighted_bin_acc,
            },
        )
    res = model_seg.predict(dataset)
    print(np.amax(res))
    return (res > tresh).astype(np.uint8).reshape(len(dataset), 128, 128, 1)


def save_res(dataset, seg, name_folder, mask=False, mask_path=None):
    assert len(dataset) == len(seg)
    if not os.path.exists(gv.PATH_RES + str(name_folder)):
        os.makedirs(gv.PATH_RES + str(name_folder))
    seg = seg.reshape(len(seg), 128, 128)
    for k in range(len(dataset)):
        if mask:
            io.imsave(mask_path + str(k) + ".png", seg[k] * 255)
        utils.border_detected(dataset, k, seg, gv.PATH_RES, name_folder)


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
    souris_new = os.path.join(gv.PATH_DATA, "Souris_Test/m2Pc_c1_10Corr_1.tif")

    # Modèle de détection #
    path_model_seg = os.path.join(gv.PATH_SAVE, "Poumons/best_small++_weighted_24.h5")

    # Modèle de détection meta #
    path_model_seg_meta = os.path.join(
        gv.PATH_SAVE, "Metastases/128model_small++_weighted1015.h5"
    )

    slist = [
        (souris_8, True),
        (souris_28, True),
        (souris_56, True),
        (souris_new, False),
    ]
    # slist = [souris_new]
    nlist = [
        "souris_8",
        "souris_28",
        "souris_56",
        "souris_new_batch",
    ]  # 28 saine, 8 petites meta, 56 grosses meta
    merged_list = zip(slist, nlist)
    for (souris, name) in merged_list:
        utils.print_red(name)
        dataset = data.get_predict_dataset(souris[0], souris[1])
        res_lungs = predict_seg(dataset, path_model_seg)
        res_meta = predict_seg(
            dataset, path_model_seg_meta
        )  # dataset * reslung if masked data
        # print(np.amax(res_meta))
        # res_lungs = postprocess_loop(res_lungs)
        save_res(dataset, res_lungs, name + "_lungs")
        save_res(dataset, res_meta, name + "_meta")
