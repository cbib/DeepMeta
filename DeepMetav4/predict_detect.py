#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os

import numpy as np
import skimage.io as io
import tensorflow.keras as keras

import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.utils as utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_images_from_mouse(path_mouse):
    mouse = io.imread(path_mouse, plugin="tifffile").astype(np.uint8)
    mouse = np.array(mouse) / np.amax(mouse)
    return data.contraste_and_reshape(mouse)  # todo: don't run this on new batch images
    # return mouse.reshape(-1, 128, 128, 1)


def predict_detect(dataset, path_model_detect):
    model_detect = keras.models.load_model(path_model_detect)
    detect = model_detect.predict(dataset)
    return detect.argmax(axis=-1)


if __name__ == "__main__":
    # Souris Test #
    souris_8 = os.path.join(gv.PATH_DATA, "Souris_Test/souris_8.tif")
    souris_28 = os.path.join(gv.PATH_DATA, "Souris_Test/souris_28.tif")
    souris_56 = os.path.join(gv.PATH_DATA, "Souris_Test/souris_56.tif")
    souris_new = os.path.join(gv.PATH_DATA, "Souris_Test/m2Pc_c1_10Corr_1.tif")

    # Modèle de détection #
    path_model_detect = os.path.join(gv.PATH_SAVE, "Poumons/model_detection.h5")

    # Modèle de détection meta #
    path_model_detect_meta = os.path.join(gv.PATH_SAVE, "Metastases/model_detection.h5")

    slist = [souris_8, souris_28, souris_56, souris_new]
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
        dataset = get_images_from_mouse(souris)
        res = predict_detect(dataset, path_model_detect)
        res_meta = predict_detect(dataset, path_model_detect_meta)
        utils.print_gre("\tRes detect lungs : {}".format(res))
        utils.print_gre("\tRes detect meta : {}\n\n".format(res_meta))
