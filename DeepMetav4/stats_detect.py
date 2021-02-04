#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os

import numpy as np
import tensorflow as tf

import DeepMetav4.predict_detect as p_detect
import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.utils as utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def process_acc(pred, label):
    metric = tf.keras.metrics.BinaryAccuracy()
    metric.update_state(label, pred)
    res = metric.result().numpy()
    metric.reset_states()
    return res


if __name__ == "__main__":
    path_model_detect_lungs = os.path.join(gv.PATH_SAVE, "Poumons/best_detection.h5")
    path_model_detect_metas = os.path.join(gv.PATH_SAVE, "Metastases/best_detection.h5")

    # Souris Test #
    souris_8 = os.path.join(gv.PATH_DATA, "Souris_Test/souris_8.tif")
    souris_28 = os.path.join(gv.PATH_DATA, "Souris_Test/souris_28.tif")
    souris_56 = os.path.join(gv.PATH_DATA, "Souris_Test/souris_56.tif")
    souris_new = os.path.join(gv.PATH_DATA, "Souris_Test/m2Pc_c1_10Corr_1.tif")
    slist = [
        (souris_8, True),
        (souris_28, True),
        (souris_56, True),
        (souris_new, False),
    ]
    nlist = [
        "souris_8",
        "souris_28",
        "souris_56",
        "souris_new_batch",
    ]  # 28 saine, 8 petites meta, 56 grosses meta
    merged_list = zip(slist, nlist, gv.label_list)
    for (souris, name, label) in merged_list:
        utils.print_red(name)
        dataset = data.get_predict_dataset(souris[0], souris[1])
        res = p_detect.predict_detect(dataset, path_model_detect_lungs)
        acc = process_acc(res, label)
        print("res : " + str(res))
        print("label : " + str(np.array(label)))
        utils.print_gre("acc = {}%".format(acc * 100))
    """
    Metas pas encore prises en compte, car peut etre masked data (à verifier)
    """
