#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os

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
    path_model_detect_metas = os.path.join(
        gv.PATH_SAVE, "Metastases/model_detection.h5"
    )

    merged_list = zip(gv.slist, gv.nlist, gv.label_list, gv.meta_label_list)
    for (souris, name, label, label_meta) in merged_list:
        utils.print_red(name)
        dataset = data.get_predict_dataset(souris[0], souris[1])
        res = p_detect.predict_detect(dataset, path_model_detect_lungs)
        acc = process_acc(res, label)
        # print("res : " + str(res))
        # print("label : " + str(np.array(label)))
        utils.print_gre("acc = {}%".format(acc * 100))

        # METAS
        res_meta = p_detect.predict_detect(dataset, path_model_detect_metas)
        acc_meta = process_acc(res_meta, label_meta)
        # print("res : " + str(res_meta))
        # print("label : " + str(np.array(label_meta)))
        utils.print_gre("acc meta = {}%".format(acc_meta * 100))

    """
    Metas pas encore prises en compte, car peut etre masked data (à verifier)
    """
