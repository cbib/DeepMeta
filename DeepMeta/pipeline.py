#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import skimage.io as io

import DeepMeta.predict as p
import DeepMeta.utils.data as data
import DeepMeta.utils.global_vars as gv
import DeepMeta.utils.stats as stats
import DeepMeta.utils.utils as utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-stats", dest="stats", action="store_false")
    parser.set_defaults(stats=True)
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.set_defaults(save=True)
    args = parser.parse_args()
    utils.print_red(args)
    return args


def get_label_path(mouse_path, name, folder="lungs"):
    return (
        "/".join(mouse_path.split("/")[:-1])
        + "/"
        + name.split(".")[0]
        + "/"
        + folder
        + "/"
    )


def get_label_masks(folder_path):
    res = []
    file_list = utils.sorted_alphanumeric(utils.list_files_path(folder_path))
    for file in file_list:
        im = io.imread(file, plugin="tifffile")
        if np.amax(im) != 0:
            im = (im / np.amax(im)).astype(np.uint8)
        res.append(im)
    return res


if __name__ == "__main__":
    flags = get_flags()
    # Model seg lungs #
    path_model_seg_lungs = os.path.join(
        gv.PATH_SAVE, "Poumons/best_seg_model_weighted.h5"
    )
    # Model seg metas #
    path_model_seg_metas = os.path.join(
        gv.PATH_SAVE, "Metastases/128model_small++_weighted1015.h5"
    )

    # MOUSE INFOS
    MOUSE_PATH = os.path.join(gv.PATH_DATA, "Souris_Test/souris_8.tif")
    name = "souris_8"
    LABEL_PATH = get_label_path(MOUSE_PATH, name)
    LABEL_PATH_METAS = get_label_path(MOUSE_PATH, name, folder="metas")

    # merged_list = zip(gv.slist, gv.nlist, gv.label_list, gv.meta_label_list)
    # for (souris, name, label, label_meta) in merged_list:

    souris = (MOUSE_PATH, True)  # True if need to change contrast

    utils.print_red(name)
    dataset = data.get_predict_dataset(souris[0], souris[1])

    seg_lungs = p.predict_seg(dataset, path_model_seg_lungs).reshape(128, 128, 128)
    seg_lungs = p.postprocess_loop(seg_lungs)

    seg_metas = seg_lungs * p.predict_seg(dataset, path_model_seg_metas).reshape(
        128, 128, 128
    )
    seg_metas = p.postprocess_meta(seg_metas, k1=3, k2=3)

    if flags.save:
        p.save_res(dataset, seg_lungs, name + "_pipeline_lungs")
        p.save_res(dataset, seg_metas, name + "_pipeline_metas")
        # todo : option save masks

    if flags.stats:
        label_lungs = get_label_masks(LABEL_PATH)
        label_metas = get_label_masks(LABEL_PATH_METAS)

        stats.do_stats(
            seg_lungs, label_lungs, gv.PATH_RES + str(name) + "_pipeline_lungs/"
        )
        stats.do_stats(
            seg_metas, label_metas, gv.PATH_RES + str(name) + "_pipeline_metas/"
        )
