#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

"""
Pipeline
==========
This file is used to run prediction on a mouse.
"""

import argparse
import os

import numpy as np
import skimage.io as io
import skimage.measure as measure
from scipy import ndimage

import DeepMeta.postprocessing.post_process_and_count as post_count
import DeepMeta.predict as p
import DeepMeta.utils.data as data
import DeepMeta.utils.global_vars as gv
import DeepMeta.utils.stats as stats
import DeepMeta.utils.utils as utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_flags():
    """
    Argument parser for the script

    :return: Object containing flags.
    :rtype: Dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats",
        dest="stats",
        action="store_true",
        help="Run stats, need to have labels in test folder",
    )
    parser.set_defaults(stats=False)
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        help="Save output (image with mask border)",
    )
    parser.set_defaults(save=False)
    parser.add_argument(
        "--mask", dest="mask", action="store_true", help="Save output masks"
    )
    parser.set_defaults(mask=False)
    parser.add_argument(
        "--contrast",
        dest="contrast",
        action="store_true",
        help="Enhance image contrast",
    )
    parser.set_defaults(contrast=False)
    parser.add_argument(
        "--csv", dest="csv", action="store_true", help="Save res in csv file"
    )
    parser.set_defaults(csv=False)
    parser.add_argument(
        "--csv_file", type=str, default=None, help="Name of the csv file"
    )
    parser.add_argument(
        "--mousename",
        type=str,
        default="m2Pc_c1_10Corr_1.tif",
        help="Name of the file to do inference on. (File in test folder)",
    )
    parser.add_argument("--day", type=str, default=None, help="Day of the mouse in csv")
    parser.add_argument(
        "--mut", type=str, default=None, help="Mutation of the mouse in csv"
    )
    args = parser.parse_args()
    utils.print_red(args)
    return args


def get_label_path(mouse_path, name, folder="lungs"):
    """
    Get the label path from mouse name.

    :param mouse_path: Path of the current segmented mouse
    :type mouse_path: str
    :param name: Name of the mouse
    :type name: str
    :param folder: "Lungs" Or "Metas", task related argument
    :type folder: str
    :return: The label folder path for the current mouse
    :rtype: str
    """
    return (
        "/".join(mouse_path.split("/")[:-1])
        + "/"
        + name.split(".")[0]
        + "/"
        + folder
        + "/"
    )


def get_label_masks(folder_path):
    """
    Create a list of images from folder_path

    :param folder_path: Path of the label folder
    :type folder_path: str
    :return: List containing all labels images
    :rtype: List
    """
    res = []
    file_list = utils.sorted_alphanumeric(utils.list_files_path(folder_path))
    for file in file_list:
        im = io.imread(file, plugin="tifffile")
        if np.amax(im) != 0:
            im = (im / np.amax(im)).astype(np.uint8)
        res.append(im)
    return res


def write_in_csv(filename, mousename, day, vol_l, vol_m, vol_pm, mutation):
    """
    Create a csv file and fill it with
    :param filename:
    :type filename:
    :param mousename:
    :type mousename:
    :param day:
    :type day:
    :param vol_l:
    :type vol_l:
    :param vol_m:
    :type vol_m:
    :param vol_pm:
    :type vol_pm:
    :param mutation:
    :type mutation:
    :return:
    :rtype:
    """
    check_and_create_file(filename)
    f = open(filename, "a")
    f.write(
        mousename
        + ","
        + day
        + ","
        + str(vol_l)
        + ","
        + str(vol_m)
        + ","
        + str(vol_pm)
        + ","
        + mutation
        + "\n"
    )
    f.close()


def write_meta_in_csv(filename, mousename, slice_nb, meta_id, vol, mutation):
    """
    Create a csv file and fill it in order to create graph number of meta
    in function of time.

    :param filename: csv filename
    :type filename: Str
    :param mousename: Name of the mouse file
    :type mousename: Str
    :param slice_nb: Id of the slice
    :type slice_nb: int
    :param meta_id:
    :type meta_id: Int
    :param vol: Meta volume
    :type vol: int
    :param mutation: Name of the mesured mutation
    :type mutation: Str
    """
    check_and_create_file_meta(filename)
    f = open(filename, "a")
    f.write(
        mousename
        + ","
        + str(slice_nb)
        + ","
        + str(meta_id)
        + ","
        + str(vol)
        + ","
        + mutation
        + "\n"
    )
    f.close()


def check_and_create_file_meta(path):
    if not os.path.isfile(path):
        f = open(path, "a+")
        f.write("name,slice_nb,meta_id,vol,mutation\n")
        f.close()


def check_and_create_file(path):
    if not os.path.isfile(path):
        f = open(path, "a+")
        f.write("name,day,vol_l,vol_m,vol_pm,mutation\n")
        f.close()


def get_vol_metaid(mask, id, vol=0.0047):
    return np.count_nonzero(mask == id) * vol


def laplace(img_stack, mask_list):  # move to post process
    """
    Remove false positives in lung segmentation. Apply a
    laplace of gaussian filter on slices, if the mean value of the
    result is <1 we remove the mask.

    .. note::
       We process only first and last slices (until we find a value >1).
       This ensure that we do not remove false
       negative slices.

    :param img_stack: Full image stack (dataset).
    :type img_stack: np.array
    :param mask_list: Full lung segmentation output
    :type mask_list: np.array
    :return: Updated mask list
    :rtype: np.array
    """
    img_stack2 = (img_stack * 255).astype(np.uint8)
    for i, img in enumerate(img_stack2):
        new_im = ndimage.gaussian_laplace(img, sigma=7)
        if np.mean(new_im) < 1:
            mask_list[i] = np.zeros((128, 128))
        else:
            break
    for i, img in enumerate(img_stack2[::-1]):
        new_im = ndimage.gaussian_laplace(img, sigma=7)
        if np.mean(new_im) < 1:
            mask_list[(len(mask_list) - 1) - i] = np.zeros((128, 128))
        else:
            break
    return mask_list


def lungs_sanity_check(mask_list):  # move to post process
    """
    Check if there is some false positive. If mask < 15px -> mask is null.
    If i-1 and i+1 do not contain mask, i does not contains a mask either.

    :param mask_list: Lungs segmentation output
    :type mask_list: np.array
    :return: Checked segmentation output
    :rtype: np.array
    """
    mask_list[0] = np.zeros((128, 128))
    mask_list[-1] = np.zeros((128, 128))
    for i in range(1, len(mask_list) - 1):
        if mask_list[i].sum() > 15:
            if mask_list[i - 1].sum() < 15 and mask_list[i + 1].sum() < 15:
                mask_list[i] = np.zeros((128, 128))
        else:
            mask_list[i] = np.zeros((128, 128))
    return mask_list


def process_mouse(path, name, contrast):
    """
    Run pipeline on mouse.

    :param path: Mouse path
    :type path: Str
    :param name: Name of the mouse file
    :type name: Str
    :param contrast: Flag to enhance contrast
    :type contrast: Bool
    :return: Image stack, Lungs segmentation, Metastasis segmentation
    :rtype: np.array, np.array, np.array
    """
    utils.print_red(name)
    dataset = data.get_predict_dataset(path, contrast)

    seg_lungs = p.predict_seg(dataset, path_model_seg_lungs).reshape(-1, 128, 128)
    seg_lungs = laplace(dataset, seg_lungs)
    seg_lungs = lungs_sanity_check(seg_lungs)
    seg_lungs = p.postprocess_loop(seg_lungs)

    seg_metas = (
        seg_lungs * p.predict_seg(dataset, path_model_seg_metas).reshape(-1, 128, 128)
    ).astype(np.uint8)
    seg_metas = p.postprocess_meta(seg_metas, k1=3, k2=3)
    return dataset, seg_lungs, seg_metas


def nb_meta_volume(meta_mask, i):
    """
    Write csv 1 line = 1 meta

    :param meta_mask: Metas mask list
    :type meta_mask: np.array
    :param i: slice number
    :type i: int
    """
    labeled_mask, num = measure.label(meta_mask, return_num=True)
    for nb in range(num):
        vol_metaid = get_vol_metaid(labeled_mask, nb + 1)
        write_meta_in_csv(
            flags.csv_file, flags.mousename, i, nb + 1, vol_metaid, flags.mut
        )


def process_for_stats(seg_lungs, seg_metas):
    """

    :param seg_lungs: Lungs mask list
    :type seg_lungs: np.array
    :param seg_metas: Metas mask list
    :type seg_metas: np.array
    :return: Lungs volume, meta volume, volume per meta
    :rtype: (float, float, float)
    """
    vol = 0
    for mask in seg_lungs:
        vol += post_count.vol_mask(mask)

    vol_meta = 0
    vol_per_meta = 0
    meta_slice = 0
    for i, meta_mask in enumerate(seg_metas):
        if np.amax(meta_mask) == 1.0:
            vol_meta += post_count.vol_mask(meta_mask)
            vol_per_meta += post_count.mean_vol_per_meta(meta_mask)
            # nb_meta_volume(meta_mask, i)
            meta_slice += 1
    vol_per_meta /= meta_slice
    print(100 * "-")
    print("\n")
    utils.print_gre(name)
    print("\n")
    utils.print_gre(
        "Lungs volume : {:.2f} mm cubed ; {:.5f} mL".format(vol, (vol * 0.001))
    )
    utils.print_gre(
        "Metas volume : {:.2f} mm cubed ; {:.5f} mL".format(
            vol_meta, (vol_meta * 0.001)
        )
    )
    utils.print_gre(
        "Volume per meta : {:.2f} mm cubed ; {:.5f} mL".format(
            vol_per_meta, (vol_per_meta * 0.001)
        )
    )
    return vol, vol_meta, vol_per_meta


if __name__ == "__main__":
    flags = get_flags()
    # Model seg lungs
    path_model_seg_lungs = os.path.join(
        gv.PATH_SAVE, "Poumons/best_seg_model_weighted.h5"
    )
    # Model seg metas
    path_model_seg_metas = os.path.join(gv.PATH_SAVE, "Metastases/best_seg_weighted.h5")

    # MOUSE INFOS
    MOUSE_PATH = os.path.join(gv.PATH_DATA, "Souris_Test/" + flags.mousename)
    name = os.path.basename(MOUSE_PATH)
    souris = (MOUSE_PATH, flags.contrast)

    # PREDICTIONS
    dataset, seg_lungs, seg_metas = process_mouse(souris[0], name, souris[1])

    # OUTPUT
    if flags.save:
        p.save_res(dataset, seg_lungs, name + "_pipeline_lungs", mask=flags.mask)
        p.save_res(dataset, seg_metas, name + "_pipeline_metas", mask=flags.mask)

    if flags.stats:
        LABEL_PATH = get_label_path(MOUSE_PATH, name)
        LABEL_PATH_METAS = get_label_path(MOUSE_PATH, name, folder="metas")
        label_lungs = get_label_masks(LABEL_PATH)
        label_metas = get_label_masks(LABEL_PATH_METAS)

        stats.do_stats(
            seg_lungs, label_lungs, gv.PATH_RES + str(name) + "_pipeline_lungs/"
        )
        stats.do_stats(
            seg_metas, label_metas, gv.PATH_RES + str(name) + "_pipeline_metas/"
        )

    vol, vol_meta, vol_per_meta = process_for_stats(seg_lungs, seg_metas)
    #########################

    if flags.csv:
        if flags.day is None:
            utils.print_red("flag --day is None")
        if flags.mousename is None:
            utils.print_red("flag --mousename is None")
        if flags.csv_file is None:
            utils.print_red("flag --csv_file is None")
        if flags.mut is None:
            utils.print_red("flag --mut is None")
        else:
            write_in_csv(
                flags.csv_file,
                flags.mousename,
                flags.day,
                vol,
                vol_meta,
                vol_per_meta,
                flags.mut,
            )

# todo: readme pour csv, move csv function in a new file (?)
# todo: clean this code -> thingz to generate graphs are polluting the code
# todo: lien vers napari-deepmeta dans le readme
