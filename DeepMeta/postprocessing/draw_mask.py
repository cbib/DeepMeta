#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os
import re

import matplotlib.pyplot as plt
import skimage
import skimage.io as io

"""
File used to draw masks on image, you can provide one or two masks folders
FIRT_LUNG_SLICE is the number of the first slice in which you can see lungs
"""

SAVE_PATH = "/home/elefevre/Desktop/double_mask/"
FIRST_LUNG_SLICE = 30
MOUSE_PATH = "/home/elefevre/Documents/Doc/Projet_Detection_Metastase_Souris/Data/iL34_1c/2PLc_day8.tif"  # noqa
MASK1_PATH = "/home/elefevre/Desktop/mask_souris4/"
MASK2_PATH = "/home/elefevre/Documents/Data/Souris_Test/Masque_Poumons/masque_8/"


def get_imgs(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def get_imgs_from_mouse(path):
    res = []
    img = io.imread(path, plugin="tifffile")
    for slice in img:
        res.append(slice)
    return res


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa
    return sorted(data, key=alphanum_key)


def border(slice, mask1, mask2=None):
    # draw mask border on img
    img_file = slice
    mask1_file = io.imread(MASK1_PATH + mask1)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    cell_contours = skimage.measure.find_contours(mask1_file, 0.8)
    for n, contour in enumerate(cell_contours):
        ax.plot(
            contour[:, 1],
            contour[:, 0],
            linewidth=1,
            color="blue",
            label="Ground truth",
        )
    if mask2 is not None:
        mask2_file = io.imread(MASK2_PATH + mask2)
        cell_contours = skimage.measure.find_contours(mask2_file, 0.8)
        for n, contour in enumerate(cell_contours):
            ax.plot(
                contour[:, 1],
                contour[:, 0],
                linewidth=1,
                color="red",
                label="Predicted",
            )
    plt.legend(loc="upper left")
    plt.xlim((0, 128))
    plt.ylim((128, 0))
    plt.imshow(img_file, cmap="gray")
    plt.savefig(SAVE_PATH + mask1)
    plt.close(fig)


if __name__ == "__main__":
    # -> Simple mask
    imgs_list = get_imgs_from_mouse(MOUSE_PATH)
    mask1_list = sorted_aphanumeric(get_imgs(MASK1_PATH))
    mask2_list = sorted_aphanumeric(get_imgs(MASK2_PATH))

    for i in range(len(mask1_list)):
        img = imgs_list[i + FIRST_LUNG_SLICE]
        mask = mask1_list[i]
        print(mask)
        border(img, mask)

    # # -> Double Mask  (assume there is the same number of masks in the two folders)
    # imgs_list = get_imgs_from_mouse(MOUSE_PATH)
    # mask1_list = sorted_aphanumeric(get_imgs(MASK1_PATH))
    # mask2_list = sorted_aphanumeric(get_imgs(MASK2_PATH))
    #
    # for i in range(len(mask1_list)):
    #     img = imgs_list[i + 21]
    #     mask = mask1_list[i]
    #     mask2 = mask2_list[i]
    #     print(mask)
    #     border(img, mask, mask2)
