#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os

import cc3d
import cv2
import numpy as np
import skimage.io as io
import skimage.measure as measure

# FOLDER_PATH = "/home/elefevre/Desktop/mask_prediction/"
# FOLDER_PATH = "/home/elefevre/Desktop/Mask_Souris8bis/"
FOLDER_PATH = "/home/elefevre/Documents/Data/Souris_Test/Masque_Poumons/masque_8/"


def get_imgs(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def remove_blobs(img, min_size=10):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8
    )
    sizes = stats[1:, -1]
    nb_components = nb_components - 1  # remove background
    img2 = np.zeros(output.shape)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2


def dilate_and_erode(img, k1=3, k2=3):
    kernel1 = np.ones((k1, k1), np.uint8)
    kernel2 = np.ones((k2, k2), np.uint8)

    img_dilation = cv2.dilate(img, kernel1, iterations=1)
    img_erosion2 = cv2.erode(img_dilation, kernel2, iterations=1)
    return img_erosion2


def calculate_volume(path=FOLDER_PATH):
    file_list = get_imgs(path)
    vol = 0
    for file in file_list:
        mask = io.imread(path + file)
        blob_removed = remove_blobs(mask)
        eroded = dilate_and_erode(blob_removed)
        vol += np.count_nonzero(eroded == 255) * 0.0047
    return vol


def post_process(mask_path):
    mask = io.imread(mask_path)
    blob_removed = remove_blobs(mask)
    eroded = dilate_and_erode(blob_removed)
    return eroded


def mean_vol_per_meta(mask, vol=0.0047):
    _, num = measure.label(mask, return_num=True)
    nb_pix = mask.sum()
    return (nb_pix * vol) / num


def vol_mask(mask, vol=0.0047):
    nb_pix = mask.sum()
    return nb_pix * vol


def process_meta_number(meta_masks):
    labels_out, N = cc3d.connected_components(
        meta_masks, return_N=True, connectivity=18
    )
    # print('N 18: {}'.format(N))
    #
    # for segid in range(1, N + 1):
    #     extracted_image = labels_out * (labels_out == segid)
    #     io.imsave("img_{}.tiff".format(segid), extracted_image, plugin="tifffile")
    return N


if __name__ == "__main__":
    print("Volume = " + str(calculate_volume()))
