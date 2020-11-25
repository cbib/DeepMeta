#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os

import pandas as pd

import DeepMetav4.models.model_detection
import DeepMetav4.models.model_resnet
import DeepMetav4.models.model_unet
import DeepMetav4.models.model_vgg16
import DeepMetav4.models.small_plus_plus
import DeepMetav4.utils.utils

# THE ONLY VAR YOU NEED TO MODIFY
# PATH_DATA = "/home/elefevre/Datasets/deepmeta/Data/"
PATH_DATA = "/home/edgar/Documents/Datasets/deepmeta/Data/"

MAX_GPU_NUMBER = 6

##########################################################
PATH_SAVE = "data/saved_models/"
PATH_RES = "data/results/"
PATH_FILTERS_FEATURES = os.path.join(PATH_RES, "Filters_and_features/")

PATH_CLASSIF = os.path.join(PATH_DATA, "Classif/")
PATH_LUNGS = os.path.join(PATH_DATA, "Poumons/")
PATH_META = os.path.join(PATH_DATA, "Metastases/")


path_souris = os.path.join(PATH_CLASSIF, "Souris/")
tab = pd.read_csv(os.path.join(PATH_CLASSIF, "Tableau_General.csv")).values
tab_meta = pd.read_csv(os.path.join(PATH_META, "tab_meta.csv")).values
path_img_classif = os.path.join(PATH_CLASSIF, "Images/")

path_mask = os.path.join(PATH_LUNGS, "Masques/")
path_img = os.path.join(PATH_LUNGS, "Images/")
path_lab = os.path.join(PATH_LUNGS, "Labels/")

path_gen_img = os.path.join(PATH_DATA, "Dataset_generated/Images/")
path_gen_lab = os.path.join(PATH_DATA, "Dataset_generated/Labels/")

path_masked_img = os.path.join(PATH_DATA, "Metastases/masked_imgs/")

# meta_path_mask = os.path.join(PATH_META, "Masques/")
meta_path_img = os.path.join(PATH_META, "Images_new/")
meta_path_lab = os.path.join(PATH_META, "Labels_new/")

numSouris = DeepMetav4.utils.utils.calcul_numSouris(path_souris)


model_list = {
    "detection": DeepMetav4.models.model_detection.model_detection,
    "detection_bn": DeepMetav4.models.model_detection.model_detection_bn,
    "detection_stride": DeepMetav4.models.model_detection.model_detection_stride,
    "small++": DeepMetav4.models.small_plus_plus.small_plus_plus,
    "unet": DeepMetav4.models.model_unet.unet,
    "vgg16": DeepMetav4.models.model_vgg16.vgg16,
    "resnet34": DeepMetav4.models.model_resnet.resnet34,
    "resnet18": DeepMetav4.models.model_resnet.resnet18,
    "resnet50": DeepMetav4.models.model_resnet.resnet50,
    "resnetv2": DeepMetav4.models.model_resnet.resnetv2,
}
