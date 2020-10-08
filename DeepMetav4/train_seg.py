#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

"""
Train seg
==========
This files is used to train networks to segment images.
"""

import os

import tensorflow.keras as keras
import argparse

import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.utils as utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=300, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument(
        "--model_name",
        type=str,
        default="small++",
        help="Name of the model you want to train (detection, small++)",
    )
    parser.add_argument(
        "--meta", type=bool, default=False, help="True if we want to segment metastasis"
    )
    parser.add_argument(
        "--weighted",
        type=bool,
        default=False,
        help="Use weighted model (default False)",
    )
    parser.add_argument(
        "--size", type=int, default=128, help="Size of the image, one number"
    )
    parser.add_argument("--w1", type=int, default=2, help="weight inside")
    parser.add_argument("--w2", type=int, default=4, help="Weight border")
    parser.add_argument(
        "--patience", type=int, default=10, help="Set patience value for early stopper"
    )
    # parser.add_argument("--load", type=str, default=gv.PATH_SAVE
    #                     \ + "Poumons/test_small++_weighted24.h5",
    #                     help="path for a model if you want to load weights")
    args = parser.parse_args()
    utils.print_red(args)
    return args

def train(path_images=gv.path_img, path_labels=gv.path_lab):
    file_path = data.save_model_name(opt, gv.PATH_SAVE)
    dataset, label, model_seg, checkpoint, metric = data.prepare_for_training(
        path_images, path_labels, file_path, opt
    )
    earlystopper = keras.callbacks.EarlyStopping(
        patience=opt.patience, verbose=1, min_delta=0.00001, restore_best_weights=True
    )

    history = model_seg.fit(
        dataset,
        label,
        validation_split=0.2,
        batch_size=opt.batch_size,
        epochs=opt.n_epochs,
        callbacks=[earlystopper, checkpoint, utils.CosLRDecay(opt.n_epochs, opt.lr)],
    )
    utils.plot_learning_curves(history, "segmentation_" + opt.model_name, metric)


if __name__ == "__main__":
    opt = get_args()
    if opt.meta:
        train(path_images=gv.meta_path_img, path_labels=gv.meta_path_lab)
    else:
        train()
