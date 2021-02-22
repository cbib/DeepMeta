#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

"""
Train seg
==========
This files is used to train networks to segment images.
"""

import os

import tensorflow.keras as keras

import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.tune_reporter as tune_rep
import DeepMetav4.utils.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def train(args, path_images=gv.path_img, path_labels=gv.path_lab, hp_search=True):
    file_path = data.save_model_name(args, gv.PATH_SAVE)
    dataset, dataset_val, model_seg, checkpoint, metric = data.new_prepare_for_training(
        path_images, path_labels, file_path, args
    )
    earlystopper = keras.callbacks.EarlyStopping(
        monitor=metric,
        patience=args["patience"],
        verbose=1,
        min_delta=0.001,
        restore_best_weights=True,
    )
    cb_list = [earlystopper, utils.CosLRDecay(args["n_epochs"], args["lr"])]
    if hp_search:
        cb_list.append(tune_rep.TuneReporter(metric=metric))
    else:
        cb_list.append(checkpoint)
    history = model_seg.fit(
        dataset,
        validation_data=dataset_val,
        epochs=args["n_epochs"],
        callbacks=cb_list,
    )
    if not hp_search:
        utils.plot_learning_curves(
            history, "segmentation_" + args["model_name"], metric
        )


def train_meta(args, path_images=gv.path_img, path_labels=gv.path_lab, hp_search=True):
    file_path = data.save_model_name(args, gv.PATH_SAVE)
    dataset, dataset_val, model_seg, checkpoint = data.meta_for_training(
        path_images, path_labels, file_path, args
    )
    earlystopper = keras.callbacks.EarlyStopping(
        patience=args["patience"],
        verbose=1,
        min_delta=0.001,
        restore_best_weights=True,
    )
    cb_list = [earlystopper, utils.CosLRDecay(args["n_epochs"], args["lr"])]
    if hp_search:
        cb_list.append(tune_rep.TuneReporter(metric="loss"))
    else:
        cb_list.append(checkpoint)
    history = model_seg.fit(
        dataset,
        validation_data=dataset_val,
        epochs=args["n_epochs"],
        callbacks=cb_list,
    )
    if not hp_search:
        utils.plot_learning_curves(
            history, "segmentation_" + args["model_name"], "loss"
        )


if __name__ == "__main__":
    opt = vars(utils.get_args())
    if opt["meta"]:
        train_meta(
            opt,
            path_images=gv.meta_path_img,
            path_labels=gv.meta_path_lab,
            hp_search=False,
        )
    else:
        train(opt, hp_search=False)
