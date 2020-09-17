#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import os

import tensorflow.keras as keras

import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
    opt = utils.get_args()
    if opt.meta:
        train(path_images=gv.meta_path_img, path_labels=gv.meta_path_lab)
    else:
        train()
