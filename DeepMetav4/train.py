#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Function used to train Lungs detection (ie is there lungs in this image)
def train_detect(args, model_name="detection"):
    utils.print_red("Training Detect : ")
    if args.meta:
        dataset, label = data.create_dataset_detect_meta(
            gv.path_gen_img, gv.path_gen_lab, gv.tab_meta, args.size
        )
        save_name = "Metastases/model_"
    else:
        dataset, label = data.create_dataset_detect(
            gv.path_img, gv.tab, gv.numSouris, args.size
        )
        save_name = "Poumons/model_"
    utils.print_gre("label 0 : {}".format(np.sum(np.transpose(label)[0])))
    utils.print_gre("label 1 : {}".format(np.sum(np.transpose(label)[1])))
    input_shape = (
        args.size,
        args.size,
        1,
    )
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model_detect = gv.model_list[model_name](input_shape, args.lr)
        es = keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            patience=opt.patience,
            min_delta=0.00001,
            restore_best_weights=True,
        )
        file_path = os.path.join(gv.PATH_SAVE, save_name + model_name + ".h5")
        checkpoint = keras.callbacks.ModelCheckpoint(
            file_path,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max",
        )
        history = model_detect.fit(
            dataset,
            label,
            validation_split=0.2,
            batch_size=args.batch_size,
            epochs=args.n_epochs,
            callbacks=[es, checkpoint, utils.CosLRDecay(args.n_epochs, args.lr)],
        )
    utils.plot_learning_curves(history, name="detect", metric="accuracy")


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
        "--meta", type=bool, default=True, help="True if we want to segment metastasis"
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


if __name__ == "__main__":
    opt = get_args()
    # train_detect(opt, opt.model_name)
    # train()
    train(path_images=gv.meta_path_img, path_labels=gv.meta_path_lab)
