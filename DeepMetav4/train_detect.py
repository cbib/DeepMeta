#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import tensorflow.keras as keras

import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.tune_reporter as tune_rep
import DeepMetav4.utils.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_save_name(args):
    if args["meta"]:
        save_name = "Metastases/model_"
    else:
        save_name = "Poumons/model_"
    return save_name


def train_detect(
    args, img_path=gv.path_classif_lungs, model_name="detection", hp_search=True
):
    utils.print_red("Training Detect : ")
    save_name = get_save_name(args)
    input_shape = (
        args["size"],
        args["size"],
        1,
    )
    # tab = gv.tab_meta
    train_ds, val_ds = data.dataset_detect(img_path, args)
    # dataset, label = data.create_dataset_detect(
    #     gv.path_img_classif, tab, args["size"], meta=args["meta"]
    # )
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model_detect = gv.model_list[model_name](
            input_shape, args["lr"], drop_r=args["drop_r"], filters=args["filters"]
        )
        es = keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            patience=args["patience"],
            restore_best_weights=True,
        )
        cb_list = [es, utils.CosLRDecay(args["n_epochs"], args["lr"])]
        if hp_search:
            cb_list.append(tune_rep.TuneReporter())
        else:
            file_path = os.path.join(gv.PATH_SAVE, save_name + model_name + ".h5")
            checkpoint = keras.callbacks.ModelCheckpoint(
                file_path,
                monitor="val_accuracy",
                verbose=1,
                save_best_only=True,
                mode="max",
            )
            cb_list.append(checkpoint)
        # history = model_detect.fit(
        #     dataset,
        #     label,
        #     validation_split=0.2,
        #     batch_size=args["batch_size"],
        #     epochs=args["n_epochs"],
        #     callbacks=cb_list,
        # )

        history = model_detect.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args["n_epochs"],
            callbacks=cb_list,
        )
    if not hp_search:
        utils.plot_learning_curves(history, name="detect", metric="val_accuracy")


if __name__ == "__main__":
    opt = vars(utils.get_args())
    if opt["meta"]:
        train_detect(opt, img_path=gv.path_classif_metas, hp_search=False)
    else:
        train_detect(opt, hp_search=False)
