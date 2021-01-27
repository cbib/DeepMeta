#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import tensorflow.keras as keras

import DeepMetav4.tune_reporter as tune_rep
import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.utils as utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Function used to train Lungs detection (ie is there lungs in this image)
def train_detect(args, model_name="detection", hp_search=True):
    utils.print_red("Training Detect : ")
    if args["meta"]:
        dataset, label = data.create_dataset_detect(
            gv.path_img_classif, gv.tab_meta, args["size"]
        )
        save_name = "Metastases/model_"
    else:
        dataset, label = data.create_dataset_detect(
            gv.path_img_classif, gv.tab, args["size"]
        )
        save_name = "Poumons/model_"
    input_shape = (
        args["size"],
        args["size"],
        1,
    )
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
            min_delta=0.00001,
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
        history = model_detect.fit(
            dataset,
            label,
            validation_split=0.2,
            batch_size=args["batch_size"],
            epochs=args["n_epochs"],
            callbacks=cb_list,
        )
        if not hp_search:
            utils.plot_learning_curves(history, name="detect", metric="accuracy")


if __name__ == "__main__":
    opt = vars(utils.get_args())
    train_detect(opt, hp_search=False)
