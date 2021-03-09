#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import os
import time

import tensorflow as tf
import tensorflow.keras as keras

import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.tune_reporter as tune_rep
import DeepMetav4.utils.utils as utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_save_name(args):
    if args["meta"]:
        save_name = "Metastases/model_"
    else:
        save_name = "Poumons/model_"
    return save_name


@tf.function
def train_step(model, x, y, loss_fn, optimizer, train_metric):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss_value = loss_fn(y, pred)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_metric.update_state(y, pred)


@tf.function
def test_step(model, x, y, test_metric):
    val_logits = model(x, training=False)
    test_metric.update_state(y, val_logits)


def train_model(model, epochs, train_dataset, val_dataset, lr):
    train_acc_metric = keras.metrics.BinaryAccuracy()
    val_acc_metric = keras.metrics.BinaryAccuracy()
    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(lr=lr)
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            train_step(
                model,
                x_batch_train,
                y_batch_train,
                loss_fn,
                optimizer,
                train_acc_metric,
            )
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        train_acc_metric.reset_states()
        for x_batch_val, y_batch_val in val_dataset:
            test_step(model, x_batch_val, y_batch_val, val_acc_metric)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))


# Function used to train Lungs detection (ie is there lungs in this image)
def train_detect(args, model_name="detection", hp_search=True):
    utils.print_red("Training Detect : ")
    save_name = get_save_name(args)
    input_shape = (
        args["size"],
        args["size"],
        1,
    )
    # tab = gv.tab_meta
    train_ds, val_ds = data.new_dataset_detect(
        "/home/edgar/Documents/Datasets/deepmeta/Data/Classif_lungs/", args
    )
    # dataset, label = data.create_dataset_detect(
    #     gv.path_img_classif, tab, args["size"], meta=args["meta"]
    # )
    # print(label)
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
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
        utils.plot_learning_curves(history, name="detect", metric="binary_accuracy")
    # train_model(model_detect, args["n_epochs"], train_ds, val_ds, args["lr"])


if __name__ == "__main__":
    opt = vars(utils.get_args())
    train_detect(opt, hp_search=False)
