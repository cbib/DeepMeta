import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
from tensorflow.python.framework import dtypes


def block_down(inputs, filters, drop, w_decay=0.0001, kernel_size=3, name=""):
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        kernel_regularizer=regularizers.l2(w_decay),
        activation="elu",
        name=name + "_conv1",
    )(inputs)
    c = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        activation="elu",
        kernel_initializer="he_normal",
        padding="same",
        kernel_regularizer=regularizers.l2(w_decay),
        name=name + "_conv2",
    )(x)
    p = layers.MaxPooling2D((2, 2), name=name + "_maxpool")(c)
    p = layers.Dropout(drop, name=name + "_dropout")(p)
    return c, p


def bridge(inputs, filters, drop, kernel_size=3):
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        activation="elu",
        name="bridge_conv1",
    )(inputs)
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        activation="elu",
        name="bridge_conv2",
    )(x)
    x = layers.Dropout(drop, name="bridge_dropout")(x)
    return x


def block_up(inputs, conc, filters, drop, w_decay=0.0001, kernel_size=3, name=""):
    x = layers.Conv2DTranspose(
        filters,
        (2, 2),
        strides=(2, 2),
        padding="same",
        kernel_regularizer=regularizers.l2(w_decay),
        name=name + "_convTranspose",
    )(inputs)
    for i in range(len(conc)):
        x = layers.concatenate([x, conc[i]], name=name + "_concatenate" + str(i))
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        kernel_regularizer=regularizers.l2(w_decay),
        activation="elu",
        name=name + "_conv1",
    )(x)
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        kernel_regularizer=regularizers.l2(w_decay),
        activation="elu",
        name=name + "_conv2",
    )(x)
    x = layers.Dropout(drop, name=name + "_dropout")(x)
    return x


def weighted_cross_entropy(y_true, y_pred):
    """
    -- Fonction de coût pondéré --
    :param y_true: vrai valeur de y (label)
    :param y_pred: valeur prédite de y par le modèle
    :return: valeur de la fonction de cout d'entropie croisée pondérée
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=3)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except Exception:
        pass

    epsilon = tf.convert_to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = tf.zeros_like(y_pred, dtype=y_pred.dtype)  # array_ops
    cond = y_pred >= zeros
    relu_logits = tf.where(cond, y_pred, zeros)
    neg_abs_logits = tf.where(cond, -y_pred, y_pred)
    entropy = tf.math.add(
        relu_logits - y_pred * seg,
        tf.math.log1p(tf.math.exp(neg_abs_logits)),
        name=None,
    )

    return K.mean(tf.multiply(weight, entropy), axis=-1)