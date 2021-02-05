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


class WeightedMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name=None, dtype=None, weighted=False):
        super(WeightedMeanIoU, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            "total_confusion_matrix",
            shape=(num_classes, num_classes),
            initializer=tf.zeros_initializer,
            dtype=dtypes.float64,
        )
        self.weighted = weighted

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.weighted:
            [seg, weight] = tf.unstack(y_true, 2, axis=3)
            y_true = tf.expand_dims(seg, -1)
            sample_weight = tf.expand_dims(weight, -1)
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast((y_pred > tf.constant(0.5)), self._dtype)
        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = tf.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=dtypes.float64,
        )
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype
        )
        denominator = sum_over_row + sum_over_col - true_positives
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype)
        )
        iou = tf.math.divide_no_nan(true_positives, denominator)
        return tf.math.divide_no_nan(
            tf.reduce_sum(iou, name="mean_iou"), num_valid_entries
        )

    def reset_states(self):
        K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {"num_classes": self.num_classes}
        base_config = super(WeightedMeanIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def mean_iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = (K.sum(y_true_f + y_pred_f)) - intersection
    return intersection / union


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection.numpy() + smooth) / (
        K.sum(y_true_f).numpy() + K.sum(y_pred_f).numpy() + smooth
    )


def soft_dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
