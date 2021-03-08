import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def model_detection(input_shape, lr, n_classes=2, filters=32, drop_r=0.2):
    inputs = layers.Input(input_shape)

    b1_c = layers.Conv2D(filters, (3, 3), activation="linear", padding="same")(inputs)
    b1_c = layers.Dropout(rate=drop_r)(b1_c)  # initially 0.5
    b1_l = layers.LeakyReLU(alpha=0.1)(b1_c)
    b1_mp = layers.MaxPool2D(2)(b1_l)

    b2_c = layers.Conv2D(filters * 2, (3, 3), activation="linear", padding="same")(
        b1_mp
    )
    b2_c = layers.Dropout(rate=drop_r)(b2_c)
    b2_l = layers.LeakyReLU(alpha=0.1)(b2_c)
    b2_mp = layers.MaxPool2D(2)(b2_l)

    b3_c = layers.Conv2D(filters * 4, (3, 3), activation="linear", padding="same")(
        b2_mp
    )
    b3_c = layers.Dropout(rate=drop_r)(b3_c)
    b3_l = layers.LeakyReLU(alpha=0.1)(b3_c)
    b3_mp = layers.MaxPool2D(2)(b3_l)

    b4_c = layers.Conv2D(filters * 8, (3, 3), activation="linear", padding="same")(
        b3_mp
    )
    b4_c = layers.Dropout(rate=drop_r)(b4_c)
    b4_l = layers.LeakyReLU(alpha=0.1)(b4_c)
    b4_mp = layers.MaxPool2D(2)(b4_l)

    flat = layers.Flatten()(b4_mp)

    b4_d = layers.Dense(filters * 4, activation="linear")(flat)
    b4_l = layers.LeakyReLU(alpha=0.1)(b4_d)

    outputs = layers.Dense(n_classes, activation="softmax")(b4_l)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=["accuracy"],
    )

    return model


def model_detection_bn(input_shape, lr, n_classes=2, w_decay=0.0001):
    inputs = layers.Input(input_shape)

    b1_c = layers.Conv2D(
        16,
        (3, 3),
        activation="linear",
        padding="same",
        kernel_initializer=tf.keras.initializers.glorot_normal(),
        kernel_regularizer=tf.keras.regularizers.l2(w_decay),
    )(inputs)
    b1_c = layers.Dropout(rate=0.4)(b1_c)  # initially 0.4
    b1_l = layers.LeakyReLU(alpha=0.1)(b1_c)
    b1_bn = layers.BatchNormalization()(b1_l)
    b1_mp = layers.MaxPool2D(2)(b1_bn)

    b2_c = layers.Conv2D(
        32,
        (3, 3),
        activation="linear",
        padding="same",
        kernel_initializer=tf.keras.initializers.glorot_normal(),
        kernel_regularizer=tf.keras.regularizers.l2(w_decay),
    )(b1_mp)
    b2_c = layers.Dropout(rate=0.4)(b2_c)
    b2_l = layers.LeakyReLU(alpha=0.1)(b2_c)
    b2_bn = layers.BatchNormalization()(b2_l)
    b2_mp = layers.MaxPool2D(2)(b2_bn)

    # b3_c = layers.Conv2D(32, (3, 3), activation="linear", padding='same',
    # kernel_initializer=tf.keras.initializers.glorot_normal(),
    #                      kernel_regularizer=tf.keras.regularizers.l2(w_decay))(b2_mp)
    # b3_c = layers.Dropout(rate=0.25)(b3_c)
    # b3_l = layers.LeakyReLU(alpha=0.1)(b3_c)
    # b3_bn = layers.BatchNormalization()(b3_l)
    # b3_mp = layers.MaxPool2D(2)(b3_bn)

    # b4_c = layers.Conv2D(64, (3, 3), activation="linear", padding='same',
    # kernel_initializer=tf.keras.initializers.glorot_normal(),
    #                      kernel_regularizer=tf.keras.regularizers.l2(w_decay))(b3_mp)
    # b4_c = layers.Dropout(rate=0.25)(b4_c)
    # b4_l = layers.LeakyReLU(alpha=0.1)(b4_c)
    # b4_bn = layers.BatchNormalization()(b4_l)
    # b4_mp = layers.MaxPool2D(2)(b4_bn)

    flat = layers.Flatten()(b2_mp)

    b4_d = layers.Dense(64, activation="linear")(flat)
    b4_l = layers.LeakyReLU(alpha=0.1)(b4_d)

    outputs = layers.Dense(n_classes, activation="softmax")(b4_l)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=["accuracy"],
    )

    return model


def model_detection_stride(input_shape, lr, n_classes=2):
    inputs = layers.Input(input_shape)

    b1_c = layers.Conv2D(32, (3, 3), activation="linear", padding="same", strides=2)(
        inputs
    )
    b1_c = layers.Dropout(rate=0.2)(b1_c)  # initially 0.5
    b1_l = layers.LeakyReLU(alpha=0.1)(b1_c)

    b2_c = layers.Conv2D(64, (3, 3), activation="linear", padding="same", strides=2)(
        b1_l
    )
    b2_c = layers.Dropout(rate=0.2)(b2_c)
    b2_l = layers.LeakyReLU(alpha=0.1)(b2_c)

    b3_c = layers.Conv2D(128, (3, 3), activation="linear", padding="same", strides=2)(
        b2_l
    )
    b3_c = layers.Dropout(rate=0.2)(b3_c)
    b3_l = layers.LeakyReLU(alpha=0.1)(b3_c)

    b4_c = layers.Conv2D(256, (3, 3), activation="linear", padding="same", strides=2)(
        b3_l
    )
    b4_c = layers.Dropout(rate=0.2)(b4_c)
    b4_l = layers.LeakyReLU(alpha=0.1)(b4_c)

    flat = layers.Flatten()(b4_l)

    b4_d = layers.Dense(128, activation="linear")(flat)
    b4_l = layers.LeakyReLU(alpha=0.1)(b4_d)

    outputs = layers.Dense(n_classes, activation="softmax")(b4_l)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=["binary_accuracy"],
    )

    return model
