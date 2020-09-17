import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers


def conv_block(inputs, filters, w_decay, strides=1, block_name=""):
    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        strides=strides,
        kernel_regularizer=regularizers.l2(w_decay),
    )(x)
    return x


def shortcut(inputs, filters, strides, block_name="l", v2=False, w_decay=(0.0001)):
    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)
    if v2:
        print(x.shape)
        x = layers.AveragePooling2D(pool_size=2, strides=strides)(x)
        x = layers.Conv2D(
            filters,
            1,
            name=block_name + "_shortcut",
            padding="same",
            kernel_regularizer=regularizers.l2(w_decay),
        )(x)
        print(x.shape)
    else:
        x = layers.Conv2D(
            filters,
            1,
            strides=strides,
            name=block_name + "_shortcut",
            kernel_regularizer=regularizers.l2(w_decay),
        )(x)
    return x


def bottleneck(inputs, filters, strides, w_decay, block_name):
    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters // 4,
        1,
        strides=strides,
        kernel_regularizer=regularizers.l2(w_decay),
        kernel_initializer=tf.keras.initializers.he_normal(),
        name=block_name + "_conv1",
        padding="same",
    )(x)
    # x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters // 4,
        3,
        padding="same",
        kernel_regularizer=regularizers.l2(w_decay),
        kernel_initializer=tf.keras.initializers.he_normal(),
        name=block_name + "_conv2",
    )(x)
    # x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters,
        1,
        kernel_regularizer=regularizers.l2(w_decay),
        kernel_initializer=tf.keras.initializers.he_normal(),
        name=block_name + "_conv3",
    )(x)
    x = layers.Dropout(0.4)(x)
    return x


def resnet_block(filters, inputs, strides=1, w_decay=0.0001, block_name=""):
    x = conv_block(
        inputs, filters, w_decay, strides=strides, block_name=block_name + "_conv1"
    )
    x = conv_block(x, filters, w_decay, block_name=block_name + "_conv2")
    if strides != 1:
        inputs = shortcut(inputs, filters, strides, block_name)
    x = layers.Add()([inputs, x])
    return x


def resnet_bottleneck_block(
    filters, inputs, block_name, strides=1, w_decay=0.0001, v2=False
):
    x = bottleneck(inputs, filters, strides, w_decay, block_name)
    # if strides != 1:
    inputs = shortcut(inputs, filters, strides, block_name, v2)
    x = layers.Add()([inputs, x])
    return x


def resnet34(input_shape, lr, nb_classes=2, base_filter=16):
    inputs = layers.Input(input_shape)

    b1_c1 = layers.Conv2D(
        filters=base_filter,
        kernel_size=7,
        strides=2,
        padding="same",
        name="block1_conv1",
    )(inputs)
    b1_bn = layers.BatchNormalization()(b1_c1)
    b1_relu = layers.ReLU()(b1_bn)
    b1_mp = layers.MaxPooling2D(pool_size=3, strides=2)(b1_relu)

    b2_c1 = resnet_block(base_filter, b1_mp, block_name="block2_1")
    b2_c2 = resnet_block(base_filter, b2_c1, block_name="block2_2")
    b2_c3 = resnet_block(base_filter, b2_c2, block_name="block2_3")

    # b3_c1 = resnet_block(base_filter * 2, b2_c3, strides=2, block_name="block3_1")
    # b3_c2 = resnet_block(base_filter * 2, b3_c1, block_name="block3_2")
    # b3_c3 = resnet_block(base_filter * 2, b3_c2, block_name="block3_3")
    # b3_c4 = resnet_block(base_filter * 2, b3_c3, block_name="block3_4")

    # b4_c1 = resnet_block(base_filter * 4, b3_c4, strides=2, block_name="block4_1")
    # b4_c2 = resnet_block(base_filter * 4, b4_c1, block_name="block4_2")
    # b4_c3 = resnet_block(base_filter * 4, b4_c2, block_name="block4_3")
    # b4_c4 = resnet_block(base_filter*4, b4_c3, block_name="block4_4")
    # b4_c5 = resnet_block(base_filter*4, b4_c4, block_name="block4_5")
    # b4_c6 = resnet_block(base_filter*4, b4_c5, block_name="block4_6")
    # #
    # b5_c1 = resnet_block(base_filter * 8, b4_c6, strides=2, block_name="block5_1")
    # b5_c2 = resnet_block(base_filter * 8, b5_c1, block_name="block5_2")
    # b5_c3 = resnet_block(base_filter * 8, b5_c2, block_name="block5_3")

    pool = layers.GlobalAveragePooling2D()(b2_c3)
    fc1 = layers.Dense(base_filter * 2, activation="relu")(pool)
    dp1 = layers.Dropout(0.5)(fc1)
    # fc2 = layers.Dense(base_filter, activation='relu')(dp1)
    # dp2 = layers.Dropout(0.5)(fc2)
    fc3 = layers.Dense(nb_classes, activation="softmax")(dp1)

    model = keras.Model(inputs=[inputs], outputs=[fc3])
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=["accuracy"],
    )
    return model


def resnet18(input_shape, lr, nb_classes=2, base_filter=16):
    inputs = layers.Input(input_shape)

    b1_c1 = layers.Conv2D(
        filters=base_filter, kernel_size=7, strides=2, padding="same"
    )(inputs)
    b1_bn = layers.BatchNormalization()(b1_c1)
    b1_relu = layers.ReLU()(b1_bn)
    b1_mp = layers.MaxPooling2D(pool_size=3, strides=2)(b1_relu)

    b2_c1 = resnet_block(base_filter, b1_mp)
    b2_c2 = resnet_block(base_filter, b2_c1)

    b3_c1 = resnet_block(base_filter * 2, b2_c2, strides=2)
    b3_c2 = resnet_block(base_filter * 2, b3_c1)

    b4_c1 = resnet_block(base_filter * 4, b3_c2, strides=2)
    b4_c2 = resnet_block(base_filter * 4, b4_c1)

    b5_c1 = resnet_block(base_filter * 8, b4_c2, strides=2)
    b5_c2 = resnet_block(base_filter * 8, b5_c1)

    pool = layers.GlobalAveragePooling2D()(b5_c2)
    fc1 = layers.Dense(base_filter * 8, activation="relu")(pool)
    dp1 = layers.Dropout(0.5)(fc1)
    # fc2 = layers.Dense(base_filter * 8, activation='relu')(dp1)
    # dp2 = layers.Dropout(0.5)(fc2)
    fc3 = layers.Dense(nb_classes, activation="softmax")(dp1)

    model = keras.Model(inputs=[inputs], outputs=[fc3])
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=["accuracy"],
    )
    return model


def resnet50(input_shape, lr, nb_classes=2, base_filter=8, w_decay=0.0001):
    inputs = layers.Input(input_shape)

    b1_c1 = layers.Conv2D(
        filters=base_filter,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=tf.keras.initializers.glorot_normal(),
        name="block1_conv1",
        kernel_regularizer=regularizers.l2(w_decay),
    )(inputs)
    b1_c1 = layers.Dropout(0.15)(b1_c1)
    b1_bn = layers.BatchNormalization()(b1_c1)
    b1_relu = layers.ReLU()(b1_bn)
    b1_mp = layers.MaxPooling2D(pool_size=3, strides=2)(b1_relu)

    base_filter = base_filter * 4

    b2_c1 = resnet_bottleneck_block(base_filter, b1_mp, "block2_1")
    b2_c2 = resnet_bottleneck_block(base_filter, b2_c1, "block2_2")
    b2_c3 = resnet_bottleneck_block(base_filter, b2_c2, "block2_3")
    #
    # b3_c1 = resnet_bottleneck_block(base_filter * 2, b2_c3, "block3_1", strides=2)
    # b3_c2 = resnet_bottleneck_block(base_filter * 2, b3_c1, "block3_2")
    # b3_c3 = resnet_bottleneck_block(base_filter * 2, b3_c2, "block3_3")
    # b3_c4 = resnet_bottleneck_block(base_filter * 2, b3_c3, "block3_4")

    # b4_c1 = resnet_bottleneck_block(base_filter * 4, b3_c4, "block4_1", strides=2)
    # b4_c2 = resnet_bottleneck_block(base_filter * 4, b4_c1, "block4_2")
    # b4_c3 = resnet_bottleneck_block(base_filter * 4, b4_c2, "block4_3")
    # b4_c4 = resnet_bottleneck_block(base_filter * 4, b4_c3, "block4_4")
    # b4_c5 = resnet_bottleneck_block(base_filter * 4, b4_c4, "block4_5")
    # b4_c6 = resnet_bottleneck_block(base_filter * 4, b4_c5, "block4_6")

    # b5_c1 = resnet_bottleneck_block(base_filter * 8, b4_c3, "block5_1", strides=2)
    # b5_c2 = resnet_bottleneck_block(base_filter * 8, b5_c1, "block5_2")
    # b5_c3 = resnet_bottleneck_block(base_filter * 8, b5_c2, "block5_3")

    pool = layers.GlobalAveragePooling2D(name="avg_pool")(b2_c3)
    fc1 = layers.Dense(
        base_filter,
        activation="relu",
        kernel_constraint=tf.keras.constraints.min_max_norm(),
    )(pool)
    dp1 = layers.Dropout(0.5)(fc1)
    # fc2 = layers.Dense(base_filter * 2, activation='relu',
    # kernel_constraint=tf.keras.constraints.min_max_norm())(dp1)
    # dp2 = layers.Dropout(0.5)(fc2)
    fc3 = layers.Dense(nb_classes, activation="softmax")(dp1)

    model = keras.Model(inputs=[inputs], outputs=[fc3])

    bce = keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    model.compile(
        loss=bce, optimizer=keras.optimizers.Adam(lr=lr), metrics=["accuracy"]
    )
    return model


def resnetv2(input_shape, lr, nb_classes=2, base_filter=8, w_decay=0.0001):
    inputs = layers.Input(input_shape)
    b1_c1 = layers.Conv2D(
        filters=base_filter,
        kernel_size=3,
        padding="same",
        kernel_initializer=tf.keras.initializers.he_normal(),
        name="block1_conv1",
        kernel_regularizer=regularizers.l2(w_decay),
    )(inputs)
    b1_c1 = layers.BatchNormalization()(b1_c1)
    b1_c1 = layers.ReLU()(b1_c1)
    b1_c2 = layers.Conv2D(
        filters=base_filter,
        kernel_size=3,
        padding="same",
        kernel_initializer=tf.keras.initializers.he_normal(),
        name="block1_conv2",
        kernel_regularizer=regularizers.l2(w_decay),
    )(b1_c1)
    b1_c2 = layers.BatchNormalization()(b1_c2)
    b1_c2 = layers.ReLU()(b1_c2)
    b1_c3 = layers.Conv2D(
        filters=base_filter,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=tf.keras.initializers.he_normal(),
        name="block1_conv3",
        kernel_regularizer=regularizers.l2(w_decay),
    )(b1_c2)
    b1_bn = layers.BatchNormalization()(b1_c3)
    b1_relu = layers.ReLU()(b1_bn)
    b1_mp = layers.MaxPooling2D(pool_size=3, strides=2)(b1_relu)

    base_filter = base_filter * 4

    b2_c1 = resnet_bottleneck_block(base_filter, b1_mp, "block2_1")
    b2_c2 = resnet_bottleneck_block(base_filter, b2_c1, "block2_2")
    b2_c3 = resnet_bottleneck_block(base_filter, b2_c2, "block2_3")

    # b3_c1 = resnet_bottleneck_block(base_filter * 2, b2_c3, "block3_1", strides=2)
    # b3_c2 = resnet_bottleneck_block(base_filter * 2, b3_c1, "block3_2")
    # b3_c3 = resnet_bottleneck_block(base_filter * 2, b3_c2, "block3_3")
    # b3_c4 = resnet_bottleneck_block(base_filter * 2, b3_c3, "block3_4")

    pool = layers.GlobalAveragePooling2D(name="avg_pool")(b2_c3)
    fc1 = layers.Dense(
        base_filter,
        activation="relu",
        kernel_constraint=tf.keras.constraints.min_max_norm(),
    )(pool)
    dp1 = layers.Dropout(0.5)(fc1)
    # fc2 = layers.Dense(base_filter * 2, activation='relu',
    # kernel_constraint=tf.keras.constraints.min_max_norm())(dp1)
    # dp2 = layers.Dropout(0.5)(fc2)
    fc3 = layers.Dense(nb_classes, activation="softmax")(dp1)

    model = keras.Model(inputs=[inputs], outputs=[fc3])

    bce = keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    model.compile(
        loss=bce, optimizer=keras.optimizers.Adam(lr=lr), metrics=["accuracy"]
    )
    return model
