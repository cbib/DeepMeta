import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def block_pred(inputs, nb_classes):
    x = layers.Flatten()(inputs)
    d1 = layers.Dense(1024, activation="relu")(x)
    d2 = layers.Dense(1024, activation="relu")(d1)
    o = layers.Dense(nb_classes, activation="softmax")(d2)
    return o


def block_conv(inputs, kernel_size, filters, nb_conv=2, drop=0.3):
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(inputs)
    x = layers.Dropout(drop)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    for i in range(nb_conv - 1):
        x = layers.Conv2D(
            filters,
            (kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
        )(x)
    p = layers.Dropout(drop)(x)
    p = layers.LeakyReLU(alpha=0.1)(p)
    p = layers.MaxPooling2D((2, 2))(p)
    return p


def vgg16(input_shape, lr, num_classes=2):
    inputs = layers.Input(input_shape)
    c1 = block_conv(inputs, 3, 32, drop=0.5)
    c2 = block_conv(c1, 3, 64, drop=0.5)

    c3 = block_conv(c2, 3, 128, 3, drop=0.5)
    c4 = block_conv(c3, 3, 256, 3, drop=0.5)
    c5 = block_conv(c4, 3, 256, 3, drop=0.5)

    outputs = block_pred(c5, num_classes)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=["accuracy"],
    )
    return model
