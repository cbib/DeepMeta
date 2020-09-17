import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from . import utils_model


def unet(input_shape, filters=16):
    inputs = layers.Input(input_shape)
    c1, p1 = utils_model.block_down(inputs, filters=filters, drop=0.1, name="block1")
    c2, p2 = utils_model.block_down(p1, filters=2 * filters, drop=0.2, name="block2")
    c3, p3 = utils_model.block_down(p2, filters=4 * filters, drop=0.2, name="block3")
    c4, p4 = utils_model.block_down(p3, filters=8 * filters, drop=0.2, name="block4")

    o = utils_model.bridge(p4, filters=16 * filters, drop=0.2)

    u4 = utils_model.block_up(o, [c4], filters=8 * filters, drop=0.2, name="block5")
    u3 = utils_model.block_up(u4, [c3], filters=4 * filters, drop=0.2, name="block6")
    u2 = utils_model.block_up(u3, [c2], filters=2 * filters, drop=0.2, name="block7")
    u1 = utils_model.block_up(u2, [c1], filters=filters, drop=0.2, name="block8")

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(u1)
    model = keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model
