import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from . import utils_model


def small_plus_plus(input_shape, filters=16):
    inputs = layers.Input(input_shape)

    c2, p2 = utils_model.block_down(
        inputs, filters=filters, drop=0.3, name="block_down_1"
    )
    c3, p3 = utils_model.block_down(
        p2, filters=2 * filters, drop=0.3, name="block_down_2"
    )
    c4, p4 = utils_model.block_down(
        p3, filters=4 * filters, drop=0.3, name="block_down_3"
    )

    o = utils_model.bridge(p4, filters=8 * filters, drop=0.3)

    u4 = utils_model.block_up(o, [c4], filters=4 * filters, drop=0.3, name="block_up_1")

    n3_1 = utils_model.block_up(
        c4, [c3], filters=2 * filters, drop=0.3, name="block_nested_2-1"
    )
    u3 = utils_model.block_up(
        u4, [n3_1, c3], filters=2 * filters, drop=0.3, name="block_up_2"
    )

    n2_1 = utils_model.block_up(
        c3, [c2], filters=filters, drop=0.3, name="block_nested_3-1"
    )
    n2_2 = utils_model.block_up(
        n3_1, [n2_1, c2], filters=filters, drop=0.3, name="block_nested_3-2"
    )
    u2 = utils_model.block_up(
        u3, [n2_2, n2_1, c2], filters=filters, drop=0.3, name="block_up_3"
    )

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", name="final_conv")(u2)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model
