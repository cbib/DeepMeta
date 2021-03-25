import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.models as models

import DeepMeta.utils.global_vars as gv
import DeepMeta.utils.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


PATH_MODEL = gv.PATH_SAVE + "Metastases/best_resnet50.h5"
PATH_IMG = gv.PATH_DATA + "Metastases/Images/7084.tif"
PATH_MASK = gv.PATH_DATA + "Metastases/Labels/7084.tif"


def show_filters(model, file_name, channels=2):
    utils.print_gre("Filters")
    layer_dict = [(layer.name, layer) for layer in model.layers]
    # Grab the filters and biases for that layer
    for (name, layer) in layer_dict:
        if "conv" in name:
            (filters, biases) = layer.get_weights()
            # Normalize filter values to a range of 0 to 1 so we can visualize them
            f_min, f_max = np.amin(filters), np.amax(filters)
            filters = (filters - f_min) / (f_max - f_min)
            # Plot first few filters
            fig = plt.figure(figsize=(10, 8))
            cols = 5
            rows = 5
            for i in range(1, cols * rows + 1):
                try:
                    fig.add_subplot(rows, cols, i)
                    plt.imshow(filters[:, :, 0, i], cmap="gray")
                except Exception:
                    print("Not enough filters to unpack")
            plt.savefig(file_name + name + "_filtres.png")
            plt.close()


def deprocess_image(x):
    # normalisation de tenseur: moyenne 0 et std 0.1
    x -= x.mean()
    x /= x.std() + 1e-5
    x *= 0.1
    # [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # RGB
    x *= 255
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def generate_filter(model, layer_name, filter_index, size=224):
    layer_output = model.get_layer(layer_name).output
    loss2 = np.mean(layer_output[:, :, :, filter_index])
    print(loss2)
    with tf.GradientTape() as tape:
        loss = K.mean(layer_output[:, :, :, filter_index])
        grads = tape.gradient(loss, model.input)
        grads /= K.sqrt(K.mean(K.square(grads))) + 1e-5
        iterate = K.function([model.input], [loss, grads])
        input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.0
        step = 1.0
        for i in range(30):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
        img = input_img_data[0]
    return deprocess_image(img)


def show_features_map(model, img, path_save):
    utils.print_gre("Features")
    nb_layers = len(model.layers)
    layer_outputs = [layer.output for layer in model.layers[1 : nb_layers + 1]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)
    layer_names = []
    for layer in model.layers[1 : nb_layers + 1]:
        layer_names.append(layer.name)
    images_per_row = 16  # nombre d'images à afficher par colonne.
    # boucle sur chaque couche du réseau
    for layer_name, layer_activation in zip(layer_names, activations):
        try:
            # nombre de filtres dans la carte
            n_filters = layer_activation.shape[-1]
            # taille de la carte (1, size, size, n_filters)
            size = layer_activation.shape[1]
            # matrice des canaux d'activation
            n_cols = n_filters // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[
                        0, :, :, col * images_per_row + row
                    ]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                    display_grid[
                        col * size : (col + 1) * size, row * size : (row + 1) * size
                    ] = channel_image
            # affichage
            scale = 1.0 / size
            plt.figure(
                figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0])
            )
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect="auto", cmap="gray")
            plt.savefig(path_save + "features_" + layer_name + ".png")
            plt.close()
        except Exception:
            print("layer : " + layer_name)
            plt.close()


def get_img_and_concat(img_path, mask_path):
    img = io.imread(img_path, plugin="tifffile")
    mask = io.imread(mask_path, plugin="tifffile")
    img = np.array(img).reshape(128, 128, 1)
    mask = np.array(mask).reshape(128, 128, 1)
    return tf.concat([img, mask], 2)


if __name__ == "__main__":
    # model = tf.keras.models.load_model(PATH_MODEL,
    # custom_objects={'weighted_cross_entropy': utils_model.weighted_cross_entropy})
    model = tf.keras.models.load_model(PATH_MODEL)
    # img = get_img_and_concat(PATH_IMG, PATH_MASK)
    img = io.imread(PATH_IMG, plugin="tifffile")
    img = np.array(img).reshape(128, 128, 1)
    img = np.expand_dims(img, 0)
    show_filters(model, gv.PATH_FILTERS_FEATURES)
    show_features_map(
        model, img, gv.PATH_FILTERS_FEATURES
    )  # save only same layers (conv, relu, etc)
