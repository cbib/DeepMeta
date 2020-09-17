import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv


def data_shape_test():
    dataset, label = data.create_dataset_detect(gv.path_img, gv.tab, gv.numSouris, 128)
    assert dataset[0].shape == (128, 128, 1)
