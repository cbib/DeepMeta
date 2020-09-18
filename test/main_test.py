import numpy as np

import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv


def test_data_lungs_classif():
    dataset, label = data.create_dataset_detect(
        gv.path_img_classif, gv.tab, gv.numSouris, 128
    )
    assert dataset[0].shape == (128, 128, 1)
    assert label[0].shape == (2,)
    assert dataset[0].dtype == np.float32
    assert label[0].dtype == np.float32
    assert np.amax(dataset[0]) <= 1
    assert np.amin(dataset[0]) >= 0


def test_data_meta_classif():
    dataset, label = data.create_dataset_detect_meta(
        gv.path_gen_img, gv.path_gen_lab, gv.tab_meta, 128
    )
    assert dataset[0].shape == (128, 128, 1)
    assert label[0].shape == (2,)
    assert dataset[0].dtype == np.float32
    assert label[0].dtype == np.float32
    assert np.amax(dataset[0]) <= 1
    assert np.amin(dataset[0]) >= 0


def test_data_seg_lungs():
    dataset, label = data.create_dataset(
        path_img=gv.path_img, path_label=gv.path_lab, size=128
    )
    assert dataset[0].shape == (1, 128, 128, 1)
    assert label[0].shape == (128, 128)
    assert dataset[0].dtype == np.float64
    assert label[0].dtype == bool
    assert np.amax(dataset[0]) <= 1
    assert np.amin(dataset[0]) >= 0


def test_data_seg_meta():
    dataset, label = data.create_dataset(
        path_img=gv.meta_path_img, path_label=gv.meta_path_lab, size=128
    )
    assert dataset[0].shape == (1, 128, 128, 1)
    assert label[0].shape == (128, 128)
    assert dataset[0].dtype == np.float64
    assert label[0].dtype == bool
    assert np.amax(dataset[0]) <= 1
    assert np.amin(dataset[0]) >= 0
