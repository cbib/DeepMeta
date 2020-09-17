# DeepMetav4

Deep learning techniques used to classify and segment lungs and metastasis on mice MRI images.

## Installation

This repo use python 3.8 and conda.

We recommand you to use a fresh virtual environment.

To install required librairies :
```shell script
conda env create -f environment.yml
```

> The environment file asssume that you have at least one Nvidia Gpu installed on your computer.


## Usage

### Set up gpus
The file `train.py` is used to train the networks, at the beginning of the file we can find Gpus ids, if you want to specify wich Gpu(s) to use just add or remove your ids in this line :
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "id0, id1, ...."
```
if you want to use all gpus, comment this line.

### Training

#### Set up training
In the file `utils/global_vars.py` you will find all the paths to the dataset and the saving paths, feel free to modify it to fit your repos.

By calling train scripts you can pass some arguments that will define the setup of the training :
```shell script
  --n_epochs N_EPOCHS   number of epochs of training
  --batch_size BATCH_SIZE
                        size of the batches
  --lr LR               adam: learning rate
  --model_name MODEL_NAME
                        Name of the model you want to train (detection,
                        small++)
  --meta META           True if we want to segment metastasis
  --weighted WEIGHTED   Use weighted model (default False)
  --size SIZE           Size of the image, one number
  --w1 W1               weight inside
  --w2 W2               Weight border
  --patience PATIENCE   Set patience value for early stopper
```

#### Run training
If you want to train a network to detect lungs :
```shell script
python -m DeepMetav4.train_detect --batch_size=32 --model_name=resnetv2 --n_epochs=200 --lr=0.01
```

If you want to train a network to segment metastasis :
```shell script
python -m DeepMetav4.train_seg  --batch_size=32 --model_name=small++ --n_epochs=200 --lr=0.0002 --patience=100 --meta=True --weighted=True --w1=10 --w2=20
```

#### Availiable models
 - Detection model
 - resnet
 - vgg
 - unet
 - small ++
 - unet ++

### Prediction

All the parameters you need to set are in the main section of the `predict.py` file.


## Lungs
### Detection
todo : here stats + img of segmented lungs
### Segmentation
## Metastasis
### Detection
### Segmentation
todo : here stats + img of segmented metas

## Issue

There is an issue with tensorflow 2.2, the graph construction takes around 10 min to be done. This is quiet
disturbing. We can't do anything against that, except waiting for the next tensorflow release.

--------------------------------------------
--------------------------------------------
--------------------------------------------

# Souvenir de ce merveilleux code

## Train seg
```shell script
python -m DeepMetav4.train_seg  --batch_size=32 --model_name=small++ --n_epochs=200 --lr=0.0002 --patience=100 --meta=True --weighted=True --w1=10 --w2=20
```

## Train classify
```shell script
 python -m DeepMetav4.train_detect --batch_size=32 --model_name=resnetv2 --n_epochs=200 --lr=0.01 --patience=1000 --meta=True
```


On avait plus ou moins fixer le model pour classifier les metas (pas de très bons res, on attend les nouvelles souris + sanity checker permettrait de ne pas louper de slice)

model resnet 50 pour classify meta -> mettre les poids dans netron pour voir le model

Rien de concluant sur la segmentation des métas, juste ça ne seg pas quand on applique le masque sur l'image.

# Sanity checker ideas

## Techniques to be sure that all slices are well segmented

The average slice between t-1 and t+1 should be almost t slice : if t and average slice really different -> t is false

## post process meta seg

multiply the two masks
