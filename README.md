# DeepMetav3

Deep learning techniques used to classify and segment lungs and metastasis on mice MRI images.


## Usage
### Requirements

This repo use python 3.6 and pip3.

We recommand you to use a virtual environement (conda or venv).

To install required librairies :
```shell script
pip3 install -r requirements.txt
```

> The requirements file asssume that you have Nvidia Gpu and Cuda 10.1 installed on your computer, otherwise remove '-gpu' from tensoflow-gpu in requirements files.
### Training
#### Set up gpus
The file `train.py` is used to train the networks, at the beginning of the file we can find Gpus ids, if you want to specify wich Gpu(s) to use just add or remove your ids in this line :
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "id0, id1, ...."
```
if you want to use all gpus, comment this line.
#### Set up training
In the file `utils/global_vars.py` you will find all the paths to the dataset and the saving paths, feel free to modify it to fit your repos.

By calling `python train.py` you can pass some arguments that will define the setup of the training :
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


### Prediction
#### Set ups gpus
See the same section in training.
#### Set up prediction
All the parameters you need to set are in the main section of the `predict.py` file.


## Lungs
todo : here stats + img of segmented lungs
## Metastasis
todo : here stats + img of segmented metas




# Souvenir de ce merveilleux code

## Train seg
python train.py --batch_size=320 --model_name=small++ --n_epochs=200 --lr=0.0002 --patience=100 --meta=True --weighted=True --w1=10 --w2=20

## Train classify
 python train.py --batch_size=640 --model_name=resnetv2 --n_epochs=200 --lr=0.01 --patience=1000 --meta=True


On avait plus ou moins fixer le model pour classifier les metas (pas de très bons res, on attend les nouvelles souris + sanity checker permettrait de ne pas louper de slice)

model resnet 50 pour classify meta -> mettre les poids dans netron pour voir le model

Rien de concluant sur la segmentation des métas, juste ça ne seg pas quand on applique le masque sur l'image.

# Sanity checker ideas

## Techniques to be sure that all slices are well segmented

The average slice between t-1 and t+1 should be almost t slice : if t and average slice really different -> t is false

## post process meta seg

multiply the two masks
