# DeepMetav4

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://gitmoji.dev">
  <img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square" alt="Gitmoji">
</a> 

Deep learning techniques used to segment lungs and metastasis on mice MRI images.

SPEAK ABOUT DATA ?

## Contents
- [Installation](#installation)
- [Performance](#performance)
- [Usage](#usage)


## Installation

We recommand you to use conda to create and manage dependencies.

To install required librairies :
```shell script
conda env create -f environment.yml
```

If you do not want to use conda : 
```shell script
pip install -f requirements.txt
```


> The environment files asssume that you have at least one Nvidia Gpu installed on your computer.

## Performance
To mesure the performance of each network, we rely on several metrics:
 - IoU (Jaccard index)
 - MCC (Matthews Correlation Coeficient)
 - AUC (AUROC).
### Lungs
![seg_lungs](docs/_static/lungs_seg.png)

- Mean IoU on test data = 0.891
- Mean MCC on test data = 0.878
- Mean AUC on test data = 0.950

### Metastasis

<img src="./docs/_static/sm_metas_seg.png" alt="sm meta seg" width="250">
<img src="./docs/_static/bg_meta_seg.png" alt="bg meta seg" width="250">

- Mean IoU on test data = 0.768
- Mean MCC on test data = 0.598
- Mean AUC on test data = 0.821

## Usage

### Set up gpus
If you have several GPUs :

At the beginning of the scripts you can find Gpus ids, if you want to specify wich Gpu(s) to use just add or remove your ids in this line :
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "id0, id1, ...."
```
if you want to use all gpus, comment this line.

### Training

#### Set up training
In the file `utils/global_vars.py` you will find all the paths to the dataset and the saving paths, feel free to modify it to fit your repos.

By calling train scripts you can pass some arguments that will define the setup of the training :
```shell script
  --n_epochs N_EPOCHS      number of epochs of training
  --batch_size BATCH_SIZE  size of the batches
  --lr LR                  adam: learning rate
  --model_name MODEL_NAME  Name of the model you want to train (detection, small++)
  --meta                   If we want to segment metas
  --weighted               If you want to use weighted crossentropy
  --size SIZE              Size of the image, one number
  --w1 W1                  weight inside
  --w2 W2                  Weight border
  --patience PATIENCE      Set patience value for early stopper
  --filters FILTERS        Number of filters in the first conv block
  --drop_r RATE            Dropout rate
```

#### Run training
To train a network to segment lungs :
```shell script
python -m DeepMetav4.train_seg  --batch_size=32 --model_name=small++ --n_epochs=200 --lr=0.001
```

To train a network to segment metastasis :
```shell script
python -m DeepMetav4.train_seg  --batch_size=32 --model_name=small++ --n_epochs=200 --lr=0.001 --meta --weighted --w1=10 --w2=20
```

#### Availiable models
 - unet
 - small ++
 - unet ++

### Prediction
If you want to segment images :
```shell script
python -m DeepMetav4.predict_seg
```

### Hyper-parameter search

To find the optimal solution to our tasks, we use a combination of `Ray Tune` and `WandB`
to implement hyper parameter search. Basically, we use the BOHB scheduler and search algorithm.
The parameters we search for are :
- batch_size
- learning rate
- dropout rate
- number of filters
- cross entropy weights

To run HP search :
```shell
python -m DeepMetav4.hp_search
```

### Pipeline

The pipeline.py script aims to run inference on one mouse, save result and do stats
if possible.

To do so, fill the paths in the script and then :
```shell
python -m Deepmetav4.pipeline
```

You can add flags to this script, if you do not want to save images or runs stats:
`--no-save` and `--no-stats`.
