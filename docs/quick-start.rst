Quick Start
================

Installation
-------------

We recommend you to use conda to create environments and manage dependencies.

.. code-block:: bash

   git clone git@github.com:EdgarLefevre/DeepMetav4.git
   cd DeepMetav4
   conda env create -f environment.yml

If you don't want to use conda, you can use python 3.6 and pip.

.. code-block:: bash

   git clone git@github.com:EdgarLefevre/DeepMeta.git
   cd DeepMeta
   pip install -r requirements.txt

Set up global vars
------------------

DeepMeta's code need some configuration before running.
Open the *utils/global_vars.py* and you will find all the paths to the dataset and the saving paths,
feel free to modify it to fit your repos.

Set up GPUS
------------

For the multi-GPUs setup :
We have set a globar var *CUDA_VISIBLE_DEVICES* which aim is to select GPUS we want to work with.
If you want all available GPU(s) you can remove the declaration on top of the scripts.
Else, you need to assign ids of GPU(s) you want to use :

.. code-block:: python

   os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,5"

If you have only one GPU, just comment this line.

Run training
------------
To train a network to segment lungs :

.. code-block:: bash

   python -m DeepMeta.train  --batch_size=32 --model_name=small++ --n_epochs=200 --lr=0.001

To train a network to segment metastasis :

.. code-block:: bash

   python -m DeepMeta.train  --batch_size=32 --model_name=small++ --n_epochs=200 --lr=0.001 --meta --weighted --w1=10 --w2=20

``After each training, a training plot will be drawn and save in the plot folder``

The training script have several flags and arguments:

.. code-block:: text

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

Available models:
   - unet
   - unet++
   - small++

Prediction
--------------

If you want to run prediction :
.. code-block:: bash

   python predict.py

Hyper-parameter search
--------------------

To find the optimal solution to our tasks, we use a combination of *Ray Tune* and *WandB* to implement hyper parameter
search. Basically, we use the BOHB scheduler and search algorithm. The parameters we search for are :
- batch_size
- learning rate
- dropout rate
- number of filters
- cross entropy weights

To run HP search :

.. code-block:: bash

   python -m DeepMeta.hp_search

``You will need to store a WandB api key in a file called .wandb_key``

Pipeline
--------

The pipeline.py script aims to run inference on one mouse, save result and do stats if possible.

To do so, fill the paths in the script and then :

.. code-block:: bash

   python -m Deepmetav4.pipeline


You can add flags to this script, if you do not want to save images or runs stats: *--no-save* and *--no-stats*.

If you want to save masks use the *--mask* flag.

``You need label masks to runs stats.``

