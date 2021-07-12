# Deepmeta's code

### Set up gpus
If you have several GPUs :

At the beginning of the scripts you can find Gpus ids, if you want to specify which Gpu(s) to use just add or remove your ids in this line :
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "id0, id1, ...."
```
if you want to use all gpus, comment this line.

### Training

#### Set up training
In the file `utils/global_vars.py` you will find all the paths to the dataset and the saving paths, feel free to modify it to fit your repos.

#### Run training
To train a network to segment lungs :
```shell script
python -m DeepMeta.train_seg  --batch_size=32 --model_name=small++ --n_epochs=200 --lr=0.001
```

To train a network to segment metastasis :
```shell script
python -m DeepMeta.train_seg  --batch_size=32 --model_name=small++ --n_epochs=200 --lr=0.001 --meta --weighted --w1=10 --w2=20
```
>After each training, a training plot will be drawn and saved in the plot folder.

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

#### Availiable models
 - unet
 - small ++

### Prediction
If you want to segment images :
```shell script
python -m DeepMeta.predict_seg
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
python -m DeepMeta.hp_search
```
> You'll need a WandB api key to see the result in the WandB interface.
### Pipeline

The pipeline.py script aims to run inference on one mouse, save result and do stats
if possible.

To do so, fill the paths in the script and then :
```shell
python -m Deepmetav4.pipeline
```

You can add flags to this script, if you do not want to save images or runs stats:
`--no-save` and `--no-stats`.

If you want to save masks use the `--mask` flag.

>You need label masks to runs stats.
