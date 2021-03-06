#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

"""
Hp Search
==========
This file is used to run hyper parameter search on a model.
Just fill all the variables, fill you search space and run the script.

.. warning::
    To see the result, you have to create a file `.wandb_key` containing your
    WandB api key.

    todo: resolve issue sphinx
"""

import os

import ray
from ray import tune
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

import DeepMeta.train as t
import DeepMeta.utils.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

num_samples = 100  # -1 -> infinite, need stop condition
experiment_name = "seg_meta_mcc"
METRIC = "val_accuracy"  # this is the name of the attribute in tune reporter
MODE = "max"


def create_folders():
    """
    This function creates the needed folders for ray and WandB (if needed).
    """
    os.makedirs("ray_result", exist_ok=True)
    os.makedirs("wandb", exist_ok=True)


if __name__ == "__main__":
    create_folders()
    ray.init(num_cpus=20, num_gpus=2)

    config = vars(utils.get_args())

    # WANDB
    # adding wandb keys
    config["wandb"] = {
        "project": experiment_name,
        "api_key_file": "/scratch/elefevre/Projects/DeepMeta/.wandb_key",
    }

    config["lr"] = tune.choice([0.01, 0.001, 0.0001])
    config["batch_size"] = tune.qrandint(32, 128, 32)
    config["model_name"] = "small++"
    config["w1"] = tune.randint(5, 20)
    config["w2"] = tune.randint(10, 20)
    config["drop_r"] = tune.quniform(0.2, 0.5, 0.005)
    config["filters"] = tune.choice([4, 8, 16])
    config["meta"] = True
    config["weighted"] = True

    utils.print_gre(config)
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric=METRIC,
        mode=MODE,
        reduction_factor=2,
    )

    search_alg = TuneBOHB(
        metric=METRIC,
        mode=MODE,
        max_concurrent=5,
    )

    analysis = tune.run(
        t.train,  # fonction de train
        loggers=DEFAULT_LOGGERS + (WandbLogger,),
        config=config,
        local_dir="ray_results",
        name=experiment_name,
        num_samples=num_samples,
        search_alg=search_alg,
        scheduler=scheduler,
        resources_per_trial={"cpu": 10, "gpu": 1},
    )
    print(
        "Best hyperparameters found were: ",
        analysis.get_best_config(
            metric=METRIC,
            mode=MODE,
        ),
    )
    ray.shutdown()
