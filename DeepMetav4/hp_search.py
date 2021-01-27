import os

import ray
from ray import tune
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

import DeepMetav4.train_detect as t_detect
import DeepMetav4.utils.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

num_samples = 100
experiment_name = "detect_metas"

if __name__ == "__main__":
    ray.init(num_cpus=20, num_gpus=2)

    config = vars(utils.get_args())

    # WANDB
    # adding wandb keys
    config["wandb"] = {
        "project": experiment_name,
        "api_key": "2087297064263382243a621b1bcdd37fcf1c6bb4",
    }

    config["lr"] = tune.uniform(0.0001, 0.1)
    config["batch_size"] = tune.randint(32, 256)
    config["drop_r"] = tune.uniform(0.2, 0.6)
    config["filters"] = tune.choice([4, 8, 16, 32, 64])
    config["model_name"] = "detection"
    config["meta"] = True

    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="val_accuracy",
        mode="max",
        reduction_factor=2,
    )

    search_alg = TuneBOHB(
        metric="val_accuracy",
        mode="max",
        max_concurrent=10,
    )

    analysis = tune.run(
        t_detect.train_detect,
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
        analysis.get_best_config(metric="val_accuracy", mode="max"),
    )
    ray.shutdown()
