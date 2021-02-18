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

num_samples = 300
experiment_name = "detect_metas"


class CustomStopper(tune.Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id, result):
        if not self.should_stop and result["val_accuracy"] > 0.95:
            self.should_stop = True
        return self.should_stop

    def stop_all(self):
        return self.should_stop


if __name__ == "__main__":
    ray.init(num_cpus=20, num_gpus=2)

    config = vars(utils.get_args())

    # WANDB
    # adding wandb keys
    config["wandb"] = {
        "project": experiment_name,
        "api_key_file": "/home/edgar/Documents/Projects/DeepMetav4/.wandb_key",
    }

    config["lr"] = tune.choice([0.01, 0.001, 0.0001])
    config["batch_size"] = tune.qrandint(32, 256, 32)
    config["drop_r"] = tune.quniform(0.2, 0.5, 0.1)
    config["filters"] = tune.choice([4, 8, 16])
    config["model_name"] = tune.choice(["detection", "resnet50", "resnetv2"])
    config["meta"] = True

    utils.print_gre(config)
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
        stop=CustomStopper(),
    )
    print(
        "Best hyperparameters found were: ",
        analysis.get_best_config(metric="val_accuracy", mode="max"),
    )
    ray.shutdown()
