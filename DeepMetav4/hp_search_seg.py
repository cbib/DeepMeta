import os

import ray
from ray import tune
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

import DeepMetav4.train_seg as t_seg
import DeepMetav4.utils.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

num_samples = -1  # -1 -> infinite, need stop condition
experiment_name = "seg_lungs_iou"
checkpoint_dir = "ray_logs"


class CustomStopper(tune.Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id, result):
        if not self.should_stop and result["val_accuracy"] < 0.1:
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
        "project": "seg_lungs_iou",
        "api_key": "2087297064263382243a621b1bcdd37fcf1c6bb4",
    }

    config["lr"] = tune.uniform(0.00001, 0.1)
    config["batch_size"] = tune.randint(16, 64)
    config["model_name"] = "small++"
    config["w1"] = tune.randint(1, 20)
    config["w2"] = tune.randint(1, 20)
    config["meta"] = True
    config["weighted"] = True

    utils.print_gre(config)
    # scheduler = AsyncHyperBandScheduler(
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="val_accuracy",
        mode="min",
        reduction_factor=1.5,
    )

    # Use bayesian optimisation with TPE implemented by hyperopt
    search_alg = TuneBOHB(
        metric="val_accuracy",
        mode="min",
        max_concurrent=5,
    )

    analysis = tune.run(
        t_seg.train,
        loggers=DEFAULT_LOGGERS + (WandbLogger,),
        config=config,
        local_dir="ray_results",
        name=experiment_name,
        num_samples=num_samples,
        search_alg=search_alg,
        scheduler=scheduler,
        resources_per_trial={"cpu": 20, "gpu": 2},
        stop=CustomStopper(),
    )
    print(
        "Best hyperparameters found were: ",
        analysis.get_best_config(metric="val_weighted_mean_io_u", mode="min"),
    )
    ray.shutdown()
