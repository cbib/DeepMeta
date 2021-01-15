import os

import ray
from ray import tune
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorflow import keras

import DeepMetav4.utils.utils as utils
from DeepMetav4.train_detect import train_detect

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


ray.init(num_cpus=10, num_gpus=1)

EPOCHS = 100
num_samples = 25
experiment_name = "detect_lungs"
checkpoint_dir = "ray_logs"

# config = {
#     "lr": tune.choice([0.01, 0.1, 0.001,]),
#     "batch_size": tune.choice([32, 64, 128]),
#     "neurons": tune.choice([64, 128, 256]),
#     "dropout": tune.choice([0.0, 0.1, 0.2]),
# }
config = vars(utils.get_args())


# WANDB
# adding wandb keys
config["wandb"] = {
    "project": "deepmeta-detect-lungs",
    "api_key_file": "2087297064263382243a621b1bcdd37fcf1c6bb4",
}

config["lr"] = tune.choice([0.01, 0.1, 0.001])
config["batch_size"] = tune.choice([32, 64, 128])


class TuneReporter(keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def on_epoch_end(self, epoch, logs=None):
        tune.report(
            keras_info=logs,
            val_loss=logs["val_loss"],
            val_accuracy=logs["val_accuracy"],
        )


if __name__ == "__main__":
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="val_accuracy", mode="max"
    )

    # Use bayesian optimisation with TPE implemented by hyperopt
    search_alg = HyperOptSearch(
        metric="val_accuracy",
        mode="max",
    )
    # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)

    analysis = tune.run(
        train_detect,
        # WANDB
        loggers=DEFAULT_LOGGERS + (WandbLogger,),
        config=config,
        local_dir="ray_results",
        name=experiment_name,
        num_samples=num_samples,
        search_alg=search_alg,
        scheduler=scheduler,
        resources_per_trial={"cpu": 4, "gpu": 1},
    )

ray.shutdown()
