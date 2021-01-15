import os

import ray
import tensorflow as tf
from ray import tune
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorflow import keras

import DeepMetav4.utils.data as data
import DeepMetav4.utils.global_vars as gv
import DeepMetav4.utils.utils as utils

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
config["model_name"] = "detection"


class TuneReporter(keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def on_epoch_end(self, epoch, logs=None):
        tune.report(
            keras_info=logs,
            val_loss=logs["val_loss"],
            val_accuracy=logs["val_accuracy"],
        )


def train_detect(args, model_name="detection"):
    utils.print_red("Training Detect : ")
    if args["meta"]:
        dataset, label = data.create_dataset_detect_meta(
            gv.path_gen_img, gv.path_gen_lab, gv.tab_meta, args["size"]
        )
    else:
        dataset, label = data.create_dataset_detect(
            gv.path_img_classif, gv.tab, gv.numSouris, args["size"]
        )
    input_shape = (
        args["size"],
        args["size"],
        1,
    )
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model_detect = gv.model_list[model_name](input_shape, args["lr"])
        es = keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            patience=args["patience"],
            min_delta=0.00001,
            restore_best_weights=True,
        )
        tr = TuneReporter()
        model_detect.fit(
            dataset,
            label,
            validation_split=0.2,
            batch_size=args["batch_size"],
            epochs=args["n_epochs"],
            callbacks=[es, utils.CosLRDecay(args["n_epochs"], args["lr"]), tr],
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
