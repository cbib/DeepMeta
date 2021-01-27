from ray import tune
from tensorflow import keras


class TuneReporter(keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def __init__(self, metric="val_accuracy"):
        super().__init__()
        self.metric = metric

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        tune.report(
            keras_info=logs,
            val_loss=logs["val_loss"],
            val_accuracy=logs[self.metric],
        )
