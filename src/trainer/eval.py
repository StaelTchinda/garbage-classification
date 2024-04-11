import keras
import tensorflow as tf

from src.trainer.config import TrainingConfig, get_default_training_config


def eval(model: keras.Model, dataset: tf.data.Dataset, config: TrainingConfig = None):
    if config is None:
        config = get_default_training_config()

    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=config.metrics)
    val_result = model.evaluate(dataset)

    return val_result
