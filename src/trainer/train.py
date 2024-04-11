import keras
import tensorflow as tf

from src.trainer.config import TrainingConfig, get_default_training_config


def train(model: keras.Model, train_data: tf.data.Dataset, val_data: tf.data.Dataset, config: TrainingConfig = None):
    if config is None:
        config = get_default_training_config()

    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=config.metrics)
    train_history = model.fit(train_data, validation_data=val_data, epochs=config.epochs, callbacks=config.callbacks)

    return train_history
