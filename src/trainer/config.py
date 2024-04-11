import abc
from pathlib import Path
from typing import Text, List, Optional

import yaml
from keras.src.callbacks import EarlyStopping, ModelCheckpoint

from src.utils import PROJECT_ROOT_PATH

CHECKPOINTS_PATH: Path = Path(__file__).parent.parent.parent / 'checkpoints'
TRAINING_DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT_PATH / 'config' / 'default_train_config.yaml'


class CallbackConfig(abc.ABCMeta):
    __callback_name__: Text

    @abc.abstractmethod
    def build_callback(self):
        pass

    @classmethod
    def from_dict(cls, callback_dict: dict):
        callback_name = callback_dict['__callback_name__']
        callback_class = __registered_callbacks__[callback_name]
        callback_params = callback_dict.copy()
        del callback_params['__callback_name__']
        return callback_class(**callback_params)


class EarlyStoppingConfig(CallbackConfig):
    __callbackname__ = 'EarlyStopping'

    def __init__(self, patience: int):
        self.patience: int = patience

    def build_callback(self):
        return EarlyStopping(patience=self.patience)


class ModelCheckpointConfig(CallbackConfig):
    __callback_name__ = 'ModelCheckpoint'

    def __init__(self, filepath: Text):
        self.filepath: int = filepath

    def build_callback(self):
        return ModelCheckpoint(self.filepath, save_best_only=True)


__registered_callbacks__ = {
    'EarlyStopping': EarlyStopping,
    'ModelCheckpoint': ModelCheckpoint
}


class TrainingConfig:
    optimizer: Text
    loss: Text
    metrics: List[Text]
    epochs: int
    callbacks: List[CallbackConfig]

    def __init__(self, optimizer: Text, loss: Text, metrics: List[Text], epochs: int, callbacks: List[CallbackConfig]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.callbacks = callbacks


def get_default_training_config() -> TrainingConfig:
    return read_training_config(TRAINING_DEFAULT_CONFIG_PATH)


def get_default_checkpoint_path() -> Path:
    config = get_default_training_config()
    checkpoint_callbacks: List[ModelCheckpoint] = list(filter(lambda callback: isinstance(callback, ModelCheckpoint), config.callbacks))
    if len(checkpoint_callbacks)==0:
        raise ValueError("ModelCheckpoint callback is not found in the training configuration")
    checkpoint_callback: ModelCheckpoint = checkpoint_callbacks[0]
    return PROJECT_ROOT_PATH / checkpoint_callback.filepath


def read_training_config(config_path: Path) -> TrainingConfig:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    callbacks = [CallbackConfig.from_dict(callback_dict) for callback_dict in config_dict['callbacks']]
    config_params = {**config_dict, 'callbacks': callbacks}
    return TrainingConfig(**config_params)