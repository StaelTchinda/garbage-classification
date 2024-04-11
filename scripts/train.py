import argparse

from src.data.config import DATASET_DEFAULT_CONFIG_PATH, DatasetConfig, read_dataset_config
from src.data.dataset import get_class_names, get_train_dataset, get_val_dataset
from src.model import get_model
from src.trainer.config import TrainingConfig, read_training_config, TRAINING_DEFAULT_CONFIG_PATH
from src.trainer.train import train


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_config', type=str, help='The path to the data config file',
                        default=str(DATASET_DEFAULT_CONFIG_PATH))
    parser.add_argument('--train_config', type=str, help='The path to the config file',
                        default=TRAINING_DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print("Training model with the following arguments:", args)

    data_config = read_dataset_config(args.data_config)
    train_config = read_training_config(args.train_config)

    classnames = get_class_names()
    model = get_model(class_count=len(classnames))
    train_data = get_train_dataset(data_config)
    val_data = get_val_dataset(data_config)

    train(model, train_data, val_data, train_config)


if __name__ == '__main__':
    main()
