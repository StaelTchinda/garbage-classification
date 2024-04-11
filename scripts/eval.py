import argparse

from src.data.config import DATASET_DEFAULT_CONFIG_PATH, DatasetConfig, read_dataset_config
from src.data.dataset import get_class_names, get_train_dataset, get_test_dataset
from src.model import get_model
from src.trainer.config import TrainingConfig, read_training_config, TRAINING_DEFAULT_CONFIG_PATH
from src.trainer.eval import eval


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--data_config', type=str, help='The path to the data config file',
                        default=str(DATASET_DEFAULT_CONFIG_PATH))
    parser.add_argument('--train_config', type=str, help='The path to the config file',
                        default=TRAINING_DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data_config = read_dataset_config(args.data_config)
    train_config = read_training_config(args.train_config)

    classnames = get_class_names()
    model = get_model(class_count=len(classnames))
    test_data = get_test_dataset(data_config)

    eval_result = eval(model, test_data, train_config)
    score = eval_result[0]
    metric_values = eval_result[1:]
    metrics = dict(zip(train_config.metrics, metric_values))
    print('Test Loss =', score)
    for metric_name, metric_value in metrics.items():
        print(f"Test {metric_name} = {metric_value}")


if __name__ == '__main__':
    main()
