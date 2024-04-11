import argparse
from pathlib import Path
from typing import Tuple, List

import keras
import tensorflow as tf
import numpy as np

from src.data.config import DATASET_DEFAULT_CONFIG_PATH, DatasetConfig, read_dataset_config, get_default_input_shape
from src.data.dataset import get_class_names, get_train_dataset, get_val_dataset
from src.model import get_model
from src.trainer.config import TrainingConfig, read_training_config, TRAINING_DEFAULT_CONFIG_PATH, get_default_checkpoint_path
from src.trainer.train import train
from src.utils import read_image


def parse_args():
    parser = argparse.ArgumentParser(description='Infer from a model for garbage classification for a specific image')
    parser.add_argument('--data_config', type=str, help='The path to the data config file',
                        default=str(DATASET_DEFAULT_CONFIG_PATH))
    parser.add_argument('--train_config', type=str, help='The path to the config file',
                        default=TRAINING_DEFAULT_CONFIG_PATH)
    parser.add_argument('--checkpoint', type=str, help='The path to the model checkpoint file',
                        default=get_default_checkpoint_path())
    parser.add_argument('--input_shape', type=str, help='The input shape of the model',
                        default=get_default_input_shape())
    parser.add_argument('image', type=str, help='The path to the image to infer')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    input_shape: Tuple = args.input_shape

    # Set up the model
    classnames = get_class_names()
    model: keras.Model = get_model(class_count=len(classnames))
    model.build(input_shape=get_default_input_shape())
    model.load_weights(args.checkpoint)
    model.compile()

    # Set the image: load and preprocess it
    input_path: Path = Path(args.image)
    images_path: List[Path] = []
    if input_path.is_dir():
        images_path = list(input_path.iterdir())
    else:
        images_path.append(input_path)

    images: List[np.ndarray] = []
    for i in range(len(images_path)):
        image = read_image(images_path[i])
        image = tf.image.resize(image, [input_shape[-3],  input_shape[-2]])
        images.append(image)
    images: np.ndarray = np.array(images)

    results = model.predict(images)
    with np.printoptions(precision=2, formatter={'float': '{:0.2f}'.format}):
        for (image_idx, (image_path, result)) in enumerate(zip(images_path, results)):
            print(f"Image {image_idx + 1}/{len(images_path)}")
            print(f"\t Image: {image_path.stem}")
            print(f"\t Prediction: {classnames[np.argmax(result)]}")
            print(f"\t Confidence: {np.max(result)}")
            print(f"\t Class probabilities: {result.flatten()}")

if __name__ == '__main__':
    main()
