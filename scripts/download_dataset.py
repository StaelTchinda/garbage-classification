import os
import argparse

from src.data.config import DATA_PATH, GARBAGE_CLASSIFICATION_DATA_KEY


def parse_args():
    # The arguments to parse are the same as kaggle.api.kaggle_api_extended.KaggleApi.dataset_download_files
    parser = argparse.ArgumentParser(description='Download dataset from kaggle')
    parser.add_argument('--dataset', type=str, help='The dataset to download', default=GARBAGE_CLASSIFICATION_DATA_KEY)
    parser.add_argument('--path', type=str, help='The path to download the dataset', default=DATA_PATH)
    parser.add_argument('--unzip', action='store_true', help='Unzip the dataset')
    parser.add_argument('--force', action='store_true', help='Force the download if the file already exists')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    return parser.parse_args()


def main():
    args = parse_args()

    if os.environ.get('KAGGLE_USERNAME') is None:
        os.environ['KAGGLE_USERNAME'] = input('Enter your kaggle username: ')
    if os.environ.get('KAGGLE_KEY') is None:
        os.environ['KAGGLE_KEY'] = input('Enter your kaggle key: ')

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()

    api.authenticate()
    api.dataset_download_files(args.dataset, path=args.path, force=args.force, quiet=args.quiet)


if __name__ == '__main__':
    main()
