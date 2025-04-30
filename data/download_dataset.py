import os, zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--kaggle-username', required=True)
parser.add_argument('--kaggle-key', required=True)
args = parser.parse_args()

os.environ['KAGGLE_USERNAME'] = args.kaggle_username
os.environ['KAGGLE_KEY'] = args.kaggle_key

api = KaggleApi()
api.authenticate()
# Download và unzip
dataset = 'paultimothymooney/chest-xray-pneumonia'
api.dataset_download_files(dataset, path='data', unzip=True)