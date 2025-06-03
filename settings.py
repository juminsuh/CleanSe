import getpass
import os
import sys

__USERNAME = getpass.getuser()
_BASE_DIR = f'/mnt/aix7101'
MODEL_PATH = f'{_BASE_DIR}/llama-7b/'
DATA_FOLDER = os.path.join(_BASE_DIR, 'minsuh-dataset')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'minsuh-output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)

