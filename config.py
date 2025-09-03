import random as rn
import numpy as np
import warnings
import pandas as pd

# 기본 설정
RANDOM_SEED = 42
KFOLD_SPLITS = 10

np.random.seed(2025)
rn.seed(2025)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    DATA_DIR = "/content/drive/MyDrive/Colab Notebooks"
except:
    DATA_DIR = "./data"

OUTPUT_DIR = "./outputs"
