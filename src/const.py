from pathlib import Path

data = Path('data')
model = Path('model')

TRAIN_DATA_PATH = data / 'train.csv'
TEST_DATA_PATH = data / 'test.csv'
MODEL_PATH = model / 'model.pkl'

CACHE_DIR = 'cache'
MAX_FEATURES = 200_000
