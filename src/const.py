from pathlib import Path

# Define the path to the data directory
data = Path('data')
model = Path('model')

TRAIN_DATA_PATH = data / 'traintest.csv'
TEST_DATA_PATH = data / 'testtest.csv'
MODEL_PATH = model / 'model.pkl'

CACHE_DIR = 'cache'
MAX_FEATURES = 200_000
