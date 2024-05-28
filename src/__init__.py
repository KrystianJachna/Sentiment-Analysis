from pathlib import Path

# Define the path to the data directory
data = Path('data')

PROCESSED_TRAIN_DATA = data / 'processed' / 'train.csv'
PROCESSED_TEST_DATA = data / 'processed' / 'test.csv'

RAW_TRAIN_DATA = data / 'raw' / 'train.csv'
RAW_TEST_DATA = data / 'raw' / 'test.csv'
