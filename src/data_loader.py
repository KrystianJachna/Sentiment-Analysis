from pathlib import Path

import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    """
    Load data from the given path and return a DataFrame.
    Preprocess the data by removing neutral reviews and converting the labels to binary.
    """
    print(f"Loading data from {path}")
    data = pd.read_csv(path, names=['label', 'review'], usecols=[0, 2])
    data = data[data['label'] != 3]
    data['label'] = data['label'].apply(lambda x: 0 if x < 3 else 1)
    return data
