import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, train_path: str, test_path: str, review_col: str = 'text', label_col: str = 'label'):
        self.train_path = train_path
        self.test_path = test_path
        self.review_col = review_col
        self.label_col = label_col

    def load_data(self):
        train_data = pd.read_csv(self.train_path)
        test_data = pd.read_csv(self.test_path)
        return train_data, test_data

    def get_train_test_split(self, train_data: pd.DataFrame, test_size: float = 0.2):
        X = train_data[self.review_col]
        y = train_data[self.label_col]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_val, y_train, y_val
