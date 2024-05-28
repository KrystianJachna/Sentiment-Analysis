from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

class Lemmatizer(BaseEstimator, TransformerMixin):
    """
    Lemmatizer class that lemmatizes the input tokens using the nltk library
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def transform(self, X: list[str], y: int = None) -> list[str]:
        return [self.lemmatizer.lemmatize(token) for token in X]

    def fit(self, data: list[str], y: int = None) -> 'Lemmatizer':
        return self
