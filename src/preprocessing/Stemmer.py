from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class Stemmer(BaseEstimator, TransformerMixin):
    """
     Stemmer class that stems the input tokens using the nltk library
    """

    def __init__(self):
        self.stemmer = PorterStemmer()

    def transform(self, X: pd.Series, y: int = None) -> pd.Series:
        return X.apply(lambda tokens: ' '.join([self.stemmer.stem(token) for token in tokens]))

    def fit(self, data: list[str], y: int = None) -> 'Stemmer':
        return self

