import nltk
from sklearn.base import BaseEstimator, TransformerMixin


class Tokenizer(BaseEstimator, TransformerMixin):
    """
    Tokenizer class that tokenizes the input text using the nltk library
    """

    def __init__(self):
        pass

    def transform(self, X: str, y: int = None) -> list[str]:
        return nltk.word_tokenize(X)

    def fit(self, data: str, y: int = None) -> 'Tokenizer':
        return self
