import nltk
from sklearn.base import BaseEstimator, TransformerMixin


class Tokenizer(BaseEstimator, TransformerMixin):
    """
    Tokenizer class that tokenizes the input text using the nltk library
    """

    def __init__(self):
        pass

    def transform(self, X: list[str], y: int = None) -> list[list[str]]:
        return [nltk.word_tokenize(text) for text in X]

    def fit(self, data: str, y: int = None) -> 'Tokenizer':
        return self
