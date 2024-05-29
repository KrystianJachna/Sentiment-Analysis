from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


class Lemmatizer(BaseEstimator, TransformerMixin):
    """
    Lemmatizer class that lemmatizes the input tokens using the nltk library
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def transform(self, X: list[list[str]], y: int = None) -> list[list[str]]:
        return [[self.lemmatizer.lemmatize(word) for word in text] for text in X]

    def fit(self, data: list[str], y: int = None) -> 'Lemmatizer':
        return self


tokens = ["running", "better"]
lemmas = WordNetLemmatizer()
lemmas = [lemmas.lemmatize(token) for token in tokens]


