from sklearn.base import BaseEstimator, TransformerMixin
import nltk


class Tokenizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, X:str, y:int=None) -> list[str]:
        return nltk.word_tokenize(X)

    def fit(self, data:str , y:int=None) -> 'Tokenizer':
        return self
