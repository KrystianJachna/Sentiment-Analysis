from src.preprocessing.Lemmatizer import Lemmatizer
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')

def test_lemmatizer():
    lemmatizer = Lemmatizer()
    tokens = ["running", "better"]
    lemmas = lemmatizer.transform(tokens)
    assert lemmas == [WordNetLemmatizer().lemmatize(token) for token in tokens]

def test_lemmatizer_empty_list():
    lemmatizer = Lemmatizer()
    tokens = []
    lemmas = lemmatizer.transform(tokens)
    assert lemmas == []

def test_lemmatizer_whitespace_only():
    lemmatizer = Lemmatizer()
    tokens = ["     "]
    lemmas = lemmatizer.transform(tokens)
    assert lemmas == ["     "]
