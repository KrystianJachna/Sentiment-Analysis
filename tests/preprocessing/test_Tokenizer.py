from src.preprocessing.Tokenizer import Tokenizer
import nltk

def test_tokenizer():
    tokenizer = Tokenizer()
    text = "This is a test sentence."
    tokens = tokenizer.transform(text)
    assert tokens == nltk.word_tokenize(text)

def test_tokenizer_empty_string():
    tokenizer = Tokenizer()
    text = ""
    tokens = tokenizer.transform(text)
    assert tokens == []

def test_tokenizer_whitespace_only():
    tokenizer = Tokenizer()
    text = "     "
    tokens = tokenizer.transform(text)
    assert tokens == []
