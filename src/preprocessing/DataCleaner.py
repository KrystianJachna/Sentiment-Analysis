import string
from re import sub

from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

from .const import *
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(
        self, *,
        replace_url: bool = True,
        replace_mention: bool = True,
        replace_hashtag: bool = True,
        replace_emoji: bool = True,
        replace_numbers: bool = True,
        replace_email: bool = True,
        punctuation: bool = True,
        word_len_threshold: int = 2
    ):
        self.replace_url = replace_url
        self.replace_mention = replace_mention
        self.replace_hashtag = replace_hashtag
        self.replace_emoji = replace_emoji
        self.replace_numbers = replace_numbers
        self.replace_email = replace_email
        self.punctuation = punctuation
        self.word_len_threshold = word_len_threshold

    def transform(self, X: pd.Series, y: int = None) -> pd.Series:
        return self._clean_text(X)

    def fit(self, X: pd.Series, y: int = None) -> 'DataCleaner':
        return self

    def _clean_text(self, X: pd.Series) -> pd.Series:
        X = X.str.lower()

        if self.replace_email:
            X = X.apply(self._replace_email)
        if self.replace_url:
            X = X.apply(self._replace_url)
        if self.replace_mention:
            X = X.apply(self._replace_mention)
        if self.replace_hashtag:
            X = X.apply(self._replace_hashtag)
        if self.replace_emoji:
            X = X.apply(self._replace_emotes)
        if self.replace_numbers:
            X = X.apply(self._replace_numbers)

        X = X.apply(self._remove_stop_words)

        if self.punctuation:
            X = X.apply(self._remove_punctuation)

        X = X.apply(word_tokenize)

        return X.apply(self._remove_short_words)

    def _remove_stop_words(self, text: str) -> str:
        return ' '.join([word for word in word_tokenize(text) if word not in STOP_WORDS])

    def _replace_email(self, text: str) -> str:
        return sub(EMAIL_REGEX, "email", text)

    def _replace_url(self, text: str) -> str:
        return sub(URL_REGEX, "url", text)

    def _replace_mention(self, text: str) -> str:
        return sub(MENTION_REGEX, "mention", text)

    def _replace_hashtag(self, text: str) -> str:
        return sub(HASHTAG_REGEX, "hashtag", text)

    def _replace_emotes(self, text: str) -> str:
        for meaning, emojis in EMOJI_MEANING.items():
            for emoji in emojis:
                text = text.replace(emoji, meaning)
        return text

    def _replace_numbers(self, text: str) -> str:
        return sub(NUMBER_REGEX, "number", text)

    def _remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))

    def _remove_short_words(self, tokens: list[str]) -> str:
        return list(filter(lambda token: len(token) >= self.word_len_threshold, tokens))
