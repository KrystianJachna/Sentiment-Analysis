import string
from re import sub

from sklearn.base import BaseEstimator, TransformerMixin

from .utils import *


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    DataCleaner is a class that cleans the text data by replacing URLs, mentions, hashtags, emojis, numbers, and emails
    with their respective tags. It also removes punctuation and extra spaces from the text.
    """

    def __init__(
        self, *,
        replace_url: bool = True,
        replace_mention: bool = True,
        replace_hashtag: bool = True,
        replace_emoji: bool = True,
        replace_numbers: bool = True,
        replace_email: bool = True,
        punctuation: bool = True
    ):
        self.replace_url = replace_url
        self.replace_mention = replace_mention
        self.replace_hashtag = replace_hashtag
        self.replace_emoji = replace_emoji
        self.replace_numbers = replace_numbers
        self.replace_email = replace_email
        self.punctuation = punctuation

    def transform(self, X: list[str], y: int = None) -> list[str]:
        return self._clean_test(X)

    def fit(self, X: list[str], y: int = None) -> 'DataCleaner':
        return self

    def _clean_test(self, texts: list[str]) -> list[str]:
        texts = [text.lower() for text in texts]
        texts = [sub(EMAIL_REGEX, "email", text) if self.replace_email else text for text in texts]
        texts = [sub(URL_REGEX, "url", text) if self.replace_url else text for text in texts]
        texts = [sub(MENTION_REGEX, "mention", text) if self.replace_mention else text for text in texts]
        texts = [sub(HASHTAG_REGEX, "hashtag", text) if self.replace_hashtag else text for text in texts]

        if self.replace_emoji:
            for emoji, meaning in EMOJI_MEANING.items():
                texts = [text.replace(emoji, meaning) for text in texts]

        texts = [sub(NUMBER_REGEX, "number", text) if self.replace_numbers else text for text in texts]
        texts = [text.translate(str.maketrans('', '', string.punctuation)) if self.punctuation else text for text in texts]
        texts = [' '.join(text.split()) for text in texts]
        return [text.strip() for text in texts]

