from sklearn.base import BaseEstimator, TransformerMixin
from re import sub
from .utils import *


class DataCleaner(BaseEstimator, TransformerMixin):

    def __init__(
        self, *,
        replace_url: bool = True,
        replace_mention: bool = True,
        replace_hashtag: bool = True,
        replace_emoji: bool = True,
        replace_numbers: bool = True,
        replace_email: bool = True,
    ):
        self.replace_url = replace_url
        self.replace_mention = replace_mention
        self.replace_hashtag = replace_hashtag
        self.replace_emoji = replace_emoji
        self.replace_numbers = replace_numbers

    def transform(self, data:str, y:int=None) -> str:
        data = data.lower()
        data = sub(URL_REGEX, "url", data) if self.replace_url else data
        data = sub(MENTION_REGEX, "mention", data) if self.replace_mention else data
        data = sub(HASHTAG_REGEX, "hashtag", data) if self.replace_hashtag else data

        if self.replace_emoji:
            for emoji, meaning in EMOJI_MEANING.items():
                data = data.replace(emoji, meaning)

        data = sub(NUMBER_REGEX, "number", data) if self.replace_numbers else data
        data = sub(EMAIL_REGEX, "email", data) if self.replace_email else data
        return data.strip()


    def fit(self, data:str , y:int=None) -> 'DataCleaner':
        return self
