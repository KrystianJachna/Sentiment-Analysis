import unittest

import pandas as pd
from nltk.corpus import stopwords

from src.preprocessing.DataCleaner import DataCleaner


class TestDataCleaner(unittest.TestCase):
    STOP_WORD = set(stopwords.words('english'))

    def setUp(self):
        self.cleaner = DataCleaner()
        self.data = pd.Series([
            "Hello, this is a test email@example.com and a URL: https://example.com",
            "Check out @username's post! #exciting 😃",
            "Here are some numbers: 12345, and punctuation!!!",
            "Short words like a, an, the should be removed."
        ])

    def test_email_replacement(self):
        cleaner = DataCleaner(replace_email=True, replace_url=False, replace_mention=False, replace_hashtag=False,
                              replace_emoji=False, replace_numbers=False, punctuation=False)
        result = cleaner.transform(self.data)
        self.assertIn("email", result.iloc[0])
        self.assertNotIn("email@example.com", result.iloc[0])

    def test_url_replacement(self):
        cleaner = DataCleaner(replace_email=False, replace_url=True, replace_mention=False, replace_hashtag=False,
                              replace_emoji=False, replace_numbers=False, punctuation=False)
        result = cleaner.transform(self.data)
        self.assertIn("url", result.iloc[0])
        self.assertNotIn("https://example.com", result.iloc[0])

    def test_mention_replacement(self):
        cleaner = DataCleaner(replace_email=False, replace_url=False, replace_mention=True, replace_hashtag=False,
                              replace_emoji=False, replace_numbers=False, punctuation=False)
        result = cleaner.transform(self.data)
        self.assertIn("mention", result.iloc[1])
        self.assertNotIn("@username", result.iloc[1])

    def test_hashtag_replacement(self):
        cleaner = DataCleaner(replace_email=False, replace_url=False, replace_mention=False, replace_hashtag=True,
                              replace_emoji=False, replace_numbers=False, punctuation=False)
        result = cleaner.transform(self.data)
        self.assertIn("hashtag", result.iloc[1])
        self.assertNotIn("#exciting", result.iloc[1])

    def test_emoji_replacement(self):
        cleaner = DataCleaner(replace_email=False, replace_url=False, replace_mention=False, replace_hashtag=False,
                              replace_emoji=True, replace_numbers=False, punctuation=False)
        result = cleaner.transform(self.data)
        self.assertIn("smile_emote", result.iloc[1])
        self.assertNotIn("😃", result.iloc[1])

    def test_numbers_replacement(self):
        cleaner = DataCleaner(replace_email=False, replace_url=False, replace_mention=False, replace_hashtag=False,
                              replace_emoji=False, replace_numbers=True, punctuation=False)
        result = cleaner.transform(self.data)
        self.assertIn("number", result.iloc[2])
        self.assertNotIn("12345", result.iloc[2])

    def test_punctuation_removal(self):
        cleaner = DataCleaner(replace_email=False, replace_url=False, replace_mention=False, replace_hashtag=False,
                              replace_emoji=False, replace_numbers=False, punctuation=True)
        result = cleaner.transform(self.data)
        self.assertNotIn("!!!", result.iloc[2])
        self.assertNotIn(",", result.iloc[0])

    def test_remove_short_words(self):
        cleaner = DataCleaner(word_len_threshold=3)
        result = cleaner.transform(self.data)
        self.assertNotIn("a", result.iloc[3])
        self.assertNotIn("an", result.iloc[3])
        self.assertNotIn("the", result.iloc[3])

    def test_combined_cleaning(self):
        cleaner = DataCleaner()
        result = cleaner.transform(self.data)
        self.assertNotIn("email@example.com", result.iloc[0])
        self.assertNotIn("https://example.com", result.iloc[0])
        self.assertNotIn("@username", result.iloc[1])
        self.assertNotIn("#exciting", result.iloc[1])
        self.assertNotIn("😃", result.iloc[1])
        self.assertNotIn("12345", result.iloc[2])
        self.assertNotIn(",", result.iloc[0])
        self.assertNotIn("a", result.iloc[3])
        self.assertNotIn("an", result.iloc[3])
        self.assertNotIn("the", result.iloc[3])

    def test_empty_data(self):
        cleaner = DataCleaner()
        result = cleaner.transform(pd.Series([]))
        self.assertEqual(0, len(result))

    def test_remove_stop_words(self):
        cleaner = DataCleaner()
        result = cleaner.transform(self.data)
        for word in self.STOP_WORD:
            self.assertNotIn(word, result.iloc[0])
            self.assertNotIn(word, result.iloc[1])
            self.assertNotIn(word, result.iloc[2])
            self.assertNotIn(word, result.iloc[3])
