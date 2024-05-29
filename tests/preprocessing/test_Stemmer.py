import unittest

import pandas as pd

from src.preprocessing.Stemmer import Stemmer


class TestStemmer(unittest.TestCase):

    def setUp(self):
        self.stemmer = Stemmer()

    def test_stemmer(self):
        tokenized_data = pd.Series([
            ["running", "runners", "ran", "easily"],
            ["connected", "connection", "connecting"],
            ["better", "best", "good", "well"],
            ["this", "is", "a", "test"]
        ])
        expected = pd.Series([
            "run runner ran easili",
            "connect connect connect",
            "better best good well",
            "thi is a test"
        ])
        result = self.stemmer.transform(tokenized_data)
        pd.testing.assert_series_equal(result, expected)

    def test_stemmer_empty_input(self):
        empty_data = pd.Series([[]])
        result = self.stemmer.transform(empty_data)
        self.assertEqual(result.iloc[0], "")

    def test_stemmer_single_word(self):
        single_word_data = pd.Series([["running"]])
        expected_single_word = pd.Series(["run"])
        result = self.stemmer.transform(single_word_data)
        pd.testing.assert_series_equal(result, expected_single_word)
