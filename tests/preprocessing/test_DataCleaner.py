from src.preprocessing.DataCleaner import DataCleaner


def test_data_cleaner():
    cleaner = DataCleaner()
    text = ("   Hello @user,        check out this  link: https://example.com :) #fun my number is "
            "1234 contact            me at krysti4nape@gmail.com")
    cleaned = cleaner.transform(text)
    assert cleaned == "hello mention check out this link url smile hashtag my number is number contact me at email"


def test_data_cleaner_no_replace():
    cleaner = DataCleaner(replace_url=False, replace_mention=False, replace_hashtag=False, replace_emoji=False,
                          replace_numbers=False, replace_email=False, punctuation=False)
    text = ("Hello @user, check out this link: https://example.com :) "
            "#fun my number is 1234 contact me at krysti4nape@gmail.com")
    cleaned = cleaner.transform(text)
    assert cleaned == text.lower()


def test_data_cleaner_empty_string():
    cleaner = DataCleaner()
    text = ""
    cleaned = cleaner.transform(text)
    assert cleaned == ""


def test_data_cleaner_whitespace_only():
    cleaner = DataCleaner()
    text = "     "
    cleaned = cleaner.transform(text)
    assert cleaned == ""


def test_data_cleaner_special_characters():
    cleaner = DataCleaner()
    text = "!@#$%^&*()"
    cleaned = cleaner.transform(text)
    assert cleaned == ''
