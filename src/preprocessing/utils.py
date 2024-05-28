EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
URL_REGEX = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
NUMBER_REGEX = r"\b\d+\b"
MENTION_REGEX = r"@\w+"
HASHTAG_REGEX = r"#\w+"

EMOJI_MEANING = {
    ":)": "smile",
    ":(": "sad",
    ":D": "laugh",
    ":/": "annoyed",
    ":P": "playful",
    ":*": "kiss",
    ";)": "wink",
    ":|": "indifferent",
    ":O": "surprised",
    ":@": "angry",
    ":S": "confused",
    ":$": "embarrassed",
    ":X": "sealed lips",
    ":&": "angry",
    ":#": "mute",
    ":^)": "smug",
    "8)": "glasses",
    "8|": "sunglasses"
}

