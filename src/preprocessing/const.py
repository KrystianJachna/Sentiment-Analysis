EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
URL_REGEX = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
NUMBER_REGEX = r"\b\d+\b"
MENTION_REGEX = r"@\w+"
HASHTAG_REGEX = r"#\w+"

EMOJI_MEANING = {
    "smile_emote": [":)", ":d", "😃", "😀", "😁", "😊", "😉", "🙂", "😄", ":-)", "=)", ":]", ":^)"],
    "laugh_emote": ["😂", "🤣", "😆", ":p", ":-p", "=p", ":b", ":d"],
    "wink_emote": ["😉", ";)", ";-)", ";]"],
    "love_emote": [":*", "😍", "😘", "😗", "😙", "😚", "❤️", "💖", "💗", "💓", "<3"],
    "sad_emote": [":(", "😢", "😭", "😞", "☹️", "🙁", "😔", ":-(", "=("],
    "angry_emote": [":@", "😠", "😡", "🤬", ">:(", ":-@"],
    "surprised_emote": [":o", "😲", "😮", "😯", "😱", ":-o", "=o"],
    "neutral_emote": [":|", "😐", "😑", "😶", ":-|"],
    "playful_emote": [":p", "😛", "😜", "😝", "😋", ":-p", "=p", ":-b"],
    "confused_emote": ["🤔", "😕", "😟", ":/", ":-/"],
    "tired_emote": ["😫", "😩", "😴", "😪"],
    "embarrassed_emote": [":$", "😳"],
    "cool_emote": ["😎", "🆒", "B)", "B-)"]
}
