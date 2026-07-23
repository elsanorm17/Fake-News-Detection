"""
tokenizers.py

Custom tokenization functions used to compare different
punctuation-splitting strategies for fake news detection.
"""

import string

PUNCTUATION = set(string.punctuation)


def tokenize_full(text: str) -> list[str]:
    """
    Tokenize text by splitting on whitespace and every punctuation mark,
    including punctuation occurring inside words.

    Example:
        "can't." -> ["can", "'", "t", "."]
    """
    tokens = []

    for word in text.lower().split():
        current = ""

        for char in word:
            if char in PUNCTUATION:
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(char)
            else:
                current += char

        if current:
            tokens.append(current)

    return tokens


def tokenize_outside(text: str) -> list[str]:
    """
    Tokenize text by splitting on whitespace and separating punctuation
    only at the beginning or end of each word.

    Example:
        "(hello)," -> ["(", "hello", ","]

        "can't" -> ["can't"]
    """
    tokens = []

    for word in text.lower().split():

        if not word:
            continue

        # Leading punctuation
        while word and word[0] in PUNCTUATION:
            tokens.append(word[0])
            word = word[1:]

        # Trailing punctuation
        trailing = []
        while word and word[-1] in PUNCTUATION:
            trailing.append(word[-1])
            word = word[:-1]

        if word:
            tokens.append(word)

        # Preserve original order
        tokens.extend(reversed(trailing))

    return tokens