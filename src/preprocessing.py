"""
preprocessing.py

Utility functions for loading and preprocessing the Fake News dataset.
"""

from pathlib import Path
import string

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# NLTK setup

def ensure_nltk_resources():
    """Download required NLTK resources if they are not already installed."""
    resources = [
        ("corpora", "stopwords"),
        ("corpora", "wordnet"),
        ("tokenizers", "punkt"),
        ("corpora", "omw-1.4"),
    ]

    for resource_type, resource in resources:
        try:
            nltk.data.find(f"{resource_type}/{resource}")
        except LookupError:
            nltk.download(resource)


# Text preprocessing

def preprocess_text(text):
    """
    Tokenize text, lowercase words, and remove stopwords and punctuation.

    Parameters
    ----------
    text : str

    Returns
    -------
    list[str]
        Cleaned list of tokens.
    """
    ensure_nltk_resources()

    stop_words = set(stopwords.words("english"))
    stop_words.update({"reuters"})

    punctuation = set(string.punctuation)

    tokens = word_tokenize(str(text))

    cleaned_tokens = [
        token.lower()
        for token in tokens
        if token.lower() not in stop_words
        and token not in punctuation
    ]

    return cleaned_tokens


# Dataset loading

def load_dataset(data_dir="../data"):
    """
    Load the fake and true news datasets into a single DataFrame.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing fake_news_data.csv and true_news_data.csv.

    Returns
    -------
    pandas.DataFrame
        Columns:
            title
            text
            classification
    """
    data_dir = Path(data_dir)

    fake_news = pd.read_csv(data_dir / "fake_news_data.csv")
    true_news = pd.read_csv(data_dir / "true_news_data.csv")

    fake_news = fake_news[["title", "text"]].copy()
    fake_news["classification"] = False

    true_news = true_news[["title", "text"]].copy()
    true_news["classification"] = True

    new_df = pd.concat([fake_news, true_news], ignore_index=True)

    return new_df


# Test

if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print(f"\nLoaded {len(df)} articles.")