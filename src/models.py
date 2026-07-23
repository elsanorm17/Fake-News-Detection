"""
models.py

Utilities for training and evaluating machine learning models for
fake news detection.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

from preprocessing import load_dataset, preprocess_text


def prepare_data(test_size=0.33, random_state=42):
    """
    Load the dataset and create train/test splits.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = load_dataset()

    X = df["text"]
    y = df["classification"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def vectorize_text(X_train, X_test, max_features=1000):
    """
    Convert raw text into bag-of-words vectors.
    """

    vectorizer = CountVectorizer(
        analyzer=preprocess_text,
        max_features=max_features,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    """
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model


def train_svm(X_train, y_train):
    """
    Train an RBF-kernel Support Vector Machine.
    """
    model = SVC()

    model.fit(X_train, y_train)

    return model


def train_linear_svc(X_train, y_train):
    """
    Train a Linear Support Vector Classifier.
    """
    model = LinearSVC()

    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_data()

    X_train_vec, X_test_vec, vectorizer = vectorize_text(
        X_train,
        X_test,
    )

    logistic = train_logistic_regression(
        X_train_vec,
        y_train,
    )

    svm = train_svm(
        X_train_vec,
        y_train,
    )

    linear = train_linear_svc(
        X_train_vec,
        y_train,
    )

    print("Models trained successfully.")