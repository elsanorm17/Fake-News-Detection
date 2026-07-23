"""
visualization.py

Visualization utilities for the Fake News Detection project.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.svm import LinearSVC
from wordcloud import WordCloud


def plot_wordcloud(
    texts,
    labels,
    target_label,
    stopwords,
    title,
    figsize=(12, 8),
):
    """
    Generate a word cloud for one class.
    """

    text = " ".join(
        article
        for article, label in zip(texts, labels)
        if label == target_label
    )

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=1,
        background_color="white",
        collocations=False,
        stopwords=stopwords,
    ).generate(text)

    plt.figure(figsize=figsize)
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def print_model_accuracies(models, X_test, y_test):
    """
    Print accuracy for a dictionary of trained models.

    Example
    -------
    models = {
        "Logistic Regression": logistic_model,
        "SVC": svm_model,
        "Linear SVC": linear_model,
    }
    """

    for name, model in models.items():
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"{name}: {accuracy:.2%}")


def plot_roc_curve(model, X_test, y_test):
    """
    Plot an ROC curve for a classifier that supports
    decision_function().
    """

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title("ROC Curve")
    plt.tight_layout()
    plt.show()


def plot_linear_svc_c_values(
    X_train,
    X_test,
    y_train,
    y_test,
    c_values=None,
):
    """
    Plot LinearSVC accuracy versus C.
    """

    if c_values is None:
        c_values = [i * 0.2 for i in range(1, 12)]

    accuracies = []

    for C in c_values:
        model = LinearSVC(
            C=C,
            max_iter=100000,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracies.append(
            accuracy_score(y_test, predictions)
        )

    plt.figure(figsize=(8, 5))
    plt.plot(c_values, accuracies, marker="o")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title("Linear SVC Accuracy vs. C")
    plt.grid(True)
    plt.tight_layout()
    plt.show()