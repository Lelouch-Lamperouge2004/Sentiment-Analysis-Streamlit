import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download("punkt")
nltk.download("stopwords")


# ---------------------------
# TEXT PREPROCESSING
# ---------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    cleaned = [
        stemmer.stem(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(cleaned)


# ---------------------------
# LOAD DATASET
# ---------------------------
def load_dataset():
    df = pd.read_csv("data/reviews.csv")

    df = df[["verified_reviews", "feedback"]]
    df.dropna(inplace=True)

    df["clean_text"] = df["verified_reviews"].apply(preprocess_text)

    return df


# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
def reset_feature():
    df = load_dataset()

    X = df["clean_text"]
    y = df["feedback"]

    vectorizer = CountVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, vectorizer


# ---------------------------
# SIMPLE PREDICTION
# ---------------------------
def simple_predict(model, user_text, X_train, y_train, vectorizer):
    model.fit(X_train, y_train)

    user_vec = vectorizer.transform(user_text)
    pred = model.predict(user_vec)[0]

    sentiment = "Positive" if pred == 1 else "Negative"

    y_pred_train = model.predict(X_train)

    report = classification_report(
        y_train, y_pred_train, output_dict=True
    )

    return sentiment, report
