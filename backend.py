import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report


# ---------------------------
# TEXT PREPROCESSING
# ---------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()

    return " ".join(tokens)


# ---------------------------
# LOAD DATASET
# ---------------------------
def load_dataset():
    df = pd.read_csv("reviews.csv")

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
        X_vec,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
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
        y_train,
        y_pred_train,
        output_dict=True
    )

    return sentiment, report
