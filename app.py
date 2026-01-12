"""
Sentiment Analysis Web Application

This application performs sentiment analysis on text reviews
using classical machine learning models and NLP techniques.
"""

import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import plotly.graph_objects as go

from wordcloud import WordCloud
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

import backend


# ---------------------------
# NLTK DOWNLOADS
# ---------------------------
nltk.download("punkt")


# ---------------------------
# DATA LOADING
# ---------------------------
@st.cache_data
def load_data():
    return backend.load_dataset()


df = load_data()


# ---------------------------
# APP LAYOUT
# ---------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.title("Sentiment Analysis System")
st.write("Analyze and predict sentiment from textual reviews.")


menu = st.sidebar.selectbox(
    "Select Option",
    ["Visualization", "Prediction"]
)


# ===========================
# VISUALIZATION SECTION
# ===========================
if menu == "Visualization":

    vis_option = st.sidebar.selectbox(
        "Select Visualization",
        ["Word Cloud", "Sentiment Distribution"]
    )

    # WORD CLOUD
    if vis_option == "Word Cloud":

        review_type = st.selectbox(
            "Select Review Type",
            ["All", "Positive", "Negative"]
        )

        if review_type == "Positive":
            text = " ".join(
                df[df["feedback"] == 1]["verified_reviews"].astype(str)
            )
        elif review_type == "Negative":
            text = " ".join(
                df[df["feedback"] == 0]["verified_reviews"].astype(str)
            )
        else:
            text = " ".join(df["verified_reviews"].astype(str))

        wordcloud = WordCloud(
            background_color="white",
            width=800,
            height=400
        ).generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)

    # SENTIMENT DISTRIBUTION
    if vis_option == "Sentiment Distribution":

        counts = df["feedback"].value_counts()

        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Negative", "Positive"],
                    y=[counts.get(0, 0), counts.get(1, 0)]
                )
            ]
        )

        fig.update_layout(
            title="Sentiment Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Count"
        )

        st.plotly_chart(fig, use_container_width=True)


# ===========================
# PREDICTION SECTION
# ===========================
if menu == "Prediction":

    st.subheader("Sentiment Prediction")

    model_name = st.sidebar.selectbox(
        "Select Model",
        [
            "Bernoulli Naive Bayes",
            "Logistic Regression",
            "Gradient Boosting",
            "Linear SVC"
        ]
    )

    user_input = st.text_area("Enter review text")

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        user_text = [user_input]

        X_train, X_test, y_train, y_test, vectorizer = backend.reset_feature()

        if st.button("Predict Sentiment"):

            with st.spinner("Processing..."):
                time.sleep(1)

                if model_name == "Bernoulli Naive Bayes":
                    model = BernoulliNB()
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_name == "Gradient Boosting":
                    model = GradientBoostingClassifier()
                else:
                    model = LinearSVC()

                prediction, report = backend.simple_predict(
                    model,
                    user_text,
                    X_train,
                    y_train,
                    vectorizer
                )

            st.success("Prediction Completed")

            st.metric("Predicted Sentiment", prediction)

            st.subheader("Model Evaluation")
            st.table(pd.DataFrame(report))
