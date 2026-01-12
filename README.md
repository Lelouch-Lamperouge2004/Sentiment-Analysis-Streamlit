# Sentiment Analysis Web Application using Streamlit

## Overview

The **Sentiment Analysis Web Application** is a machine learning–based system designed to analyze and classify the sentiment of textual reviews. The application helps businesses, researchers, and individuals understand user opinions by categorizing text data into **Positive** or **Negative** sentiment using Natural Language Processing (NLP) techniques.

The project is built using **Python** and **Streamlit**, providing an interactive web-based dashboard that allows users to visualize sentiment distributions and perform real-time sentiment predictions on custom text input.

---

## Live Application

The application is deployed on **Streamlit Community Cloud** and can be accessed using the link below:

**Live Demo:**  
https://sentiment-analysis-app-wlhpdgtxsrujcuepwjimzk.streamlit.app

---

## Features

- Interactive web-based dashboard built with Streamlit
- Text preprocessing using NLP techniques (tokenization, stopword removal, stemming)
- Sentiment visualization using:
  - Word Cloud
  - Sentiment distribution bar chart
- Real-time sentiment prediction for user-entered text
- Multiple machine learning models for sentiment classification:
  - Bernoulli Naive Bayes
  - Logistic Regression
  - Gradient Boosting Classifier
  - Linear Support Vector Classifier (Linear SVC)
- Clean and minimal user interface
- Deployed for real-time access using Streamlit Cloud

---

## Technologies Used

- **Programming Language:** Python  
- **Web Framework:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Natural Language Processing:** NLTK  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Matplotlib, Plotly, WordCloud  

---

## Project Structure

sentiment-analysis-app/
│
├── app.py # Streamlit frontend
├── backend.py # Data processing and ML logic
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files


---

## How the System Works

1. Text reviews are preprocessed using NLP techniques such as tokenization, stopword removal, and stemming.
2. The cleaned text is converted into numerical features using Count Vectorization.
3. Machine learning models are trained on labeled review data.
4. The trained models predict sentiment for new user-provided text.
5. Results are displayed through an interactive Streamlit dashboard.

---

## How to Run the Project Locally

### Step 1: Clone the Repository

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

### Step 2: Create a Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Run the Application
streamlit run app.py


The application will open automatically in your web browser.
Use Cases :
Customer feedback analysis
Product review analysis
Opinion mining
Academic and learning projects
Business sentiment insights

<br>Future Enhancements :
Add neutral sentiment classification
Integrate deep learning models (LSTM, BERT)
Allow CSV file upload for bulk analysis
Add sentiment trend analysis over time

<br>Conclusion :
The Sentiment Analysis Web Application using Streamlit provides an easy-to-use and effective solution for understanding textual sentiment. With its clean interface, real-time predictions, and visual insights, the project demonstrates practical application of machine learning and NLP concepts in a real-world scenario.
