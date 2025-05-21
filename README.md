# Check out the live demo of the app here:  

[Spam-detector](https://hj9cbiz3dk7pqrq7mkirwo.streamlit.app/)



# 🗂️ Text Classification Tool - Spam vs Ham

This is a simple web-based text classification tool developed using **Streamlit**. It allows users to classify a given text input (such as an email or message) as either **Spam** or **Ham** (not spam) using one of three machine learning models.

## 🚀 Features

- Classify input text as **Spam** or **Ham**.
- Choose between three different models:
  - Naive Bayes
  - Decision Tree
  - Support Vector Machine (SVM)
- Clean and interactive web interface built with Streamlit.

## 🧠 Models Used

All models are pre-trained and loaded using `joblib` from the `models/` directory:

- `vectorizer.pkl` – Text vectorization model (e.g., CountVectorizer or TfidfVectorizer).
- `naive_bayes_model.pkl`
- `decision_tree_model.pkl`
- `svm_model.pkl`


## ▶️ How to Run

1. Install the required dependencies:
   ```bash
   pip install streamlit scikit-learn joblib




