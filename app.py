import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained vectorizer
try:
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("❌ models/vectorizer.pkl not found.")
    st.stop()

# Load models
try:
    with open("models/decision_tree_model.pkl", "rb") as f:
        dt_model = pickle.load(f)

    with open("models/naive_bayes_model.pkl", "rb") as f:
        nb_model = pickle.load(f)

    with open("models/svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
    st.stop()

# Streamlit UI
st.title("📄 Text Classification App")
st.markdown("Use one of three ML models to classify your input text.")

user_input = st.text_area("✍️ Enter text for classification", "")

model_choice = st.selectbox("🧠 Choose a model", ["Naive Bayes", "Decision Tree", "SVM"])

if st.button("🚀 Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])

        if model_choice == "Naive Bayes":
            prediction = nb_model.predict(input_vector)[0]
        elif model_choice == "Decision Tree":
            prediction = dt_model.predict(input_vector)[0]
        else:
            prediction = svm_model.predict(input_vector)[0]

        st.success(f"✅ Predicted Class: **{prediction}**")
