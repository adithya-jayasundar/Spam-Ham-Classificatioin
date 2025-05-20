import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
with open("decision_tree_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("naive_bayes_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Load or define the vectorizer (assuming you used TfidfVectorizer for training)
# Replace this with the actual vectorizer used during training if available
vectorizer = TfidfVectorizer()

# Dummy fit to avoid error (you must use the real vectorizer used in training)
vectorizer.fit(["sample text for dummy fit"])

# Streamlit app interface
st.title("Text Classification App")

user_input = st.text_area("Enter text for classification", "")

model_choice = st.selectbox("Choose a model", ["Naive Bayes", "Decision Tree", "SVM"])

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])

        if model_choice == "Naive Bayes":
            prediction = nb_model.predict(input_vector)[0]
        elif model_choice == "Decision Tree":
            prediction = dt_model.predict(input_vector)[0]
        else:
            prediction = svm_model.predict(input_vector)[0]

        st.success(f"Predicted Class: {prediction}")
