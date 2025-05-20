import streamlit as st
import joblib

# Load vectorizer and models from models/ folder
@st.cache_resource(show_spinner=False)
def load_models():
    vectorizer = joblib.load("models/vectorizer.pkl")
    nb_model = joblib.load("models/naive_bayes_model.pkl")
    dt_model = joblib.load("models/decision_tree_model.pkl")
    svm_model = joblib.load("models/svm_model.pkl")
    return vectorizer, nb_model, dt_model, svm_model

vectorizer, nb_model, dt_model, svm_model = load_models()

# Page setup
st.set_page_config(page_title="Text Classification App", page_icon="🗂️", layout="centered")

st.markdown("<h1 style='text-align: center; color: navy;'>Text Classification Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter your text below and select a model to classify it.</p>", unsafe_allow_html=True)
st.markdown("---")

# User input form
with st.form("text_form"):
    user_input = st.text_area("Enter text here:", height=150, placeholder="Type or paste your text...")
    model_choice = st.radio("Select model:", ["Naive Bayes", "Decision Tree", "SVM"], horizontal=True)
    submitted = st.form_submit_button("Classify")

# Classification on submit
if submitted:
    if user_input.strip() == "":
        st.warning("Please enter some text before classification.")
    else:
        input_vector = vectorizer.transform([user_input]).toarray()

        if model_choice == "Naive Bayes":
            prediction = nb_model.predict(input_vector)[0]
        elif model_choice == "Decision Tree":
            prediction = dt_model.predict(input_vector)[0]
        else:
            prediction = svm_model.predict(input_vector)[0]

        st.markdown("---")
        st.success("Classification Complete ✅")
        st.markdown(f"<h3 style='color: darkgreen;'>Predicted Class: {prediction}</h3>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
