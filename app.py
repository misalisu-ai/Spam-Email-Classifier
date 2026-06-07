"""
TechCrush AI/ML Bootcamp - Capstone Project
Group 13: Muhammad Ibrahim Salisu, Orjiakor Favour, Victor Chimbo, Godstime Ekan

File: app.py (Streamlit Web Interface)
"""

import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf

# Set up the title and page icon
st.set_page_config(page_title="Group 13 Spam Classifier", page_icon="🛡️")

# Safely load our saved models and vectorizer
@st.cache_resource
def load_saved_files():
    v_path = os.path.join("models", "tfidf_vectorizer.pkl")
    nb_path = os.path.join("models", "nb_model.pkl")
    nn_path = os.path.join("models", "spam_nn_model.h5")
    
    # Check if files exist to avoid crashing
    if not (os.path.exists(v_path) and os.path.exists(nb_path) and os.path.exists(nn_path)):
        return None, None, None

    with open(v_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(nb_path, "rb") as f:
        nb_model = pickle.load(f)
    nn_model = tf.keras.models.load_model(nn_path)
    
    return vectorizer, nn_model, nb_model

vectorizer, nn_model, nb_model = load_saved_files()

# Create the Sidebar with Group Info (Great for Grading!)
with st.sidebar:
    st.header("👥 Group 13 Members")
    st.write("- Muhammad Ibrahim Salisu")
    st.write("- Orjiakor Favour")
    st.write("- Victor Chimbo")
    st.write("- Godstime Ekan")
    st.markdown("---")
    st.write("**Course:** TechCrush AI/ML Cohort 6")

# Design the Main User Interface
st.title("Intelligent Spam Email Classifier")
st.write("Welcome to the Group 13 deployment portal. Paste an email below to test our models.")

if vectorizer is None:
    st.error("Missing files! Please make sure your 'models' folder contains your vectorizer and model files.")
    st.stop()

# Text box for user to paste their email
email_input = st.text_area("Paste the email text here:", height=150, placeholder="Type something like: Win free money now!")

# Model Selection Toggle (Shows the instructors you compared multiple approaches!)
chosen_model = st.radio("Choose which model to use for prediction:", ("Neural Network (Core Solution)", "Naive Bayes (Baseline)"))

# Run the Prediction Logic when the user clicks the button
if st.button("Classify Email"):
    if email_input.strip() == "":
        st.warning("Please type or paste some text first.")
    else:
        # Step A: Turn the input text into numbers using our saved vectorizer
        transformed_text = vectorizer.transform([email_input]).toarray()
        
        # Step B: Get prediction based on the chosen model
        if chosen_model == "Neural Network (Core Solution)":
            raw_score = nn_model.predict(transformed_text)
            is_spam = raw_score >= 0.5
            certainty = raw_score if is_spam else (1.0 - raw_score)
        else:
            is_spam = nb_model.predict(transformed_text)
            probabilities = nb_model.predict_proba(transformed_text)
            certainty = probabilities[is_spam]

        # Step C: Show the results nicely on screen
        st.markdown("### Classification Result:")
        if is_spam == 1:
            st.error(f" **SPAM DETECTED** (Confidence: {certainty:.2%})")
            st.write("This email looks like unsolicited advertising, spam, or a scam.")
        else:
            st.success(f" **HAM (SAFE EMAIL)** (Confidence: {certainty:.2%})")
            st.write("This looks like a normal, safe message.")
