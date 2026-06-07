"""
TechCrush AI/ML Bootcamp - Capstone Project
Group 13: Muhammad Ibrahim Salisu, Orjiakor Favour, Victor Chimbo, Godstime Ekan

File: app.py (Streamlit Web Interface - Unified Multi-Model Layout)
Assisted by AI Tools (ChatGPT) for user interface structure per Section 4.1.
"""

import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf

# 1. Set up the page layout
st.set_page_config(page_title="Group 13 Spam Classifier", page_icon="🛡️", layout="wide")

# 2. Safely load our saved models and vectorizer
@st.cache_resource
def load_saved_files():
    v_path = os.path.join("models", "tfidf_vectorizer.pkl")
    nb_path = os.path.join("models", "nb_model.pkl")
    nn_path = os.path.join("models", "spam_nn_model.h5")
    
    if not (os.path.exists(v_path) and os.path.exists(nb_path) and os.path.exists(nn_path)):
        return None, None, None

    with open(v_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(nb_path, "rb") as f:
        nb_model = pickle.load(f)
    nn_model = tf.keras.models.load_model(nn_path)
    
    return vectorizer, nn_model, nb_model

vectorizer, nn_model, nb_model = load_saved_files()

# 3. Create the Sidebar with Group Info for Grading
with st.sidebar:
    st.image("https://img.shields.io/badge/TechCrush_Cohort_5-Capstone_Project-blue?style=for-the-badge")
    st.header("👥 Group 13 Members")
    st.write("- Muhammad Ibrahim Salisu")
    st.write("- Orjiakor Favour")
    st.write("- Victor Chimbo")
    st.write("- Godstime Ekan")
    st.markdown("---")
    st.write("**Course:** TechCrush AI/ML Bootcamp")

# 4. Design the Main User Interface
st.title("✉️ Intelligent Multi-Model Spam Email Classifier")
st.write("Paste an email message below. Our pipeline will instantly analyze the text tokens across both of our trained architectures.")

if vectorizer is None:
    st.error("Missing files! Please make sure your 'models' folder contains your vectorizer and model files.")
    st.stop()

# Big text area for user input (Simple and straightforward)
email_input = st.text_area("Paste the email text here:", height=180, placeholder="Type or paste something like: Urgent! Click here to claim your cash bonus lottery now...")

# 5. Run the Prediction Logic when the user clicks the button
if st.button("Classify Email", type="primary", use_container_width=True):
    if email_input.strip() == "":
        st.warning("⚠️ Please type or paste some text first.")
    else:
        with st.spinner("Processing text and calculating model probabilities..."):
            # Step A: Turn the input text into a numerical matrix matching our 5,695 features
            transformed_text = vectorizer.transform([email_input]).toarray()
            
            # Step B: Model 1 - Neural Network Execution
            nn_raw_score = nn_model.predict(transformed_text)
            nn_is_spam = 1 if nn_raw_score >= 0.5 else 0
            nn_certainty = float(nn_raw_score) if nn_is_spam == 1 else float(1.0 - nn_raw_score)
            
            # Step C: Model 2 - Multinomial Naive Bayes Execution
            nb_is_spam = int(nb_model.predict(transformed_text))
            nb_probabilities = nb_model.predict_proba(transformed_text)
            nb_certainty = float(nb_probabilities[nb_is_spam])

        st.markdown("---")
        st.subheader("📊 Comparative Prediction Analysis")
        st.write("To fulfill the project requirements completely, your input text was processed independently by both models:")

        # Create 2 layout columns side-by-side to display both model findings beautifully
        col1, col2 = st.columns(2)

        # Left Column: Neural Network Output (Core Solution)
        with col1:
            st.markdown("### 🧠 Deep Learning Neural Network")
            if nn_is_spam == 1:
                st.error(f"🚨 **RESULT: SPAM DETECTED**")
                st.metric(label="Prediction Confidence Strength", value=f"{nn_certainty:.2%}")
            else:
                st.success(f"✅ **RESULT: HAM (SAFE EMAIL)**")
                st.metric(label="Prediction Confidence Strength", value=f"{nn_certainty:.2%}")
            st.caption("Architectural structure: Input (5695) → Dense (128) → Dropout (0.5) → Dense (64) → Sigmoid Output.")

        # Right Column: Naive Bayes Output (Baseline)
        with col2:
            st.markdown("### 🧮 Multinomial Naive Bayes")
            if nb_is_spam == 1:
                st.error(f"🚨 **RESULT: SPAM DETECTED**")
                st.metric(label="Prediction Confidence Strength", value=f"{nb_certainty:.2%}")
            else:
                st.success(f"✅ **RESULT: HAM (SAFE EMAIL)**")
                st.metric(label="Prediction Confidence Strength", value=f"{nb_certainty:.2%}")
            st.caption("Architectural structure: Traditional probabilistic model using word frequencies and Laplace Smoothing (alpha=1.0).")
