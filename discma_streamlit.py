import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import streamlit as st
import re
import textstat
import joblib

# ---- Load Trained Model and Scaler ----
@st.cache_resource
def load_trained_model():
    model = lgb.Booster(model_file='model/discma.txt')  # Trained LightGBM model
    scaler = joblib.load('model/scaler.pkl')  # Pre-fitted StandardScaler
    return model, scaler

# ---- Feature Extractor ----
def extract_features(question_text):
    features = {
        "length": 0,
        "word_count": 0,
        "avg_word_length": 0,
        "num_numbers": 0,
        "num_math_symbols": 0,
        "num_variables": 0,
        "readability": 0,
        "num_keywords": 0
    }

    if not isinstance(question_text, str) or question_text.strip() == "":
        return features

    text = question_text.strip()
    words = text.split()
    keywords = ["sequence", "term", "sum", "summation", "series", "pattern", "next", "following", "arithmetic", "geometric"]

    features["length"] = len(text)
    features["word_count"] = len(words)
    features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
    features["num_numbers"] = len(re.findall(r'\d+', text))
    features["num_math_symbols"] = len(re.findall(r'[+\-*/=^]', text))
    features["num_variables"] = len(re.findall(r'\b[a-zA-Z]\b', text))  # only standalone letters like "n", "k", "x"
    features["readability"] = textstat.flesch_reading_ease(text)
    features["num_keywords"] = sum(1 for word in words if word.lower() in keywords)

    return features

# ---- Predict Difficulty ----
def predict_difficulty(model, scaler, new_questions):
    features_list = [extract_features(q) for q in new_questions]
    X_new = pd.DataFrame(features_list)
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    return predictions

# ---- Streamlit UI ----
def main():
    st.title("🧮 Discrete Math Difficulty Predictor (Sequences & Summations)")

    # Load trained model and scaler
    model, scaler = load_trained_model()

    # ---- Upload CSV ----
    st.header("📄 Upload a CSV with Your Questions")
    uploaded_file = st.file_uploader("Upload a CSV file (columns named like 'Question 1', 'Question 2', etc.)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        question_columns = [col for col in df.columns if col.lower().startswith('question')]

        if question_columns:
            questions = df[question_columns].iloc[0].tolist()
            predictions = predict_difficulty(model, scaler, questions)

            st.subheader("📊 Predicted Difficulty for Uploaded Questions")
            for i, q in enumerate(questions):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"📈 **Predicted Difficulty:** `{predictions[i]:.2f}`")
                st.markdown("💡 _Suggestion placeholder: You can simplify this question by..._")
                st.markdown("---")
        else:
            st.error("No 'Question' columns found. Please check your CSV formatting.")

    # ---- Manual Input ----
    st.header("✍️ Or Input Questions Manually")
    question_input = st.text_area("Enter one or more questions (separated by new lines):")

    if question_input:
        new_questions = [q.strip() for q in question_input.split("\n") if q.strip()]
        predictions = predict_difficulty(model, scaler, new_questions)

        st.subheader("📊 Predicted Difficulty for Manual Questions")
        for i, q in enumerate(new_questions):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"📈 **Predicted Difficulty:** `{predictions[i]:.2f}`")
            st.markdown("💡 _Suggestion placeholder: You can simplify this question by..._")
            st.markdown("---")

if __name__ == "__main__":
    main()
