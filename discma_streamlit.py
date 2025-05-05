import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import re
import textstat
import streamlit as st

# ---- Load Trained Model and Scaler ----
@st.cache_resource
def load_model_and_scaler():
    model = lgb.Booster(model_file='model/discma.txt')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

# ---- Custom Feature Extractor (exact match with training) ----
def extract_features(question_text):
    features = {}
    if isinstance(question_text, str):
        features['num_words'] = len(question_text.split())
        features['num_numbers'] = len(re.findall(r'\d+', question_text))
        features['num_math_symbols'] = len(re.findall(r'[+\-*/=]', question_text))
        features['question_length'] = len(question_text)
        features['has_sequence_word'] = int(bool(re.search(r'\b(sequence|summation|terms|series|sum)\b', question_text, re.IGNORECASE)))
    else:
        features = {key: 0 for key in ['num_words', 'num_numbers', 'num_math_symbols', 'question_length', 'has_sequence_word']}
    return features

# ---- Prediction Function ----
def predict_difficulty(model, scaler, question_text):
    features = extract_features(question_text)
    X_new = pd.DataFrame([features])
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)
    return prediction[0]

# ---- Streamlit Interface ----
def main():
    st.title('Question Difficulty Prediction')

    model, scaler = load_model_and_scaler()

    st.subheader("Enter a Question Text to Predict Difficulty")

    question_text = st.text_area("Enter your question here:")

    if question_text:
        prediction = predict_difficulty(model, scaler, question_text)
        st.write(f"Predicted Difficulty: {prediction:.2f}")

    uploaded_file = st.file_uploader("Upload Question Data (CSV)", type=["csv"])

    if uploaded_file is not None:
        df_questions = pd.read_csv(uploaded_file)
        st.write("Uploaded Questions:")
        st.write(df_questions.head())

        predictions = []
        for q_text in df_questions.iloc[:, 0]:
            prediction = predict_difficulty(model, scaler, q_text)
            predictions.append(prediction)

        df_questions['Predicted Difficulty'] = predictions
        st.write("Questions with Predicted Difficulty:")
        st.write(df_questions)

# ---- Run Streamlit App ----
if __name__ == "__main__":
    main()
