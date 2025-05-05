import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import re
import textstat
import streamlit as st
import openai

# Streamlit Sidebar Settings
with st.sidebar:
    st.sidebar.title("⚙️ Settings")
    use_gpt4 = st.sidebar.toggle("Use GPT-4", value=False)
    selected_model = "gpt-4" if use_gpt4 else "gpt-3.5-turbo"

openai.api_key = st.secrets["OPENAI_API_KEY"]

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
        
        st.subheader("Generate Similar Questions")

    if st.button("Generate Similar Questions using OpenAI"):
        with st.spinner("Generating questions..."):
            generated_questions = generate_similar_questions(
                base_question=question_text,
                difficulty=prediction,
                num_questions=3,
                model=selected_model
            )
            if generated_questions:
                st.success("Similar questions generated:")
                for i, q in enumerate(generated_questions, 1):
                    st.markdown(f"**{i}.** {q}")

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
