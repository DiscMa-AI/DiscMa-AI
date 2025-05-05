import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import re
import textstat
import streamlit as st
import openai
import seaborn as sns
import matplotlib.pyplot as plt

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
    return prediction[0], features

# ---- Generate Explanation ----
def generate_difficulty_explanation(question, features, difficulty_score):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    feature_list = "\n".join([f"- {k}: {v}" for k, v in features.items()])

    prompt = (
        f"Given the following discrete math question and its extracted features:\n"
        f"Question: \"{question}\"\n\n"
        f"Features:\n{feature_list}\n\n"
        f"The model predicted a difficulty score of {difficulty_score:.2f}.\n"
        f"Explain why this question is considered to have that difficulty based on the features above."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error (explanation): {e}")
        return None

# ---- Generate Adjusted Questions ----
def generate_adjusted_questions(base_question, difficulty, intent, model="gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    prompt = (
        f"Given the following discrete math question and its difficulty rating of {difficulty:.2f}:\n"
        f"\"{base_question}\"\n\n"
        f"Generate a {intent} version of this question. Keep it clear and concise."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error (adjusted question): {e}")
        return []

# ---- Feature Table ----
def show_feature_table(features):
    st.subheader("📋 Extracted Features")
    df = pd.DataFrame([features]).T
    df.columns = ['Value']
    st.table(df)

# ---- Streamlit App ----
def main():
    st.title('📊 Discrete Math Question Difficulty Analyzer')
    model, scaler = load_model_and_scaler()

    st.subheader("🔤 Enter a Question")
    question_text = st.text_area("Enter your question here:")

    if question_text:
        difficulty, features = predict_difficulty(model, scaler, question_text)

        st.markdown(f"<h4>🧠 Predicted Difficulty: <span style='color:#ff4b4b'><b>{difficulty:.2f}</b></span></h4>", unsafe_allow_html=True)
        show_feature_table(features)

        explanation = generate_difficulty_explanation(question_text, features, difficulty)
        if explanation:
            st.info(f"📝 Explanation: {explanation}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Generate Similar"):
                sim = generate_adjusted_questions(question_text, difficulty, "similar")
                st.success(f"Similar Question: {sim}")

        with col2:
            if st.button("Generate Easier"):
                easier = generate_adjusted_questions(question_text, difficulty, "easier")
                st.success(f"Easier Question: {easier}")

        with col3:
            if st.button("Generate Harder"):
                harder = generate_adjusted_questions(question_text, difficulty, "harder")
                st.success(f"Harder Question: {harder}")

    st.divider()
    st.subheader("📁 Or Upload a CSV of Questions")
    uploaded_file = st.file_uploader("Upload Question Data (CSV)", type=["csv"])

    if uploaded_file is not None:
        df_questions = pd.read_csv(uploaded_file)
        st.write("Uploaded Questions Preview:")
        st.write(df_questions.head())

        difficulties, feature_rows = [], []
        for q in df_questions.iloc[:, 0]:
            d, f = predict_difficulty(model, scaler, q)
            difficulties.append(d)
            feature_rows.append(f)

        df_features = pd.DataFrame(feature_rows)
        df_questions['Predicted Difficulty'] = difficulties
        df_full = pd.concat([df_questions, df_features], axis=1)

        st.write("📊 Questions with Features and Predicted Difficulty:")
        st.dataframe(df_full)

# ---- Run the App ----
if __name__ == "__main__":
    main()
