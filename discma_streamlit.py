# Enhanced Streamlit App with Advanced Features
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import re
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import json

@st.cache_resource
def load_model_and_scaler():
    model = lgb.Booster(model_file='model/discma1.txt')
    scaler = joblib.load('model/scaler1.pkl')
    return model, scaler

# Load embedding examples (precomputed vectors of in-scope questions)
def load_embedding_examples():
    with open("model/sequence_examples_embeddings.json") as f:
        return json.load(f)

# Extract features
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

# Get OpenAI embedding
@st.cache_data(show_spinner=False)
def get_embedding(text):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    try:
        response = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.warning(f"Embedding error: {e}")
        return np.zeros(1536)  # default size for fallback

# Check if question is in-scope
def is_in_scope(question_text, embedding_examples, threshold=0.8):
    emb = get_embedding(question_text)
    emb = normalize([emb])[0]
    example_embeddings = np.array(embedding_examples["embeddings"])
    sims = cosine_similarity([emb], example_embeddings)[0]
    return max(sims) >= threshold

# Predict difficulty
def predict_difficulty(model, scaler, question_text):
    features = extract_features(question_text)
    X_new = pd.DataFrame([features])
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)
    return prediction[0], features

# Generate explanation
def generate_explanation(question_text, difficulty_score, features, model="gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = (
        f"Question: \"{question_text}\"\n"
        f"Predicted Difficulty: {difficulty_score:.2f}\n"
        f"Features: {features}\n"
        f"Explain why this question is considered to have this difficulty level."
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
        return f"Explanation error: {e}"

# Generate related questions
def generate_custom_questions(base_question, difficulty, difficulty_type="similar", num_questions=3, model="gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    adjustment = {
        "similar": f"with a difficulty level of approximately {difficulty:.2f}",
        "easier": f"that are slightly easier than difficulty {difficulty:.2f}",
        "harder": f"that are slightly harder than difficulty {difficulty:.2f}"
    }
    prompt = (
        f"Generate {num_questions} discrete math questions based on the following, {adjustment[difficulty_type]}:\n"
        f"Question: {base_question}\n\nQuestions with brief notes explaining why each is {difficulty_type}:\n"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=700,
        )
        return response.choices[0].message.content.strip().split("\n")
    except Exception as e:
        return [f"Generation error: {e}"]

# Generate heatmap
def generate_feature_heatmap(questions):
    feature_data = [extract_features(q) for q in questions]
    labels = [q[:30] + '...' if len(q) > 30 else q for q in questions]
    df = pd.DataFrame(feature_data, index=labels)
    st.subheader("🔍 Feature Heatmap")
    fig, ax = plt.subplots(figsize=(10, len(questions)*0.5 + 2))
    sns.heatmap(df, annot=True, cmap="viridis", fmt="d", ax=ax)
    st.pyplot(fig)

# Main app

def main():
    st.title("📊 Discrete Math Question Difficulty Predictor")
    model, scaler = load_model_and_scaler()
    emb_examples = load_embedding_examples()

    st.subheader("🔤 Enter a Question")
    question_text = st.text_area("Enter your question:")

    if question_text:
        in_scope = is_in_scope(question_text, emb_examples)
        prediction, features = predict_difficulty(model, scaler, question_text)
        st.markdown(f"**Predicted Difficulty:** `{prediction:.2f}`")
        st.subheader("📌 Features")
        st.table(pd.DataFrame([features]))

        if in_scope:
            explanation = generate_explanation(question_text, prediction, features)
            st.subheader("🧠 Explanation")
            st.write(explanation)
            generate_feature_heatmap([question_text])

            st.subheader("🤖 Generate Questions")
            for diff_type in ["similar", "easier", "harder"]:
                if st.button(f"Generate {diff_type.capitalize()} Questions"):
                    with st.spinner("Generating..."):
                        questions = generate_custom_questions(question_text, prediction, diff_type)
                    st.success(f"{diff_type.capitalize()} Questions:")
                    for q in questions:
                        st.markdown(f"- {q}")
        else:
            st.warning("This question appears to be outside the scope (not about sequences or summations). No generation performed.")

    st.divider()
    st.subheader("📁 Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV of questions", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        predictions, scopes = [], []
        for q in df.iloc[:, 0]:
            pred, _ = predict_difficulty(model, scaler, q)
            predictions.append(pred)
            scopes.append(is_in_scope(q, emb_examples))

        df['Predicted Difficulty'] = predictions
        df['In Scope'] = scopes
        st.write("Processed Data:", df)
        generate_feature_heatmap(df.iloc[:, 0].tolist())
        st.download_button("📥 Download Processed CSV", df.to_csv(index=False), "processed_questions.csv")

if __name__ == '__main__':
    main()
