import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import re
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import textstat

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = lgb.Booster(model_file='model/discma1.txt')
    scaler = joblib.load('model/scaler1.pkl')
    return model, scaler

# Feature extraction from a question
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
    features["num_variables"] = len(re.findall(r'\b[a-zA-Z]\b', text))
    features["readability"] = textstat.flesch_reading_ease(text)
    features["num_keywords"] = sum(1 for word in words if word.lower() in keywords)

    return features

# Feature Weights
FEATURE_WEIGHTS = {
    "length": 0.4,
    "word_count": 0.3,
    "avg_word_length": 0.25,
    "num_numbers": 0.4,
    "num_math_symbols": 0.5,
    "num_variables": 0.3,
    "readability": -0.4,
    "num_keywords": 0.2,
}

# Adjusted difficulty calculation
def calculate_adjusted_difficulty(model, scaler, question_text):
    features = extract_features(question_text)
    X_new = pd.DataFrame([features])
    X_new_scaled = scaler.transform(X_new)
    predicted_difficulty = model.predict(X_new_scaled)[0]

    normalized_features = {
        "length": min(features["length"] / 100, 1),
        "word_count": min(features["word_count"] / 20, 1),
        "avg_word_length": min(features["avg_word_length"] / 6, 1),
        "num_numbers": min(features["num_numbers"] / 5, 1),
        "num_math_symbols": min(features["num_math_symbols"] / 5, 1),
        "num_variables": min(features["num_variables"] / 5, 1),
        "readability": 1 - (features["readability"] / 100),
        "num_keywords": min(features["num_keywords"] / 5, 1),
    }

    adjusted_difficulty = predicted_difficulty
    for feature, weight in FEATURE_WEIGHTS.items():
        adjusted_difficulty += weight * normalized_features[feature]

    adjusted_difficulty = max(0, min(10, adjusted_difficulty))
    return predicted_difficulty, adjusted_difficulty, features

# Explanation generation via GPT
def generate_explanation_with_feature_impact_adjustment(question_text, adjusted_difficulty, features, model="gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = (
        f"Question: \"{question_text}\"\n"
        f"Adjusted Difficulty: {adjusted_difficulty:.2f}\n"
        f"Features: {features}\n"
        f"Based on the following feature values (length, word_count, avg_word_length, num_numbers, "
        f"num_math_symbols, num_variables, readability, num_keywords), explain how each feature impacts the difficulty. "
        f"Provide a final difficulty score recommendation after considering these impacts."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Explanation error: {e}"

# Question generation
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

# Feature heatmap
def generate_feature_heatmap(questions):
    feature_data = [extract_features(q) for q in questions]
    labels = [q[:30] + '...' if len(q) > 30 else q for q in questions]
    df = pd.DataFrame(feature_data, index=labels)
    st.subheader("\U0001F50D Feature Heatmap")
    fig, ax = plt.subplots(figsize=(10, len(questions) * 0.5 + 2))
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f", ax=ax)
    st.pyplot(fig)

# Main application
def main():
    st.set_page_config(page_title="Discrete Math Difficulty Predictor", layout="wide")
    st.title("\U0001F4CA Discrete Math Question Difficulty Predictor")

    st.markdown("""
    ### \U0001F4CC Instructions for Use

    #### Manual Input
    - Enter a **single discrete math question** focused on **sequences and summations**.

    #### \U0001F4CB CSV Upload
    - Upload a **CSV file** with a **single column** of questions under the header `Question`.
    """)

    model, scaler = load_model_and_scaler()

    # Manual Input Section
    st.subheader("\U0001F524 Manual Input")
    question_text = st.text_area("Enter your discrete math question:")

    if question_text:
        cleaned_input = question_text.strip()
        if not cleaned_input:
            st.warning("\u26A0\uFE0F Please enter a non-empty question.")
            return

        _, adjusted_difficulty, features = calculate_adjusted_difficulty(model, scaler, cleaned_input)

        st.subheader("\U0001F4CC Feature Analysis")
        st.table(pd.DataFrame([features]))

        explanation = generate_explanation_with_feature_impact_adjustment(cleaned_input, adjusted_difficulty, features)
        st.subheader("\U0001F9E0 Explanation with Feature Impact")
        st.write(explanation)
        st.markdown(f"**\U0001F4C8 Adjusted Difficulty:** `{adjusted_difficulty:.2f}`")

        generate_feature_heatmap([cleaned_input])

        st.subheader("\U0001F916 Generate Related Questions")
        for diff_type in ["similar", "easier", "harder"]:
            if st.button(f"Generate {diff_type.capitalize()} Questions"):
                with st.spinner("Generating..."):
                    questions = generate_custom_questions(cleaned_input, adjusted_difficulty, diff_type)
                st.success(f"{diff_type.capitalize()} Questions:")
                for q in questions:
                    st.markdown(f"- {q}")

    st.divider()

    # CSV Upload Section
    st.subheader("\U0001F4C1 Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV of discrete math questions", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("\U0001F4C4 Preview of Uploaded File:")
        st.dataframe(df.head())

        for index, row in df.iterrows():
            q = row[0]
            if not isinstance(q, str) or not q.strip():
                continue

            cleaned_q = q.strip()
            _, adjusted_difficulty, features = calculate_adjusted_difficulty(model, scaler, cleaned_q)
            explanation = generate_explanation_with_feature_impact_adjustment(cleaned_q, adjusted_difficulty, features)

            st.markdown(f"### Question {index + 1}: {cleaned_q}")
            st.markdown(f"**Adjusted Difficulty:** `{adjusted_difficulty:.2f}`")
            st.markdown(f"**Explanation:** {explanation}")

            for diff_type in ["similar", "easier", "harder"]:
                if st.button(f"Generate {diff_type.capitalize()} for Q{index + 1}"):
                    with st.spinner("Generating..."):
                        questions = generate_custom_questions(cleaned_q, adjusted_difficulty, diff_type)
                    st.success(f"{diff_type.capitalize()} Questions for Q{index + 1}:")
                    for q in questions:
                        st.markdown(f"- {q}")

if __name__ == "__main__":
    main()
