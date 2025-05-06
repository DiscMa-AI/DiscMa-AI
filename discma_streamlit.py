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

# Feature extraction
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

# Feature weights
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

# Adjusted difficulty score
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

# GPT explanation
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
    st.subheader("🔍 Feature Heatmap")
    fig, ax = plt.subplots(figsize=(10, len(questions) * 0.5 + 2))
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f", ax=ax)
    st.pyplot(fig)

# Main app
def main():
    st.set_page_config(page_title="Discrete Math Difficulty Predictor", layout="wide")
    st.title("📊 Discrete Math Question Difficulty Predictor")

    st.markdown("""
### 📌 Instructions for Use

#### Manual Input
- **Input a single discrete math question** related to **sequences and summations**.
- Supported topics include:
  - Arithmetic sequences
  - Geometric series
  - Pattern identification
  - Summation notation (Σ notation)
  - Next-term prediction in sequences
- The question must be clear and precise. Avoid overly general or vague questions.

#### Example of a Valid Question:
- **"What is the 10th term in the arithmetic sequence: 3, 7, 11, 15, ...?"**
- **"Find the sum of the first 20 terms of the geometric series: 5, 15, 45, ..."**

#### Examples of Questions **NOT** Processed:
These questions **will not be processed** because they fall outside the scope of sequences and summations:
- **"What is the derivative of x^2?"** (calculus-related)
- **"Solve for x in the equation 2x + 3 = 5."** (algebra-related)
- **"What is the area of a circle with radius 7?"** (geometry-related)
- **"Write a program to compute factorials."** (programming-related)
- **"What is the capital of France?"** (general knowledge)

#### 📋 CSV Upload
- Upload a **CSV file** containing a column of **discrete math questions** that follow the sequences and summations theme.
- The file should **not** contain any non-relevant questions or other types of math problems.

#### Example CSV Format:

| Question                                           |
|----------------------------------------------------|
| What is the sum of the first 10 terms in the arithmetic sequence: 2, 5, 8, 11, ...? |
| Find the next term in the geometric sequence: 3, 6, 12, 24, ... |
| What is the sum of the first 15 terms in the series: 10, 20, 40, ...? |
| Identify the pattern and find the 10th term: 5, 10, 20, 40, ... |
| What is the 8th term in the arithmetic sequence 1, 4, 7, 10, ...? |

#### Notes:
- Each row in the CSV should only contain a single question in plain text.
- Questions that are too short, malformed, or unrelated to the sequence and summation topics will be flagged as invalid.
- The maximum number of questions processed at once will depend on the file size and question length.

#### Important:
- **Questions that are too short** (e.g., "Find the sum of 1, 2, and 3.") or **unrelated** to sequences and summations will not be processed.
- Ensure your CSV file contains questions relevant to discrete math and summations.

### ❗ Example Invalid Questions in CSV:
| Question                                           |
|----------------------------------------------------|
| What is the derivative of x^2?                     |
| Solve for x in the equation 2x + 3 = 5.            |
| What is the area of a circle with radius 7?        |
| Write a program to compute factorials.             |
| What is the capital of France?                     |

These types of questions will be **ignored** by the app, as they do not pertain to sequences and summations.

""")


    model, scaler = load_model_and_scaler()

    st.subheader("🔤 Manual Input")
    question_text = st.text_area("Enter your discrete math question:")

    if question_text:
        cleaned_q = question_text.strip()
        keywords = ["sequence", "term", "sum", "summation", "series", "pattern", "next", "following", "arithmetic", "geometric"]

        if len(cleaned_q.split()) < 4:
            st.error("❌ The question is too short or malformed.")
        elif not any(kw in cleaned_q.lower() for kw in keywords):
            st.error("❌ Question is not about sequences or summations.")
        else:
            _, adjusted_difficulty, features = calculate_adjusted_difficulty(model, scaler, cleaned_q)
            explanation = generate_explanation_with_feature_impact_adjustment(cleaned_q, adjusted_difficulty, features)

            st.markdown(f"**Adjusted Difficulty:** {adjusted_difficulty:.2f}")
            st.markdown("**🧠 Explanation:**")
            st.write(explanation)

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Generate Similar Question"):
                    with st.spinner("Generating similar questions..."):
                        gen_questions = generate_custom_questions(cleaned_q, adjusted_difficulty, "similar")
                    st.markdown("**Similar Questions:**")
                    for gq in gen_questions:
                        st.markdown(f"- {gq}")

            with col2:
                if st.button("Generate Easier Question"):
                    with st.spinner("Generating easier questions..."):
                        gen_questions = generate_custom_questions(cleaned_q, adjusted_difficulty, "easier")
                    st.markdown("**Easier Questions:**")
                    for gq in gen_questions:
                        st.markdown(f"- {gq}")

            with col3:
                if st.button("Generate Harder Question"):
                    with st.spinner("Generating harder questions..."):
                        gen_questions = generate_custom_questions(cleaned_q, adjusted_difficulty, "harder")
                    st.markdown("**Harder Questions:**")
                    for gq in gen_questions:
                        st.markdown(f"- {gq}")

    st.divider()

    st.subheader("📁 Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV of discrete math questions", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Preview of Uploaded File:")
        st.dataframe(df.head())

        keywords = ["sequence", "term", "sum", "summation", "series", "pattern", "next", "following", "arithmetic", "geometric"]

        for q in df.iloc[:, 0]:
            st.markdown("---")
            st.markdown(f"### 📌 Question: {q}")
            if not isinstance(q, str) or not q.strip():
                st.warning("❌ Empty or non-text input.")
                continue

            cleaned_q = q.strip()
            if len(cleaned_q.split()) < 4:
                st.warning("❌ Too short or malformed question.")
                continue

            if not any(kw in cleaned_q.lower() for kw in keywords):
                st.warning("❌ Question is not about sequences or summations.")
                continue

            try:
                _, adjusted_difficulty, features = calculate_adjusted_difficulty(model, scaler, cleaned_q)
                explanation = generate_explanation_with_feature_impact_adjustment(cleaned_q, adjusted_difficulty, features)

                st.markdown(f"**Adjusted Difficulty:** {adjusted_difficulty:.2f}")
                st.markdown("**🧠 Explanation:**")
                st.write(explanation)

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button(f"Generate Similar Question for {cleaned_q[:10]}..."):
                        with st.spinner("Generating similar questions..."):
                            gen_questions = generate_custom_questions(cleaned_q, adjusted_difficulty, "similar")
                        st.markdown("**Similar Questions:**")
                        for gq in gen_questions:
                            st.markdown(f"- {gq}")

                with col2:
                    if st.button(f"Generate Easier Question for {cleaned_q[:10]}..."):
                        with st.spinner("Generating easier questions..."):
                            gen_questions = generate_custom_questions(cleaned_q, adjusted_difficulty, "easier")
                        st.markdown("**Easier Questions:**")
                        for gq in gen_questions:
                            st.markdown(f"- {gq}")

                with col3:
                    if st.button(f"Generate Harder Question for {cleaned_q[:10]}..."):
                        with st.spinner("Generating harder questions..."):
                            gen_questions = generate_custom_questions(cleaned_q, adjusted_difficulty, "harder")
                        st.markdown("**Harder Questions:**")
                        for gq in gen_questions:
                            st.markdown(f"- {gq}")
            except Exception as e:
                st.warning(f"❌ Error processing question: {e}")
    
    # Generate feature heatmap
    st.subheader("📊 Feature Heatmap of Questions")
    if st.button("Generate Feature Heatmap"):
        generate_feature_heatmap(df.iloc[:, 0].dropna())

# Run app
if __name__ == "__main__":
    main()
