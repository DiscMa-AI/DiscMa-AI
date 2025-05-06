import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import re
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import textstat  # Required for readability metric
import json

@st.cache_resource
def load_model_and_scaler():
    model = lgb.Booster(model_file='model/discma1.txt')
    scaler = joblib.load('model/scaler1.pkl')
    return model, scaler

# Extract features
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

# Predict difficulty
def predict_difficulty(model, scaler, question_text):
    features = extract_features(question_text)
    X_new = pd.DataFrame([features])
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)
    return prediction[0], features

# Generate explanation based on feature insights
# Generate explanation based on feature insights
def generate_explanation_with_features(question_text, difficulty_score, features, model="gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Extract the feature values
    length = features["length"]
    word_count = features["word_count"]
    avg_word_length = features["avg_word_length"]
    num_numbers = features["num_numbers"]
    num_math_symbols = features["num_math_symbols"]
    num_variables = features["num_variables"]
    readability = features["readability"]
    num_keywords = features["num_keywords"]
    
    # Build a prompt that will guide GPT-3 to explain why the question has this level of difficulty.
    prompt = (
        f"Question: \"{question_text}\"\n"
        f"Predicted Difficulty: {difficulty_score:.2f}\n"
        f"Features:\n"
        f"Length: {length} characters\n"
        f"Word Count: {word_count} words\n"
        f"Average Word Length: {avg_word_length:.2f} characters\n"
        f"Number of Numbers: {num_numbers}\n"
        f"Number of Math Symbols: {num_math_symbols}\n"
        f"Number of Variables: {num_variables}\n"
        f"Readability (Flesch Score): {readability:.2f}\n"
        f"Number of Keywords: {num_keywords}\n\n"
        f"Given these features, explain in detail why this question is considered to have a difficulty level of {difficulty_score:.2f}. "
        f"Consider how the length of the question, word count, presence of mathematical symbols and variables, readability, and use of keywords impact the overall difficulty. "
        f"For example, discuss how complex the mathematical content is (e.g., the number of symbols and variables), the clarity or difficulty of the language (based on readability), "
        f"and how the length or word count might affect how challenging the question is."
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
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
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f", ax=ax)
    st.pyplot(fig)

# Main app
def main():
    st.title("📊 Discrete Math Question Difficulty Predictor")
    model, scaler = load_model_and_scaler()

    st.subheader("🔤 Enter a Question")
    question_text = st.text_area("Enter your question:")

    if question_text:
        prediction, features = predict_difficulty(model, scaler, question_text)
        st.markdown(f"**Predicted Difficulty:** {prediction:.2f}")
        st.subheader("📌 Features")
        st.table(pd.DataFrame([features]))

        # Generate explanation with feature insights
        explanation = generate_explanation_with_features(question_text, prediction, features)
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

    st.divider()
    st.subheader("📁 Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV of questions", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        # Iterate through each question in the CSV and provide interactive elements
        for idx, row in df.iterrows():
            question_text = row[0]
            st.subheader(f"Question {idx+1}: {question_text}")

            prediction, features = predict_difficulty(model, scaler, question_text)
            st.markdown(f"**Predicted Difficulty:** {prediction:.2f}")
            st.subheader("📌 Features")
            st.table(pd.DataFrame([features]))

            # Generate explanation with feature insights
            explanation = generate_explanation_with_features(question_text, prediction, features)
            st.subheader("🧠 Explanation")
            st.write(explanation)

            generate_feature_heatmap([question_text])

            st.subheader("🤖 Generate Related Questions")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"Generate Similar Questions for Question {idx+1}"):
                    with st.spinner("Generating..."):
                        similar_qs = generate_custom_questions(question_text, prediction, "similar")
                    st.success(f"Similar Questions:")
                    for q in similar_qs:
                        st.markdown(f"- {q}")
            with col2:
                if st.button(f"Generate Easier Questions for Question {idx+1}"):
                    with st.spinner("Generating..."):
                        easier_qs = generate_custom_questions(question_text, prediction, "easier")
                    st.success(f"Easier Questions:")
                    for q in easier_qs:
                        st.markdown(f"- {q}")
            with col3:
                if st.button(f"Generate Harder Questions for Question {idx+1}"):
                    with st.spinner("Generating..."):
                        harder_qs = generate_custom_questions(question_text, prediction, "harder")
                    st.success(f"Harder Questions:")
                    for q in harder_qs:
                        st.markdown(f"- {q}")

        st.divider()
        # Export processed data with explanations and related questions
        results = []
        for idx, row in df.iterrows():
            question_text = row[0]
            pred, features = predict_difficulty(model, scaler, question_text)
            explanation = generate_explanation_with_features(question_text, pred, features)

            similar_qs = generate_custom_questions(question_text, pred, "similar")
            easier_qs = generate_custom_questions(question_text, pred, "easier")
            harder_qs = generate_custom_questions(question_text, pred, "harder")

            results.append({
                "Question": question_text,
                "Predicted Difficulty": round(pred, 2),
                "Features": json.dumps(features),
                "Explanation": explanation,
                "Similar Questions": " | ".join(similar_qs),
                "Easier Questions": " | ".join(easier_qs),
                "Harder Questions": " | ".join(harder_qs),
            })

        processed_df = pd.DataFrame(results)
        st.write("📋 Processed Data with Features and Explanations")
        st.dataframe(processed_df)
        st.download_button("📥 Download Processed CSV", processed_df.to_csv(index=False), "processed_questions.csv")

if __name__ == '__main__':
    main()
