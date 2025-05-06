# Enhanced Streamlit App with Difficulty Adjustment Based on Features
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

# Feature Weights for Difficulty Adjustment
FEATURE_WEIGHTS = {
    "length": 0.2,
    "word_count": 0.1,
    "avg_word_length": 0.05,
    "num_numbers": 0.1,
    "num_math_symbols": 0.25,
    "num_variables": 0.15,
    "readability": -0.2,  # negative because higher readability should lower difficulty
    "num_keywords": 0.05,
}

# Calculate final adjusted difficulty
def calculate_adjusted_difficulty(model, scaler, question_text):
    # Get model's initial prediction and features
    features = extract_features(question_text)
    X_new = pd.DataFrame([features])
    X_new_scaled = scaler.transform(X_new)
    predicted_difficulty = model.predict(X_new_scaled)[0]
    
    # Normalize features
    normalized_features = {
        "length": min(features["length"] / 100, 1),
        "word_count": min(features["word_count"] / 20, 1),
        "avg_word_length": min(features["avg_word_length"] / 6, 1),
        "num_numbers": min(features["num_numbers"] / 5, 1),
        "num_math_symbols": min(features["num_math_symbols"] / 5, 1),
        "num_variables": min(features["num_variables"] / 5, 1),
        "readability": 1 - (features["readability"] / 100),  # Inverted for readability
        "num_keywords": min(features["num_keywords"] / 5, 1),
    }
    
    # Calculate the adjusted difficulty based on feature impact
    adjusted_difficulty = predicted_difficulty
    for feature, weight in FEATURE_WEIGHTS.items():
        adjusted_difficulty += weight * normalized_features[feature]
    
    return predicted_difficulty, adjusted_difficulty, features

# Generate explanation and adjustment based on features
def generate_explanation_with_feature_impact_adjustment(question_text, adjusted_difficulty, features, model="gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Construct the prompt to analyze feature impact on difficulty
    prompt = (
        f"Question: \"{question_text}\"\n"
        f"Adjusted Difficulty: {adjusted_difficulty:.2f}\n"
        f"Features: {features}\n"
        f"Based on the following feature values (length, word_count, avg_word_length, num_numbers, "
        f"num_math_symbols, num_variables, readability, num_keywords), explain how each feature impacts the difficulty."
        f"Provide a final difficulty score recommendation after considering these impacts."
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        return f"Explanation error: {e}"

# Main app
def main():
    st.title("📊 Discrete Math Question Difficulty Predictor")
    model, scaler = load_model_and_scaler()

    st.subheader("🔤 Enter a Question")
    question_text = st.text_area("Enter your question:")

    if question_text:
        # Calculate the final adjusted difficulty
        _, adjusted_difficulty, features = calculate_adjusted_difficulty(model, scaler, question_text)
        
        st.subheader("📌 Features")
        st.table(pd.DataFrame([features]))
        
        # Generate and show explanation
        explanation = generate_explanation_with_feature_impact_adjustment(question_text, adjusted_difficulty, features)
        st.subheader("🧠 Explanation with Feature Impact")
        st.write(explanation)
        st.markdown(f"**Adjusted Difficulty:** {adjusted_difficulty:.2f}")
        
        # Optionally show heatmap of features
        generate_feature_heatmap([question_text])

        st.subheader("🤖 Generate Questions")
        for diff_type in ["similar", "easier", "harder"]:
            if st.button(f"Generate {diff_type.capitalize()} Questions"):
                with st.spinner("Generating..."):
                    questions = generate_custom_questions(question_text, adjusted_difficulty, diff_type)
                st.success(f"{diff_type.capitalize()} Questions:")
                for q in questions:
                    st.markdown(f"- {q}")

    st.divider()
    st.subheader("📁 Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV of questions", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        predictions = []
        adjusted_difficulties = []
        explanations = []
        
        for q in df.iloc[:, 0]:
            _, adjusted_difficulty, features = calculate_adjusted_difficulty(model, scaler, q)
            explanation = generate_explanation_with_feature_impact_adjustment(q, adjusted_difficulty, features)
            
            predictions.append("N/A")  # No need for original prediction anymore
            adjusted_difficulties.append(adjusted_difficulty)
            explanations.append(explanation)
        
        # Add results to the DataFrame
        df['Adjusted Difficulty'] = adjusted_difficulties
        df['Explanation'] = explanations

        st.write("Processed Data:", df)
        generate_feature_heatmap(df.iloc[:, 0].tolist())
        
        st.download_button("📥 Download Processed CSV", df.to_csv(index=False), "processed_questions.csv")
