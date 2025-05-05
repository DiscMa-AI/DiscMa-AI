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
    return prediction[0]

def generate_similar_questions(base_question, difficulty, num_questions=3, model="gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    prompt = (
        f"Generate {num_questions} discrete math questions similar to the following, "
        f"with a difficulty level of approximately {difficulty:.2f}:\n\n"
        f"Question: {base_question}\n\n"
        "Similar Questions:"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip().split("\n")
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return []

def generate_feedback_from_gpt(question_text, difficulty_score, model="gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    if difficulty_score < 4:
        level = "too easy"
    elif difficulty_score > 7:
        level = "too difficult"
    else:
        level = "moderately difficult"

    prompt = (
        f"A discrete math question was submitted:\n"
        f"\"{question_text}\"\n\n"
        f"The predicted difficulty is {difficulty_score:.2f}, which is considered {level}.\n"
        f"Give a short and constructive suggestion for how to improve or adjust the question accordingly. "
        f"The suggestion should be clear, helpful, and ideally actionable."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error (feedback): {e}")
        return None

# ---- Feature Heatmap ----
def generate_feature_heatmap(questions):
    feature_data = []
    question_labels = []

    for idx, q in enumerate(questions):
        features = extract_features(q)
        feature_data.append(features)
        label = f"Q{idx+1}" if len(q) > 30 else q
        question_labels.append(label[:30] + '...' if len(label) > 30 else label)

    feature_df = pd.DataFrame(feature_data, index=question_labels)

    st.subheader("🔍 Feature Heatmap of Questions")
    fig, ax = plt.subplots(figsize=(10, max(2, len(questions)*0.5)))
    sns.heatmap(feature_df, annot=True, cmap="viridis", fmt="d", cbar_kws={'label': 'Value'}, ax=ax)
    ax.set_xlabel("Features")
    ax.set_ylabel("Question")
    st.pyplot(fig)

# ---- Streamlit App ----
def main():
    st.title('📊 Discrete Math Question Difficulty Predictor')

    model, scaler = load_model_and_scaler()

    st.subheader("🔤 Enter a Question to Predict Difficulty")
    question_text = st.text_area("Enter your question here:")

    if question_text:
        prediction = predict_difficulty(model, scaler, question_text)
        st.write(f"Predicted Difficulty: **{prediction:.2f}**", unsafe_allow_html=True)

        features = extract_features(question_text)
        feature_df = pd.DataFrame([features])
        st.write("### Extracted Features:")
        st.write(feature_df)

        st.subheader("🔍 Explanation of Predicted Difficulty")
        feedback = generate_feedback_from_gpt(question_text, prediction)
        if feedback:
            st.info(f"💡Suggestion:\n\n{feedback}")

        generate_feature_heatmap([question_text])

        st.subheader("🤖 Generate Similar, Easier, or Harder Questions")
        option = st.radio("Select the type of questions to generate", ("Similar", "Easier", "Harder"))

        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                if option == "Similar":
                    generated_questions = generate_similar_questions(
                        base_question=question_text,
                        difficulty=prediction,
                        num_questions=3,
                    )
                    st.success("Similar questions generated:")
                elif option == "Easier":
                    generated_questions = generate_similar_questions(
                        base_question=question_text,
                        difficulty=prediction - 1,
                        num_questions=3,
                    )
                    st.success("Easier questions generated:")
                elif option == "Harder":
                    generated_questions = generate_similar_questions(
                        base_question=question_text,
                        difficulty=prediction + 1,
                        num_questions=3,
                    )
                    st.success("Harder questions generated:")

                if generated_questions:
                    for q in generated_questions:
                        st.markdown(f"- {q}")

    st.divider()

    st.subheader("📁 Or Upload a CSV of Questions")
    uploaded_file = st.file_uploader("Upload Question Data (CSV)", type=["csv"])

    if uploaded_file is not None:
        df_questions = pd.read_csv(uploaded_file)
        st.write("Uploaded Questions Preview:")
        st.write(df_questions.head())

        generate_button = st.button("Generate Difficulty and Explanations for All Questions")

        if generate_button:
            predictions = []
            explanations = []

            for q_text in df_questions.iloc[:, 0]:
                prediction = predict_difficulty(model, scaler, q_text)
                explanation = generate_feedback_from_gpt(q_text, prediction)
                predictions.append(prediction)
                explanations.append(explanation)

            df_questions['Predicted Difficulty'] = predictions
            df_questions['Explanation'] = explanations
            st.write("📊 Questions with Predicted Difficulty and Explanations:")
            st.write(df_questions)

            generate_feature_heatmap(df_questions.iloc[:, 0].tolist())

# ---- Run the App ----
if __name__ == "__main__":
    main()
