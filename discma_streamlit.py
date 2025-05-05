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

# ---- Custom Feature Extractor ----
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

# ---- Explanation Generation ----
def generate_explanation(question_text, difficulty_score, features, model="gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    prompt = (
        f"Question: \"{question_text}\"\n"
        f"Predicted Difficulty: {difficulty_score:.2f}\n"
        f"Features: {features}\n"
        f"Explain why this question is considered to have this difficulty level.\n"
        f"Base your explanation on the feature values."
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
        st.error(f"OpenAI API error (explanation): {e}")
        return "Explanation could not be generated."

# ---- Question Generation ----
def generate_custom_questions(base_question, difficulty, difficulty_type="similar", num_questions=3, model="gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    adjustment = {
        "similar": f"with a difficulty level of approximately {difficulty:.2f}",
        "easier": f"that are slightly easier than difficulty {difficulty:.2f}",
        "harder": f"that are slightly harder than difficulty {difficulty:.2f}"
    }

    explanation_note = {
        "similar": "These are similar in structure and complexity.",
        "easier": "These are simpler in structure, shorter, or more straightforward.",
        "harder": "These are more complex, longer, or require deeper analysis."
    }

    prompt = (
        f"Generate {num_questions} discrete math questions based on the following, {adjustment[difficulty_type]}:\n"
        f"Question: {base_question}\n\n"
        f"Questions with brief notes explaining why each is {difficulty_type}:\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=700,
        )
        return explanation_note[difficulty_type], response.choices[0].message.content.strip().split("\n")
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return explanation_note[difficulty_type], []

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
        prediction, features = predict_difficulty(model, scaler, question_text)
        st.markdown(f"**Predicted Difficulty:** <span style='color:blue; font-weight:bold;'>{prediction:.2f}</span>", unsafe_allow_html=True)

        st.subheader("📌 Extracted Features")
        feature_df = pd.DataFrame([features])
        st.table(feature_df)

        if features['has_sequence_word']:
            explanation = generate_explanation(question_text, prediction, features)
            st.subheader("🧠 Explanation")
            st.write(explanation)

            generate_feature_heatmap([question_text])

            st.subheader("🤖 Generate Questions Based on This")
            for diff_type in ["similar", "easier", "harder"]:
                st.markdown(f"### {diff_type.capitalize()} Questions")
                if st.button(f"Generate {diff_type} questions"):
                    with st.spinner(f"Generating {diff_type} questions..."):
                        note, generated_questions = generate_custom_questions(
                            base_question=question_text,
                            difficulty=prediction,
                            difficulty_type=diff_type,
                            num_questions=3
                        )
                    if generated_questions:
                        st.success(note)
                        for q in generated_questions:
                            st.markdown(f"- {q}")
        else:
            st.warning("⚠️ This question appears to be outside the scope of sequences and summations, so no explanation or generation will be provided.")

    st.divider()

    st.subheader("📁 Or Upload a CSV of Questions")
    uploaded_file = st.file_uploader("Upload Question Data (CSV)", type=["csv"])

    if uploaded_file is not None:
        df_questions = pd.read_csv(uploaded_file)
        st.write("Uploaded Questions Preview:")
        st.write(df_questions.head())

        detailed_results = []

        for idx, q_text in enumerate(df_questions.iloc[:, 0]):
            if not isinstance(q_text, str) or not q_text.strip():
                continue

            st.markdown(f"---\n### Question {idx + 1}:")
            st.markdown(f"**Text:** {q_text}")

            prediction, features = predict_difficulty(model, scaler, q_text)
            st.markdown(f"**Predicted Difficulty:** <span style='color:blue; font-weight:bold;'>{prediction:.2f}</span>", unsafe_allow_html=True)

            feature_df = pd.DataFrame([features])
            st.markdown("**Extracted Features:**")
            st.table(feature_df)

            if features['has_sequence_word']:
                explanation = generate_explanation(q_text, prediction, features)
                st.markdown("**Explanation:**")
                st.write(explanation)

                with st.expander("🤖 Generate Questions Based on This"):
                    for diff_type in ["similar", "easier", "harder"]:
                        if st.button(f"Generate {diff_type} questions for Q{idx + 1}", key=f"{diff_type}_{idx}"):
                            with st.spinner(f"Generating {diff_type} questions..."):
                                note, generated_questions = generate_custom_questions(
                                    base_question=q_text,
                                    difficulty=prediction,
                                    difficulty_type=diff_type,
                                    num_questions=3
                                )
                            if generated_questions:
                                st.success(note)
                                for q in generated_questions:
                                    st.markdown(f"- {q}")
            else:
                st.warning("⚠️ This question appears to be outside the scope of sequences and summations, so no explanation or generation will be provided.")

            detailed_results.append({
                "Question": q_text,
                "Predicted Difficulty": prediction,
                **features
            })

        # Show heatmap if any results
        if detailed_results:
            question_texts = [r["Question"] for r in detailed_results]
            generate_feature_heatmap(question_texts)

# ---- Run the App ----
if __name__ == "__main__":
    main()
