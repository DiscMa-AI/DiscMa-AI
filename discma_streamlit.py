import streamlit as st
import pandas as pd
import re
from model import predict_difficulty, explain_prediction, generate_question_variation

# Title
st.title("📊 Question Difficulty Predictor")
st.markdown("Use this app to predict the difficulty of math questions involving **sequences and summations** only.")

# Instructions
st.markdown("""
### ❗ Important Instructions
- This tool only accepts **math questions** about **sequences** or **summations**.
- Examples of **accepted** topics:
  - What is the 10th term of the sequence 2, 4, 6, ...?
  - What is the sum of the first 100 positive integers?
- **Rejected** topics include:
  - General knowledge (e.g. *What is the capital of France?*)
  - Arithmetic operations, geometry, algebra (unless tied to sequences/summations)

### ✅ CSV Format Sample
| Question                             |
|--------------------------------------|
| What is the 15th term of the sequence 3, 6, 9, ...? |
| Find the sum of the first 20 even numbers.         |

Ensure your file is a `.csv` with a **Question** column containing only valid questions.
""")

# Utility function
def is_valid_math_question(question: str) -> bool:
    return bool(re.search(r'\b(sequence|term|summation|sum of|nth term)\b', question, re.IGNORECASE))

# File uploader
st.header("📂 Upload CSV of Questions")
uploaded_file = st.file_uploader("Upload your .csv file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Question' not in df.columns:
        st.error("The CSV file must contain a 'Question' column.")
    else:
        valid_questions = df[df['Question'].apply(is_valid_math_question)]
        invalid_questions = df[~df['Question'].apply(is_valid_math_question)]

        if not valid_questions.empty:
            st.success(f"{len(valid_questions)} valid question(s) loaded.")
            for i, row in valid_questions.iterrows():
                question = row['Question']
                difficulty = predict_difficulty(question)
                explanation = explain_prediction(question)
                st.markdown(f"**Q:** {question}")
                st.markdown(f"**Predicted Difficulty:** {difficulty:.2f}%")
                st.markdown(f"**Explanation:** {explanation}")
                with st.expander("Generate Similar, Easier, or Harder Versions"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"Similar #{i}"):
                            st.markdown(generate_question_variation(question, "similar"))
                    with col2:
                        if st.button(f"Easier #{i}"):
                            st.markdown(generate_question_variation(question, "easier"))
                    with col3:
                        if st.button(f"Harder #{i}"):
                            st.markdown(generate_question_variation(question, "harder"))
        if not invalid_questions.empty:
            st.warning(f"{len(invalid_questions)} question(s) were skipped because they are outside the supported scope.")
            st.dataframe(invalid_questions)

# Manual entry
st.header("✍️ Manual Question Input")
manual_question = st.text_area("Enter a math question involving sequences or summations:")

if st.button("Predict Difficulty"):
    if not is_valid_math_question(manual_question):
        st.error("❌ This question is out of scope. Please input only questions about sequences or summations.")
    else:
        difficulty = predict_difficulty(manual_question)
        explanation = explain_prediction(manual_question)
        st.markdown(f"**Predicted Difficulty:** {difficulty:.2f}%")
        st.markdown(f"**Explanation:** {explanation}")
        with st.expander("Generate Similar, Easier, or Harder Versions"):
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Similar"):
                    st.markdown(generate_question_variation(manual_question, "similar"))
            with col2:
                if st.button("Easier"):
                    st.markdown(generate_question_variation(manual_question, "easier"))
            with col3:
                if st.button("Harder"):
                    st.markdown(generate_question_variation(manual_question, "harder"))
