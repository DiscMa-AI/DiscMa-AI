import streamlit as st
import pandas as pd
from difficulty_model import predict_difficulty, generate_custom_questions, extract_features_from_text

st.set_page_config(page_title="Question Difficulty Analyzer", layout="wide")
st.title("📊 Question Difficulty Analyzer")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Manual Input", "CSV Upload"])

# ---------- Manual Input Section ----------
if page == "Manual Input":
    st.header("📝 Analyze a Question")
    question_text = st.text_area("Enter your question:")

    if st.button("Predict Difficulty"):
        if question_text.strip():
            with st.spinner("Analyzing question..."):
                features = extract_features_from_text(question_text)
                difficulty = predict_difficulty(features)

            st.success(f"Predicted Difficulty: {difficulty:.2f}%")

            for diff_type in ["similar", "easier", "harder"]:
                if st.button(f"Generate {diff_type.capitalize()} Questions"):
                    with st.spinner("Generating questions..."):
                        questions = generate_custom_questions(question_text, difficulty, difficulty_type=diff_type)
                    st.success(f"{diff_type.capitalize()} Questions:")
                    for q in questions:
                        st.markdown(f"- {q}")
        else:
            st.warning("Please enter a question to analyze.")

# ---------- CSV Upload Section ----------
else:
    st.header("📁 Batch Analyze from CSV")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'Question' column:", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "Question" not in df.columns:
                st.error("CSV must contain a 'Question' column.")
            else:
                st.success("CSV successfully loaded!")

                results = []
                for q in df["Question"]:
                    if isinstance(q, str) and q.strip():
                        try:
                            features = extract_features_from_text(q)
                            difficulty = predict_difficulty(features)
                            results.append({"Question": q, "Adjusted Difficulty": round(difficulty, 2)})
                        except Exception as e:
                            results.append({"Question": q, "Adjusted Difficulty": "Error"})
                    else:
                        results.append({"Question": q, "Adjusted Difficulty": "Invalid question"})

                result_df = pd.DataFrame(results)
                st.dataframe(result_df)

                # -- Select question from processed rows for generation --
                st.subheader("📑 Select a Question for Generation")
                valid_questions = [row["Question"] for row in results if isinstance(row["Adjusted Difficulty"], float)]

                if valid_questions:
                    selected_q = st.selectbox("Choose a question to generate related items:", valid_questions)
                    selected_row = next(r for r in results if r["Question"] == selected_q)

                    st.markdown(f"**Selected Question:** {selected_q}")
                    st.markdown(f"**Adjusted Difficulty:** `{selected_row['Adjusted Difficulty']}`")

                    for diff_type in ["similar", "easier", "harder"]:
                        if st.button(f"Generate {diff_type.capitalize()} Questions for Selected"):
                            with st.spinner("Generating..."):
                                questions = generate_custom_questions(
                                    selected_q,
                                    selected_row["Adjusted Difficulty"],
                                    difficulty_type=diff_type
                                )
                            st.success(f"{diff_type.capitalize()} Questions:")
                            for q in questions:
                                st.markdown(f"- {q}")
        except Exception as e:
            st.error(f"An error occurred while processing the CSV: {e}")
