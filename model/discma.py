import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re
import textstat
import streamlit as st




# ---- Load Data ----
def load_data(question_path, rate_path):
    df_questions = pd.read_csv(question_path)
    df_rates = pd.read_csv(rate_path)

    # Clean column names
    df_questions.columns = df_questions.columns.str.strip()
    df_rates.columns = df_rates.columns.str.strip()

    return df_questions, df_rates

# ---- Custom Feature Extractor ----
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
    features["num_variables"] = len(re.findall(r'[a-zA-Z]', text))
    features["readability"] = textstat.flesch_reading_ease(text)
    features["num_keywords"] = sum(1 for word in words if word.lower() in keywords)

    return features

# ---- Feature Preparation ----
def prepare_features_and_target(df_questions, df_rates):
    question_columns = [col for col in df_questions.columns if col.startswith('Question')]
    feature_list = []
    y = []
    skipped = 0
    total = 0

    for q_col, r_col in zip(question_columns, df_rates.columns):
        for q_text, rate in zip(df_questions[q_col], df_rates[r_col]):
            total += 1
            if pd.notna(q_text) and pd.notna(rate):
                try:
                    rate = float(rate)
                    features = extract_features(q_text)
                    feature_list.append(features)
                    y.append(rate)
                except ValueError:
                    skipped += 1
                    print(f"Skipping rate value: {rate} - could not convert to float.")
            else:
                skipped += 1

    print(f"Processed {total - skipped} valid question-rate pairs. Skipped {skipped} due to missing or invalid data.")

    X = pd.DataFrame(feature_list)
    y = np.array(y)
    return X, y

# ---- Train Model with Cross-Validation ----
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = lgb.LGBMRegressor()
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
    print(f"Average RMSE (CV): {-np.mean(scores):.3f}")

    # Final training
    model.fit(X_scaled, y)
    return model, scaler

# ---- Evaluate Model ----
def evaluate_model(model, scaler, X, y):
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(f"Final RMSE: {rmse:.3f}")
    st.write(f"Final R² Score: {r2:.3f}")

    # Plot Results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.set_xlabel('Actual Incorrect Rate (%)')
    ax.set_ylabel('Predicted Incorrect Rate (%)')
    ax.set_title('Prediction vs. Actual')
    ax.grid(True)
    st.pyplot(fig)

# ---- Feature Extraction Visualization ----
def visualize_features(df_questions):
    question_columns = [col for col in df_questions.columns if col.lower().startswith('question')]
    feature_data = []
    question_labels = []

    for column in question_columns:
        question_text = df_questions[column].iloc[0]
        features = extract_features(question_text)
        feature_data.append(features)
        question_labels.append(column)

    feature_df = pd.DataFrame(feature_data, index=question_labels)

    plt.figure(figsize=(12, 6))
    sns.heatmap(feature_df, annot=True, cmap="viridis", fmt="g", cbar_kws={'label': 'Feature Value'})
    plt.title("Feature Extraction for All Questions (First Row Only)")
    plt.xlabel("Features")
    plt.ylabel("Question")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

# ---- Streamlit Interface ----
def main():
    st.title('Question Difficulty Prediction Model')

    uploaded_question_file = st.file_uploader("Upload Question Data", type=["csv"])
    uploaded_rate_file = st.file_uploader("Upload Rate Data", type=["csv"])

    if uploaded_question_file is not None and uploaded_rate_file is not None:
        # Load data
        df_questions = pd.read_csv(uploaded_question_file)
        df_rates = pd.read_csv(uploaded_rate_file)

        st.write("Data Preview - Questions")
        st.write(df_questions.head())
        st.write("Data Preview - Rates")
        st.write(df_rates.head())

        # Visualize Features
        st.subheader("Feature Visualization")
        visualize_features(df_questions)

        X, y = prepare_features_and_target(df_questions, df_rates)

        if len(X) == 0 or len(y) == 0:
            st.error("No valid data to train on. Exiting.")
            return

        # Train Model
        model, scaler = train_model(X, y)

        # Evaluate Model
        st.subheader("Model Evaluation")
        evaluate_model(model, scaler, X, y)

# ---- Run Streamlit App ----
if __name__ == "__main__":
    main()