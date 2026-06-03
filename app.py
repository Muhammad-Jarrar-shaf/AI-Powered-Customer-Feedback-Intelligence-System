"""
==================================================
AI-Powered Customer Feedback Intelligence System
Streamlit Application
==================================================
"""

import streamlit as st
import joblib

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        "Models/sentiment_model.pkl"
    )

    vectorizer = joblib.load(
        "Models/tfidf_vectorizer.pkl"
    )

    return model, vectorizer


model, vectorizer = load_artifacts()

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

# ==========================================
# HEADER
# ==========================================

st.title(
    "📊 AI-Powered Customer Feedback Intelligence System"
)

st.markdown(
    """
Analyze customer reviews using a
Machine Learning sentiment classifier.

Model:
- TF-IDF Vectorization
- Logistic Regression
"""
)

# ==========================================
# INPUT BOX
# ==========================================

review = st.text_area(
    "Enter Customer Review",
    height=150
)

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Analyze Sentiment"):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        review_vector = (
            vectorizer.transform([review])
        )

        prediction = (
            model.predict(review_vector)[0]
        )

        probabilities = (
            model.predict_proba(review_vector)[0]
        )

        negative_prob = (
            probabilities[0] * 100
        )

        positive_prob = (
            probabilities[1] * 100
        )

        st.subheader("Prediction")

        if prediction == "Positive":

            st.success(
                f"✅ Sentiment: {prediction}"
            )

        else:

            st.error(
                f"❌ Sentiment: {prediction}"
            )

        st.subheader("Confidence Scores")

        st.write(
            f"Positive: {positive_prob:.2f}%"
        )

        st.write(
            f"Negative: {negative_prob:.2f}%"
        )

        st.progress(
            int(max(
                positive_prob,
                negative_prob
            ))
        )