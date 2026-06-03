"""
===========================================================
DAY 6 - MODEL INTERPRETABILITY & ADVANCED ERROR ANALYSIS
===========================================================

Objective:
----------
1. Understand what the Logistic Regression model learned.
2. Identify the most important positive words.
3. Identify the most important negative words.
4. Explain individual predictions.
5. Investigate model mistakes.
6. Generate visualizations for presentation/interviews.

Author: Your Name
Project: AI-Powered Customer Feedback Intelligence System
===========================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# CREATE RESULTS DIRECTORY IF NOT EXISTS
# ==========================================================

os.makedirs("Results", exist_ok=True)

# ==========================================================
# LOAD SAVED MODEL & TF-IDF VECTORIZER
# ==========================================================

print("=" * 60)
print("LOADING MODEL AND VECTORIZER")
print("=" * 60)

model = joblib.load("Models/sentiment_model.pkl")
vectorizer = joblib.load("Models/tfidf_vectorizer.pkl")

print("✓ Logistic Regression model loaded")
print("✓ TF-IDF vectorizer loaded")

# ==========================================================
# EXTRACT FEATURE NAMES
# ==========================================================

print("\n" + "=" * 60)
print("EXTRACTING FEATURES")
print("=" * 60)

feature_names = vectorizer.get_feature_names_out()

print(f"Total Features: {len(feature_names)}")

# ==========================================================
# EXTRACT MODEL COEFFICIENTS
# ==========================================================

print("\n" + "=" * 60)
print("EXTRACTING MODEL COEFFICIENTS")
print("=" * 60)

coefficients = model.coef_[0]

print(f"Total Coefficients: {len(coefficients)}")

# ==========================================================
# CREATE FEATURE IMPORTANCE DATAFRAME
# ==========================================================

feature_importance = pd.DataFrame({
    "Word": feature_names,
    "Coefficient": coefficients
})

# ==========================================================
# TOP POSITIVE WORDS
# ==========================================================

print("\n" + "=" * 60)
print("TOP POSITIVE WORDS")
print("=" * 60)

top_positive_words = feature_importance.sort_values(
    by="Coefficient",
    ascending=False
).head(20)

print(top_positive_words)

# ==========================================================
# TOP NEGATIVE WORDS
# ==========================================================

print("\n" + "=" * 60)
print("TOP NEGATIVE WORDS")
print("=" * 60)

top_negative_words = feature_importance.sort_values(
    by="Coefficient",
    ascending=True
).head(20)

print(top_negative_words)

# ==========================================================
# SAVE RESULTS TO CSV
# ==========================================================

top_positive_words.to_csv(
    "Results/top_positive_words.csv",
    index=False
)

top_negative_words.to_csv(
    "Results/top_negative_words.csv",
    index=False
)

print("\n✓ CSV files saved successfully")

# ==========================================================
# VISUALIZATION - TOP POSITIVE WORDS
# ==========================================================

plt.figure(figsize=(10, 6))

plt.barh(
    top_positive_words["Word"][::-1],
    top_positive_words["Coefficient"][::-1]
)

plt.title("Top Positive Words")
plt.xlabel("Coefficient Strength")
plt.ylabel("Words")

plt.tight_layout()

plt.savefig(
    "Results/top_positive_words.png",
    bbox_inches="tight"
)

plt.show()

print("✓ Positive words plot saved")

# ==========================================================
# VISUALIZATION - TOP NEGATIVE WORDS
# ==========================================================

plt.figure(figsize=(10, 6))

plt.barh(
    top_negative_words["Word"],
    top_negative_words["Coefficient"]
)

plt.title("Top Negative Words")
plt.xlabel("Coefficient Strength")
plt.ylabel("Words")

plt.tight_layout()

plt.savefig(
    "Results/top_negative_words.png",
    bbox_inches="tight"
)

plt.show()

print("✓ Negative words plot saved")

# ==========================================================
# INDIVIDUAL PREDICTION EXPLANATION FUNCTION
# ==========================================================

def explain_prediction(review_text):
    """
    Explains why the model predicted a sentiment.

    Parameters:
    ----------
    review_text : str
        Input review text

    Returns:
    -------
    Prints:
        Prediction
        Probabilities
        Important contributing words
    """

    print("\n" + "=" * 60)
    print("REVIEW ANALYSIS")
    print("=" * 60)

    print(f"\nReview:")
    print(review_text)

    # Transform review using TF-IDF vectorizer
    review_vector = vectorizer.transform([review_text])

    # Prediction
    prediction = model.predict(review_vector)[0]

    # Probability scores
    probabilities = model.predict_proba(review_vector)[0]

    print(f"\nPredicted Sentiment: {prediction}")

    print("\nPrediction Probabilities:")

    print(
        f"Negative: {probabilities[0] * 100:.2f}%"
    )

    print(
        f"Positive: {probabilities[1] * 100:.2f}%"
    )

    print("\nContributing Words:")
    print("-" * 40)

    # Find active words in review
    active_indices = review_vector.nonzero()[1]

    if len(active_indices) == 0:
        print("No vocabulary words found.")
        return

    contributions = []

    for idx in active_indices:

        word = feature_names[idx]

        coef = coefficients[idx]

        tfidf_score = review_vector[0, idx]

        contribution = coef * tfidf_score

        contributions.append(
            [word, coef, tfidf_score, contribution]
        )

    contributions_df = pd.DataFrame(
        contributions,
        columns=[
            "Word",
            "Coefficient",
            "TFIDF",
            "Contribution"
        ]
    )

    contributions_df = contributions_df.sort_values(
        by="Contribution",
        ascending=False
    )

    print(contributions_df)

# ==========================================================
# TEST REVIEW 1
# ==========================================================

test_review_1 = (
    "This product is amazing and works perfectly"
)

explain_prediction(test_review_1)

# ==========================================================
# TEST REVIEW 2
# ==========================================================

test_review_2 = (
    "Terrible quality and complete waste of money"
)

explain_prediction(test_review_2)

# ==========================================================
# TEST REVIEW 3
# ==========================================================

test_review_3 = (
    "Very disappointed with this purchase"
)

explain_prediction(test_review_3)

# ==========================================================
# CHECK IF 'DISAPPOINTED' EXISTS IN VOCABULARY
# ==========================================================

print("\n" + "=" * 60)
print("VOCABULARY INVESTIGATION")
print("=" * 60)

target_word = "disappointed"

if target_word in feature_names:
    print(f"'{target_word}' EXISTS in vocabulary")
else:
    print(f"'{target_word}' DOES NOT EXIST in vocabulary")

# ==========================================================
# FIND SIMILAR NEGATIVE WORDS
# ==========================================================

print("\nPossible Related Words:")

similar_words = [
    word
    for word in feature_names
    if "disappoint" in word
]

if similar_words:
    print(similar_words)
else:
    print("No related words found")

# ==========================================================
# GLOBAL MODEL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("MODEL INTERPRETABILITY SUMMARY")
print("=" * 60)

print(f"Total Features Learned: {len(feature_names)}")

print(
    f"Most Positive Word: "
    f"{top_positive_words.iloc[0]['Word']}"
)

print(
    f"Most Negative Word: "
    f"{top_negative_words.iloc[0]['Word']}"
)

print("\nFiles Generated:")

print("Results/top_positive_words.csv")
print("Results/top_negative_words.csv")
print("Results/top_positive_words.png")
print("Results/top_negative_words.png")

print("\n✓ Day 6 Completed Successfully")


print("\nTop Positive Features")
print(top_positive_words.to_string(index=False))

print("\nTop Negative Features")
print(top_negative_words.to_string(index=False))


feature_importance.to_csv(
    "Results/all_feature_importance.csv",
    index=False
)