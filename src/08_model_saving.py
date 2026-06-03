# ============================================================
# Day 5: Model Saving and Inference Pipeline
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("Data/cleaned_reviews.csv")

X = df["clean_review"]
y = df["sentiment"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)

# ============================================================
# FINAL MODEL
# ============================================================

model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000,
    C=1
)

model.fit(
    X_train_tfidf,
    y_train
)

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "Models/sentiment_model.pkl"
)

print("Model Saved Successfully")

# ============================================================
# SAVE VECTORIZER
# ============================================================

joblib.dump(
    vectorizer,
    "Models/tfidf_vectorizer.pkl"
)

print("Vectorizer Saved Successfully")

# ============================================================
# LOAD SAVED OBJECTS
# ============================================================

loaded_model = joblib.load(
    "Models/sentiment_model.pkl"
)

loaded_vectorizer = joblib.load(
    "Models/tfidf_vectorizer.pkl"
)

# ============================================================
# TEST INFERENCE
# ============================================================

sample_reviews = [

    "This product is amazing and works perfectly",

    "Terrible quality, waste of money",

    "I love this device and would recommend it",

    "Very disappointed with this purchase"
]

sample_tfidf = loaded_vectorizer.transform(
    sample_reviews
)

predictions = loaded_model.predict(
    sample_tfidf
)

print("\n")
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for review, prediction in zip(
    sample_reviews,
    predictions
):

    print(f"\nReview: {review}")
    print(f"Predicted Sentiment: {prediction}")