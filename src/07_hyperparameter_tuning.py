# ============================================================
# Day 4: Hyperparameter Tuning and Model Optimization
# AI-Powered Customer Feedback Intelligence System
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("Data/cleaned_reviews.csv")

print(f"Dataset Shape: {df.shape}")

# ============================================================
# STEP 2: DEFINE FEATURES AND TARGET
# ============================================================

X = df["clean_review"]
y = df["sentiment"]

# ============================================================
# STEP 3: TRAIN TEST SPLIT
# ============================================================

# Stratified split preserves class distribution

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed")

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")

# ============================================================
# STEP 4: TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    min_df=2
)

# Learn vocabulary from training data only
X_train_tfidf = vectorizer.fit_transform(X_train)

# Apply learned vocabulary to test data
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Completed")

print(f"Train Shape: {X_train_tfidf.shape}")
print(f"Test Shape : {X_test_tfidf.shape}")

# ============================================================
# STEP 5: BASELINE MODEL
# ============================================================

print("\n")
print("=" * 60)
print("BASELINE LOGISTIC REGRESSION")
print("=" * 60)

baseline_model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)

baseline_model.fit(
    X_train_tfidf,
    y_train
)

baseline_pred = baseline_model.predict(
    X_test_tfidf
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

baseline_precision = precision_score(
    y_test,
    baseline_pred,
    pos_label="Negative"
)

baseline_recall = recall_score(
    y_test,
    baseline_pred,
    pos_label="Negative"
)

baseline_f1 = f1_score(
    y_test,
    baseline_pred,
    pos_label="Negative"
)

print(f"Accuracy : {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall   : {baseline_recall:.4f}")
print(f"F1 Score : {baseline_f1:.4f}")

# ============================================================
# STEP 6: HYPERPARAMETER GRID
# ============================================================

print("\n")
print("=" * 60)
print("GRID SEARCH")
print("=" * 60)

# C controls regularization strength
# Smaller C = stronger regularization
# Larger C = weaker regularization

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]
}

# ============================================================
# STEP 7: GRID SEARCH CROSS VALIDATION
# ============================================================

grid_search = GridSearchCV(
    estimator=LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    ),

    param_grid=param_grid,

    cv=5,

    scoring="f1_macro",

    n_jobs=-1
)

grid_search.fit(
    X_train_tfidf,
    y_train
)

# ============================================================
# STEP 8: BEST PARAMETERS
# ============================================================

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Score:")
print(f"{grid_search.best_score_:.4f}")

# ============================================================
# STEP 9: BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_

# Predict using optimized model
best_pred = best_model.predict(
    X_test_tfidf
)

# ============================================================
# STEP 10: EVALUATE TUNED MODEL
# ============================================================

print("\n")
print("=" * 60)
print("TUNED MODEL PERFORMANCE")
print("=" * 60)

tuned_accuracy = accuracy_score(
    y_test,
    best_pred
)

tuned_precision = precision_score(
    y_test,
    best_pred,
    pos_label="Negative"
)

tuned_recall = recall_score(
    y_test,
    best_pred,
    pos_label="Negative"
)

tuned_f1 = f1_score(
    y_test,
    best_pred,
    pos_label="Negative"
)

print(f"Accuracy : {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall   : {tuned_recall:.4f}")
print(f"F1 Score : {tuned_f1:.4f}")

# ============================================================
# STEP 11: CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        best_pred
    )
)

# ============================================================
# STEP 12: CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")

print(
    classification_report(
        y_test,
        best_pred
    )
)

# ============================================================
# STEP 13: BASELINE VS TUNED COMPARISON
# ============================================================

comparison_df = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Baseline Model": [
        baseline_accuracy,
        baseline_precision,
        baseline_recall,
        baseline_f1
    ],

    "Tuned Model": [
        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1
    ]
})

print("\n")
print("=" * 60)
print("BASELINE VS TUNED MODEL")
print("=" * 60)

print(comparison_df)

# ============================================================
# STEP 14: FINAL CONCLUSION
# ============================================================

print("\n")
print("=" * 60)
print("MODEL OPTIMIZATION SUMMARY")
print("=" * 60)

print(
    "Hyperparameter tuning was performed using "
    "GridSearchCV with 5-fold cross validation. "
    "The best model was selected based on "
    "macro F1 score and evaluated on the test set."
)