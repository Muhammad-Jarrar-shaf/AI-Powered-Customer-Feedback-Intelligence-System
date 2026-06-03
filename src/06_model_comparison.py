# ============================================================
# Day 3: Model Comparison and Selection
# AI-Powered Customer Feedback Intelligence System
# ============================================================

# Import required libraries
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# STEP 1: Load Dataset
# ============================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("Data/cleaned_reviews.csv")

print(f"Dataset Shape: {df.shape}")

# ============================================================
# STEP 2: Define Features and Target
# ============================================================

X = df["clean_review"]
y = df["sentiment"]

# ============================================================
# STEP 3: Train-Test Split
# ============================================================

# Stratify ensures class distribution remains similar
# in both training and testing sets

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
# STEP 4: TF-IDF Vectorization
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    min_df=2
)

# Fit only on training data to avoid data leakage
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data using learned vocabulary
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Completed")

print(f"Train Shape: {X_train_tfidf.shape}")
print(f"Test Shape : {X_test_tfidf.shape}")

# ============================================================
# STEP 5: Train Logistic Regression
# ============================================================

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)

lr_model.fit(X_train_tfidf, y_train)

lr_pred = lr_model.predict(X_test_tfidf)

# ============================================================
# STEP 6: Train Naive Bayes
# ============================================================

print("Training Multinomial Naive Bayes...")

nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

nb_pred = nb_model.predict(X_test_tfidf)

# ============================================================
# STEP 7: Train Linear SVM
# ============================================================

print("Training Linear SVM...")

svm_model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

svm_model.fit(X_train_tfidf, y_train)

svm_pred = svm_model.predict(X_test_tfidf)

# ============================================================
# STEP 8: Evaluation Function
# ============================================================

def evaluate_model(model_name, y_true, y_pred):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="Negative"
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="Negative"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="Negative"
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report")
    print(classification_report(y_true, y_pred))

    return [
        model_name,
        accuracy,
        precision,
        recall,
        f1
    ]

# ============================================================
# STEP 9: Evaluate All Models
# ============================================================

results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        lr_pred
    )
)

results.append(
    evaluate_model(
        "Multinomial Naive Bayes",
        y_test,
        nb_pred
    )
)

results.append(
    evaluate_model(
        "Linear SVM",
        y_test,
        svm_pred
    )
)

# ============================================================
# STEP 10: Create Comparison Table
# ============================================================

comparison_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison_df)

# ============================================================
# STEP 11: Cross Validation
# ============================================================

print("\n")
print("=" * 60)
print("CROSS VALIDATION")
print("=" * 60)

# Logistic Regression
lr_cv = cross_val_score(
    lr_model,
    X_train_tfidf,
    y_train,
    cv=5,
    scoring="f1_macro"
)

# Naive Bayes
nb_cv = cross_val_score(
    nb_model,
    X_train_tfidf,
    y_train,
    cv=5,
    scoring="f1_macro"
)

# Linear SVM
svm_cv = cross_val_score(
    svm_model,
    X_train_tfidf,
    y_train,
    cv=5,
    scoring="f1_macro"
)

print(f"Logistic Regression CV Mean : {lr_cv.mean():.4f}")
print(f"Naive Bayes CV Mean         : {nb_cv.mean():.4f}")
print(f"Linear SVM CV Mean          : {svm_cv.mean():.4f}")

# ============================================================
# STEP 12: Final Model Selection
# ============================================================

print("\n")
print("=" * 60)
print("PROJECT NOTE")
print("=" * 60)

print(
    "Select the final model based primarily on "
    "Negative-class Recall and F1 Score rather "
    "than Accuracy because the dataset is highly imbalanced."
)