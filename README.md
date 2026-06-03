# AI-Powered Customer Feedback Intelligence System

## Overview

This project aims to build a machine learning system that automatically classifies product reviews as **Positive** or **Negative** based on their textual content.

The long-term goal is to evolve this project into a customer feedback intelligence platform capable of:

* Sentiment classification
* Customer complaint detection
* Review summarization
* Business insight generation

---

## Problem Statement

Companies receive thousands of customer reviews across products and services. Manually analyzing this feedback is time-consuming and difficult to scale.

This project explores how Natural Language Processing (NLP) and Machine Learning can be used to automatically determine customer sentiment from product reviews.

Example:

| Review                                 | Sentiment |
| -------------------------------------- | --------- |
| "Amazing product. Highly recommended." | Positive  |
| "Stopped working after two days."      | Negative  |

---

## Dataset

Dataset: Amazon Product Reviews

Relevant columns used:

* `reviews.text` → Customer review text
* `reviews.rating` → Numerical rating (1–5 stars)

---

## Day 1 Objectives

The goal of Day 1 was to understand the dataset and prepare it for machine learning.

### 1. Data Inspection

Tasks completed:

* Loaded dataset using Pandas
* Inspected dataset structure
* Analyzed available columns
* Explored rating distribution

### Initial Dataset Size

* Reviews: 1597
* Features: 27

---

### 2. Label Engineering

Converted review ratings into sentiment labels.

#### Mapping Strategy

| Rating | Sentiment |
| ------ | --------- |
| 4–5    | Positive  |
| 1–2    | Negative  |
| 3      | Neutral   |

Because this project focuses on binary classification, neutral reviews were removed.

#### Final Dataset

* Reviews: 1053

---

### 3. Class Distribution Analysis

After removing neutral reviews:

| Sentiment | Count |
| --------- | ----: |
| Positive  |   977 |
| Negative  |    76 |

Percentage Distribution:

* Positive: 92.8%
* Negative: 7.2%

### Key Finding

The dataset is highly imbalanced.

This means accuracy alone will not be a reliable evaluation metric during model training.

Future evaluation will include:

* Precision
* Recall
* F1 Score
* Confusion Matrix

---

### 4. Feature Relevance Analysis

Inspected dataset columns and identified features relevant to sentiment prediction.

Important columns:

* `reviews.text`
* `reviews.rating`
* `sentiment`

Irrelevant columns such as location and sizing information were removed from further analysis.

---

### 5. Text Preprocessing

Implemented a text cleaning pipeline.

Steps:

1. Convert text to lowercase
2. Remove punctuation
3. Remove extra whitespace

Example:

Original:

This Product is AMAZING!!!

Cleaned:

this product is amazing

A new column called `clean_review` was created and stored.

---

### 6. Vocabulary Analysis

Vocabulary statistics:

* Total Words: 135,093
* Unique Words: 6,350

Most frequent words:

* the
* to
* and
* is
* of

Observation:

The most common words were stopwords and carry limited sentiment information.

---

### 7. Feature Engineering

Implemented TF-IDF Vectorization.

#### TF-IDF Version 1

Default TF-IDF configuration:

Feature Matrix Shape:

(1053, 6328)

---

#### TF-IDF Version 2

Configuration:

* stop_words = 'english'
* min_df = 2

Feature Matrix Shape:

(1053, 3409)

Result:

Reduced feature dimensionality by approximately 46%.

---

## Project Structure

```text
project1/
│
├── Data/
│   ├── reviews.csv
│   ├── labeled_reviews.csv
│   └── cleaned_reviews.csv
│
├── src/
│   ├── 01_data_inspection.py
│   ├── 02_label_creation.py
│   ├── 03_text_preprocessing.py
│   └── 04_feature_engineering.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Regular Expressions (re)

---

## Key Learnings

* Dataset exploration is critical before model training.
* Class imbalance can significantly impact evaluation.
* Text data must be converted into numerical representations before machine learning.
* TF-IDF provides a strong baseline representation for sentiment analysis tasks.
* Feature engineering decisions directly affect model complexity and performance.

---
## Day 2 - Model Training and Evaluation

Implemented a complete machine learning pipeline using TF-IDF and Logistic Regression.

Key steps:
- Train-test split with stratification
- TF-IDF vectorization
- Logistic Regression baseline
- Class imbalance handling using class_weight="balanced"
- Evaluation using Precision, Recall, F1 Score, and Confusion Matrix
- Error analysis of misclassified reviews
- Extraction of most influential positive and negative words

Results:

Balanced Logistic Regression:
- Accuracy: 95.73%
- Precision (Negative): 66.7%
- Recall (Negative): 80.0%
- F1 Score (Negative): 72.7%

Key Insight:
A baseline model achieved high accuracy but failed to identify any negative reviews. Using class-weight balancing significantly improved detection of minority-class reviews.

## Day 3 – Model Comparison & Selection

### Objective

The goal of Day 3 was to compare multiple machine learning algorithms and identify the most suitable model for sentiment classification on an imbalanced Amazon reviews dataset.

### Models Evaluated

The following models were trained and evaluated using TF-IDF features:

* Logistic Regression (Balanced)
* Multinomial Naive Bayes
* Linear Support Vector Machine (Linear SVM)

### Evaluation Metrics

Because the dataset is highly imbalanced (approximately 93% Positive and 7% Negative reviews), model selection was based primarily on:

* Recall (Negative Class)
* F1 Score (Negative Class)
* Cross-Validation Performance

Accuracy was not used as the primary decision metric because it can be misleading on imbalanced datasets.

### Results

| Model                   | Accuracy | Precision | Recall | F1 Score |
| ----------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression     |   95.73% |      0.67 |   0.80 |     0.73 |
| Multinomial Naive Bayes |   92.89% |      0.00 |   0.00 |     0.00 |
| Linear SVM              |   94.79% |      0.83 |   0.33 |     0.48 |

### Cross-Validation Results

| Model                   | Mean CV Score (F1 Macro) |
| ----------------------- | -----------------------: |
| Logistic Regression     |                   0.7127 |
| Linear SVM              |                   0.6819 |
| Multinomial Naive Bayes |                   0.4812 |

### Key Findings

#### Logistic Regression

* Correctly identified 12 out of 15 negative reviews.
* Achieved the highest recall and F1 score for the minority class.
* Produced the strongest cross-validation performance.

#### Multinomial Naive Bayes

* Failed to detect any negative reviews.
* Predicted all reviews as positive.
* Demonstrated the limitations of relying on accuracy for imbalanced datasets.

#### Linear SVM

* Achieved high precision but very low recall.
* Missed a significant number of negative reviews.
* Underperformed Logistic Regression on minority-class detection.

### Final Model Selection

Logistic Regression with class weighting was selected as the final model because it achieved:

* Highest Negative Recall (80%)
* Highest Negative F1 Score (0.73)
* Best Cross-Validation Performance (0.7127)

### Skills Demonstrated

* Model Comparison
* Imbalanced Classification
* Performance Evaluation
* Cross Validation
* Confusion Matrix Analysis
* Classification Report Interpretation
* Production Model Selection
## Day 4 – Hyperparameter Tuning

### Objective

The objective of Day 4 was to optimize the selected Logistic Regression model using hyperparameter tuning and cross-validation.

### Approach

GridSearchCV was used with 5-fold cross-validation to evaluate different values of the regularization parameter C.

Parameter Grid:

```python
{
    "C": [0.01, 0.1, 1, 10, 100]
}
```

Scoring Metric:

```python
f1_macro
```

### Best Hyperparameters

```python
{
    "C": 1
}
```

### Best Cross-Validation Score

```text
0.7127
```

### Tuned Model Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 95.73% |
| Precision | 66.67% |
| Recall    | 80.00% |
| F1 Score  | 72.73% |

### Findings

The tuning process revealed that the default Logistic Regression configuration was already optimal for this dataset. None of the tested regularization strengths produced a higher cross-validation score than the baseline configuration.

### Key Takeaways

* Implemented GridSearchCV for automated hyperparameter optimization.
* Applied 5-fold cross-validation for robust model selection.
* Validated that the chosen Logistic Regression model was already well-configured.
* Confirmed model stability through consistent cross-validation and test-set performance.

### Skills Demonstrated

* Hyperparameter Tuning
* GridSearchCV
* Cross Validation
* Model Optimization
* Regularization Analysis
* Performance Validation

## Author

Muhammad Jarrar Shaf

Project: AI-Powered Customer Feedback Intelligence System
