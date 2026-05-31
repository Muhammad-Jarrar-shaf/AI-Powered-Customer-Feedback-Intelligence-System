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

## Next Steps (Day 2)

* Train/Test Split
* Logistic Regression Model
* Model Evaluation
* Precision, Recall, F1 Score
* Confusion Matrix
* Handling Class Imbalance
* Error Analysis

---

## Author

Muhammad Jarrar Shaf

Project: AI-Powered Customer Feedback Intelligence System
