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
## Day 5 – Model Persistence & Inference Pipeline

### Objective

The objective of Day 5 was to make the sentiment analysis model reusable without retraining by implementing model persistence and an inference pipeline.

### Approach

The final Logistic Regression model and TF-IDF vectorizer were serialized using Joblib and stored for future use.

Saved Artifacts:

* sentiment_model.pkl
* tfidf_vectorizer.pkl

### Model Persistence

The trained model was saved using:

```python
joblib.dump(model, "Models/sentiment_model.pkl")
```

The TF-IDF vectorizer was saved using:

```python
joblib.dump(vectorizer, "Models/tfidf_vectorizer.pkl")
```

### Inference Pipeline

The saved artifacts were loaded back into memory and used to generate predictions for completely new reviews without retraining the model.

Example Predictions:

| Review                                      | Predicted Sentiment |
| ------------------------------------------- | ------------------- |
| This product is amazing and works perfectly | Positive            |
| Terrible quality, waste of money            | Negative            |
| I love this device and would recommend it   | Positive            |
| Very disappointed with this purchase        | Positive            |

### Key Findings

The inference pipeline successfully generated predictions on unseen text. Testing also revealed examples where human interpretation differed from model predictions, highlighting opportunities for future improvement through additional training data and advanced feature engineering.

### Skills Demonstrated

* Model Serialization
* Joblib
* Inference Pipeline Development
* Production Readiness
* Model Deployment Preparation
* End-to-End Machine Learning Workflow

# Day 6 – Model Interpretability & Error Analysis

## Overview

On Day 6, the focus shifted from model performance to understanding **why the model makes its predictions**. While previous stages concentrated on training, evaluation, tuning, and persistence, this phase explored the internal behavior of the final Logistic Regression model.

The goal was to identify the most influential words driving sentiment predictions, explain individual predictions, and analyze classification errors to better understand the strengths and limitations of a TF-IDF + Logistic Regression approach.

---

## Objectives

* Interpret the trained Logistic Regression model
* Extract feature importance using model coefficients
* Identify the strongest positive and negative sentiment indicators
* Explain individual review predictions
* Investigate model misclassifications
* Generate visualizations for model transparency
* Develop interview-ready explanations of model behavior

---

## Methodology

### Feature Importance Analysis

The coefficients learned by the Logistic Regression model were extracted and mapped to TF-IDF vocabulary terms.

**Interpretation:**

* Positive coefficients push predictions toward the **Positive** class.
* Negative coefficients push predictions toward the **Negative** class.
* Larger absolute coefficient values indicate stronger influence.

---

## Top Positive Features

The model identified the following words as the strongest indicators of positive sentiment:

| Word       | Coefficient |
| ---------- | ----------: |
| great      |        2.69 |
| alexa      |        2.02 |
| love       |        1.41 |
| sound      |        1.41 |
| tap        |        1.30 |
| gift       |        1.21 |
| easy       |        1.18 |
| echo       |        1.08 |
| headphones |        1.07 |
| speaker    |        1.06 |

### Key Observation

Words associated with product satisfaction, recommendation, ease of use, and Amazon Echo devices strongly influenced positive predictions.

---

## Top Negative Features

The model identified the following words as the strongest indicators of negative sentiment:

| Word     | Coefficient |
| -------- | ----------: |
| remote   |       -2.21 |
| terrible |       -1.97 |
| app      |       -1.54 |
| months   |       -1.54 |
| netflix  |       -1.53 |
| horrible |       -1.40 |
| charger  |       -1.39 |
| return   |       -1.33 |
| useless  |       -1.32 |
| waste    |       -1.31 |

### Key Observation

Words commonly associated with product failures, defects, returns, and customer dissatisfaction heavily influenced negative predictions.

---

## Individual Prediction Analysis

A prediction explanation module was implemented to inspect the contribution of individual words within a review.

### Example 1

**Review**

```text
This product is amazing and works perfectly
```

**Prediction**

```text
Positive
```

**Probability**

```text
Positive: 59.12%
Negative: 40.88%
```

**Contributing Words**

| Word      | Contribution |
| --------- | -----------: |
| product   |       +0.115 |
| works     |       +0.074 |
| amazing   |       -0.026 |
| perfectly |       -0.176 |

### Finding

Surprisingly, the words **"amazing"** and **"perfectly"** carried slightly negative weights. This highlights a limitation of small datasets and sparse TF-IDF representations, where words may receive unintuitive coefficients based on training distribution.

---

### Example 2

**Review**

```text
Terrible quality and complete waste of money
```

**Prediction**

```text
Negative
```

**Probability**

```text
Negative: 85.50%
Positive: 14.50%
```

### Strongest Contributors

```text
terrible
waste
money
complete
```

The model correctly identified multiple highly negative sentiment indicators.

---

### Example 3

**Review**

```text
Very disappointed with this purchase
```

**Prediction**

```text
Positive
```

**Probability**

```text
Positive: 60.25%
Negative: 39.75%
```

**Contributing Words**

| Word         | Contribution |
| ------------ | -----------: |
| purchase     |       +0.173 |
| disappointed |       -0.138 |

### Finding

Although the word **"disappointed"** contributed negatively, the word **"purchase"** had a stronger positive influence, resulting in an incorrect positive prediction.

This demonstrates how TF-IDF models rely on individual word statistics rather than contextual meaning.

---

## Key Insights

### Insight 1: Correlation vs Meaning

The model learns statistical relationships rather than true semantic understanding.

Example:

```text
remote
```

became the strongest negative feature, not because the word itself is negative, but because reviews mentioning "remote" were frequently negative in the training data.

---

### Insight 2: Lack of Context Awareness

TF-IDF treats words independently.

Examples:

```text
good
not good
```

Both contain the word:

```text
good
```

The model cannot fully understand the contextual difference.

---

### Insight 3: No Phrase Understanding

The model processes:

```text
very disappointed
```

as separate words:

```text
very
disappointed
```

rather than a single sentiment-bearing phrase.

---

### Insight 4: Small Dataset Effects

With only:

```text
1,053 reviews
```

and severe class imbalance:

```text
Positive: 977
Negative: 76
```

some coefficients may not accurately represent real-world sentiment.

---

## Files Generated

```text
Results/

top_positive_words.csv
top_negative_words.csv

top_positive_words.png
top_negative_words.png

all_feature_importance.csv
```

---

## Skills Demonstrated

* Model Interpretability
* Feature Importance Analysis
* Logistic Regression Coefficient Analysis
* Prediction Explainability
* Error Analysis
* Data Visualization
* Business-Oriented Model Evaluation

---

## Key Takeaway

Day 6 transformed the project from a simple sentiment classifier into an interpretable machine learning system. By analyzing feature importance and explaining individual predictions, deeper insights were gained into model behavior, strengths, limitations, and potential areas for future improvement.

The analysis revealed that while the Logistic Regression model performs well overall, TF-IDF-based approaches remain limited by their inability to capture context, semantics, and phrase-level meaning. These findings provide a strong foundation for discussing model decisions, trade-offs, and real-world deployment considerations during technical interviews.

---

## Day 6 Deliverables

* ✅ Feature Importance Extraction
* ✅ Positive & Negative Feature Analysis
* ✅ Prediction Explanation System
* ✅ Misclassification Investigation
* ✅ Visualization of Important Features
* ✅ Model Transparency & Interpretability
* ✅ Interview-Ready Discussion Points

**Project Status:** Day 6 Complete


## Author

Muhammad Jarrar Shaf

Project: AI-Powered Customer Feedback Intelligence System
