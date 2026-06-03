# AI-Powered Customer Feedback Intelligence System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)

## Overview

An end-to-end Machine Learning and NLP application that analyzes customer reviews and automatically classifies them as **Positive** or **Negative**.

The project demonstrates the complete machine learning lifecycle, including:

* Data preprocessing
* Feature engineering
* Model training
* Model evaluation
* Hyperparameter tuning
* Model interpretability
* Business intelligence reporting
* Streamlit deployment

The final system provides real-time sentiment predictions through an interactive web application while also generating business insights from customer feedback.

---

## Problem Statement

Organizations receive thousands of customer reviews across products and services. Manually analyzing this feedback is expensive, time-consuming, and difficult to scale.

This project leverages Natural Language Processing (NLP) and Machine Learning to automatically identify customer sentiment and uncover actionable business insights.

### Example

| Review                               | Predicted Sentiment |
| ------------------------------------ | ------------------- |
| Amazing product. Highly recommended. | Positive            |
| Terrible quality. Waste of money.    | Negative            |

---

## Dataset

**Amazon Product Reviews Dataset**

### Final Dataset Size

| Metric           | Value |
| ---------------- | ----: |
| Total Reviews    | 1,053 |
| Positive Reviews |   977 |
| Negative Reviews |    76 |

### Class Distribution

* Positive: 92.8%
* Negative: 7.2%

This project addresses the challenge of imbalanced classification through model selection and class-weight balancing.

---

## Machine Learning Pipeline

```text
Raw Reviews
      │
      ▼
Text Cleaning
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Logistic Regression
      │
      ▼
Sentiment Prediction
      │
      ▼
Business Insights
      │
      ▼
Streamlit Application
```

---

## Technologies Used

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* Logistic Regression
* TF-IDF Vectorization

### Visualization

* Matplotlib

### Deployment

* Streamlit

### Model Persistence

* Joblib

---

## Model Development

### Models Evaluated

* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine (SVM)

### Final Model

```python
LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000,
    C=1
)
```

### Why Logistic Regression?

The model achieved the best balance between precision, recall, and F1 score on the highly imbalanced dataset.

---

## Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 95.73% |
| Precision | 66.67% |
| Recall    | 80.00% |
| F1 Score  | 72.73% |

### Key Achievement

The final model successfully detected **80% of negative reviews**, despite negative samples representing only **7.2%** of the dataset.

---

## Model Interpretability

The project includes feature importance analysis using Logistic Regression coefficients.

### Strong Positive Indicators

* great
* alexa
* love
* sound
* easy
* speaker

### Strong Negative Indicators

* remote
* terrible
* charger
* return
* useless
* waste

This analysis helps explain model decisions and increases transparency.

---

## Business Insights

### Positive Drivers

Customers frequently praised:

* Ease of use
* Device quality
* Sound performance
* Kindle ecosystem
* Amazon product experience

### Common Complaints

Customers frequently reported issues related to:

* Remote controls
* Chargers
* Accessories
* Device reliability
* Product returns

### Recommendations

* Improve accessory reliability
* Investigate remote-control issues
* Enhance customer support processes
* Continue emphasizing ease of use and sound quality in marketing campaigns

---

## Streamlit Application

The project includes an interactive Streamlit application for real-time sentiment analysis.

### Features

* Customer review input
* Real-time sentiment prediction
* Confidence score generation
* Positive/Negative probability display

### Run Locally

```bash
git clone <repository-url>

cd AI-Powered-Customer-Feedback-Intelligence-System

pip install -r requirements.txt

streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Project Structure

```text
AI-Powered-Customer-Feedback-Intelligence-System/

├── Data/
├── Models/
├── Results/
├── src/
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Skills Demonstrated

* Natural Language Processing (NLP)
* Feature Engineering
* Imbalanced Classification
* Model Evaluation
* Hyperparameter Tuning
* Model Interpretability
* Business Analytics
* Streamlit Deployment
* End-to-End Machine Learning Development

---

## Future Improvements

* BERT-based sentiment analysis
* Multi-class sentiment classification
* Aspect-based sentiment analysis
* Cloud deployment
* Real-time feedback dashboard
* Review summarization using LLMs

---

## Author

**Muhammad Jarrar Shaf**

AI / Machine Learning Engineer

GitHub: https://github.com/Muhammad-Jarrar-shaf

LinkedIn: https://www.linkedin.com/in/muhammad-jarrar-33a239362/
