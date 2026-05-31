"""
04_feature_engineering.py

Purpose:
Convert cleaned text into numerical features
using TF-IDF vectorization.

Also performs:
- Vocabulary analysis
- Word frequency analysis
- Feature inspection
"""

import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("Data/cleaned_reviews.csv")

print("Dataset Shape:")
print(df.shape)

# Vocabulary Analysis
all_text = " ".join(df["clean_review"])

words = all_text.split()

print("\nVocabulary Statistics")
print("Total Words:", len(words))
print("Unique Words:", len(set(words)))

# Most common words
word_freq = Counter(words)

print("\nTop 20 Most Common Words:")
print(word_freq.most_common(20))

# TF-IDF Version 1
print("\n==========================")
print("TF-IDF Version 1")
print("==========================")

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(df["clean_review"])

print("Shape:", X.shape)

print("Number of Features:")
print(len(tfidf.get_feature_names_out()))

print("\nFirst 50 Features:")
print(tfidf.get_feature_names_out()[:50])

# TF-IDF Version 2
print("\n==========================")
print("TF-IDF Version 2")
print("==========================")

tfidf_v2 = TfidfVectorizer(
    stop_words="english",
    min_df=2
)

X_v2 = tfidf_v2.fit_transform(df["clean_review"])

print("Shape:", X_v2.shape)

print("Number of Features:")
print(len(tfidf_v2.get_feature_names_out()))

print("\nFirst 50 Features:")
print(tfidf_v2.get_feature_names_out()[:50])

# Feature inspection
feature_names = tfidf_v2.get_feature_names_out()

print("\nFeature Sample (100-150):")
print(feature_names[100:150])

print("\nFeature Sample (500-550):")
print(feature_names[500:550])

print("\nFeature Engineering Complete.")