"""
02_label_creation.py

Purpose:
Convert review ratings into sentiment labels.

Ratings:
4-5 -> Positive
1-2 -> Negative
3   -> Neutral

Neutral reviews are removed because this project
focuses on binary sentiment classification.
"""

import pandas as pd

# Load dataset
df = pd.read_csv("Data/reviews.csv")

print("Original Dataset Shape:")
print(df.shape)

# Create sentiment labels
def create_sentiment(rating):

    if rating >= 4:
        return "Positive"

    elif rating <= 2:
        return "Negative"

    else:
        return "Neutral"


df["sentiment"] = df["reviews.rating"].apply(create_sentiment)

print("\nSentiment Distribution (Before Removing Neutral):")
print(df["sentiment"].value_counts())

# Remove neutral reviews
df = df[df["sentiment"] != "Neutral"]

print("\nSentiment Distribution (After Removing Neutral):")
print(df["sentiment"].value_counts())

print("\nPercentage Distribution:")
print(df["sentiment"].value_counts(normalize=True) * 100)

print("\nDataset Shape After Removing Neutral Reviews:")
print(df.shape)

print("\nRating Distribution:")
print(df["reviews.rating"].value_counts().sort_index())

# Save intermediate dataset
df.to_csv("labeled_reviews.csv", index=False)

print("\nLabeled dataset saved successfully.")