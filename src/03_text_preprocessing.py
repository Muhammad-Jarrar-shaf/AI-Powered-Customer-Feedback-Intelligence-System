"""
03_text_preprocessing.py

Purpose:
Clean review text before feature extraction.

Cleaning Steps:
1. Convert text to lowercase
2. Remove punctuation
3. Remove extra spaces

Output:
cleaned_reviews.csv
"""

import re
import pandas as pd

# Load labeled dataset
df = pd.read_csv("Data/labeled_reviews.csv")

print("Dataset Shape:")
print(df.shape)

# Remove irrelevant columns
df = df.drop(
    columns=[
        "reviews.userCity",
        "reviews.userProvince",
        "sizes"
    ],
    errors="ignore"
)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Text cleaning function
def clean_text(text):

    text = str(text).lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra spaces
    text = " ".join(text.split())

    return text


# Create cleaned review column
df["clean_review"] = df["reviews.text"].apply(clean_text)

# Compare original vs cleaned text
print("\nSample Cleaning Results:\n")

for i in range(5):

    print("ORIGINAL:")
    print(df["reviews.text"].iloc[i])

    print("\nCLEANED:")
    print(df["clean_review"].iloc[i])

    print("-" * 80)

# Save cleaned dataset
df.to_csv("cleaned_reviews.csv", index=False)

print("\nCleaned dataset saved successfully.")