import pandas as pd

df = pd.read_csv("Data/reviews.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df["reviews.rating"].value_counts().sort_index())