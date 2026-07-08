"""
Step 1: Load the dataset and explore it.

We use the classic Iris dataset (built into scikit-learn, no download
needed). 150 flowers, 3 species, 4 measurements each. This is the
"hello world" of classification: small, clean, and a good place to
learn multi-class prediction end-to-end.
"""
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("Shape (rows, cols):", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst 5 rows:\n", df.head())
print("\nClass balance:\n", df["species"].value_counts())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nSummary stats:\n", df.describe())

df.to_csv("data/iris.csv", index=False)
print("\nSaved to data/iris.csv")
