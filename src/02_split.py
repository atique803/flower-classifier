"""
Step 2: Train/test split.

Iris has no missing values or outliers to clean, so we go straight from
loading to splitting. Stratified split keeps the same species ratio in
both train and test sets - important for small, balanced datasets like
this one (50 flowers per species).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/iris.csv")

feature_cols = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
X = df[feature_cols]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("\nTrain class balance:\n", y_train.value_counts())
print("\nTest class balance:\n", y_test.value_counts())

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
print("\nSaved train/test splits to data/")
