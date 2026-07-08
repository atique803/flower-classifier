"""
Step 3: Train classification models.

Two models to compare:
1. K-Nearest Neighbors - classifies a flower by looking at its closest
   neighbors in the training data (distance-based, no assumptions).
2. Decision Tree - learns a series of yes/no splits on the features
   (e.g. "petal length < 2.5?") and is easy to read after training.

Both wrapped in a Pipeline with StandardScaler (matters for KNN, which
is distance-based; harmless for the tree).
"""
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()

models = {
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "decision_tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ]),
}

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f"data/{name}.joblib")
    print(f"Trained and saved: {name}")
