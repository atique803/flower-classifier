"""
Step 4: Evaluate models on the held-out test set.

Metrics (classification, not regression, so different from the house
price project):
- Accuracy: fraction of flowers correctly classified.
- Confusion matrix: rows = actual species, columns = predicted species,
  shows exactly which species get confused with which.
- Precision/recall/F1 (classification report): per-species breakdown,
  useful when classes could be imbalanced (not the case here, but good
  habit to check).
"""
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

for name in ["knn", "decision_tree"]:
    model = joblib.load(f"data/{name}.joblib")
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n{name}")
    print(f"  Accuracy: {acc:.4f}")
    print("  Confusion matrix (rows=actual, cols=predicted):")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, preds, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))
    print("\n  Classification report:\n", classification_report(y_test, preds))
