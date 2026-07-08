# Iris Flower Classifier

A beginner machine learning project that classifies iris flowers into one of three species — **setosa**, **versicolor**, or **virginica** — based on four measurements: sepal length, sepal width, petal length, and petal width.

This is the classic "hello world" of classification, used here to learn the fundamentals of a multi-class ML pipeline end to end: loading a dataset, splitting it fairly, training more than one model, and judging which one actually performs better. The goal isn't a production tool — it's to build intuition for how a model tells categories apart, and to see where classification gets genuinely hard (versicolor and virginica overlap in measurements, while setosa is always trivially separable).

## How it works

The project is a pipeline of four scripts, each one building on the output of the last:

| Step | Script | What it does |
|---|---|---|
| 1. Load & explore | [`src/01_load_and_explore.py`](src/01_load_and_explore.py) | Loads the built-in scikit-learn Iris dataset (150 flowers, no download needed), checks for missing values, and looks at class balance. |
| 2. Split | [`src/02_split.py`](src/02_split.py) | Splits the data into 120 training rows and 30 test rows. Uses a **stratified** split so each species keeps its 40/10 ratio in both sets — important since we only have 50 examples per species to begin with. |
| 3. Train | [`src/03_train.py`](src/03_train.py) | Trains two different classifiers on the training set: K-Nearest Neighbors and a Decision Tree. Both are saved to `data/` so they don't need retraining to evaluate. |
| 4. Evaluate | [`src/04_evaluate.py`](src/04_evaluate.py) | Runs both trained models on the held-out test set (data they've never seen) and reports accuracy, a confusion matrix, and a per-species precision/recall breakdown. |

## Setup

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

## Usage

Run the scripts in order:

```bash
venv\Scripts\python src\01_load_and_explore.py
venv\Scripts\python src\02_split.py
venv\Scripts\python src\03_train.py
venv\Scripts\python src\04_evaluate.py
```

## Results

| Model | Accuracy |
|---|---|
| K-Nearest Neighbors (k=5) | 93.3% |
| Decision Tree (max_depth=3) | 96.7% |

Both models classify setosa perfectly every time. Every misclassification happens between versicolor and virginica — the two species whose measurements genuinely overlap in real life, not a flaw in either model.

## Glossary (for beginners)

- **Feature** — an input column the model reads (here: the four measurements). In code, this is `X`.
- **Target / label** — the column the model is trying to predict (here: species). In code, this is `y`.
- **Train/test split** — data is split into two piles: one the model learns from, one it's tested on afterward. This checks whether the model actually generalizes, rather than just memorizing the training data (a failure mode called **overfitting**).
- **Stratified split** — a train/test split that preserves the original class ratios in both pieces, so a small dataset like this one doesn't end up with an unbalanced test set by chance.
- **K-Nearest Neighbors (KNN)** — classifies a new flower by looking at the species of the most similar flowers (its "neighbors") in the training data.
- **Decision Tree** — learns a series of yes/no questions on the features (e.g. "is petal length < 2.5 cm?") to arrive at a species.
- **Accuracy** — the percentage of test flowers classified correctly.
- **Confusion matrix** — a table showing exactly which species get mistaken for which. Rows are the actual species, columns are what the model predicted.
- **Precision / recall / F1** — a more detailed, per-species breakdown of correctness. Precision asks "when the model says virginica, how often is it right?"; recall asks "of all the actual virginica flowers, how many did the model catch?"; F1 is a balance of the two.
