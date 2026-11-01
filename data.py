"""Synthetic + sklearn binary datasets for reproducible experiments."""
from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_problem(name: str = "synthetic", seed: int = 0):
    rng = np.random.default_rng(seed)
    if name == "breast_cancer":
        data = load_breast_cancer()
        X, y01 = data.data, data.target
    else:
        X, y01 = make_classification(
            n_samples=2000, n_features=50, n_informative=20,
            n_redundant=5, n_classes=2, random_state=seed,
        )
    y = np.where(y01 == 1, 1.0, -1.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y01
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # add bias column
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    return X_train, y_train, X_test, y_test
