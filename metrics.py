"""Shared metrics and history helpers."""
from __future__ import annotations
import numpy as np
from problem import objective, sigmoid

def accuracy(w, X, y):
    pred = np.sign(X @ w)
    pred[pred == 0] = 1
    return float(np.mean(pred == y))

def pack_history(obj_hist, time_hist, w, X_tr, y_tr, X_te, y_te, lam):
    return {
        "objective": list(map(float, obj_hist)),
        "time_s": list(map(float, time_hist)),
        "final_obj": float(objective(w, X_tr, y_tr, lam)),
        "train_acc": accuracy(w, X_tr, y_tr),
        "test_acc": accuracy(w, X_te, y_te),
        "w_norm": float(np.linalg.norm(w)),
    }
