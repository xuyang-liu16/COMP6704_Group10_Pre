"""Shared convex problem: L2-regularized binary logistic regression.

minimize_w  (1/n) sum log(1+exp(-y_i x_i^T w)) + (lam/2)||w||_2^2
y_i in {-1,+1}. Strictly convex for lam>0.
"""
from __future__ import annotations
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

def objective(w: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float) -> float:
    # y in {-1,+1}
    margins = y * (X @ w)
    loss = np.mean(np.logaddexp(0.0, -margins))
    return float(loss + 0.5 * lam * np.dot(w, w))

def grad(w: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    # d/dw mean softplus(-y Xw) = mean( -y * sigmoid(-y Xw) * x ) = mean( (sigmoid(Xw)- (y+1)/2 ) * ... )
    # For y in {-1,+1}: grad_loss = (1/n) X^T (sigmoid(Xw) - (y+1)/2) wait:
    # p = sigmoid(X @ w); target = (y+1)/2 in {0,1}; grad = X.T @ (p-target)/n + lam w
    p = sigmoid(X @ w)
    target = (y + 1.0) / 2.0
    return (X.T @ (p - target)) / X.shape[0] + lam * w

def lipschitz_hint(X: np.ndarray, lam: float) -> float:
    # ||X||_2^2 / (4n) + lam upper-bounds smooth logistic Hessian
    # use Frobenius proxy for speed
    n = X.shape[0]
    return float((np.linalg.norm(X, ord=2) ** 2) / (4.0 * n) + lam)
