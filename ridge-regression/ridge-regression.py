import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights: w = (X^T X + λI)^-1 X^T y
    """
    X   = np.asarray(X,   dtype=float)
    y   = np.asarray(y,   dtype=float)

    d   = X.shape[1]
    A   = X.T @ X + lam * np.eye(d)   # (d, d) regularised matrix
    b   = X.T @ y                      # (d,)

    return np.linalg.solve(A, b).tolist()