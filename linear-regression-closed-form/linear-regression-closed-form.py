import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute optimal weights using the normal equation: w = (X^T X)^-1 X^T y
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    XtX = X.T @ X
    Xty = X.T @ y

    return np.linalg.solve(XtX, Xty)