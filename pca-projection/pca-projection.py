import numpy as np

def pca_projection(X, k):
    """
    Project data onto top-k principal components.
    """
    X   = np.asarray(X, dtype=float)        # (n, d)

    # 1. Centre the data
    Xc  = X - X.mean(axis=0)               # (n, d)

    # 2. Sample covariance matrix (divide by n-1)
    C   = (Xc.T @ Xc) / (len(X) - 1)      # (d, d)

    # 3. Eigen-decomposition; eigh returns values in ascending order
    vals, vecs = np.linalg.eigh(C)         # vals (d,), vecs (d, d)

    # 4. Sort by descending eigenvalue, take top-k eigenvectors
    idx = np.argsort(vals)[::-1]           # descending
    W   = vecs[:, idx[:k]]                 # (d, k)

    # 5. Project centred data
    return (Xc @ W).tolist()               # (n, k)