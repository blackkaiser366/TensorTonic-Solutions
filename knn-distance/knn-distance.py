import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)

    # Correct 1D reshape: (n,) → (n, 1)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]

    # Vectorised pairwise Euclidean distances
    diff  = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))   # (n_test, n_train)

    # Stable full sort
    sorted_idx = np.argsort(dists, axis=1, kind='stable')

    k_actual  = min(k, n_train)
    neighbors = sorted_idx[:, :k_actual]

    # Pad with -1 if k > n_train
    if k_actual < k:
        pad       = np.full((n_test, k - k_actual), -1, dtype=int)
        neighbors = np.hstack([neighbors, pad])

    return neighbors.astype(int)