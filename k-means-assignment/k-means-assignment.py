import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid using squared Euclidean distance.
    """
    P = np.asarray(points,    dtype=float)  # (n, d)
    C = np.asarray(centroids, dtype=float)  # (k, d)

    # Squared distances: (n, 1, d) - (1, k, d) → (n, k, d) → (n, k)
    diff      = P[:, np.newaxis, :] - C[np.newaxis, :, :]
    sq_dists  = np.sum(diff ** 2, axis=-1)  # (n, k)

    # argmin: ties broken by smallest index (argmin returns first occurrence)
    return np.argmin(sq_dists, axis=1).tolist()