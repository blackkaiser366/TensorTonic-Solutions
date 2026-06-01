import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    P = np.asarray(points,      dtype=float)  # (n, d)
    A = np.asarray(assignments, dtype=int)    # (n,)
    d = P.shape[1]

    centroids = np.zeros((k, d))              # empty clusters → zero vector

    for j in range(k):
        mask = A == j
        if mask.any():
            centroids[j] = P[mask].mean(axis=0)

    return centroids.tolist()