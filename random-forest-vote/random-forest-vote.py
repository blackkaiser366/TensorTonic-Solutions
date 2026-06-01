import numpy as np

def random_forest_vote(predictions):
    """
    Compute majority vote across trees for each sample.
    Ties broken by smallest class label.
    """
    P = np.asarray(predictions, dtype=int)  # (T, N)
    T, N = P.shape
    n_classes = P.max() + 1

    # Count votes: bins per sample → (N, n_classes)
    votes = np.zeros((N, n_classes), dtype=int)
    for t in range(T):
        votes[np.arange(N), P[t]] += 1

    # argmax returns first (smallest) index on ties
    return np.argmax(votes, axis=1).tolist()