import numpy as np

def information_gain(y, split_mask):
    """
    Compute information gain of a binary split using base-2 entropy.
    """
    y          = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    y_left  = y[ split_mask]
    y_right = y[~split_mask]

    # Edge case: either side empty → no gain
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0

    def entropy(labels):
        n = len(labels)
        if n == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / n
        # Stable: 0·log(0) = 0 by convention
        return -np.sum(p * np.log2(np.where(p > 0, p, 1.0)))

    n       = len(y)
    n_left  = len(y_left)
    n_right = len(y_right)

    h_parent   = entropy(y)
    h_weighted = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right)

    return float(h_parent - h_weighted)
