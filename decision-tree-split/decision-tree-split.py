import numpy as np

def decision_tree_split(X, y):
    """
    Find the best (feature, threshold) split maximising Gini information gain.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, d = X.shape

    def gini(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        return 1.0 - np.sum(p ** 2)

    parent_gini = gini(y)
    best_gain   = -np.inf
    best_feat   = None
    best_thresh = None

    for f in range(d):
        vals      = np.sort(np.unique(X[:, f]))
        if len(vals) < 2:
            continue
        thresholds = (vals[:-1] + vals[1:]) / 2  # midpoints

        for thresh in thresholds:
            left_mask  = X[:, f] <= thresh
            right_mask = ~left_mask

            n_l, n_r = left_mask.sum(), right_mask.sum()
            if n_l == 0 or n_r == 0:
                continue

            weighted = (n_l / n) * gini(y[left_mask]) + \
                       (n_r / n) * gini(y[right_mask])
            gain = parent_gini - weighted

            # Tie-break: largest gain, then smallest f, then smallest thresh
            if (gain > best_gain or
               (gain == best_gain and (f < best_feat or
               (f == best_feat and thresh < best_thresh)))):
                best_gain   = gain
                best_feat   = f
                best_thresh = thresh

    return [best_feat, best_thresh]