import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    def gini_node(y):
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / n
        return 1.0 - np.sum(p ** 2)

    y_left  = np.asarray(y_left)
    y_right = np.asarray(y_right)

    n_left  = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_total == 0:
        return 0.0

    weighted = (n_left  / n_total) * gini_node(y_left) + \
               (n_right / n_total) * gini_node(y_right)

    return float(weighted)