import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    
    # Stable: only compute log where p > 0 (0*log(0) = 0 by convention)
    log_p = np.where(p > 0, np.log2(p), 0.0)
    
    return float(-np.sum(p * log_p))