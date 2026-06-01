import numpy as np

def majority_classifier(y_train, y_right):
    """
    Predict the most frequent class for all test samples.
    """
    y_train = np.asarray(y_train)
    X_test  = np.asarray(y_right)

    # Edge case: empty training set
    if len(y_train) == 0:
        return np.array([], dtype=int)

    # Find majority class — stable: argmax returns first max on tie
    classes, counts = np.unique(y_train, return_counts=True)
    majority_class  = classes[np.argmax(counts)]

    # Return array of majority class repeated for every test sample
    n_test = X_test.shape[0] if X_test.ndim >= 1 else 1
    return np.full(n_test, majority_class, dtype=int)