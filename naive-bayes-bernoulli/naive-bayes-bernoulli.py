import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):  # ← renamed
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test  = np.asarray(X_test,  dtype=float)

    classes   = np.sort(np.unique(y_train))
    n_train   = len(y_train)
    n_classes = len(classes)

    # Log priors
    log_priors = np.array([
        np.log(np.sum(y_train == c) / n_train)
        for c in classes
    ])

    # Laplace-smoothed theta: P(x_f=1 | y=c)
    theta = np.zeros((n_classes, X_train.shape[1]))
    for i, c in enumerate(classes):
        X_c      = X_train[y_train == c]
        n_c      = X_c.shape[0]
        theta[i] = (X_c.sum(axis=0) + 1) / (n_c + 2)

    # Log-space Bernoulli likelihood
    log_theta    = np.log(theta)
    log_1m_theta = np.log(1 - theta)

    X_test_3d    = X_test[:, np.newaxis, :]
    log_liks     = (X_test_3d      * log_theta[np.newaxis, :, :]
                 + (1 - X_test_3d) * log_1m_theta[np.newaxis, :, :])

    return log_liks.sum(axis=2) + log_priors  # (n_test, n_classes)