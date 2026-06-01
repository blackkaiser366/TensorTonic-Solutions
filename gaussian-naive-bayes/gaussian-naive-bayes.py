import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Gaussian Naive Bayes classifier using log posteriors.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test  = np.asarray(X_test,  dtype=float)

    classes  = np.sort(np.unique(y_train))
    n        = len(y_train)
    eps      = 1e-9

    # ── Per-class statistics ───────────────────────────────────────────
    log_priors = []
    means      = []
    variances  = []

    for c in classes:
        X_c = X_train[y_train == c]
        log_priors.append(np.log(len(X_c) / n))          # log P(c)
        means.append(X_c.mean(axis=0))                    # μ_cj
        variances.append(X_c.var(axis=0) + eps)           # σ²_cj + ε

    log_priors = np.array(log_priors)                     # (n_classes,)
    means      = np.array(means)                          # (n_classes, d)
    variances  = np.array(variances)                      # (n_classes, d)

    # ── Log-likelihood: Gaussian log pdf summed over features ─────────
    # X_test   : (n_test, 1,        d)
    # means    : (1,      n_classes, d)  →  broadcast
    X_test_3d  = X_test[:, np.newaxis, :]                 # (n_test, 1, d)

    log_liks   = (
        -0.5 * np.log(2 * np.pi * variances)              # (n_classes, d)
        - 0.5 * (X_test_3d - means) ** 2 / variances      # (n_test, n_classes, d)
    ).sum(axis=-1)                                         # (n_test, n_classes)

    # ── Log posterior = log prior + log likelihood ────────────────────
    log_post   = log_liks + log_priors                    # (n_test, n_classes)

    # ── Predict class with highest log posterior ──────────────────────
    return classes[np.argmax(log_post, axis=1)].tolist()