import numpy as np

TAU_REL = 1e-12  # numerical-zero threshold factor: tau = TAU_REL * max|lambda|


# ================================= pca_fit ==================================
def pca_fit(X, k, standardize=False, Gram_trick=True):
    """PCA fit.

    :param X: np.array((n, m)), n >= 2 observations in rows, m features in columns
    :param k: int, number of components, 1 <= k <= r_num (numerical rank)
    :param standardize: bool, correlation PCA if True, covariance PCA if False
    :param Gram_trick: bool -- True: build components via the n x n Gram matrix if n < m. 
                               False: build only through m x m matrix Sigma.

    :return: (mu, scale, U, lam)
        mu    - np.array((m,)), mean vector
        scale - np.array((m,)), column scale (std if standardize, else ones; 1 for constant)
        U     - np.array((m, k)), orthonormal columns, variance-descending, sign-fixed
        lam   - np.array((m,)), full spectrum of Sigma_s, descending, zero-padded
    """
    # your code here \/
    return ...
    # your code here /\


# ============================== pca_transform ===============================
def pca_transform(X, mu, scale, U):
    """
    :param X: np.array((n, m)), data matrix
    :param mu, scale, U: fit parameters returned by pca_fit

    :return: np.array((n, k)) -- P = ((X - mu) / scale) @ U
    """
    # your code here \/
    return ...
    # your code here /\


# =============================== pca_inverse ================================
def pca_inverse(P, mu, scale, U):
    """
    :param P: np.array((n, k)), projected coordinates
    :param mu, scale, U: fit parameters returned by pca_fit

    :return: np.array((n, m)) -- X_hat = (P @ U^T) * scale + mu
    """
    # your code here \/
    return ...
    # your code here /\


# ========================= explained_variance_ratio =========================
def explained_variance_ratio(lam, k):
    """
    :param lam: np.array((m,)), covariance spectrum, descending
    :param k: int, number of leading components

    :return: float -- sum(lam[:k]) / sum(lam) when the total is positive
    """
    # your code here \/
    return ...
    # your code here /\
