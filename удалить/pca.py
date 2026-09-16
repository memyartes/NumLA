import numpy as np

TAU_REL = 1e-12  # numerical-zero threshold factor: tau = TAU_REL * max|lambda|


def _mgs_orthonormalize(V):
    m, q = V.shape
    Q = np.zeros_like(V)
    for j in range(q):
        v = V[:, j].copy()
        for _ in range(2):
            for i in range(j):
                v = v - (Q[:, i] @ v) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return Q


def _fix_signs(U):
    U = U.copy()
    for j in range(U.shape[1]):
        idx = np.argmax(np.abs(U[:, j]))
        if U[idx, j] < 0:
            U[:, j] = -U[:, j]
    return U


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
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"X must be a 2D array (n, m), got array with ndim={X.ndim}."
        )

    n, m = X.shape

    if n < 2:
        raise ValueError(
            f"Invalid sample count: n={n}, but at least n=2 observations are required."
        )
    if m < 1:
        raise ValueError(
            f"Invalid feature count: m={m}, but at least m=1 feature is required."
        )

    # 1. среднее и центрирование
    mu = X.mean(axis=0)
    const_mask = np.all(X == X[0, :], axis=0)
    if np.any(const_mask):
        mu = mu.copy()
        mu[const_mask] = X[0, const_mask]
    Xc = X - mu
    Xc[:, const_mask] = 0.0

    # 2. масштаб
    if standardize:
        std = np.sqrt(np.sum(Xc ** 2, axis=0) / (n - 1))
        scale = np.where(const_mask, 1.0, std)
    else:
        scale = np.ones(m)

    Xs = Xc / scale

    if Gram_trick and n < m:
        G = (Xs @ Xs.T) / (n - 1)
        eigval, eigvec = np.linalg.eigh(G)
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = eigval[order], eigvec[:, order]

        tau = TAU_REL * np.max(np.abs(eigval)) if eigval.size else 0.0
        if np.any(eigval < -tau):
            raise ValueError("Negative eigenvalue beyond numerical tolerance encountered.")

        pos = eigval > tau
        lam_pos, w_pos = eigval[pos], eigvec[:, pos]

        U_raw = (Xs.T @ w_pos) / np.sqrt((n - 1) * lam_pos)[None, :]
        U_full = _mgs_orthonormalize(U_raw) if U_raw.shape[1] else U_raw
        U_full = _fix_signs(U_full)

        rnum = U_full.shape[1]
        lam = np.zeros(m)
        lam[:rnum] = lam_pos

    else:
        Sigma_s = (Xs.T @ Xs) / (n - 1)
        eigval, eigvec = np.linalg.eigh(Sigma_s)
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = eigval[order], eigvec[:, order]

        tau = TAU_REL * np.max(np.abs(eigval)) if eigval.size else 0.0
        if np.any(eigval < -tau):
            raise ValueError("Negative eigenvalue beyond numerical tolerance encountered.")

        lam = np.where(np.abs(eigval) <= tau, 0.0, eigval)
        rnum = int(np.sum(lam > 0))
        U_full = _fix_signs(eigvec)

    if not (1 <= k <= rnum):
        raise ValueError(
            f"k={k} is invalid: 1 <= k <= r_num={rnum} is required "
            f"(numerical rank r_num={rnum})."
        )

    U = U_full[:, :k]
    return mu, scale, U, lam


# ============================== pca_transform ===============================
def pca_transform(X, mu, scale, U):
    """
    :return: np.array((n, k)) -- P = ((X - mu) / scale) @ U
    """
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    scale = np.asarray(scale, dtype=float)
    U = np.asarray(U, dtype=float)

    Xs = (X - mu) / scale
    return Xs @ U


# =============================== pca_inverse ================================
def pca_inverse(P, mu, scale, U):
    """
    :return: np.array((n, m)) -- X_hat = (P @ U^T) * scale + mu
    """
    P = np.asarray(P, dtype=float)
    mu = np.asarray(mu, dtype=float)
    scale = np.asarray(scale, dtype=float)
    U = np.asarray(U, dtype=float)

    return (P @ U.T) * scale + mu


# ========================= explained_variance_ratio =========================
def explained_variance_ratio(lam, k):
    """
    :return: float -- sum(lam[:k]) / sum(lam) при положительной сумме lam
    """
    lam = np.asarray(lam, dtype=float)
    total = lam.sum()
    if total <= 0:
        raise ValueError("Суммарная дисперсия (sum(lam)) должна быть положительной.")
    return float(lam[:k].sum() / total)
