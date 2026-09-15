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
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("not 2d matrix")

    n, m = X.shape

    if n < 2 or m < 1:
        raise ValueError("ops..") #придумать нормальное название ошибки на английском 

    if not (1 <= k <= m):
        raise ValueError("k must be between 1 and m")

    u = np.mean(X, axis=0) #размерность
    X_center = X - u
    if standardize:
        scale = np.ones(m, dtype=float)
        for j in range(m):
            if np.all(X[:, j] == X[0, j]):
                u[j] = X[0, j]
                X_center[:, j] = 0.0    #проверить
                scale[j] = 1.0
            else:
                scale[j] = np.sqrt(
                    np.sum(X_center[:, j] ** 2) / (n - 1)
                )

        X_s = X_center / scale

    else:
        scale = np.ones(m, dtype=float)
        Xs = X_centered

    if Gram_trick and n < m:
        G = (X_s @ X.T) / (n - 1)
        eigenvalues, eigenvektors = np.linalg.eigh(G)
        ord = np.argsort(eigenvalues)[::-1] #тк порядок обратный 
        eigenvalues = eigenvalues[ord]
        #прикол грамма
        eigenvektors = eigenvektors[:, ord]

        lambdas = np.zeros(m, dtype=float)
        lambdas[:n] = eigenvalues[:n]

        tau_scale = np.max(np.abs(lambdas)) * TAU_REL # дадада масштабируем границу
        if np.any(lambdas < -tau_scale):
            raise ValueError("") #нада чет написать
        lambdas[np.abs(lambdas) <= tau_scale] = 0.0

        r_num = int(np.count_nonzero(lambdas > 0.0))

        if k > r_num:
            raise ValueError(f"{k} < {r_num}")

        U = np.zeros((m, r_num), dtype=float)
        colum = 0
        for i in range(n):
            l = lambdas[i]
            if l <= 0.0:
                continue
            u = (Xs.T @ eigenvektors[:, i]) / np.sqrt((n - 1) * l)
            U[:, colum] = u
            colum += 1
        for i in range(r_num):
            for _ in range(2):
                for j in range(i):
                    U[:, i] -= np.dot(U[:, j], U[:, i]) * U[:, j]

            norm = np.linalg.norm(U[:, i])
            if norm <= 0:
                raise ValueError("") #чет написать
            U[:, i] /= norm

        for i in range(r_num):
            j = int(np.argmax(np.abs(U[:, i])))
            if U[j, i] < 0:
                U[:, i] *= -1.0






        



        
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
