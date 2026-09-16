import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

from pca import pca_fit, pca_transform, pca_inverse, explained_variance_ratio, TAU_REL

IMG = (64, 64)
RNG_SEED = 42


def show_images(images, titles, filename, ncols=8):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.5 * ncols, 1.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.reshape(IMG), cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()


def task_1_mean_face(X):
    mu = X.mean(axis=0)
    show_images([mu], ["mean face"], "mean_face.png", ncols=1)

    idx = [0, 1, 2, 3]
    centered = X[idx] - mu
    show_images(centered, [f"centered #{i}" for i in idx], "centered_faces.png", ncols=4)
    return mu


def task_2_eigenfaces(X):
    mu, scale, U, lam = pca_fit(X, k=16, standardize=False, Gram_trick=True)
    show_images(U.T, [f"eig {i+1}" for i in range(16)], "eigenfaces_16.png", ncols=4)
    return mu, scale, U, lam


def task_3_spectrum(lam):
    tau = TAU_REL * np.max(np.abs(lam))
    pos = lam[lam > tau]
    n_zero = np.sum(lam <= tau)
    print(f"Non-zero eigenvalues above threshold: {len(pos)} (n-1 = {399})")
    print(f"Number of numerically-zero eigenvalues: {n_zero}")

    plt.figure()
    plt.plot(np.arange(1, len(pos) + 1), pos, "o-")
    plt.yscale("log")
    plt.xlabel("i")
    plt.ylabel("lambda_i")
    plt.title("Eigenfaces spectrum (log scale, lambda_i > tau)")
    plt.savefig("eigenfaces_spectrum.png", dpi=120)
    plt.close()


def task_4_reconstruction(X, mu_full, scale_full):
    ks = [1, 5, 10, 25, 50, 100, 200, 399]
    face_idx = 10

    images, titles = [X[face_idx]], ["original"]
    for k in ks:
        mu, scale, U, lam = pca_fit(X, k=k, standardize=False, Gram_trick=True)
        P = pca_transform(X[face_idx:face_idx + 1], mu, scale, U)
        X_hat = pca_inverse(P, mu, scale, U)
        images.append(X_hat[0])
        titles.append(f"k={k}")
    show_images(images, titles, "reconstruction_row.png", ncols=len(images))

    # ошибка восстановления для всей выборки
    mu_full2, scale_full2, U_full, lam_full = pca_fit(X, k=max(ks), standardize=False, Gram_trick=True)
    Xc = X - mu_full2
    Xc_norm = np.linalg.norm(Xc, ord="fro")

    tau = TAU_REL * np.max(np.abs(lam_full))
    rnum = int(np.sum(lam_full > tau))
    ks_valid = [k for k in ks if k <= rnum]

    ek_empirical, ek_theoretical = [], []
    total_var = lam_full.sum()
    for k in ks_valid:
        Uk = U_full[:, :k]
        P = Xc @ Uk
        X_hat = P @ Uk.T + mu_full2
        e_emp = np.linalg.norm(X - X_hat, ord="fro") / Xc_norm
        e_theo = np.sqrt(max(0.0, 1 - explained_variance_ratio(lam_full, k)))
        ek_empirical.append(e_emp)
        ek_theoretical.append(e_theo)

    plt.figure()
    plt.plot(ks_valid, ek_empirical, "o-", label="empirical")
    plt.plot(ks_valid, ek_theoretical, "x--", label="theoretical sqrt(1-EVR_k)")
    plt.xlabel("k")
    plt.ylabel("e_k")
    plt.legend()
    plt.title("Reconstruction error vs k")
    plt.savefig("reconstruction_error.png", dpi=120)
    plt.close()

    return rnum


def task_5_compression(n, m, rnum):
    naive = n * m
    print("\nCompression:")
    for k in [1, 5, 10, 25, 50, 100, 200, min(399, rnum)]:
        stored = m + n * m + k + n * k  # mu + U_k (m*k) + lam (k) + P (n*k); U_k учтено как m*k
        stored = m + m * k + n * k      # mu + U_k + P, без lam (не обязателен для восстановления)
        ratio = naive / stored
        print(f"  k={k:4d}  naive={naive}  stored={stored}  ratio={ratio:.2f}")
    print("Выигрыш (ratio > 1) исчезает примерно при k, для которого "
          "m*k + n*k + m >= n*m, т.е. k >= m*(n-1)/(n+m) (грубая оценка).")


def task_6_recognition(X, y):
    rng = np.random.default_rng(RNG_SEED)
    train_idx, test_idx = [], []
    for person in np.unique(y):
        idx = np.where(y == person)[0]
        idx = rng.permutation(idx)
        train_idx.extend(idx[:8])
        test_idx.extend(idx[8:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mu_full, scale_full, U_full, lam_full = pca_fit(
        X_train, k=min(319, X_train.shape[0] - 1), standardize=False, Gram_trick=True
    )
    tau = TAU_REL * np.max(np.abs(lam_full))
    rnum = int(np.sum(lam_full > tau))

    ks = [1, 2, 5, 10, 25, 50, 75, 100, 150, 200, min(rnum, 319)]
    ks = sorted(set(k for k in ks if 1 <= k <= rnum))
    accs = []
    for k in ks:
        Uk = U_full[:, :k]
        P_train = pca_transform(X_train, mu_full, scale_full, Uk)
        P_test = pca_transform(X_test, mu_full, scale_full, Uk)

        preds = []
        for p in P_test:
            dists = np.linalg.norm(P_train - p[None, :], axis=1)
            preds.append(y_train[np.argmin(dists)])
        preds = np.array(preds)
        acc = np.mean(preds == y_test)
        accs.append(acc)

    print("\nRecognition accuracy vs k:")
    for k, a in zip(ks, accs):
        print(f"  k={k:4d}  acc={a:.4f}")

    plt.figure()
    plt.plot(ks, accs, "o-")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.title("Nearest-neighbor recognition accuracy")
    plt.savefig("recognition_accuracy.png", dpi=120)
    plt.close()
    print(f"RNG seed used: {RNG_SEED}")

    return X_train, y_train, mu_full, scale_full, U_full


def task_7_distance_to_face_space(X_train, mu_full, scale_full, U_full):
    k = 50
    Uk = U_full[:, :k]
    mu_train = mu_full  # уже среднее обучающей части

    def dist(x):
        xc = x - mu_train
        proj = Uk @ (Uk.T @ xc)
        return np.linalg.norm(xc - proj)

    rng = np.random.default_rng(RNG_SEED)

    # "посторонние" изображения: шум и однотонные прямоугольники
    n_outliers = 40
    noise_imgs = rng.uniform(0, 1, size=(n_outliers // 2, IMG[0] * IMG[1]))
    flat_imgs = np.tile(rng.uniform(0, 1, size=(n_outliers // 2, 1)), (1, IMG[0] * IMG[1]))
    outliers = np.vstack([noise_imgs, flat_imgs])

    # часть лиц (не из обучающей выборки для честности можно взять hold-out; тут для примера X_train)
    faces_sample = X_train[:n_outliers]

    d_faces = np.array([dist(x) for x in faces_sample])
    d_outliers = np.array([dist(x) for x in outliers])

    # разбиваем каждую группу на подбор порога / проверку
    half_f, half_o = len(d_faces) // 2, len(d_outliers) // 2
    d_faces_fit, d_faces_val = d_faces[:half_f], d_faces[half_f:]
    d_out_fit, d_out_val = d_outliers[:half_o], d_outliers[half_o:]

    print(f"\nMedian d(x) faces (fit): {np.median(d_faces_fit):.3f}")
    print(f"Median d(x) outliers (fit): {np.median(d_out_fit):.3f}")

    d0 = (np.median(d_faces_fit) + np.median(d_out_fit)) / 2  # простой порог посередине

    rejected_faces = np.mean(d_faces_val > d0)
    accepted_outliers = np.mean(d_out_val <= d0)
    print(f"Threshold d0 = {d0:.3f}")
    print(f"Validation: rejected faces fraction = {rejected_faces:.3f} "
          f"(n_faces_val={len(d_faces_val)})")
    print(f"Validation: outliers accepted as faces fraction = {accepted_outliers:.3f} "
          f"(n_outliers_val={len(d_out_val)})")

    plt.figure()
    plt.hist(d_faces, bins=15, alpha=0.6, label="faces")
    plt.hist(d_outliers, bins=15, alpha=0.6, label="outliers")
    plt.axvline(d0, color="k", ls="--", label="threshold")
    plt.legend()
    plt.xlabel("d(x)")
    plt.title("Distance to face space")
    plt.savefig("distance_to_face_space.png", dpi=120)
    plt.close()


def task_8_timing(X, k=50):
    n, m = X.shape

    t0 = time.perf_counter()
    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / (n - 1)
    eigval, eigvec = np.linalg.eigh(Sigma)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    U_direct = eigvec[:, :k]
    t_direct = time.perf_counter() - t0

    t0 = time.perf_counter()
    G = (Xc @ Xc.T) / (n - 1)
    eigval_g, eigvec_g = np.linalg.eigh(G)
    order_g = np.argsort(eigval_g)[::-1]
    eigval_g, eigvec_g = eigval_g[order_g], eigvec_g[:, order_g]
    lam_g = eigval_g[:k]
    U_gram = (Xc.T @ eigvec_g[:, :k]) / np.sqrt((n - 1) * lam_g)[None, :]
    t_gram = time.perf_counter() - t0

    print(f"\nDirect (Sigma {m}x{m}): {t_direct:.4f} s")
    print(f"Gram trick (G {n}x{n}):  {t_gram:.4f} s")

    # сравнение спектров и направлений с учётом знака
    lam_direct = eigval[:k]
    print("Max |lambda_direct - lambda_gram|:", np.max(np.abs(lam_direct - lam_g)))

    U_direct_s = _fix_sign_local(U_direct)
    U_gram_s = _fix_sign_local(U_gram / np.linalg.norm(U_gram, axis=0, keepdims=True))
    cos_sim = np.abs(np.sum(U_direct_s * U_gram_s, axis=0))
    print("Min |cos angle| between corresponding directions:", cos_sim.min())


def _fix_sign_local(U):
    U = U.copy()
    for j in range(U.shape[1]):
        idx = np.argmax(np.abs(U[:, j]))
        if U[idx, j] < 0:
            U[:, j] = -U[:, j]
    return U


def main():
    data = fetch_olivetti_faces(shuffle=False)
    X, y = data.data, data.target
    print(X.shape, y.shape)

    mu = task_1_mean_face(X)
    mu_full, scale_full, U16, lam = task_2_eigenfaces(X)
    task_3_spectrum(lam)
    rnum = task_4_reconstruction(X, mu_full, scale_full)
    task_5_compression(X.shape[0], X.shape[1], rnum)
    X_train, y_train, mu_train, scale_train, U_train = task_6_recognition(X, y)
    task_7_distance_to_face_space(X_train, mu_train, scale_train, U_train)
    task_8_timing(X, k=50)


if __name__ == "__main__":
    main()