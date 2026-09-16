import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from pca import pca_fit, pca_transform, pca_inverse, explained_variance_ratio

RNG_SEED = 42


def print_top_weights(u, feature_names, title, topn=5):
    order = np.argsort(-np.abs(u))[:topn]
    print(f"\n{title}")
    for idx in order:
        print(f"  {feature_names[idx]:35s} {u[idx]: .4f}")


def task_1_2(X, feature_names):
    """Два режима PCA + диагноз первой компоненты."""
    results = {}
    for standardize in (False, True):
        mu, scale, U, lam = pca_fit(X, k=3, standardize=standardize, Gram_trick=False)
        evr1 = explained_variance_ratio(lam, 1)
        evr2 = explained_variance_ratio(lam, 2)
        evr3 = explained_variance_ratio(lam, 3)
        label = "standardized" if standardize else "raw"
        results[label] = dict(mu=mu, scale=scale, U=U, lam=lam)

        print(f"\n=== Regime: {label} ===")
        print("lambda[:10] =", lam[:10])
        print(f"EVR1={evr1:.4f}  EVR2={evr2:.4f}  EVR3={evr3:.4f}")
        print_top_weights(U[:, 0], feature_names, "u1 top weights")
        print_top_weights(U[:, 1], feature_names, "u2 top weights")

        # scree plot
        plt.figure()
        plt.plot(np.arange(1, len(lam) + 1), lam, "o-")
        plt.xlabel("i")
        plt.ylabel("lambda_i")
        plt.title(f"Scree plot ({label})")
        plt.yscale("log")
        plt.savefig(f"scree_{label}.png", dpi=120)
        plt.close()

    # --- Диагноз: без стандартизации первая компонента определяется
    # признаками с наибольшей абсолютной дисперсией (обычно площадь / периметр).
    U_raw = results["raw"]["U"]
    u1 = U_raw[:, 0]
    top2 = np.argsort(-np.abs(u1))[:2]
    print("\nДва признака с наибольшими |весами| в u1 (raw):")
    for idx in top2:
        w = u1[idx]
        e_j = np.zeros_like(u1)
        e_j[idx] = 1.0
        cos_angle = abs(u1 @ e_j)  # = |w|, т.к. e_j -- орт
        print(f"  {feature_names[idx]:35s} weight={w: .4f}  |u1^T e_j|={cos_angle:.4f}")

    frac = np.sum(u1[top2] ** 2) / np.sum(u1 ** 2)  # ||u1||=1, поэтому знаменатель = 1
    print(f"Доля кв. нормы u1, приходящаяся на эти 2 признака: {frac:.4f}")
    print("(u1 близка к плоскости этих двух признаков, но не совпадает ни с одной "
          "координатной осью, т.к. остальные веса ненулевые, хоть и малы)")

    return