import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.datasets import load_digits

from pca import pca_fit, pca_transform, explained_variance_ratio


def task_1_scatter_and_distances(X, y):
    mu, scale, U, lam = pca_fit(X, k=2, standardize=False, Gram_trick=False)
    P = pca_transform(X, mu, scale, U)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(P[:, 0], P[:, 1], c=y, cmap="tab10", s=10, alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="digit", loc="best", fontsize=8)
    plt.xlabel("P1")
    plt.ylabel("P2")
    plt.title("Digits PCA scatter (2D)")
    plt.savefig("digits_scatter_2d.png", dpi=120)
    plt.close()

    # матрица попарных расстояний между средними по классам в 2D-проекции
    classes = np.unique(y)
    centroids = np.array([P[y == c].mean(axis=0) for c in classes])
    dist_matrix = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :], axis=-1
    )
    print("Pairwise centroid distances (2D projection):")
    header = "     " + " ".join(f"{c:5d}" for c in classes)
    print(header)
    for c, row in zip(classes, dist_matrix):
        print(f"{c:3d}  " + " ".join(f"{v:5.2f}" for v in row))

    return dist_matrix


def task_2_dimension(X):
    m = X.shape[1]
    mu, scale, U, lam = pca_fit(X, k=m, standardize=False, Gram_trick=False)
    evr = np.array([explained_variance_ratio(lam, k) for k in range(1, m + 1)])
    k90 = int(np.searchsorted(evr, 0.90) + 1)
    k95 = int(np.searchsorted(evr, 0.95) + 1)
    print(f"\nk for 90% EVR: {k90} (of {m} original dims)")
    print(f"k for 95% EVR: {k95} (of {m} original dims)")
    print("Это число линейных компонент, сохраняющих заданную долю ДИСПЕРСИИ, "
          "а не истинная (возможно нелинейная) размерность многообразия, на "
          "котором лежат данные -- PCA видит только линейные корреляции.")
    return k90, k95


def task_3_3d_projection(X, y):
    mu, scale, U, lam = pca_fit(X, k=3, standardize=False, Gram_trick=False)
    P = pca_transform(X, mu, scale, U)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=y, cmap="tab10", s=10, alpha=0.7)
    ax.set_xlabel("P1")
    ax.set_ylabel("P2")
    ax.set_zlabel("P3")
    ax.set_title("Digits PCA scatter (3D)")
    plt.savefig("digits_scatter_3d.png", dpi=120)
    plt.close()
    print("\nСравните digits_scatter_2d.png и digits_scatter_3d.png визуально: "
          "третье измерение обычно лучше разделяет пары цифр, которые в 2D "
          "накладываются (например, часто 3/5/8 или 4/9). Наблюдение по "
          "картинке -- не строгий вывод о качестве классификации.")


def main():
    digits = load_digits()
    X, y = digits.data, digits.target
    print(X.shape, y.shape)

    task_1_scatter_and_distances(X, y)
    task_2_dimension(X)
    task_3_3d_projection(X, y)


if __name__ == "__main__":
    main()