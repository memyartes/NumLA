import ast
import re
import sys
from pathlib import Path

import numpy as np
import pca
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pca import explained_variance_ratio, pca_fit, pca_inverse, pca_transform

TAU_REL = 1e-12
EPS = 1e-8
ORTHO = 1e-10


def _coordinates(X, standardize=False):
    """Exact constant-column rule; retain small but nonzero variation."""
    constant = np.all(X == X[0], axis=0)
    mu = X.mean(axis=0)
    mu[constant] = X[0, constant]
    centered = X - mu
    centered[:, constant] = 0.0
    scale = np.ones(X.shape[1])
    if standardize:
        scale[~constant] = np.sqrt(
            np.sum(centered[:, ~constant] ** 2, axis=0) / (len(X) - 1)
        )
    return mu, scale, centered / scale


def _data(n, m, seed=13):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n, m)) * np.geomspace(0.1, 10, m) + np.arange(m) * 7
    )


def _known_spectrum(values, wide=False, rotate=True):
    """Paired centered rows with prescribed covariance eigenvalues."""
    values = np.asarray(values, dtype=float)
    r = len(values)
    n, m = 2 * r, 2 * r + 3 if wide else r
    if rotate:
        Q, _ = np.linalg.qr(np.random.default_rng(41).normal(size=(m, r)))
    else:
        Q = np.eye(m, r)
    half = np.sqrt((n - 1) * values / 2)[:, None] * Q.T
    return np.vstack((half, -half)), Q


def _assert_contract(X, k, standardize=False, Gram_trick=True, spectrum=None):
    n, m = X.shape
    mu, scale, U, lam = pca_fit(
        X, k, standardize=standardize, Gram_trick=Gram_trick
    )
    for value, shape in ((mu, (m,)), (scale, (m,)), (U, (m, k)), (lam, (m,))):
        assert isinstance(value, np.ndarray)
        assert value.shape == shape
        assert np.all(np.isfinite(value))
    ref_mu, ref_scale, Xs = _coordinates(X, standardize)
    assert_allclose(mu, ref_mu, rtol=1e-12, atol=1e-12)
    assert_allclose(scale, ref_scale, rtol=1e-12, atol=0)
    assert np.all(scale > 0)
    assert np.all(np.diff(lam) <= 0)
    assert np.all(lam >= 0)
    assert np.all(lam[:k] > 0)
    assert np.linalg.norm(U.T @ U - np.eye(k)) <= ORTHO
    for col in U.T:
        assert col[np.argmax(np.abs(col))] > 0

    raw = np.zeros(m)
    if spectrum is None:
        singular_values = np.linalg.svd(Xs, compute_uv=False)
        raw[: len(singular_values)] = singular_values**2 / (n - 1)
    else:
        raw[: len(spectrum)] = spectrum
    tau = TAU_REL * raw.max()
    ref_lam = np.where(raw <= tau, 0.0, raw)
    assert_allclose(lam, ref_lam, rtol=EPS, atol=EPS * raw[0])
    T = np.sum(Xs**2) / (n - 1)
    eta = m * tau / T
    assert abs(lam.sum() - T) / T <= eta + EPS

    P = pca_transform(X, mu, scale, U)
    assert P.shape == (n, k)
    assert np.all(np.isfinite(P))
    assert_allclose(P, Xs @ U, rtol=1e-10, atol=1e-10)
    assert np.linalg.norm(P.T @ P / (n - 1) - np.diag(lam[:k])) / T <= EPS
    covariance_times_U = Xs.T @ (Xs @ U) / (n - 1)
    assert np.linalg.norm(covariance_times_U - U * lam[:k]) / T <= EPS

    X_hat = pca_inverse(P, mu, scale, U)
    assert X_hat.shape == (n, m)
    assert np.all(np.isfinite(X_hat))
    assert_allclose(X_hat, (P @ U.T) * scale + mu, rtol=1e-10, atol=1e-10)
    residual = np.linalg.norm(Xs - P @ U.T) ** 2 / np.linalg.norm(Xs) ** 2
    evr = explained_variance_ratio(lam, k)
    assert abs(evr - ref_lam[:k].sum() / ref_lam.sum()) <= EPS
    assert abs(residual - (1 - evr)) <= eta + EPS
    if k == m:
        assert np.linalg.norm(Xs - P @ U.T) / np.linalg.norm(Xs) <= ORTHO
    return mu, scale, U, lam


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
@pytest.mark.parametrize(
    "n,m,k",
    [
        (45, 8, 3),
        (45, 8, 8),
        (8, 8, 4),
        (12, 25, 6),
        (12, 25, 11),
        (2, 5, 1),
        (5, 1, 1),
    ],
)
def test_numerical_contract(n, m, k, standardize, Gram_trick):
    _assert_contract(_data(n, m), k, standardize, Gram_trick)


@pytest.mark.parametrize("standardize", [False, True])
def test_defaults(standardize):
    X = _data(10, 17)
    default = pca_fit(X, 3, standardize=standardize)
    explicit = pca_fit(X, 3, standardize=standardize, Gram_trick=True)
    for left, right in zip(default, explicit):
        assert_allclose(left, right, rtol=1e-10, atol=1e-10)
    if not standardize:
        for left, right in zip(pca_fit(X, 3), explicit):
            assert_allclose(left, right, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("rows", [1, 4])
def test_transform_on_new_data_uses_supplied_parameters(rows):
    X = np.arange(rows * 3, dtype=float).reshape(rows, 3) + 20
    mu, scale = np.array([2.0, -3.0, 5.0]), np.array([2.0, 0.5, 4.0])
    U = np.array([[0.6, 0.0], [0.8, 0.0], [0.0, 1.0]])
    assert_allclose(
        pca_transform(X, mu, scale, U),
        ((X - mu) / scale) @ U,
        rtol=1e-12,
        atol=1e-12,
    )


def test_inverse_uses_supplied_parameters():
    P = np.array([[2.0, -1.0], [0.0, 0.0], [-3.0, 4.0]])
    mu, scale = np.array([2.0, -3.0, 5.0]), np.array([2.0, 0.5, 4.0])
    U = np.array([[0.6, 0.0], [0.8, 0.0], [0.0, 1.0]])
    expected = np.array(
        [[4.4, -2.2, 1.0], [2.0, -3.0, 5.0], [-1.6, -4.2, 21.0]]
    )
    assert_allclose(
        pca_inverse(P, mu, scale, U), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("factor", [1e-20, 1.0, 1e20])
@pytest.mark.parametrize(
    "k,expected", [(0, 0.0), (1, 0.5), (2, 0.8), (3, 1.0), (5, 1.0)]
)
def test_explained_variance_ratio_independently(k, expected, factor):
    lam = np.array([5.0, 3.0, 2.0, 0.0, 0.0]) * factor
    result = explained_variance_ratio(lam, k)
    assert np.ndim(result) == 0
    assert result == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_exact_constant_columns(standardize, Gram_trick):
    X = np.full((100, 103), 3.14)
    X[:, 0] = np.linspace(-1e-15, 1e-15, len(X))
    X[:, 2] = 0.0
    mu, scale, U, lam = _assert_contract(X, 1, standardize, Gram_trick)
    assert_array_equal(mu[1:], X[0, 1:])
    assert_array_equal(scale[1:], np.ones(X.shape[1] - 1))
    assert_array_equal(X[:, 1:] - mu[1:], np.zeros_like(X[:, 1:]))
    assert np.count_nonzero(lam) == 1
    restored = pca_inverse(pca_transform(X, mu, scale, U), mu, scale, U)
    assert_allclose(restored[:, 1:], X[:, 1:], rtol=0, atol=1e-14)


@pytest.mark.parametrize("Gram_trick", [False, True])
def test_small_nonconstant_feature_is_standardized(Gram_trick):
    X = np.array(
        [[1, 1e-14], [1, -1e-14], [-1, 1e-14], [-1, -1e-14]], dtype=float
    )
    _, scale, _, lam = _assert_contract(X, 2, True, Gram_trick)
    assert_allclose(
        scale, [np.sqrt(4 / 3), 1e-14 * np.sqrt(4 / 3)], rtol=1e-12, atol=0
    )
    assert_allclose(lam, [1, 1], rtol=1e-12, atol=1e-12)


def _assert_explicit_error(action, rank=None):
    with pytest.raises(Exception) as caught:
        action()
    assert not isinstance(
        caught.value,
        (
            TypeError,
            AttributeError,
            NameError,
            IndexError,
            NotImplementedError,
        ),
    )
    message = str(caught.value).strip()
    assert message, "An explicit error must explain the invalid request"
    if rank is not None:
        assert re.search(
            rf"(?<!\d){rank}(?!\d)", message
        ), f"The error must state the allowed component count ({rank}): {message}"


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_rank_deficient_data(wide, Gram_trick):
    X, _ = _known_spectrum([9, 4, 1], wide=wide)
    X = np.column_stack((X, X[:, 0] + X[:, 1]))
    _, _, _, lam = _assert_contract(X, 3, Gram_trick=Gram_trick)
    assert np.count_nonzero(lam) == 3
    assert explained_variance_ratio(lam, 3) == pytest.approx(1, abs=1e-12)
    _assert_explicit_error(
        lambda: pca_fit(X, 4, Gram_trick=Gram_trick), rank=3
    )


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
@pytest.mark.parametrize("value", [0.0, 3.14])
def test_zero_numerical_rank(value, standardize, Gram_trick):
    _assert_explicit_error(
        lambda: pca_fit(
            np.full((10, 13), value),
            1,
            standardize=standardize,
            Gram_trick=Gram_trick,
        ),
        rank=0,
    )


@pytest.mark.parametrize("k", [-1, 0, 4])
def test_invalid_component_count(k):
    _assert_explicit_error(lambda: pca_fit(_data(10, 3), k))


@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (3, 0)])
def test_invalid_sample_or_feature_count(shape):
    _assert_explicit_error(lambda: pca_fit(np.ones(shape), 1))


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
@pytest.mark.parametrize("factor", [1e-4, 1.0, 1e4])
def test_relative_numerical_zero_threshold(wide, Gram_trick, factor):
    X, _ = _known_spectrum([1, 1e-8, 4e-12, 0.25e-12], wide=wide, rotate=False)
    X *= factor
    _, _, _, lam = pca_fit(X, 3, Gram_trick=Gram_trick)
    assert lam.shape == (X.shape[1],)
    assert np.count_nonzero(lam) == 3
    assert_allclose(
        lam[:3], factor**2 * np.array([1, 1e-8, 4e-12]), rtol=1e-4, atol=0
    )
    assert_array_equal(lam[3:], np.zeros(len(lam) - 3))
    _assert_explicit_error(
        lambda: pca_fit(X, 4, Gram_trick=Gram_trick), rank=3
    )


@pytest.mark.parametrize("standardize", [False, True])
def test_gram_matches_direct_full_spectrum_and_subspace(standardize):
    X = _data(12, 25)
    mu_g, s_g, Ug, lg = pca_fit(X, 6, standardize=standardize, Gram_trick=True)
    mu_d, s_d, Ud, ld = pca_fit(
        X, 6, standardize=standardize, Gram_trick=False
    )
    assert_allclose(mu_g, mu_d, rtol=1e-12, atol=1e-12)
    assert_allclose(s_g, s_d, rtol=1e-12, atol=1e-12)
    assert lg.shape == ld.shape == (X.shape[1],)
    assert_allclose(lg, ld, rtol=EPS, atol=EPS * lg[0])
    assert np.linalg.norm(Ug @ Ug.T - Ud @ Ud.T) <= EPS


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_repeated_eigenvalues(k, Gram_trick):
    X, Q = _known_spectrum([9, 9, 2, 1], wide=True)
    _, _, U, _ = _assert_contract(X, k, Gram_trick=Gram_trick)
    if k == 2:
        assert np.linalg.norm(U @ U.T - Q[:, :2] @ Q[:, :2].T) <= EPS


def test_gram_orthonormality_with_small_retained_eigenvalues():
    rng = np.random.default_rng(19)
    left = rng.normal(size=(18, 6))
    left -= left.mean(axis=0)
    left, _ = np.linalg.qr(left)
    right, _ = np.linalg.qr(rng.normal(size=(25, 6)))
    X = (left * np.sqrt(17 * np.geomspace(1, 1e-10, 6))) @ right.T
    _assert_contract(X, 6, Gram_trick=True)


@pytest.fixture
def intercept_eigh(monkeypatch):
    """Observe both allowed backends, including module-level from-imports."""
    import scipy.linalg

    def install(wrapper):
        for module in (np.linalg, scipy.linalg):
            original = module.eigh

            def patched(a, *args, _original=original, **kwargs):
                return wrapper(_original, a, *args, **kwargs)

            source_root = Path(pca.__file__).resolve().parent
            for loaded in list(sys.modules.values()):
                filename = getattr(loaded, "__file__", None)
                if filename and Path(filename).resolve().is_relative_to(
                    source_root
                ):
                    for name, value in list(vars(loaded).items()):
                        if value is original:
                            monkeypatch.setattr(loaded, name, patched)
            monkeypatch.setattr(module, "eigh", patched)

    return install


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
@pytest.mark.parametrize("n,m", [(7, 13), (13, 7), (7, 7)])
@pytest.mark.algorithm
def test_eigh_receives_required_matrix(
    intercept_eigh, n, m, Gram_trick, standardize
):
    X = _data(n, m)
    _, _, Xs = _coordinates(X, standardize)
    expected = (Xs @ Xs.T if Gram_trick and n < m else Xs.T @ Xs) / (n - 1)
    seen = []

    def observe(original, matrix, *args, **kwargs):
        seen.append(np.array(matrix, copy=True))
        return original(matrix, *args, **kwargs)

    intercept_eigh(observe)
    pca_fit(X, 3, standardize=standardize, Gram_trick=Gram_trick)
    assert seen, "PCA must use numpy.linalg.eigh or scipy.linalg.eigh"
    for matrix in seen:
        assert matrix.shape == expected.shape
        assert_allclose(matrix, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("Gram_trick", [False, True])
@pytest.mark.parametrize("relative_value", [-1e-12, 1e-12, -1e-6])
@pytest.mark.algorithm
def test_eigh_threshold_boundary_and_negative_result(
    intercept_eigh, Gram_trick, relative_value
):
    X, _ = _known_spectrum([1, 0.1, 0.01, 0.0], wide=True, rotate=False)
    observed = []

    def perturb(original, matrix, *args, **kwargs):
        lam, vectors = original(matrix, *args, **kwargs)
        lam[np.argmin(lam)] = relative_value * np.max(np.abs(lam))
        observed.append(True)
        return lam, vectors

    intercept_eigh(perturb)
    if relative_value < -TAU_REL:
        _assert_explicit_error(lambda: pca_fit(X, 3, Gram_trick=Gram_trick))
    else:
        _, _, _, lam = pca_fit(X, 3, Gram_trick=Gram_trick)
        assert np.count_nonzero(lam) == 3
        assert_array_equal(lam[3:], np.zeros(len(lam) - 3))
    assert observed


@pytest.mark.algorithm
def test_sign_tie_uses_first_coordinate(intercept_eigh):
    X = np.array([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]])

    def tied(original, matrix, *args, **kwargs):
        return np.array([0.0, 2.0]), np.array(
            [[1.0, -1.0], [1.0, 1.0]]
        ) / np.sqrt(2)

    intercept_eigh(tied)
    _, _, U, _ = pca_fit(X, 1, Gram_trick=False)
    assert_allclose(
        U[:, 0], np.array([1.0, -1.0]) / np.sqrt(2), rtol=0, atol=1e-15
    )


def _mixed_spectrum(values, n, m, seed=71):
    """Mix both observations and features, retaining a known spectrum."""
    rng = np.random.default_rng(seed)
    r = len(values)
    assert r <= min(n - 1, m)
    left = rng.normal(size=(n, r))
    left -= left.mean(axis=0)
    left, _ = np.linalg.qr(left)
    right, _ = np.linalg.qr(rng.normal(size=(m, r)))
    return (left * np.sqrt((n - 1) * np.asarray(values))) @ right.T, right


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("n,m", [(61, 9), (13, 47), (17, 17)])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_seeded_matrices(seed, n, m, standardize, Gram_trick):
    _assert_contract(_data(n, m, seed), 5, standardize, Gram_trick)


SPECTRA = [
    pytest.param([100, 1, 0.9, 0.8, 0.7, 0.6], id="one-dominant-direction"),
    pytest.param(np.geomspace(1, 1e-8, 6), id="eight-orders-of-variance"),
    pytest.param([9, 9, 9, 2, 2, 1], id="two-repeated-groups"),
    pytest.param([1, 1, 1, 1, 1, 1], id="isotropic-subspace"),
    pytest.param([9, 9 - 1e-10, 2, 1, 0.5, 0.1], id="almost-repeated-pair"),
    pytest.param([1, 0.1, 1e-5, 1e-7, 1e-9, 1e-11], id="small-retained-tail"),
    pytest.param([1, 0.1, 1e-5, 1e-7, 1e-14, 1e-15], id="discarded-tail"),
]


@pytest.mark.parametrize("values", SPECTRA)
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_spectrum_zoo(values, wide, Gram_trick):
    X, _ = _mixed_spectrum(values, 18, 31 if wide else 8)
    rank = np.count_nonzero(np.asarray(values) > TAU_REL * max(values))
    _, _, _, lam = _assert_contract(X, rank, Gram_trick=Gram_trick)
    assert np.count_nonzero(lam) == rank
    assert explained_variance_ratio(lam, rank) == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
@pytest.mark.parametrize(
    "operation",
    ["translate", "shuffle-rows", "permute-features", "reflect-features"],
)
def test_geometry_invariances(operation, standardize, Gram_trick):
    X = _data(19, 29)
    _, _, U, lam = pca_fit(
        X.copy(), 4, standardize=standardize, Gram_trick=Gram_trick
    )
    rng = np.random.default_rng(25)
    expected_U = U
    if operation == "translate":
        changed = X + np.arange(X.shape[1]) * 32
    elif operation == "shuffle-rows":
        changed = X[rng.permutation(len(X))]
    elif operation == "permute-features":
        order = rng.permutation(X.shape[1])
        changed, expected_U = X[:, order], U[order]
    else:
        signs = rng.choice([-1.0, 1.0], X.shape[1])
        changed, expected_U = X * signs, U * signs[:, None]
    _, _, changed_U, changed_lam = _assert_contract(
        changed, 4, standardize, Gram_trick
    )
    assert_allclose(changed_lam, lam, rtol=EPS, atol=EPS * lam[0])
    assert (
        np.linalg.norm(changed_U @ changed_U.T - expected_U @ expected_U.T)
        <= EPS
    )


@pytest.mark.parametrize("Gram_trick", [False, True])
def test_rotation_of_feature_space(Gram_trick):
    X = _data(14, 23)
    rotation, _ = np.linalg.qr(np.random.default_rng(51).normal(size=(23, 23)))
    _, _, U, lam = pca_fit(X.copy(), 4, Gram_trick=Gram_trick)
    _, _, rotated_U, rotated_lam = _assert_contract(
        X @ rotation, 4, Gram_trick=Gram_trick
    )
    expected = rotation.T @ U
    assert_allclose(rotated_lam, lam, rtol=EPS, atol=EPS * lam[0])
    assert (
        np.linalg.norm(rotated_U @ rotated_U.T - expected @ expected.T) <= EPS
    )


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
@pytest.mark.parametrize("factor", [-1e40, -1.0, 1e-40, 7.0])
def test_global_change_of_units(factor, standardize, Gram_trick):
    X = _data(15, 27)
    mu, scale, U, lam = pca_fit(
        X.copy(), 4, standardize=standardize, Gram_trick=Gram_trick
    )
    new_mu, new_scale, new_U, new_lam = _assert_contract(
        X * factor, 4, standardize, Gram_trick
    )
    assert_allclose(new_mu / factor, mu, rtol=1e-12, atol=1e-12)
    assert_allclose(
        new_scale / abs(factor) if standardize else new_scale,
        scale,
        rtol=1e-12,
        atol=0,
    )
    expected_lam = new_lam if standardize else new_lam / factor**2
    assert_allclose(expected_lam, lam, rtol=EPS, atol=EPS * lam[0])
    assert np.linalg.norm(new_U @ new_U.T - U @ U.T) <= EPS


@pytest.mark.parametrize("Gram_trick", [False, True])
def test_standardization_removes_individual_positive_units(Gram_trick):
    X = _data(19, 29)
    factors = np.geomspace(1e-8, 1e8, X.shape[1])
    _, _, U, lam = pca_fit(
        X.copy(), 4, standardize=True, Gram_trick=Gram_trick
    )
    _, _, new_U, new_lam = _assert_contract(X * factors, 4, True, Gram_trick)
    assert_allclose(new_lam, lam, rtol=EPS, atol=EPS * lam[0])
    assert np.linalg.norm(new_U @ new_U.T - U @ U.T) <= EPS


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
@pytest.mark.parametrize("repeat", [2, 5])
def test_repeating_observations_and_bessel_correction(
    repeat, standardize, Gram_trick
):
    X = _data(11, 19)
    _, _, U, lam = pca_fit(
        X.copy(), 4, standardize=standardize, Gram_trick=Gram_trick
    )
    repeated = np.repeat(
        X, repeat, axis=0
    )  # Also crosses n=m and switches strategy.
    _, _, new_U, new_lam = _assert_contract(
        repeated, 4, standardize, Gram_trick
    )
    factor = (
        1 if standardize else repeat * (len(X) - 1) / (repeat * len(X) - 1)
    )
    assert_allclose(new_lam, lam * factor, rtol=EPS, atol=EPS * lam[0])
    assert np.linalg.norm(new_U @ new_U.T - U @ U.T) <= EPS


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_adding_mean_observations(standardize, Gram_trick):
    X = _data(13, 23)
    mu, _, U, lam = pca_fit(
        X.copy(), 4, standardize=standardize, Gram_trick=Gram_trick
    )
    augmented = np.vstack((X, np.tile(mu, (12, 1))))
    _, _, new_U, new_lam = _assert_contract(
        augmented, 4, standardize, Gram_trick
    )
    factor = 1 if standardize else (len(X) - 1) / (len(augmented) - 1)
    assert_allclose(new_lam, lam * factor, rtol=EPS, atol=EPS * lam[0])
    assert np.linalg.norm(new_U @ new_U.T - U @ U.T) <= EPS


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_nested_components_and_monotone_training_error(
    standardize, Gram_trick
):
    X = _data(20, 31)
    previous_U, previous_error, previous_evr = None, np.inf, 0.0
    full_lam = None
    for k in (1, 2, 5, 10, 19):
        mu, scale, U, lam = _assert_contract(X, k, standardize, Gram_trick)
        if full_lam is not None:
            assert_allclose(lam, full_lam, rtol=EPS, atol=EPS * lam[0])
            assert np.linalg.norm(previous_U - U @ (U.T @ previous_U)) <= EPS
        Xs = (X - mu) / scale
        P = pca_transform(X, mu, scale, U)
        error = np.linalg.norm(Xs - P @ U.T) / np.linalg.norm(Xs)
        evr = explained_variance_ratio(lam, k)
        assert error <= previous_error + EPS
        assert evr >= previous_evr - EPS
        previous_U, previous_error, previous_evr, full_lam = U, error, evr, lam


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_new_data_batching_and_projection_round_trip(standardize, Gram_trick):
    mu, scale, U, _ = pca_fit(
        _data(23, 31), 5, standardize=standardize, Gram_trick=Gram_trick
    )
    new = _data(7, 31, seed=64) + 50
    P = pca_transform(new, mu, scale, U)
    parts = [
        pca_transform(part, mu, scale, U)
        for part in (new[:1], new[1:3], new[3:])
    ]
    assert_allclose(np.vstack(parts), P, rtol=1e-10, atol=1e-10)
    restored = pca_inverse(P, mu, scale, U)
    assert_allclose(
        pca_transform(restored, mu, scale, U), P, rtol=1e-10, atol=1e-10
    )
    residual = (new - restored) / scale
    assert (
        np.linalg.norm(residual @ U) / np.linalg.norm((new - mu) / scale)
        <= EPS
    )


@pytest.mark.parametrize(
    "layout", ["fortran", "strided", "reversed", "transposed"]
)
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_array_memory_layouts(layout, standardize, Gram_trick):
    X = _data(38, 54)
    if layout == "fortran":
        X = np.asfortranarray(X)
    elif layout == "strided":
        X = X[::2, ::2]
    elif layout == "reversed":
        X = X[::-1, ::-1]
    else:
        X = X.T
    _assert_contract(X, 5, standardize, Gram_trick)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_input_dtypes_computed_in_double_precision(
    dtype, standardize, Gram_trick
):
    X = (
        np.random.default_rng(68)
        .integers(-30, 40, size=(21, 37))
        .astype(dtype)
    )
    mu, scale, U, lam = pca_fit(
        X, 4, standardize=standardize, Gram_trick=Gram_trick
    )
    ref_mu, ref_scale, Xs = _coordinates(X.astype(np.float64), standardize)
    assert_allclose(mu, ref_mu, rtol=1e-12, atol=1e-12)
    assert_allclose(scale, ref_scale, rtol=1e-12, atol=0)
    assert np.linalg.norm(U.T @ U - np.eye(4)) <= ORTHO
    singular = np.linalg.svd(Xs, compute_uv=False)
    expected = singular[:4] ** 2 / (len(X) - 1)
    assert_allclose(lam[:4], expected, rtol=EPS, atol=EPS * expected[0])


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_large_offset_does_not_create_variance(standardize, Gram_trick):
    X = np.random.default_rng(6).integers(-16, 17, size=(32, 15)).astype(float)
    _assert_contract(X + 2.0**35, 5, standardize, Gram_trick)


@pytest.mark.parametrize("n,m", [(41, 9), (13, 37)])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_against_sklearn_full_svd(n, m, standardize, Gram_trick):
    from sklearn.decomposition import PCA

    X = _data(n, m, seed=32)
    mu, scale, U, lam = pca_fit(
        X.copy(), 4, standardize=standardize, Gram_trick=Gram_trick
    )
    ref_mu, ref_scale, Xs = _coordinates(X, standardize)
    model = PCA(n_components=4, svd_solver="full").fit(Xs)
    V = model.components_.T
    assert_allclose(mu, ref_mu, rtol=1e-12, atol=1e-12)
    assert_allclose(scale, ref_scale, rtol=1e-12, atol=0)
    assert_allclose(
        lam[:4], model.explained_variance_, rtol=EPS, atol=EPS * lam[0]
    )
    assert np.linalg.norm(U @ U.T - V @ V.T) <= EPS
    own = pca_inverse(pca_transform(X, mu, scale, U), mu, scale, U)
    library = model.inverse_transform(model.transform(Xs)) * ref_scale + ref_mu
    assert_allclose(own, library, rtol=EPS, atol=EPS)


@pytest.mark.stress
@pytest.mark.parametrize("n,m", [(4096, 24)])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("Gram_trick", [False, True])
def test_large_tall_matrices(n, m, standardize, Gram_trick):
    _assert_contract(_data(n, m, seed=5), 8, standardize, Gram_trick)


@pytest.mark.stress
@pytest.mark.algorithm
@pytest.mark.parametrize("n,m,rank", [(48, 2048, 12)])
def test_large_wide_uses_small_eigenproblem(intercept_eigh, n, m, rank):
    values = np.geomspace(10, 0.01, rank)
    X, _ = _mixed_spectrum(values, n, m)
    seen = []

    def require_small_problem(original, matrix, *args, **kwargs):
        assert matrix.shape == (
            n,
            n,
        ), "Wide PCA must diagonalize the n x n Gram matrix"
        seen.append(matrix.shape)
        return original(matrix, *args, **kwargs)

    intercept_eigh(require_small_problem)
    _, _, _, lam = _assert_contract(X, 8, Gram_trick=True, spectrum=values)
    assert seen
    assert np.count_nonzero(lam) == rank


PCA_SRC = Path(__file__).resolve().parents[1] / "src" / "pca.py"
LINALG_FORBIDDEN = {
    "svd",
    "svdvals",
    "eig",
    "eigvals",
    "eigvalsh",
    "qr",
    "pinv",
    "pinvh",
    "lstsq",
    "matrix_rank",
    "solve",
    "inv",
    "cholesky",
    "schur",
    "orth",
    "eigs",
    "eigsh",
    "svds",
}
READY_MADE_PREFIXES = (
    "sklearn.decomposition",
    "sklearn.preprocessing",
)
READY_MADE_FUNCTIONS = {
    "numpy.cov",
    "numpy.corrcoef",
    "numpy.std",
    "numpy.var",
    "numpy.nanstd",
    "numpy.nanvar",
    "scipy.stats.zscore",
    "scipy.stats.zmap",
}


def _is_forbidden(name):
    parts = name.split(".")
    return (
        name in READY_MADE_FUNCTIONS
        or any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in READY_MADE_PREFIXES
        )
        or (
            parts[0] in {"numpy", "scipy"}
            and "linalg" in parts
            and parts[-1] in LINALG_FORBIDDEN
        )
    )


def _forbidden_references(source):
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.aliases = {}
            self.bad = set()

        def resolve(self, node):
            if isinstance(node, ast.Name):
                return self.aliases.get(node.id, "")
            if isinstance(node, ast.Attribute):
                parent = self.resolve(node.value)
                return f"{parent}.{node.attr}" if parent else ""
            return ""

        def check(self, name):
            if name and _is_forbidden(name):
                self.bad.add(name)

        def visit_Import(self, node):
            for item in node.names:
                self.aliases[item.asname or item.name.split(".")[0]] = (
                    item.name if item.asname else item.name.split(".")[0]
                )
                self.check(item.name)

        def visit_ImportFrom(self, node):
            for item in node.names:
                full = f"{node.module or ''}.{item.name}"
                if item.name == "*":
                    candidates = READY_MADE_FUNCTIONS | {
                        f"{node.module}.{name}"
                        for name in LINALG_FORBIDDEN
                        if "linalg" in (node.module or "").split(".")
                    }
                    for candidate in candidates:
                        module, _, name = candidate.rpartition(".")
                        if module == node.module:
                            self.aliases[name] = candidate
                else:
                    self.aliases[item.asname or item.name] = full
                    self.check(full)

        def visit_Attribute(self, node):
            self.check(self.resolve(node))
            self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                self.check(self.resolve(node))

        def visit_Assign(self, node):
            self.visit(node.value)
            full = self.resolve(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = full

        def visit_FunctionDef(self, node):
            outer = self.aliases.copy()
            args = (
                node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            )
            args += [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
            for arg in args:
                self.aliases[arg.arg] = ""
            self.generic_visit(node)
            self.aliases = outer

        visit_AsyncFunctionDef = visit_FunctionDef

    visitor = Visitor()
    visitor.visit(ast.parse(source))
    return sorted(visitor.bad)


@pytest.mark.algorithm
def test_pca_uses_permitted_tools():
    bad = _forbidden_references(PCA_SRC.read_text(encoding="utf-8"))
    assert not bad, f"Disallowed ready-made operations in src/pca.py: {bad}"


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np; np.linalg.svd(X)",
        "import numpy.linalg as la; la.eig(X)",
        "from scipy import linalg as la; la.qr(X)",
        "from numpy.linalg import svd as decompose; decompose(X)",
        "import numpy as np; la = np.linalg; la.pinv(X)",
        "import scipy.sparse.linalg as la; la.eigsh(X)",
        "import numpy as np; np.cov(X)",
        "from numpy import std as spread; spread(X)",
        "from sklearn.decomposition import PCA as Model; Model()",
        "from sklearn.preprocessing import StandardScaler; StandardScaler()",
        "from scipy.stats import zscore; zscore(X)",
        "from numpy import *; cov(X)",
        "from scipy.linalg import *; svd(X)",
    ],
)
def test_checker_detects_disallowed_operations(source):
    assert _forbidden_references(source)


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np; np.linalg.eigh(X)",
        "from scipy.linalg import eigh as decompose; decompose(X)",
        "from numpy import linalg as la; la.eigh(X); la.norm(X)",
        "import numpy as np; np.mean(X, axis=0); np.sqrt(X); np.sum(X)",
        "from sklearn.datasets import load_breast_cancer, load_digits, fetch_olivetti_faces, make_swiss_roll",
        "from sklearn.model_selection import train_test_split",
        "import matplotlib.pyplot as plt; plt.scatter(x, y)",
        "def svd(X): return X\nsvd(X)",
        "# np.linalg.svd(X)\ntext = 'np.cov(X)'",
        "import numpy as np\ndef helper(np): return np.std()",
        "from numpy import *; mean(X, axis=0)",
    ],
)
def test_checker_accepts_permitted_code(source):
    assert _forbidden_references(source) == []
