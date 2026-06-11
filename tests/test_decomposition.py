"""Unit tests for the core decomposition math in :mod:`fgvb.decomposition`.

These exercise the linear-algebra primitives directly (subset ``R^2``, the group
Shapley game, Johnson relative weights, cross-validated ``R^2``) and the
:func:`decompose_latent_space` entry point, checking the additive identities the
method guarantees and that it recovers known structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from fgvb.decomposition import (
    DecompositionResult,
    _GramCache,
    _shapley_groups,
    cv_r2,
    decompose_latent_space,
    feature_grounded_codes,
    relative_weights,
)


def _standardize(F):
    Fmu, Fsd = F.mean(0), F.std(0)
    Fsd[Fsd < 1e-9] = 1.0
    return (F - Fmu) / Fsd


# --------------------------------------------------------------------- _GramCache
def test_gram_r2_perfect_fit():
    """A score that is an exact linear combination of the features has R^2 == 1."""
    rng = np.random.default_rng(0)
    X = _standardize(rng.normal(size=(300, 4)))
    s = X @ np.array([1.0, -2.0, 0.5, 3.0])
    s = s - s.mean()
    cache = _GramCache.build(X)
    r2 = cache.r2(X.T @ s, float(s @ s), np.arange(4))
    # The tiny ridge load shrinks a perfect fit by a hair below 1.
    assert r2 == pytest.approx(1.0, abs=1e-4)


def test_gram_r2_pure_noise_is_low():
    """A target independent of the features explains almost nothing."""
    rng = np.random.default_rng(1)
    X = _standardize(rng.normal(size=(500, 3)))
    s = rng.normal(size=500)
    s = s - s.mean()
    cache = _GramCache.build(X)
    r2 = cache.r2(X.T @ s, float(s @ s), np.arange(3))
    assert 0.0 <= r2 < 0.1


def test_gram_r2_clipped_to_unit_interval():
    """Empty subsets and degenerate targets return 0, never out of [0, 1]."""
    rng = np.random.default_rng(2)
    X = _standardize(rng.normal(size=(100, 3)))
    cache = _GramCache.build(X)
    s = rng.normal(size=100)
    assert cache.r2(X.T @ s, float(s @ s), np.array([], dtype=int)) == 0.0
    assert cache.r2(X.T @ s, 0.0, np.arange(3)) == 0.0


# --------------------------------------------------------------------- Shapley
def test_shapley_sums_to_full_model_r2():
    """The group Shapley values are an exact additive split of the full R^2."""
    rng = np.random.default_rng(3)
    X = _standardize(rng.normal(size=(400, 6)))
    s = X @ rng.normal(size=6) + 0.3 * rng.normal(size=400)
    s = s - s.mean()
    cache = _GramCache.build(X)
    syy = float(s @ s)
    c = X.T @ s
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    phi = _shapley_groups(cache, c, syy, groups)
    full = cache.r2(c, syy, np.arange(6))
    assert phi.sum() == pytest.approx(full, abs=1e-8)


def test_shapley_attributes_to_driving_group():
    """The group that actually drives the score gets the largest Shapley share."""
    rng = np.random.default_rng(4)
    X = _standardize(rng.normal(size=(400, 6)))
    # Score driven only by the first group (cols 0,1).
    s = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=400)
    s = s - s.mean()
    cache = _GramCache.build(X)
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    phi = _shapley_groups(cache, X.T @ s, float(s @ s), groups)
    assert phi.argmax() == 0
    assert phi[0] > phi[1] + phi[2]


# --------------------------------------------------------------------- Johnson
def test_relative_weights_sum_to_r2_and_nonneg():
    """Johnson weights are non-negative and sum to the regression R^2."""
    rng = np.random.default_rng(5)
    X = _standardize(rng.normal(size=(400, 5)))
    y = X @ rng.normal(size=5) + 0.4 * rng.normal(size=400)
    eps, r2 = relative_weights(X, y)
    assert np.all(eps >= -1e-12)
    assert eps.sum() == pytest.approx(r2, abs=1e-6)
    # The reported R^2 agrees with a direct OLS R^2.
    cache = _GramCache.build(X)
    yc = y - y.mean()
    direct = cache.r2(X.T @ yc, float(yc @ yc), np.arange(5))
    assert r2 == pytest.approx(direct, abs=1e-6)


def test_relative_weights_constant_target():
    """A constant (zero-variance) target yields zero weights and zero R^2."""
    rng = np.random.default_rng(6)
    X = _standardize(rng.normal(size=(50, 3)))
    eps, r2 = relative_weights(X, np.full(50, 7.0))
    assert r2 == 0.0
    assert np.allclose(eps, 0.0)


# --------------------------------------------------------------------- CV R^2
def test_cv_r2_signal_vs_noise():
    """Cross-validated R^2 is high for real signal and near zero for noise."""
    rng = np.random.default_rng(7)
    X = _standardize(rng.normal(size=(300, 4)))
    y_signal = X @ np.array([1.0, 0.5, -1.0, 2.0]) + 0.2 * rng.normal(size=300)
    y_noise = rng.normal(size=300)
    assert cv_r2(X, y_signal, folds=5, seed=0) > 0.8
    assert cv_r2(X, y_noise, folds=5, seed=0) < 0.1


# --------------------------------------------------------- decompose_latent_space
def test_decompose_basic_shapes_and_identities(dataset):
    """Every direction's group shares plus noise equal its variance (sum to R^2)."""
    df_latent, df_feat, feat_to_group = dataset
    groups = {}
    for f, g in feat_to_group.items():
        groups.setdefault(g, []).append(f)
    res = decompose_latent_space(
        df_latent.values, df_feat.values, list(df_feat.columns), groups, seed=0,
    )
    assert isinstance(res, DecompositionResult)
    assert len(res.directions) == df_latent.shape[1]
    for d in res.directions:
        assert sum(d.group_shares.values()) == pytest.approx(d.r2, abs=1e-8)
        assert d.noise_frac == pytest.approx(1.0 - d.r2, abs=1e-12)
        assert 0.0 <= d.r2 <= 1.0
    # Variance fractions sum to 1 across all directions.
    assert sum(d.variance_frac for d in res.directions) == pytest.approx(1.0)


def test_decompose_recovers_high_rho(dataset):
    """A latent that is a linear mixture of the features is mostly explained."""
    df_latent, df_feat, feat_to_group = dataset
    groups = {}
    for f, g in feat_to_group.items():
        groups.setdefault(g, []).append(f)
    res = decompose_latent_space(
        df_latent.values, df_feat.values, list(df_feat.columns), groups, seed=0,
    )
    assert res.global_r2 > 0.8
    assert res.global_r2_cv > 0.7
    # Global group shares plus noise partition the whole budget.
    total = sum(res.global_group_shares.values()) + res.global_noise
    assert total == pytest.approx(1.0, abs=1e-8)


def test_decompose_n_directions_limits():
    """``n_directions`` keeps only the top-k directions by variance."""
    rng = np.random.default_rng(8)
    Z = rng.normal(size=(150, 5))
    F = rng.normal(size=(150, 4))
    names = [f"f{i}" for i in range(4)]
    groups = {n: [n] for n in names}
    res = decompose_latent_space(Z, F, names, groups, n_directions=2, seed=0)
    assert len(res.directions) == 2


def test_decompose_axes_mode_matches_columns():
    """``mode='axes'`` analyses the raw latent columns rather than eigenvectors."""
    rng = np.random.default_rng(9)
    Z = rng.normal(size=(120, 3))
    F = rng.normal(size=(120, 3))
    names = [f"f{i}" for i in range(3)]
    groups = {n: [n] for n in names}
    res = decompose_latent_space(Z, F, names, groups, mode="axes", seed=0)
    assert res.mode == "axes"
    assert len(res.directions) == 3


def test_decompose_invalid_mode_raises():
    rng = np.random.default_rng(10)
    Z = rng.normal(size=(50, 2))
    F = rng.normal(size=(50, 2))
    names = ["a", "b"]
    with pytest.raises(ValueError, match="unknown mode"):
        decompose_latent_space(Z, F, names, {"a": ["a"], "b": ["b"]}, mode="bogus")


def test_decompose_bootstrap_ci():
    """``n_bootstrap`` attaches an ordered 95% CI around each direction's R^2."""
    rng = np.random.default_rng(11)
    F = rng.normal(size=(200, 3))
    Z = F @ rng.normal(size=(3, 2)) + 0.3 * rng.normal(size=(200, 2))
    names = [f"f{i}" for i in range(3)]
    groups = {n: [n] for n in names}
    res = decompose_latent_space(Z, F, names, groups, n_bootstrap=25, seed=0)
    for d in res.directions:
        assert d.r2_ci is not None
        lo, hi = d.r2_ci
        assert lo <= hi


def test_summary_table_columns(dataset):
    df_latent, df_feat, feat_to_group = dataset
    groups = {}
    for f, g in feat_to_group.items():
        groups.setdefault(g, []).append(f)
    res = decompose_latent_space(
        df_latent.values, df_feat.values, list(df_feat.columns), groups, seed=0,
    )
    table = res.summary_table()
    assert len(table) == len(res.directions)
    for col in ("direction", "variance_frac", "R2", "R2_cv", "noise"):
        assert col in table.columns
    for g in res.group_names:
        assert f"grp_{g}" in table.columns


# --------------------------------------------------- feature_grounded_codes
def test_feature_grounded_codes_shape_and_reconstruction(dataset):
    """The feature-grounded projection has the same shape and is close when R^2 high."""
    df_latent, df_feat, feat_to_group = dataset
    groups = {}
    for f, g in feat_to_group.items():
        groups.setdefault(g, []).append(f)
    res = decompose_latent_space(
        df_latent.values, df_feat.values, list(df_feat.columns), groups, seed=0,
    )
    zhat = feature_grounded_codes(df_latent.values, df_feat.values, res)
    assert zhat.shape == df_latent.values.shape
    # Since the latent is mostly feature-driven, the reconstruction tracks it.
    num = np.var(df_latent.values - zhat)
    den = np.var(df_latent.values)
    assert num / den < 0.5
