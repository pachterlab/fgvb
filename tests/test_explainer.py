"""Unit tests for the dataframe-first API :class:`FeatureGroundedDecomposition`.

These cover the public surface used in the tutorial: the headline scalars, the
SHAP and Johnson budget tables (and the additive identities that every row sums
to 1), the object-level vectors, the override hooks, input alignment, and the
figure-returning report methods.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fgvb import FeatureGroundedDecomposition
from fgvb.explainer import _NOISE_COL, _WEIGHTED_SUM_LABEL


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------- scalars
def test_rho_in_unit_interval(explainer):
    assert 0.0 <= explainer.rho <= 1.0
    assert explainer.rho > 0.8  # mixture data is mostly explained
    assert explainer.rho_cv <= explainer.rho + 1e-9


def test_repr_reports_shapes_and_fit_state(dataset):
    df_latent, df_feat, feat_to_group = dataset
    exp = FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group)
    assert "fitted=False" in repr(exp)
    _ = exp.rho  # triggers the lazy fit
    r = repr(exp)
    assert "fitted=True" in r
    assert f"n={len(df_latent)}" in r


# ---------------------------------------------------------------- SHAP table
def test_shap_table_shape_and_rows_sum_to_one(explainer, dataset):
    df_latent, df_feat, _ = dataset
    table, rho, fig = explainer.shap()
    d = df_latent.shape[1]
    g = len(explainer.group_names)
    assert table.shape == (d + 1, g + 1)
    assert _NOISE_COL in table.columns
    assert _WEIGHTED_SUM_LABEL in table.index
    assert np.allclose(table.sum(axis=1), 1.0)
    assert rho == pytest.approx(explainer.rho)
    assert fig is not None


def test_shap_noise_matches_r2(explainer):
    table, _, _ = explainer.shap()
    assert np.allclose(explainer.r2.values, 1.0 - table[_NOISE_COL].values)


def test_shap_override_grouping_recomputes(explainer):
    """Passing ``feat_to_group`` regroups for that call only."""
    base, _, _ = explainer.shap()
    # Pool everything into a single group: that column should carry the whole R^2.
    one = {f: "all" for f in explainer.feature_names}
    table, _, _ = explainer.shap(feat_to_group=one)
    assert list(table.columns) == ["all", _NOISE_COL]
    assert np.allclose(table.sum(axis=1), 1.0)
    # The grouping fixed at construction is unchanged afterwards.
    again, _, _ = explainer.shap()
    assert list(again.columns) == list(base.columns)


def test_shap_heatmap_figure_type(explainer):
    _, _, fig = explainer.shap(figure_type="heatmap")
    assert fig is not None


def test_shap_invalid_figure_type_raises(explainer):
    with pytest.raises(ValueError, match="figure_type"):
        explainer.shap(figure_type="bogus")


def test_shap_many_groups_warns(dataset):
    """Attributing to more than the warn threshold of groups warns the user."""
    df_latent, df_feat, _ = dataset
    # Each feature its own group is < threshold; build a wide one to exceed it.
    rng = np.random.default_rng(0)
    F = rng.normal(size=(150, 15))
    names = [f"f{i}" for i in range(15)]
    df_f = pd.DataFrame(F, columns=names)
    Z = F @ rng.normal(size=(15, 3)) + rng.normal(size=(150, 3))
    df_l = pd.DataFrame(Z, columns=[f"z{i}" for i in range(3)])
    exp = FeatureGroundedDecomposition(df_l, df_f)  # 15 singleton groups
    with pytest.warns(UserWarning, match="exponential"):
        exp.shap()


# ---------------------------------------------------------------- Johnson table
def test_johnson_table_shape_and_rows_sum_to_one(explainer, dataset):
    df_latent, df_feat, _ = dataset
    table, rho, fig = explainer.johnson()
    d = df_latent.shape[1]
    q = df_feat.shape[1]
    assert table.shape == (d + 1, q + 1)
    assert np.allclose(table.sum(axis=1), 1.0)
    assert np.allclose(explainer.r2.values, 1.0 - table[_NOISE_COL].values)
    assert rho == pytest.approx(explainer.rho)
    assert fig is not None


def test_johnson_bar_figure_type(explainer):
    _, _, fig = explainer.johnson(figure_type="bar", top_n=3)
    assert fig is not None


# ---------------------------------------------------------------- vectors
def test_variance_fraction_and_r2_vectors(explainer, dataset):
    d = dataset[0].shape[1]
    vf = explainer.variance_fraction
    r2 = explainer.r2
    assert len(vf) == d + 1
    assert len(r2) == d + 1
    assert vf[_WEIGHTED_SUM_LABEL] == pytest.approx(1.0)  # all directions kept
    assert r2[_WEIGHTED_SUM_LABEL] == pytest.approx(explainer.rho)


def test_directions_and_eigh_internals(explainer, dataset):
    d = dataset[0].shape[1]
    assert len(explainer.directions) == d
    assert explainer.eigvals.shape == (d,)
    assert explainer.eigvecs.shape == (d, d)
    # Eigenvalues come out in descending order.
    assert np.all(np.diff(explainer.eigvals) <= 1e-9)


# ---------------------------------------------------------------- global budget / scree
def test_global_budget_group_and_feature(explainer):
    assert explainer.global_budget(by="group") is not None
    assert explainer.global_budget(by="feature", top_n=4) is not None


def test_global_budget_invalid_by_raises(explainer):
    with pytest.raises(ValueError, match="group.*feature|by must"):
        explainer.global_budget(by="bogus")


def test_scree_returns_figure(explainer):
    assert explainer.scree() is not None


# ---------------------------------------------------------------- modes / n_directions
def test_axes_mode_uses_column_labels(dataset):
    df_latent, df_feat, feat_to_group = dataset
    exp = FeatureGroundedDecomposition(
        df_latent, df_feat, feat_to_group, mode="axes",
    )
    table, _, _ = exp.shap()
    dir_rows = [i for i in table.index if i != _WEIGHTED_SUM_LABEL]
    assert dir_rows == [str(c) for c in df_latent.columns]


def test_n_directions_limits_rows(dataset):
    df_latent, df_feat, feat_to_group = dataset
    exp = FeatureGroundedDecomposition(
        df_latent, df_feat, feat_to_group, n_directions=2,
    )
    table, _, _ = exp.shap()
    assert table.shape[0] == 2 + 1  # 2 directions + weighted_sum


# ---------------------------------------------------------------- grouping defaults
def test_default_grouping_is_per_feature(dataset):
    """``feat_to_group=None`` makes each feature its own singleton group."""
    df_latent, df_feat, _ = dataset
    exp = FeatureGroundedDecomposition(df_latent, df_feat)
    assert set(exp.group_names) == set(df_feat.columns)


def test_ungrouped_features_warn(dataset):
    """Features missing from the mapping land in an 'ungrouped' group with a warning."""
    df_latent, df_feat, feat_to_group = dataset
    partial = dict(feat_to_group)
    dropped = df_feat.columns[0]
    del partial[dropped]
    with pytest.warns(UserWarning, match="ungrouped"):
        exp = FeatureGroundedDecomposition(df_latent, df_feat, partial)
    assert "ungrouped" in exp.group_names


# ---------------------------------------------------------------- alignment
def test_align_by_shared_index():
    rng = np.random.default_rng(0)
    idx = [f"s{i}" for i in range(40)]
    df_l = pd.DataFrame(rng.normal(size=(40, 3)), index=idx)
    df_f = pd.DataFrame(rng.normal(size=(40, 2)), index=idx[::-1])  # same labels, shuffled
    exp = FeatureGroundedDecomposition(df_l, df_f)
    # Rows are aligned by label, so the two frames share an index after construction.
    assert exp.df_latent.index.equals(exp.df_feat.index)


def test_align_partial_overlap_warns():
    rng = np.random.default_rng(1)
    df_l = pd.DataFrame(rng.normal(size=(30, 2)), index=[f"s{i}" for i in range(30)])
    df_f = pd.DataFrame(rng.normal(size=(30, 2)), index=[f"s{i}" for i in range(10, 40)])
    with pytest.warns(UserWarning, match="share only"):
        exp = FeatureGroundedDecomposition(df_l, df_f)
    assert len(exp.df_latent) == 20  # the intersection


def test_align_no_shared_index_assumes_order():
    rng = np.random.default_rng(2)
    df_l = pd.DataFrame(rng.normal(size=(25, 2)), index=range(25))
    df_f = pd.DataFrame(rng.normal(size=(25, 2)), index=range(100, 125))
    with pytest.warns(UserWarning, match="same order"):
        exp = FeatureGroundedDecomposition(df_l, df_f)
    assert len(exp.df_latent) == 25


def test_align_mismatched_length_raises():
    rng = np.random.default_rng(3)
    df_l = pd.DataFrame(rng.normal(size=(20, 2)), index=range(20))
    df_f = pd.DataFrame(rng.normal(size=(15, 2)), index=range(100, 115))
    with pytest.raises(ValueError, match="rows"):
        FeatureGroundedDecomposition(df_l, df_f)
