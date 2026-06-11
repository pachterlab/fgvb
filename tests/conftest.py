"""Shared pytest configuration for the fgvb test suite.

Forces a headless Matplotlib backend so the figure-returning report methods can be
exercised without a display, and provides small deterministic data fixtures used
by the unit tests.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display needed to build report figures

import numpy as np
import pandas as pd
import pytest


def make_dataset(n: int = 200, d: int = 4, seed: int = 0):
    """A small, fully deterministic dataset with known group structure.

    The latent space is a linear mixture of three feature groups
    (``shape`` / ``texture`` / ``intensity``) plus additive noise, so the
    decomposition should recover a high ``rho`` and attribute most variance to
    the groups that drive it.  Returns ``(df_latent, df_feat, feat_to_group)``.
    """
    rng = np.random.default_rng(seed)
    feature_names = [
        "shape_0", "shape_1", "shape_2",
        "texture_0", "texture_1",
        "intensity_0", "intensity_1",
    ]
    F = rng.normal(size=(n, len(feature_names)))
    df_feat = pd.DataFrame(F, columns=feature_names)
    feat_to_group = {name: name.split("_")[0] for name in feature_names}

    mixing = rng.normal(size=(len(feature_names), d))
    Z = F @ mixing + 0.5 * rng.normal(size=(n, d))
    df_latent = pd.DataFrame(Z, columns=[f"z{i}" for i in range(d)])
    return df_latent, df_feat, feat_to_group


@pytest.fixture
def dataset():
    """``(df_latent, df_feat, feat_to_group)`` for a small deterministic problem."""
    return make_dataset()


@pytest.fixture
def explainer(dataset):
    """A fitted-on-demand :class:`FeatureGroundedDecomposition` over ``dataset``."""
    from fgvb import FeatureGroundedDecomposition

    df_latent, df_feat, feat_to_group = dataset
    return FeatureGroundedDecomposition(df_latent, df_feat, feat_to_group, seed=0)
