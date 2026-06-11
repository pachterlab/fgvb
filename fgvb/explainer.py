"""Dataframe-first API for feature-grounded variance decomposition.

This module wraps the core method in :mod:`decomposition` behind a single class
so that the analysis can be applied to any latent space in a few lines.  All
inputs are pandas objects:

* ``df_latent`` -- (n x d) latent codes, one column per latent dimension.
* ``df_feat``   -- (n x q) interpretable features, one column per feature.
* ``feat_to_group`` -- mapping ``feature name -> group name`` (q -> g).

The class fits the decomposition once and exposes three reports.  Each latent
direction is a principal direction of the latent covariance (``mode="eigen"``)
or a raw latent axis (``mode="axes"``).  For every direction the fraction of its
variance the features explain is ``R^2``; the remainder ``1 - R^2`` is the
irreducible (unexplained) part.  Weighting directions by their variance gives
the headline scalar ``rho``, the fraction of the whole latent space explained.

* :meth:`scree`   -- variance carried by each latent direction (figure only).
* :meth:`shap`    -- group-level Shapley budget: a ``(d+1) x (g+1)`` table, the
  scalar ``rho``, and a stacked-bar figure.  The ``g`` group columns are the
  Shapley shares of ``R^2`` (collinearity-aware, non-negative); the final
  ``noise`` column is the unexplained fraction ``1 - R^2``.  Groups plus noise
  partition each direction's variance, so **every row sums to 1**.  The final row
  is the eigenvalue-weighted sum across directions (the global budget).
* :meth:`johnson` -- per-feature Johnson (2000) relative weights: a
  ``(d+1) x (q+1)`` table, the scalar ``rho``, and a heatmap figure.  Same layout
  as :meth:`shap` but the attribution is refined to individual features.
* :meth:`global_budget` -- the whole latent space's budget as a single stacked
  bar (the ``weighted_sum`` row of the :meth:`shap` or :meth:`johnson` table),
  ``by="group"`` or ``by="feature"`` (figure only).

Both attributions are exact additive decompositions of the same ``R^2``: the
group Shapley shares and the per-feature relative weights each sum to ``R^2`` for
every direction, so the two tables are consistent refinements of one another.

The budget tables carry only quantities on the share scale (group/feature shares
and noise, summing to 1).  The two quantities on a different scale -- the
eigenvalue *weight* of each direction and its total explained fraction ``R^2`` --
are shared across both views and are exposed as object-level vectors,
:attr:`variance_fraction` and :attr:`r2`, rather than duplicated as columns in
each table.
"""

from __future__ import annotations

import logging
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from pydantic import ConfigDict, validate_call

from . import viz
from .decomposition import DecompositionResult, decompose_latent_space

logger = logging.getLogger(__name__)


class Report(NamedTuple):
    """Return value of :meth:`FeatureGroundedDecomposition.shap` / ``.johnson``.

    Unpack as ``table, rho, fig = explainer.shap()`` or access by name.
    """

    table: pd.DataFrame   # (d+1) x (g+1) [shap] or (d+1) x (q+1) [johnson]
    rho: float            # fraction of total latent variance explained by features
    figure: object        # matplotlib Figure


_WEIGHTED_SUM_LABEL = "weighted_sum"
_NOISE_COL = "noise"

# Above this group count the exact 2^G Shapley game is slow and the stacked-bar
# figure becomes unreadable; shap() warns and suggests pooling features.
_SHAP_GROUP_WARN = 12

class FeatureGroundedDecomposition:
    """Feature-grounded variance decomposition of a latent space.

    Parameters
    ----------
    df_latent : DataFrame, shape (n, d)
        Latent codes, one row per sample and one column per latent dimension.
    df_feat : DataFrame, shape (n, q)
        Interpretable features, one row per sample and one column per feature.
        Must describe the same samples as ``df_latent`` (aligned by index if both
        carry a shared index, otherwise assumed to be in the same row order).
    feat_to_group : dict, optional
        Maps each feature name (a column of ``df_feat``) to its group name.
        Features absent from this mapping are collected into an ``"ungrouped"``
        group with a warning.  ``None`` (the default) makes each feature its own
        singleton group, reducing the group budget to a per-feature one -- handy
        when groups are not needed or when there are few enough features to treat
        each individually.
    mode : {"eigen", "axes"}, default "eigen"
        ``"eigen"`` analyses the principal directions of the latent covariance;
        ``"axes"`` analyses the raw latent columns of ``df_latent``.
    n_directions : int, optional
        Analyse only the top-``k`` directions by variance.  ``None`` uses all
        ``d`` directions.
    cv_folds : int, default 5
        Folds for the cross-validated ``R^2`` (exposed via :attr:`rho_cv`; the
        report tables use the in-sample ``R^2`` so that the additive budgets are
        exact).
    seed : int, default 0
        Random seed for the cross-validation fold assignment.

    Notes
    -----
    The decomposition is computed lazily on first use and cached.  Its quantities
    are exposed directly on the object: the headline scalars :attr:`rho` and
    :attr:`rho_cv`, the per-direction vectors :attr:`variance_fraction` and
    :attr:`r2`, and the raw internals :attr:`directions`, :attr:`eigvals`,
    :attr:`eigvecs`, :attr:`global_group_shares`, and :attr:`global_noise`.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        df_latent: pd.DataFrame,
        df_feat: pd.DataFrame,
        feat_to_group: dict | None = None,
        *,
        mode: str = "eigen",
        n_directions: int | None = None,
        cv_folds: int = 5,
        seed: int = 0,
    ):
        df_latent, df_feat = self._align(df_latent, df_feat)

        self.df_latent = df_latent
        self.df_feat = df_feat
        self.feat_to_group = dict(feat_to_group) if feat_to_group is not None else None
        self.mode = mode
        self.n_directions = n_directions
        self.cv_folds = cv_folds
        self.seed = seed

        self.feature_names = list(df_feat.columns)
        self.groups = self._invert_mapping(self.feature_names, feat_to_group)
        self.group_names = list(self.groups.keys())

        # Labels for the latent directions (rows of the report tables).
        if mode == "axes":
            self._dir_labels = [str(c) for c in df_latent.columns]
        else:
            self._dir_labels = None  # filled in after fitting (PC0, PC1, ...)

        self._result: DecompositionResult | None = None

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _align(df_latent: pd.DataFrame, df_feat: pd.DataFrame):
        """Align the two frames by shared index, or by row order as a fallback."""
        if df_latent.index.equals(df_feat.index):
            return df_latent, df_feat
        common = df_latent.index.intersection(df_feat.index)
        if len(common) > 0:
            if len(common) < max(len(df_latent), len(df_feat)):
                warnings.warn(
                    f"df_latent and df_feat share only {len(common)} of "
                    f"{len(df_latent)}/{len(df_feat)} rows; using the intersection.",
                    stacklevel=3,
                )
            return df_latent.loc[common], df_feat.loc[common]
        if len(df_latent) != len(df_feat):
            raise ValueError(
                f"df_latent has {len(df_latent)} rows but df_feat has "
                f"{len(df_feat)}, and their indices do not overlap."
            )
        warnings.warn(
            "df_latent and df_feat have no shared index labels; assuming the rows "
            "are already in the same order.",
            stacklevel=3,
        )
        df_feat = df_feat.copy()
        df_feat.index = df_latent.index
        return df_latent, df_feat

    @staticmethod
    def _invert_mapping(feature_names: list[str], feat_to_group: dict | None) -> dict:
        """Turn ``feature -> group`` into ``group -> [features]`` (order-stable).

        ``feat_to_group=None`` makes each feature its own singleton group, so the
        group Shapley budget reduces to a per-feature budget.  This is the natural
        default when groups are not needed (the Johnson view is per-feature
        regardless) or when there are few enough features to treat individually.
        """
        if feat_to_group is None:
            return {f: [f] for f in feature_names}
        groups: dict[str, list[str]] = {}
        ungrouped: list[str] = []
        for feat in feature_names:
            g = feat_to_group.get(feat)
            if g is None:
                ungrouped.append(feat)
                g = "ungrouped"
            groups.setdefault(g, []).append(feat)
        if ungrouped:
            warnings.warn(
                f"{len(ungrouped)} feature(s) are not in feat_to_group and were "
                f"placed in an 'ungrouped' group: {ungrouped[:5]}"
                + (" ..." if len(ungrouped) > 5 else ""),
                stacklevel=3,
            )
        return groups

    def _fit(self) -> "FeatureGroundedDecomposition":
        """Run (and cache) the decomposition.  Called automatically when needed."""
        if self._result is None:
            logger.info(
                "Fitting FeatureGroundedDecomposition: %d samples, %d latent dims, "
                "%d features, %d groups.",
                len(self.df_latent), self.df_latent.shape[1],
                len(self.feature_names), len(self.group_names),
            )
            self._result = decompose_latent_space(
                self.df_latent.values,
                self.df_feat.values,
                self.feature_names,
                self.groups,
                mode=self.mode,
                n_directions=self.n_directions,
                cv_folds=self.cv_folds,
                compute_feature_weights=True,
                seed=self.seed,
            )
            if self._dir_labels is None:
                self._dir_labels = [f"PC{d.index}" for d in self._result.directions]
        return self

    @property
    def _res(self) -> DecompositionResult:
        """Internal :class:`~decomposition.DecompositionResult` (fits if needed)."""
        return self._fit()._result

    @property
    def rho(self) -> float:
        """Fraction of total latent variance explained by the features."""
        return self._res.global_r2

    @property
    def rho_cv(self) -> float:
        """Cross-validated estimate of :attr:`rho`."""
        return self._res.global_r2_cv

    @property
    def directions(self) -> list:
        """Per-direction results (variance, ``R^2``, attributions); one per direction."""
        return self._res.directions

    @property
    def eigvals(self) -> np.ndarray:
        """Eigenvalues of the latent covariance (variance carried by each direction)."""
        return self._res.eigvals

    @property
    def eigvecs(self) -> np.ndarray:
        """Eigenvectors of the latent covariance; columns are directions (``d x k``)."""
        return self._res.eigvecs

    @property
    def global_group_shares(self) -> dict:
        """Eigenvalue-weighted Shapley share of each group across all directions."""
        return self._res.global_group_shares

    @property
    def global_noise(self) -> float:
        """Eigenvalue-weighted unexplained fraction across all directions (``1 - rho``)."""
        return self._res.global_noise

    @property
    def variance_fraction(self) -> pd.Series:
        """Eigenvalue weight of each direction (``lambda_k / sum lambda``).

        Indexed by direction label with a final ``weighted_sum`` entry equal to
        the total analysed variance fraction (1.0 when all directions are kept).
        This is the weight used by the eigenvalue-weighted-sum rows of the budget
        tables; it lives on the object rather than as a table column because it is
        on the variance scale, not the share scale, and is shared by both views.
        """
        res = self._res
        vals = [d.variance_frac for d in res.directions]
        idx = list(self._dir_labels) + [_WEIGHTED_SUM_LABEL]
        return pd.Series(vals + [float(sum(vals))], index=idx, name="variance_fraction")

    @property
    def r2(self) -> pd.Series:
        """Total explained fraction ``R^2`` of each direction (``1 - noise``).

        Indexed by direction label with a final ``weighted_sum`` entry equal to
        :attr:`rho`.  Equals the row sum of the attribution columns in either
        budget table; kept as a vector to avoid a column redundant with ``noise``.
        """
        res = self._res
        vals = [d.r2 for d in res.directions]
        idx = list(self._dir_labels) + [_WEIGHTED_SUM_LABEL]
        return pd.Series(vals + [res.global_r2], index=idx, name="r2")

    # ------------------------------------------------------------------ reports
    def scree(self, *, figsize=(8, 2.8), fig_out=None):
        """Plot the variance carried by each latent direction.  Returns a Figure.

        ``fig_out`` controls saving: ``None`` (default) returns the figure
        without writing it; a path saves to that path; ``True`` saves to the
        default location ``figures/scree.png``.
        """
        fig = viz.plot_scree(self._res, figsize=figsize)
        viz.save_figure(fig, fig_out, "scree")
        return fig

    def shap(self, *, figure_type: str = "bar", feat_to_group: dict | None = None,
             top_k: int | None = None, top_n: int = 12, figsize=(10, 6),
             fig_out=None) -> Report:
        """Group-level Shapley variance budget.

        Returns a :class:`Report` ``(table, rho, figure)`` where ``table`` is a
        ``(d+1) x (g+1)`` DataFrame: one row per latent direction plus a final
        eigenvalue-weighted-sum row, one column per group, and a final ``noise``
        column (``1 - R^2``).  The group columns are the Shapley shares of ``R^2``;
        groups plus noise partition each direction's variance, so every row sums
        to 1.  The per-direction ``R^2`` and eigenvalue weight are available as
        :attr:`r2` and :attr:`variance_fraction`.

        ``feat_to_group`` overrides the grouping fixed at construction for this
        call only, recomputing the Shapley group attribution for the supplied
        ``{feature -> group}`` mapping (``None`` uses the constructor's grouping).

        ``figure_type`` selects the figure: ``"bar"`` (default) draws one
        stacked bar per direction; ``"heatmap"`` draws a direction-by-group grid
        of the shares.  ``top_k`` limits the directions shown in the bar; ``top_n``
        limits the groups shown in the heatmap (the table always carries all ``g``).
        """
        res, group_names = self._group_result(feat_to_group)
        if len(group_names) > _SHAP_GROUP_WARN:
            warnings.warn(
                f"shap() is attributing to {len(group_names)} groups; the exact "
                f"Shapley game is exponential in the number of groups (2^G "
                f"coalitions) and the figure is hard to read with this many. Pass "
                f"feat_to_group to pool features into fewer groups, or use "
                f"johnson() for the per-feature view.",
                stacklevel=2,
            )
        attributions = [d.group_shares for d in res.directions]
        table = self._build_table(group_names, attributions, res.global_group_shares)
        fig = self._budget_figure(
            table, group_names, figure_type=figure_type,
            color_of=viz._color, col_label="group", top_k=top_k, top_n=top_n,
            figsize=figsize,
        )
        viz.save_figure(fig, fig_out, "shap")
        return Report(table=table, rho=res.global_r2, figure=fig)

    # # Capitalised alias matching the requested API.
    # SHAP = shap

    def johnson(self, *, figure_type: str = "heatmap", top_n: int = 20,
                top_k: int | None = None, figsize=(10, 6), fig_out=None) -> Report:
        """Per-feature Johnson relative-weight budget.

        Returns a :class:`Report` ``(table, rho, figure)`` where ``table`` is a
        ``(d+1) x (q+1)`` DataFrame: one row per latent direction plus a final
        eigenvalue-weighted-sum row, one column per feature, and a final
        ``noise`` column.  The feature columns are the Johnson relative weights;
        features plus noise partition each direction's variance, so every row sums
        to 1.

        ``figure_type`` selects the figure: ``"heatmap"`` (default) draws a
        direction-by-feature grid; ``"bar"`` draws one stacked bar per direction.
        ``top_n`` limits the features shown (the bar lumps the rest into an
        ``"other"`` segment; the table always carries all ``q``).  ``top_k`` limits
        the directions shown in the bar.

        ``fig_out`` controls saving: ``None`` (default) returns the figure
        without writing it; a path saves to that path; ``True`` saves to the
        default location ``figures/johnson.png``.
        """
        res = self._res
        attributions = [d.feature_weights for d in res.directions]
        table = self._build_table(self.feature_names, attributions,
                                  self._global_feature_shares())
        fig = self._budget_figure(
            table, self.feature_names, figure_type=figure_type,
            color_of=self._feature_color(), col_label="feature",
            top_k=top_k, top_n=top_n, figsize=figsize,
        )
        viz.save_figure(fig, fig_out, "johnson")
        return Report(table=table, rho=res.global_r2, figure=fig)

    # Johnson = johnson

    def global_budget(self, *, by: str = "group", top_n: int = 12, figsize=(6, 5),
                      fig_out=None):
        """Plot the whole latent space's variance budget as a single stacked bar.

        This visualizes the ``weighted_sum`` row of the :meth:`shap` table
        (``by="group"``) or the :meth:`johnson` table (``by="feature"``): the
        eigenvalue-weighted shares across all directions plus the noise remainder,
        so the bar sums to 1.  ``by="feature"`` keeps the ``top_n`` largest
        features and lumps the rest into an ``"other"`` segment.  Returns a Figure.

        ``fig_out`` controls saving: ``None`` (default) returns the figure
        without writing it; a path saves to that path; ``True`` saves to the
        default location ``figures/global_budget_<by>.png``.
        """
        res = self._res
        rho_pct = res.global_r2 * 100
        if by == "group":
            shares = res.global_group_shares
            title = (f"Total latent variance budget by group\n"
                     f"(features explain {rho_pct:.1f}%)")
            fig = self._global_bar(shares, res.global_noise, top_n=None,
                                   color_of=viz._color, figsize=figsize, title=title)
        elif by == "feature":
            shares = self._global_feature_shares()
            title = (f"Total latent variance budget by feature\n"
                     f"(features explain {rho_pct:.1f}%)")
            fig = self._global_bar(shares, res.global_noise, top_n=top_n,
                                   color_of=self._feature_color(), figsize=figsize,
                                   title=title)
        else:
            raise ValueError(f"by must be 'group' or 'feature', got {by!r}")
        viz.save_figure(fig, fig_out, f"global_budget_{by}")
        return fig

    # ------------------------------------------------------------------ helpers
    def _group_result(self, feat_to_group: dict | None):
        """Result and group names for a grouping, recomputing if overridden.

        ``feat_to_group=None`` (or equal to the constructor value) returns the
        cached fit and its groups.  Any other ``{feature -> group}`` mapping
        triggers a fresh Shapley group attribution for that grouping; the
        eigendecomposition and per-direction ``R^2`` are unchanged, so only the
        group-level shares (and ``global_group_shares``) differ.
        """
        self._fit()  # populate direction labels and the grouping-invariant summaries
        if feat_to_group is None or feat_to_group == self.feat_to_group:
            return self._result, self.group_names
        groups = self._invert_mapping(self.feature_names, feat_to_group)
        result = decompose_latent_space(
            self.df_latent.values, self.df_feat.values,
            self.feature_names, groups,
            mode=self.mode, n_directions=self.n_directions,
            cv_folds=self.cv_folds, compute_feature_weights=False, seed=self.seed,
        )
        return result, list(groups.keys())

    def _build_table(
        self,
        columns: list[str],
        attributions: list[dict],
        global_attr: dict,
    ) -> pd.DataFrame:
        """Assemble a ``(d+1) x (len(columns)+1)`` budget table.

        ``attributions`` is one ``{column: share}`` dict per direction (shares sum
        to that direction's ``R^2``); the appended ``noise`` column is the
        unexplained remainder, so every row sums to 1.  ``global_attr`` is the
        eigenvalue-weighted aggregate used for the final ``weighted_sum`` row.
        """
        res = self._result
        rows = []
        index = []
        for label, attr in zip(self._dir_labels, attributions):
            shares = [float(attr.get(c, 0.0)) for c in columns]
            rows.append(shares + [1.0 - float(sum(shares))])  # ... + noise
            index.append(label)

        # Eigenvalue-weighted sum across directions (the global budget).
        global_shares = [float(global_attr.get(c, 0.0)) for c in columns]
        rows.append(global_shares + [res.global_noise])
        index.append(_WEIGHTED_SUM_LABEL)

        table = pd.DataFrame(rows, index=index, columns=list(columns) + [_NOISE_COL])
        table.index.name = "direction"
        return table

    def _global_feature_shares(self) -> dict:
        """Variance-weighted global relative weight of each feature.

        Mirrors :attr:`DecompositionResult.global_group_shares` at the feature
        level: ``sum_k (lambda_k / sum lambda) * eps_{k,j}``, summing over features
        to the global ``R^2``.
        """
        res = self._res
        vfrac = np.array([d.variance_frac for d in res.directions])
        vfrac = vfrac / vfrac.sum()
        return {
            f: float(sum(w * d.feature_weights.get(f, 0.0)
                         for w, d in zip(vfrac, res.directions)))
            for f in self.feature_names
        }

    def _global_bar(self, shares: dict, noise: float, *, top_n, color_of,
                    figsize, title):
        """Draw one stacked bar: ``shares`` (largest first) + ``other`` + noise."""
        import matplotlib.pyplot as plt

        items = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        other = 0.0
        if top_n is not None and len(items) > top_n:
            other = float(sum(v for _, v in items[top_n:]))
            items = items[:top_n]

        fig, ax = plt.subplots(figsize=figsize)
        bottom = 0.0
        for i, (name, h) in enumerate(items):
            ax.bar(0, h, bottom=bottom, width=0.6, color=color_of(name, i),
                   edgecolor="white", label=f"{name} ({h*100:.1f}%)")
            bottom += h
        if other > 0:
            ax.bar(0, other, bottom=bottom, width=0.6, color="#bbbbbb",
                   edgecolor="white", label=f"other ({other*100:.1f}%)")
            bottom += other
        ax.bar(0, noise, bottom=bottom, width=0.6, color="#dddddd",
               edgecolor="white", label=f"noise ({noise*100:.1f}%)")
        ax.set_xlim(-1, 1)
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of total latent variance")
        if title:
            ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        return fig

    def _feature_color(self):
        """Color function keying each feature to its group's stable color."""
        feat_group = {f: g for g, fs in self.groups.items() for f in fs}
        group_idx = {g: i for i, g in enumerate(self.group_names)}
        def color_of(name, i):
            g = feat_group.get(name, name)
            return viz._color(g, group_idx.get(g, i))
        return color_of

    @staticmethod
    def _rank_columns(table: pd.DataFrame, columns: list[str], top_n: int | None):
        """Split ``columns`` into (top-``n``, rest) by their ``weighted_sum`` share."""
        global_row = table.loc[_WEIGHTED_SUM_LABEL, columns].astype(float)
        ordered = list(global_row.sort_values(ascending=False).index)
        if top_n is None or top_n >= len(ordered):
            return ordered, []
        return ordered[:top_n], ordered[top_n:]

    def _budget_figure(self, table, columns, *, figure_type, color_of, col_label,
                       top_k, top_n, figsize):
        """Dispatch to the stacked-bar or heatmap renderer for a budget table."""
        if figure_type == "bar":
            return self._budget_bar(table, columns, color_of=color_of, top_k=top_k,
                                    top_n=top_n, figsize=figsize)
        if figure_type == "heatmap":
            return self._budget_heatmap(table, columns, col_label=col_label,
                                        top_n=top_n, figsize=figsize)
        raise ValueError(
            f"figure_type must be 'bar' or 'heatmap', got {figure_type!r}")

    def _budget_bar(self, table, columns, *, color_of, top_k, top_n, figsize):
        """Stacked bar per direction: attribution shares (+ ``other``) + noise."""
        import matplotlib.pyplot as plt

        top, rest = self._rank_columns(table, columns, top_n)
        dir_rows = [lbl for lbl in table.index if lbl != _WEIGHTED_SUM_LABEL]
        if top_k:
            dir_rows = dir_rows[:top_k]
        vfrac = self.variance_fraction
        x = np.arange(len(dir_rows))
        bottom = np.zeros(len(dir_rows))

        fig, ax = plt.subplots(figsize=figsize)
        for i, c in enumerate(top):
            h = table.loc[dir_rows, c].astype(float).values
            ax.bar(x, h, bottom=bottom, label=c, color=color_of(c, i), width=0.85)
            bottom += h
        if rest:
            other = table.loc[dir_rows, rest].astype(float).sum(axis=1).values
            ax.bar(x, other, bottom=bottom, label="other", color="#bbbbbb",
                   width=0.85, edgecolor="white")
            bottom += other
        noise = table.loc[dir_rows, _NOISE_COL].astype(float).values
        ax.bar(x, noise, bottom=bottom, label="noise (unexplained)",
               color="#dddddd", width=0.85, edgecolor="white")
        ax.set_xlabel("latent direction (ordered by variance)")
        ax.set_ylabel("fraction of direction variance")
        ax.set_title("Feature-grounded variance budget per latent direction")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lbl}\n{vfrac[lbl]*100:.0f}%" for lbl in dir_rows],
                           fontsize=8)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig

    def _budget_heatmap(self, table, columns, *, col_label, top_n, figsize):
        """Heatmap of per-direction shares for the top-``n`` ``columns``."""
        import matplotlib.pyplot as plt

        top, _ = self._rank_columns(table, columns, top_n)
        dir_rows = [lbl for lbl in table.index if lbl != _WEIGHTED_SUM_LABEL]
        mat = table.loc[dir_rows, top].astype(float).values

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels([str(t).replace("_", " ") for t in top],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(dir_rows)))
        ax.set_yticklabels(dir_rows, fontsize=8)
        ax.set_xlabel(col_label)
        ax.set_ylabel("latent direction")
        ax.set_title(f"Variance budget by {col_label} (share of $R^2$) per direction")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of $R^2$")
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        n, d = self.df_latent.shape
        q = self.df_feat.shape[1]
        g = len(self.group_names)
        fitted = self._result is not None
        return (f"FeatureGroundedDecomposition(n={n}, d={d}, q={q}, groups={g}, "
                f"mode={self.mode!r}, fitted={fitted})")
