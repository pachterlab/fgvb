r"""Feature-grounded variance decomposition of latent directions.

This module implements the paper's core method.  Given

* latent codes ``Z`` (n x d) from a trained encoder, and
* a bank of interpretable features ``F`` (n x p),

we ask, for every principal direction of the latent space, *what fraction of
its variance is explained by the interpretable features, and how is that share
distributed across feature families?*  The unexplained remainder is the
"irreducible" part of the direction --- structure the encoder represents that
our vocabulary of radiomic features cannot name.

Method
------
1.  **Directions.**  Eigendecompose the latent covariance
    ``C = (1/n) Z^T Z = V diag(lambda) V^T``.  The k-th direction is the
    eigenvector ``v_k`` with variance ``lambda_k``; the projected score is
    ``s_k = Z v_k`` with ``Var(s_k) = lambda_k``.  (Raw latent axes can be used
    instead via ``mode="axes"``.)

2.  **Explained variance.**  Regress each score on the (standardised) features,
    ``s_k = F beta_k + eps_k``.  The coefficient of determination ``R^2_k`` is
    the fraction of the direction's variance the features explain; ``1 - R^2_k``
    is the noise/unexplained fraction.  We report both the in-sample ``R^2`` and
    an honest K-fold cross-validated ``R^2`` (which cannot be inflated by the
    large feature count).

3.  **Attribution.**  Radiomic features are highly collinear, so raw regression
    weights do not give a meaningful variance split.  We use two principled
    additive decompositions of ``R^2`` that *sum back to* ``R^2``:

    * **Group Shapley** -- treat the feature *families* (shape, curvature,
      intensity, texture, ...) as players in a cooperative game whose value is
      the regression ``R^2``.  The Shapley value of each family is its average
      marginal contribution over all orderings.  With a handful of families the
      game is solved exactly.  This yields a per-direction "budget":
      ``Var = shape% + curvature% + ... + noise%``.

    * **Johnson's relative weights** (Johnson, 2000) -- a per-*feature*
      attribution that is exact, non-negative and collinearity-aware, obtained
      by regressing the score on the orthonormal basis closest to ``F``.

4.  **Global summary.**  Weighting directions by their variance ``lambda_k``
    gives the fraction of the *whole* latent space explained by features,
    ``rho = sum_k lambda_k R^2_k / sum_k lambda_k``, and an analogous per-family
    breakdown of the total latent variance budget.

All regression ``R^2`` values for arbitrary feature subsets are computed from a
single precomputed Gram matrix, so the exact group-Shapley game (``2^G``
coalitions) is cheap even with bootstrap resampling.

References
----------
Lindeman, Merenda & Gold (1980); Gromping (2007), *relaimpo*; Johnson (2000),
*A heuristic method for estimating the relative weight of predictor variables*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from math import factorial

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Linear-algebra core: R^2 of any feature subset from a shared Gram matrix
# ---------------------------------------------------------------------------
@dataclass
class _GramCache:
    """Precomputed quantities for fast subset-R^2 evaluation.

    Features ``X`` are standardised (zero mean, unit variance) and scores are
    centred, so no intercept is needed.
    """

    G: np.ndarray          # X^T X  (p x p)
    ridge: float           # tiny diagonal load for numerical stability
    p: int

    @classmethod
    def build(cls, X: np.ndarray, ridge: float = 1e-6) -> "_GramCache":
        G = X.T @ X
        return cls(G=G, ridge=ridge * np.trace(G) / max(X.shape[1], 1), p=X.shape[1])

    def r2(self, c: np.ndarray, syy: float, idx: np.ndarray) -> float:
        """R^2 of regressing a score on the feature columns in ``idx``.

        ``c = X^T s`` (length p), ``syy = s^T s`` (s centred).
        """
        if len(idx) == 0 or syy <= 0:
            return 0.0
        Gss = self.G[np.ix_(idx, idx)].copy()
        Gss[np.diag_indices_from(Gss)] += self.ridge
        cs = c[idx]
        try:
            beta = np.linalg.solve(Gss, cs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Gss, cs, rcond=None)[0]
        explained = float(cs @ beta)
        return max(0.0, min(1.0, explained / syy))


# ---------------------------------------------------------------------------
# Group (family) Shapley decomposition of R^2
# ---------------------------------------------------------------------------
def _shapley_groups(cache: _GramCache, c: np.ndarray, syy: float,
                    group_cols: list[np.ndarray]) -> np.ndarray:
    """Exact Shapley value of each feature group; entries sum to full-model R^2."""
    G = len(group_cols)
    players = list(range(G))
    # Cache coalition R^2 keyed by frozenset of group indices.
    r2_cache: dict[frozenset, float] = {}

    def coalition_r2(S: tuple[int, ...]) -> float:
        key = frozenset(S)
        if key not in r2_cache:
            if not S:
                r2_cache[key] = 0.0
            else:
                idx = np.concatenate([group_cols[g] for g in S]) if S else np.array([], int)
                r2_cache[key] = cache.r2(c, syy, idx)
        return r2_cache[key]

    phi = np.zeros(G)
    for g in players:
        others = [q for q in players if q != g]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(G - r - 1) / factorial(G)
            for S in combinations(others, r):
                phi[g] += w * (coalition_r2(S + (g,)) - coalition_r2(S))
    return phi


# ---------------------------------------------------------------------------
# Johnson's relative weights (per-feature attribution)
# ---------------------------------------------------------------------------
def relative_weights(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Johnson (2000) relative weights for standardised X and centred y.

    Returns ``(weights, r2)`` where ``weights`` are non-negative, length p, and
    ``weights.sum() == r2`` (up to numerical error).
    """
    n = X.shape[0]
    # Standardise the response to unit variance (X columns are already
    # standardised by the caller).
    yc = y - y.mean()
    ystd = yc.std()
    if ystd < 1e-12:
        return np.zeros(X.shape[1]), 0.0
    ys = yc / ystd
    # Correlation structure of predictors.
    Rxx = (X.T @ X) / n
    w, V = np.linalg.eigh(Rxx)
    w = np.clip(w, 1e-10, None)
    # Z* = closest set of orthonormal predictors to X; X = Z* Lambda with the
    # symmetric Lambda = V diag(sqrt(w)) V^T.  Columns of Z* have unit variance.
    Lambda = V @ np.diag(np.sqrt(w)) @ V.T
    Zstar = X @ (V @ np.diag(1.0 / np.sqrt(w)) @ V.T)
    # Orthonormal predictors -> standardised regression weights are correlations.
    beta_star = (Zstar.T @ ys) / n
    # Column sums of Lambda^2 are 1, so eps sums to R^2.
    eps = (Lambda ** 2) @ (beta_star ** 2)
    r2 = float(eps.sum())
    return eps, r2


# ---------------------------------------------------------------------------
# Cross-validated R^2 (honest explained variance)
# ---------------------------------------------------------------------------
_RIDGE_GRID = np.logspace(-2, 5, 10)


def _cv_r2_multi(X: np.ndarray, S: np.ndarray, *, folds: int = 5,
                 alphas: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """K-fold cross-validated R^2 of ridge regression for many targets at once.

    ``S`` is (n, k); returns a length-k array of CV R^2, one per column. A single
    SVD of the training design per fold is shared across all targets and all
    ridge penalties, and the best penalty (lowest pooled held-out error) is
    selected per target. Sharing the SVD is what makes wide banks (e.g. raw
    pixels, p~4096) tractable, and tuning the penalty is what stops the CV score
    from collapsing when p is large.
    """
    if alphas is None:
        alphas = _RIDGE_GRID
    n, p = X.shape
    k = S.shape[1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_id = np.array_split(order, folds)
    sse = np.zeros((len(alphas), k))
    sst = np.zeros(k)
    for f in range(folds):
        te = fold_id[f]
        tr = np.concatenate([fold_id[j] for j in range(folds) if j != f])
        Xtr, Xte = X[tr], X[te]
        Str, Ste = S[tr], S[te]
        mu = Str.mean(0)
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)  # U (ntr,r) s (r) Vt (r,p)
        UtY = U.T @ (Str - mu)        # (r, k)
        XteV = Xte @ Vt.T             # (nte, r)
        for ai, al in enumerate(alphas):
            filt = s / (s ** 2 + al)  # (r,)
            coef = filt[:, None] * UtY      # (r, k)
            pred = XteV @ coef + mu         # (nte, k)
            sse[ai] += np.sum((Ste - pred) ** 2, axis=0)
        sst += np.sum((Ste - Ste.mean(0)) ** 2, axis=0)
    r2 = 1.0 - sse / (sst[None, :] + 1e-12)
    return np.clip(r2.max(axis=0), -1.0, 1.0)  # best penalty per target


def cv_r2(X: np.ndarray, y: np.ndarray, *, folds: int = 5,
          alphas: np.ndarray | None = None, seed: int = 0) -> float:
    """K-fold cross-validated R^2 of penalty-tuned ridge regression of y on X."""
    return float(_cv_r2_multi(X, np.asarray(y, float)[:, None],
                              folds=folds, alphas=alphas, seed=seed)[0])


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class DirectionResult:
    index: int
    variance: float            # lambda_k (eigenvalue) or Var(axis)
    variance_frac: float       # lambda_k / sum lambda
    r2: float                  # in-sample R^2
    r2_cv: float               # cross-validated R^2
    group_shares: dict[str, float]   # Shapley share of R^2 per family (sum=r2)
    feature_weights: dict[str, float]  # relative weight per feature (sum=r2)
    noise_frac: float          # 1 - r2
    r2_ci: tuple[float, float] | None = None  # bootstrap CI on r2


@dataclass
class DecompositionResult:
    directions: list[DirectionResult]
    eigvals: np.ndarray
    eigvecs: np.ndarray        # columns are directions in latent space (d x k)
    feature_names: list[str]
    group_names: list[str]
    mode: str
    # Global summaries
    global_r2: float = 0.0                       # variance-weighted mean R^2
    global_r2_cv: float = 0.0                    # variance-weighted mean CV R^2
    global_group_shares: dict[str, float] = field(default_factory=dict)
    global_noise: float = 0.0

    def summary_table(self):
        """Return a pandas DataFrame, one row per direction."""
        import pandas as pd
        rows = []
        for d in self.directions:
            row = {"direction": d.index, "variance_frac": d.variance_frac,
                   "R2": d.r2, "R2_cv": d.r2_cv, "noise": d.noise_frac}
            row.update({f"grp_{g}": d.group_shares.get(g, 0.0) for g in self.group_names})
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def decompose_latent_space(
    Z: np.ndarray,
    F: np.ndarray,
    feature_names: list[str],
    groups: dict[str, list[str]],
    *,
    mode: str = "eigen",
    n_directions: int | None = None,
    cv_folds: int = 5,
    n_bootstrap: int = 0,
    compute_feature_weights: bool = True,
    seed: int = 0,
) -> DecompositionResult:
    """Decompose every latent direction into feature-family variance shares.

    Parameters
    ----------
    Z : (n, d) latent codes (posterior means).
    F : (n, p) raw feature matrix (will be standardised internally).
    feature_names : list of p feature column names matching ``F``.
    groups : mapping family name -> list of feature names in that family.
    mode : {"eigen", "axes"}
        ``"eigen"`` analyses eigenvectors of the latent covariance;
        ``"axes"`` analyses the raw latent coordinates.
    n_directions : analyse only the top-``k`` directions (by variance).
    cv_folds : folds for the honest cross-validated R^2.
    n_bootstrap : if > 0, bootstrap resamples for a 95% CI on each R^2.
    compute_feature_weights : if False, skip the (per-feature) relative-weights
        attribution -- much faster for very wide feature banks (e.g. raw pixels).
    """
    Z = np.asarray(Z, dtype=float)
    F = np.asarray(F, dtype=float)
    n, d = Z.shape
    logger.info(
        "decompose_latent_space: n=%d samples, d=%d latent dims, p=%d features, "
        "mode=%r, cv_folds=%d, n_bootstrap=%d",
        n, d, F.shape[1], mode, cv_folds, n_bootstrap,
    )

    # Standardise features; center latents.
    Fmu = F.mean(0)
    Fsd = F.std(0)
    Fsd[Fsd < 1e-9] = 1.0
    X = (F - Fmu) / Fsd
    Zc = Z - Z.mean(0)

    # Build directions.
    if mode == "eigen":
        C = (Zc.T @ Zc) / n
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        scores = Zc @ eigvecs            # (n, d), Var(col k) = eigvals[k]
    elif mode == "axes":
        eigvecs = np.eye(d)
        eigvals = Zc.var(0)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        scores = Zc[:, order]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    total_var = float(eigvals.sum())
    k = d if n_directions is None else min(n_directions, d)
    logger.info("Analysing top %d/%d directions (mode=%r).", k, d, mode)

    # Group -> column-index arrays into X.
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    group_names = list(groups.keys())
    group_cols = [
        np.array([name_to_idx[c] for c in groups[g] if c in name_to_idx], dtype=int)
        for g in group_names
    ]

    cache = _GramCache.build(X)
    rng = np.random.default_rng(seed)

    # Cross-validated R^2 for all analysed directions at once (shared SVD).
    r2cv_all = _cv_r2_multi(X, scores[:, :k], folds=cv_folds, seed=seed)

    directions: list[DirectionResult] = []
    for kk in range(k):
        s = scores[:, kk]
        syy = float(s @ s)
        c = X.T @ s

        r2_full = cache.r2(c, syy, np.arange(X.shape[1]))
        r2cv = float(r2cv_all[kk])
        phi = _shapley_groups(cache, c, syy, group_cols)
        group_shares = {g: float(v) for g, v in zip(group_names, phi)}

        if compute_feature_weights:
            eps, _ = relative_weights(X, s)
            feat_weights = {feature_names[i]: float(eps[i]) for i in range(len(feature_names))}
        else:
            feat_weights = {}

        logger.debug(
            "direction %d/%d: var_frac=%.4f R^2=%.3f R^2_cv=%.3f",
            kk, k, eigvals[kk] / total_var, r2_full, r2cv,
        )

        r2_ci = None
        if n_bootstrap > 0:
            logger.debug("direction %d: bootstrapping R^2 over %d resamples", kk, n_bootstrap)
            boots = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                bi = rng.integers(0, n, n)
                Xb = X[bi]
                sb = s[bi] - s[bi].mean()
                cb = Xb.T @ sb
                cacheb = _GramCache.build(Xb)
                boots[b] = cacheb.r2(cb, float(sb @ sb), np.arange(X.shape[1]))
            r2_ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

        directions.append(DirectionResult(
            index=kk,
            variance=float(eigvals[kk]),
            variance_frac=float(eigvals[kk] / total_var),
            r2=float(r2_full),
            r2_cv=float(r2cv),
            group_shares=group_shares,
            feature_weights=feat_weights,
            noise_frac=float(1.0 - r2_full),
            r2_ci=r2_ci,
        ))

    # Global, variance-weighted summaries.
    vfrac = eigvals[:k] / total_var
    vfrac = vfrac / vfrac.sum()
    global_r2 = float(np.sum([vf * dr.r2 for vf, dr in zip(vfrac, directions)]))
    global_r2_cv = float(np.sum([vf * max(dr.r2_cv, 0.0) for vf, dr in zip(vfrac, directions)]))
    global_group = {}
    for g in group_names:
        global_group[g] = float(np.sum([vf * dr.group_shares[g]
                                        for vf, dr in zip(vfrac, directions)]))
    global_noise = float(np.sum([vf * dr.noise_frac for vf, dr in zip(vfrac, directions)]))
    logger.info(
        "Decomposition complete: global rho=%.3f (cv=%.3f), noise=%.3f over %d directions.",
        global_r2, global_r2_cv, global_noise, k,
    )

    return DecompositionResult(
        directions=directions,
        eigvals=eigvals,
        eigvecs=eigvecs,
        feature_names=feature_names,
        group_names=group_names,
        mode=mode,
        global_r2=global_r2,
        global_r2_cv=global_r2_cv,
        global_group_shares=global_group,
        global_noise=global_noise,
    )


# ---------------------------------------------------------------------------
# Feature-grounded latent reconstruction
# ---------------------------------------------------------------------------
def feature_grounded_codes(Z: np.ndarray, F: np.ndarray, result: DecompositionResult,
                           *, ridge: float = 1e-3) -> np.ndarray:
    """Project latent codes onto the feature-explainable subspace.

    For each analysed direction we predict its score from the features and
    reassemble the latent code, ``zhat = sum_k (features -> s_k) v_k``.  Decoding
    ``zhat`` shows what the digit looks like when only the *nameable* latent
    structure is kept.  Returns ``zhat`` with the same shape as ``Z``.
    """
    Z = np.asarray(Z, float)
    F = np.asarray(F, float)
    n, d = Z.shape
    Fmu, Fsd = F.mean(0), F.std(0)
    Fsd[Fsd < 1e-9] = 1.0
    X = (F - Fmu) / Fsd
    zmean = Z.mean(0)
    Zc = Z - zmean

    V = result.eigvecs[:, : len(result.directions)]  # (d, k)
    scores = Zc @ V                                   # (n, k)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    shat = np.empty_like(scores)
    for kk in range(scores.shape[1]):
        beta = np.linalg.solve(A, X.T @ scores[:, kk])
        shat[:, kk] = X @ beta
    Zhat = shat @ V.T + zmean
    return Zhat
