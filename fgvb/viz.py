"""Plotting helpers for feature-grounded variance decomposition.

Kept deliberately small: the budget figures themselves live on
:class:`~fgvb.explainer.FeatureGroundedDecomposition`; this module only owns the
stand-alone scree plot and the shared colour function so that a group keeps the
same colour across the scree, bar, and heatmap views.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps

from .decomposition import DecompositionResult

logger = logging.getLogger(__name__)

# A categorical colour wheel; indices beyond its length wrap around.
_PALETTE = list(colormaps["tab20"].colors)

# Where ``fig_out=True`` writes figures, named after the report method.
_DEFAULT_FIG_DIR = Path("figures")


def save_figure(fig, fig_out, default_name: str, *, dpi: int = 150):
    """Optionally save ``fig`` according to a ``fig_out`` argument.

    ``fig_out`` follows a three-way convention shared by all report methods:

    * ``None`` (or ``False``) -- do nothing (the default; figures are returned,
      not written).
    * a path (``str`` / :class:`~pathlib.Path`) -- save to exactly that path.
    * ``True`` -- save to the default location ``figures/<default_name>.png``.

    Parent directories are created as needed.  Returns the path written, or
    ``None`` if nothing was saved.
    """
    if fig_out is None or fig_out is False:
        return None
    if fig_out is True:
        path = _DEFAULT_FIG_DIR / f"{default_name}.png"
    else:
        path = Path(fig_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure to %s", path)
    return path


def _color(name: str, index: int):
    """Stable colour for a group/feature, keyed by its position ``index``.

    Callers pass a consistent index per name (the group's position in
    ``group_names``), so the same group keeps the same colour across figures.
    """
    return _PALETTE[index % len(_PALETTE)]


def plot_scree(result: DecompositionResult, *, figsize=(8, 2.8)):
    """Bar plot of the variance fraction carried by each latent direction.

    Directions are ordered by variance (as produced by the decomposition); the
    height of each bar is ``lambda_k / sum lambda``.  Returns the Figure.
    """
    fracs = [d.variance_frac for d in result.directions]
    labels = [
        f"PC{d.index}" if result.mode == "eigen" else str(d.index)
        for d in result.directions
    ]
    x = range(len(fracs))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, fracs, color="#4c72b0", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_xlabel("latent direction (ordered by variance)")
    ax.set_ylabel("fraction of total\nlatent variance")
    ax.set_title("Scree: variance carried by each latent direction")
    for xi, h in zip(x, fracs):
        ax.text(xi, h, f"{h*100:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(fracs) * 1.15 if fracs else 1)
    fig.tight_layout()
    return fig
