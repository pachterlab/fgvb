"""Interpretable directions in latent spaces via feature-grounded variance decomposition.

Logging
-------
``fgvb`` logs through the standard :mod:`logging` module under the ``"fgvb"``
logger (each submodule uses its own ``logging.getLogger(__name__)`` child).
Following library convention, the package attaches a :class:`logging.NullHandler`
so it stays *silent by default* and never configures handlers or levels on the
application's behalf.

To see what the library is doing, configure logging in your own code::

    import logging
    logging.basicConfig(level=logging.INFO)      # INFO: high-level progress
    # logging.getLogger("fgvb").setLevel(logging.DEBUG)  # DEBUG: per-direction detail

INFO-level messages report coarse progress (shape of the inputs, number of
directions analysed, global R^2). DEBUG-level messages are emitted inside the
hot loops (per-direction R^2, bootstrap resampling) and can be verbose.
"""

import logging

from . import (  # noqa: F401
    decomposition,
    explainer,
    viz,
)
from .explainer import FeatureGroundedDecomposition  # noqa: F401

__version__ = "0.1.0"

__all__ = ["decomposition", "explainer", "viz", "FeatureGroundedDecomposition"]

# Silence-by-default: a library must not configure logging output itself.
# Applications opt in via logging.basicConfig() / getLogger("fgvb").setLevel(...).
logging.getLogger(__name__).addHandler(logging.NullHandler())
