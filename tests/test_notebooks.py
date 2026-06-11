"""End-to-end Jupyter notebook execution tests via ``nbval``.

Two flavours, mirroring the convention used across the lab's repos:

* **lax** (``--nbval-lax``) -- the notebook must run top to bottom without error;
  cell outputs are *not* compared.  Outputs are cleared first so a stale run does
  not leak into the check.  Used for the narrated ``tutorial.ipynb``, whose
  figures and floating-point values are not meant to be reproduced bit-for-bit.
* **strict** (``--nbval``) -- the notebook must run *and* reproduce its stored
  outputs exactly.  Used for ``quickstart.ipynb``, which is authored to emit only
  structurally stable output (shapes, labels, boolean identity checks); its one
  float-printing cell is tagged ``# NBVAL_IGNORE_OUTPUT``.

Each notebook is copied into pytest's ``tmp_path`` and run there (the working
directory is switched too) so any files it writes -- e.g. ``figures/`` -- land in
the temp dir rather than the repo.  Both tests are marked ``notebook`` so they can
be selected or skipped with ``-m notebook`` / ``-m "not notebook"``.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager

import nbformat
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_DIR = os.path.join(REPO_ROOT, "notebooks")

# Run top-to-bottom, outputs not compared.
file_list_lax = ["tutorial.ipynb"]
# Run AND reproduce stored outputs exactly.
file_list_strict = ["quickstart.ipynb"]


def clear_notebook_output(notebook_path: str) -> None:
    """Strip all outputs and execution counts from a notebook in place."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    for cell in nb["cells"]:
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


@contextmanager
def _chdir(path):
    """Temporarily switch the working directory (so notebook side-effects stay local)."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _copy_into(tmp_path, file_name: str) -> str:
    src = os.path.join(NOTEBOOK_DIR, file_name)
    dst = os.path.join(tmp_path, file_name)
    shutil.copy(src, dst)
    return dst


@pytest.mark.notebook
@pytest.mark.parametrize("file_name", file_list_lax)
def test_run_lax(tmp_path, file_name):
    """Notebook executes end-to-end without error (outputs not compared)."""
    notebook = _copy_into(tmp_path, file_name)
    clear_notebook_output(notebook)  # avoid comparing against a stale run
    with _chdir(tmp_path):
        result = pytest.main(["--nbval-lax", "--tb=short", "-vv", notebook], plugins=[])
    assert result == 0, f"{file_name} failed to execute under --nbval-lax"


@pytest.mark.notebook
@pytest.mark.parametrize("file_name", file_list_strict)
def test_run_identical_strict(tmp_path, file_name):
    """Notebook executes AND reproduces its stored outputs exactly (--nbval)."""
    notebook = _copy_into(tmp_path, file_name)  # keep outputs for the strict compare
    with _chdir(tmp_path):
        result = pytest.main(["--nbval", "--tb=short", "-vv", notebook], plugins=[])
    assert result == 0, f"{file_name} did not reproduce its stored outputs under --nbval"
