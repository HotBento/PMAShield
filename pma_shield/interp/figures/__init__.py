"""Pure plotting helpers for the §3 *Mechanistic Interpretability* figures.

Every ``fig_*.py`` module exposes one function that accepts plain NumPy arrays
(or dictionaries thereof) and returns the resulting ``matplotlib.figure.Figure``
**after** calling :func:`style.save_fig`. The module is data-source agnostic
so the same code paths are used by both
:mod:`mcp_eval.interp.scripts.make_figures` and
:mod:`mcp_eval.interp.scripts.make_mock_figures`.
"""

from pma_shield.interp.figures import style  # noqa: F401
