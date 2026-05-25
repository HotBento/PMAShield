"""Mechanistic interpretability experiments for LLM tool selection (paper §3).

Submodules
----------
* :mod:`config`          — output paths, registered model list, figure size constants.
* :mod:`probe_data`      — YAML loader for ``data/probe_scenarios``.
* :mod:`patching`        — layer-level and head-level activation patching (REMOTE / GPU only).
* :mod:`head_roles`      — classify selection heads into intent / summary / matching roles.
* :mod:`pattern_extract` — pull attention matrices for the ``fig:head-patterns`` panels.
* :mod:`figures`         — pure-plotting helpers; accept ndarrays, produce PDFs.
* :mod:`scripts`         — CLI entry points (see ``scripts/<name>.py`` docstrings).

Layout
------

``data/probe_scenarios/*.yaml``  →  :mod:`probe_data`
                                    │
                                    ▼
``mcp_eval.interp.scripts.run_patching``      (REMOTE)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
``results/interp/<model>/        results/interp/<model>/   results/mcptox/...``
patching/layer_attn_mlp.npz      head_roles.json           disagreement/...
patching/head_importance.npz
patching/circuit_matrix.npz
                                    │
                                    ▼
``mcp_eval.interp.scripts.make_figures``      (LOCAL ok — no GPU)
                                    │
                                    ▼
``figures/interp/*.pdf``

Local-only quick start::

    python -m mcp_eval.interp.scripts.make_mock_figures
"""
