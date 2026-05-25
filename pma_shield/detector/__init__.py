"""
pma_shield.detector — PMAShield detection pipeline.

Implements the two-phase detection framework from the paper:

    Offline phase  (run once per model on a benign calibration set)
        Stage 0  data_mcptox / data_mpma  — load attack datasets
        Stage 1  capture                  — extract per-head attention features
        Stage 3  selection                — discover tool-selection heads via
                                           threshold τ on selection rate r(h)

    Online phase  (at inference time)
        Stage 4  disagreement             — compute A, E, D_JS, O metrics
        Stage 5  detection                — logistic classifier (LOSO)

    Transfer evaluation
        transferability/matrix.py         — RQ2 cross-attack AUROC matrix

Each module is independently usable; reach into submodules directly for
fine-grained APIs.
"""

__all__: list[str] = []
