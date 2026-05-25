"""Convert Stage-4 ``disagreement.csv`` files into the per-point ``.npz`` layout
that :mod:`mcp_eval.interp.scripts.make_figures` consumes.

LOCAL / REMOTE: **LOCAL** — pure CSV → npz, no model.

The MCPTox/MPMA detection pipeline writes one CSV per attack with paired
benign/malicious disagreement metrics (columns ``A_benign/A_mal``,
``E_benign/E_mal``, …). For the §3 Finding-4 scatter we need, per scenario, a
2-D point ``(agreement, entropy)``. We map agreement→``A`` and entropy→``E``.

Example::

    python -m mcp_eval.interp.scripts.disagreement_from_csv \
        --csv results/mcptox/mcptox/disagreement.csv \
        --attack mcptox --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pma_shield.interp.config import model_safe


def _read_points(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    benign, attacked = [], []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("parse_ok_benign") not in ("True", "true", "1"):
                continue
            if row.get("parse_ok_mal") not in ("True", "true", "1"):
                continue
            try:
                benign.append((float(row["A_benign"]), float(row["E_benign"])))
                attacked.append((float(row["A_mal"]), float(row["E_mal"])))
            except (KeyError, ValueError):
                continue
    return np.asarray(benign), np.asarray(attacked)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--attack", required=True, help="mcptox | mpma | injecagent")
    p.add_argument("--model", required=True)
    args = p.parse_args()

    benign, attacked = _read_points(args.csv)
    if benign.size == 0:
        raise SystemExit(f"no parsed pairs in {args.csv}")

    safe = model_safe(args.model)
    out_dir = Path("results") / args.attack / safe / "disagreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "points.npz", r_bar=attacked[:, 0], H=attacked[:, 1])
    # benign points are reused by every attack panel; mcptox is the canonical source.
    if args.attack == "mcptox":
        np.savez(out_dir / "benign.npz", r_bar=benign[:, 0], H=benign[:, 1])
    print(f"wrote {out_dir}/points.npz  (benign n={len(benign)}, attacked n={len(attacked)})")


if __name__ == "__main__":
    main()
