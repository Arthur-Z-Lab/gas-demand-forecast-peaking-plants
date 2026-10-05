"""Paired significance tests for two forecasting runs (paper protocol).

Both runs must contain `preds_*.csv` files produced by `run.py`, either directly in the given
directory or in `H<horizon>/` sub-directories. Predictions of the same target day that come from
different seeds or origins are averaged before pairing, so the test compares the two models on the
same forecast origins. Reported quantities:

* Diebold-Mariano statistic with Newey-West correction at lag = horizon,
* moving-block bootstrap p value and the 95% percentile interval of the mean error difference,
* two-sided Wilcoxon signed-rank p value on the paired daily absolute errors.

Example
-------
    python tools/dm_test.py --a "CBR-TD3=output/main/HDQS/H7" \
                            --b "w/o chaotic encoder=output/ablation/HDQS/H7" \
                            --horizon 7 --out output/main/HDQS/dm_H7.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import block_bootstrap_test, dm_test


def load_preds(spec: str) -> pd.DataFrame:
    """Load one run: label=directory, with preds_*.csv in the directory or in H<h>/ sub-directories."""
    label, _, path = spec.partition("=")
    if not label or not path:
        raise SystemExit("--a/--b must be given as label=directory")
    files = sorted(glob.glob(str(Path(path) / "H*" / "preds_*.csv")))
    if not files:
        files = sorted(glob.glob(str(Path(path) / "preds_*.csv")))
    if not files:
        raise SystemExit(f"no preds_*.csv found under {path}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    grouped = (df.groupby(["target_idx", "h"])
                 .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
                 .reset_index())
    grouped["model"] = label
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired significance tests on two forecasting runs.")
    ap.add_argument("--a", action="append", required=True, help="label=directory of the reference run")
    ap.add_argument("--b", action="append", default=[], help="label=directory of the comparison run(s)")
    ap.add_argument("--horizon", type=int, required=True, help="forecast horizon of the compared runs")
    ap.add_argument("--out", default="", help="optional CSV path for the result table")
    args = ap.parse_args()

    frames = [load_preds(s) for s in args.a + args.b]
    ref = frames[0]
    rows = []
    print(f"Paired tests at horizon H = {args.horizon}; reference = {ref.model.iloc[0]}")
    print(f"{'comparison':>24} {'MAE ref':>10} {'MAE other':>10} {'DM':>7} {'p_DM':>9} "
          f"{'p_block':>9} {'p_wilcox':>10}")
    for frame in frames[1:]:
        merged = ref.merge(frame, on=["target_idx", "h"], suffixes=("_r", "_c"))
        if merged.empty:
            continue
        e_ref = np.abs(merged.y_pred_r - merged.y_true_r).to_numpy(float)
        e_oth = np.abs(merged.y_pred_c - merged.y_true_c).to_numpy(float)
        dm = dm_test(e_ref, e_oth, h=args.horizon)
        boot = block_bootstrap_test(e_ref, e_oth, block=max(5, 2 * args.horizon))
        try:
            from scipy.stats import wilcoxon
            p_wilcox = float(wilcoxon(e_ref, e_oth).pvalue)
        except Exception:  # pragma: no cover
            p_wilcox = float("nan")
        rows.append(dict(reference=ref.model.iloc[0], comparison=frame.model.iloc[0],
                         horizon=args.horizon, n_pairs=len(merged),
                         mae_reference=float(e_ref.mean()), mae_comparison=float(e_oth.mean()),
                         dm_statistic=float(dm["dm"]), p_dm=float(dm["p"]),
                         mean_diff=float(boot["mean_diff"]), ci_low=float(boot["lo"]),
                         ci_high=float(boot["hi"]), p_block=float(boot["p"]),
                         p_wilcoxon=p_wilcox))
        print(f"{frame.model.iloc[0]:>24} {e_ref.mean():>10,.0f} {e_oth.mean():>10,.0f} "
              f"{dm['dm']:>7.2f} {dm['p']:>9.3g} {boot['p']:>9.3g} {p_wilcox:>10.3g}")
    if rows:
        table = pd.DataFrame(rows)
        out = Path(args.out) if args.out else None
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(out, index=False)
            print(f"written: {out}")


if __name__ == "__main__":
    main()
