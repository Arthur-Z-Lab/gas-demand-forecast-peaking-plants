#!/usr/bin/env bash
# Protocol used in the paper: 5 horizons x 5 folds x 5 seeds on the primary dataset.
#
#   PY=/path/to/python bash scripts/run_protocol.sh
#
set -euo pipefail
PY="${PY:-python}"
DATA="${DATA:-data/HDQS.xlsx}"
COL="${COL:-HDQS}"
OUT="${OUT:-output/main/HDQS}"
SEEDS="${SEEDS:-42,0,1,2,3}"
EPISODES="${EPISODES:-40000}"

for H in 1 3 5 7 15; do
    echo "=== horizon ${H} days ==="
    "$PY" run.py --data "$DATA" --col "$COL" --horizon "$H" --seeds "$SEEDS" \
        --n_folds 5 --total_episodes "$EPISODES" --outdir "${OUT}/H${H}"
done
echo "finished. results: ${OUT}"
