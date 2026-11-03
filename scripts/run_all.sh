#!/usr/bin/env bash
# Full pipeline: download public data, run the 12-month backtest, draw figures, run tests.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/fetch_data.py reserve          # regelleistung.net results (local only, ~5 min)
[ -f data/de_lu_da_15min.csv.gz ] || python scripts/fetch_data.py da
python scripts/run_backtest.py
python scripts/make_figures.py
python -m pytest -q
