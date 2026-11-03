"""Download all public inputs (re-runnable; raw regelleistung files are cached locally, git-ignored)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stack.data import fetch_da_prices, fetch_reserve_prices  # noqa: E402

START, END = "2025-10-01", "2026-10-31"

if __name__ == "__main__":
    what = sys.argv[1:] or ["da", "reserve"]
    if "reserve" in what:
        df = fetch_reserve_prices(START, END, ROOT / "data/raw_regelleistung", ROOT / "data/reserve_prices_local.csv")
        print("reserve rows", len(df))
    if "da" in what:
        s = fetch_da_prices(START, END, ROOT / "data/de_lu_da_15min.csv.gz")
        print("da values", len(s), s.index.min(), s.index.max())
