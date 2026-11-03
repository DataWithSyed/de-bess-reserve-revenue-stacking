"""Public data access.

* Day-ahead prices (DE-LU, native 15-min MTU since 1 Oct 2025) from the Energy-Charts API
  (Fraunhofer ISE); licence CC BY 4.0, source Bundesnetzagentur | SMARD.de. Cached in data/.
* FCR and aFRR capacity tender results from the German TSOs' regelleistung.net Data Center
  (results overview download). The site states no open reuse licence, so the raw files are
  NOT redistributed in this repo: ``scripts/fetch_data.py`` downloads them locally into
  ``data/raw_regelleistung/`` (git-ignored) and builds a local tidy CSV (also git-ignored).
"""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

EC_API = "https://api.energy-charts.info"
RL_API = ("https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/"
          "resultsoverview")
TZ = "Europe/Berlin"


def _ec_get(endpoint: str, params: dict, max_tries: int = 12) -> dict:
    for _ in range(max_tries):
        r = requests.get(f"{EC_API}/{endpoint}", params=params, timeout=120)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "40")) + 2
            print(f"  429 on {endpoint}, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up on {endpoint} {params}")


def fetch_da_prices(start: str, end: str, out: Path | str) -> pd.Series:
    """15-min DE-LU day-ahead prices [EUR/MWh], UTC index, fetched in monthly chunks."""
    parts = []
    for a in pd.date_range(start, end, freq="MS"):
        b = min(a + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        p = _ec_get("price", {"bzn": "DE-LU", "start": a.strftime("%Y-%m-%d"),
                              "end": b.strftime("%Y-%m-%d")})
        parts.append(pd.Series(p["price"], index=pd.to_datetime(p["unix_seconds"], unit="s", utc=True),
                               name="price", dtype="float64"))
        time.sleep(2)
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    s.index.name = "time_utc"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s.round(2).to_csv(out, compression="gzip")
    return s


def _rl_download(day: str, product: str, raw_dir: Path) -> Path:
    f = raw_dir / f"{product}_{day}.xlsx"
    if f.exists() and f.stat().st_size > 0:
        return f
    params = {"date": day, "exportFormat": "xlsx", "market": "CAPACITY", "productTypes": product}
    for k in range(6):
        try:
            r = requests.get(RL_API, params=params, timeout=60)
            if r.status_code == 200 and r.content[:2] == b"PK":
                f.write_bytes(r.content)
                return f
        except requests.RequestException:
            pass
        time.sleep(2 + 3 * k)
    raise RuntimeError(f"failed {product} {day}")


def _parse_fcr(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path)
    # on a few days a second (cross-border only) FCR tender is published without a German price
    d = d[pd.to_numeric(d["GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"], errors="coerce").notna()]
    d = d.drop_duplicates("PRODUCTNAME", keep="first")
    return pd.DataFrame({
        "date": pd.to_datetime(d["DATE_FROM"]).dt.strftime("%Y-%m-%d"),
        "block": d["PRODUCTNAME"].str[-5:-3].astype(int) // 4,
        # FCR is pay-as-cleared; the German settlement price is in EUR/MW per 4-h block
        "fcr_eur_mw_h": d["GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"].astype(float) / 4.0,
    })


def _parse_afrr(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path)
    prod = d["PRODUCT"].astype(str)
    out = pd.DataFrame({
        "date": pd.to_datetime(d["DATE_FROM"]).dt.strftime("%Y-%m-%d"),
        "dir": prod.str.split("_").str[0].str.lower(),
        "block": prod.str.split("_").str[1].astype(int) // 4,
        "avg": d["GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]"].astype(float),
        "marg": d["GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]"].astype(float),
    })
    w = out.pivot_table(index=["date", "block"], columns="dir", values=["avg", "marg"])
    w.columns = [f"afrr_{d}_{v}" for v, d in w.columns]
    return w.reset_index()


def fetch_reserve_prices(start: str, end: str, raw_dir: Path | str, out: Path | str,
                         workers: int = 4) -> pd.DataFrame:
    """Daily FCR + aFRR capacity results (6 x 4-h blocks per day) -> tidy local CSV."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]
    jobs = [(d, p) for d in days for p in ("FCR", "aFRR")]
    with ThreadPoolExecutor(workers) as ex:
        files = list(ex.map(lambda j: _rl_download(j[0], j[1], raw_dir), jobs))
    fcr = pd.concat([_parse_fcr(f) for f, (_, p) in zip(files, jobs) if p == "FCR"])
    afrr = pd.concat([_parse_afrr(f) for f, (_, p) in zip(files, jobs) if p == "aFRR"])
    df = fcr.merge(afrr, on=["date", "block"], how="outer").sort_values(["date", "block"])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def load_da(path: Path | str) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True)["price"]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def da_by_local_day(s: pd.Series) -> dict[str, pd.Series]:
    """Group a UTC 15-min series by Europe/Berlin delivery day (92/96/100 steps on DST days)."""
    loc = s.tz_convert(TZ)
    return {str(d): g for d, g in loc.groupby(loc.index.date)}
