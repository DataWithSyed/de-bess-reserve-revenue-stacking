"""Run the 12-month backtest (1 Nov 2025 - 31 Oct 2026) and the sensitivity grid."""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stack.backtest import STRATEGIES, backtest  # noqa: E402
from stack.data import da_by_local_day, load_da  # noqa: E402
from stack.model import Battery  # noqa: E402

EVAL_START, EVAL_END = "2025-11-01", "2026-10-31"
OUT = ROOT / "results"


def load():
    da = da_by_local_day(load_da(ROOT / "data/de_lu_da_15min.csv.gz"))
    res = pd.read_csv(ROOT / "data/reserve_prices_local.csv")
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(EVAL_START, EVAL_END)]
    return da, res, days


def job(args):
    name, bat_kw, strategies = args
    da, res, days = load()
    df = backtest(da, res, Battery(**bat_kw), days, strategies=strategies)
    df["case"] = name
    for k, v in bat_kw.items():
        df[k] = v
    return df


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    da, res, days = load()
    assert all(d in da for d in days), "missing DA days"
    chk = res[res.date.isin(days)]
    assert len(chk) == 6 * len(days) and chk.notna().all().all(), "missing reserve data"
    jobs = [(f"base_{h:g}h", {"duration_h": h}, STRATEGIES) for h in (1.0, 2.0, 4.0)]
    sens = ("pf_stack", "pf_stack_avg", "xa_stack")
    for e in (0.25, 0.5):
        jobs.append((f"e_afrr_{e}", {"duration_h": 2.0, "e_afrr_h": e}, sens))
    for c in (0.0, 25.0):
        jobs.append((f"deg_{c:g}", {"duration_h": 2.0, "deg_cost_eur_mwh": c}, sens))
    for n in (1.0, 1.5):
        jobs.append((f"cycles_{n:g}", {"duration_h": 2.0, "max_cycles_per_day": n}, sens))
    jobs.append(("e_fcr_0.5", {"duration_h": 2.0, "e_fcr_h": 0.5}, sens))
    with ProcessPoolExecutor(min(8, len(jobs))) as ex:
        daily = pd.concat(list(ex.map(job, jobs)), ignore_index=True)
    daily.to_csv(OUT / "daily_results.csv.gz", index=False, compression="gzip", float_format="%.3f")

    cols = ["da", "fcr", "afrr_pos", "afrr_neg", "deg", "total", "throughput_mwh"]
    annual = daily.groupby(["case", "strategy"])[cols].sum().reset_index()
    annual["cycles_per_year"] = annual["throughput_mwh"] / daily.groupby(["case", "strategy"])["duration_h"].first().values
    annual.to_csv(OUT / "annual_eur_per_mw.csv", index=False, float_format="%.0f")

    base = annual[annual.case.str.startswith("base")].copy()
    summary = {"eval_period": [EVAL_START, EVAL_END], "n_days": len(days), "runtime_s": round(time.time() - t0, 1),
               "base": {}}
    for case, g in base.groupby("case"):
        t = g.set_index("strategy")["total"]
        summary["base"][case] = {s: round(float(t[s]), 0) for s in t.index}
        summary["base"][case]["xa_da_fcr_capture_of_pf_da_fcr_pct"] = round(100 * t["xa_da_fcr"] / t["pf_da_fcr"], 1)
        summary["base"][case]["xa_stack_capture_pct"] = round(100 * t["xa_stack"] / t["pf_stack"], 1)
        summary["base"][case]["xa_stack_capture_of_pf_avg_pct"] = round(100 * t["xa_stack"] / t["pf_stack_avg"], 1)
        summary["base"][case]["xa_stack_vs_pf_da_pct"] = round(100 * t["xa_stack"] / t["pf_da"], 1)
        summary["base"][case]["xa_stack_vs_xa_da_pct"] = round(100 * t["xa_stack"] / t["xa_da"], 1)
    xa = daily[(daily.strategy == "xa_stack") & daily.case.str.startswith("base")]
    acc = xa.groupby("case")[["afrr_pos_offered_mwh", "afrr_pos_accepted_mwh", "afrr_neg_offered_mwh",
                              "afrr_neg_accepted_mwh", "fcr_mw_avg", "afrr_pos_mw_avg", "afrr_neg_mw_avg"]].agg(
        {"afrr_pos_offered_mwh": "sum", "afrr_pos_accepted_mwh": "sum", "afrr_neg_offered_mwh": "sum",
         "afrr_neg_accepted_mwh": "sum", "fcr_mw_avg": "mean", "afrr_pos_mw_avg": "mean", "afrr_neg_mw_avg": "mean"})
    acc["afrr_pos_acceptance"] = acc.afrr_pos_accepted_mwh / acc.afrr_pos_offered_mwh
    acc["afrr_neg_acceptance"] = acc.afrr_neg_accepted_mwh / acc.afrr_neg_offered_mwh
    acc.round(3).to_csv(OUT / "ex_ante_reserve_allocation.csv")
    summary["ex_ante_allocation"] = json.loads(acc.round(3).to_json(orient="index"))
    sens = annual[~annual.case.str.startswith("base") | (annual.case == "base_2h")]
    sens = sens[sens.strategy.isin(["pf_stack", "pf_stack_avg", "xa_stack"])].pivot(index="case", columns="strategy", values="total")
    sens["capture_pct"] = 100 * sens.xa_stack / sens.pf_stack
    sens.round(1).to_csv(OUT / "sensitivity_2h.csv")
    summary["sensitivity_2h"] = json.loads(sens.round(1).to_json(orient="index"))

    r = res[res.date.isin(days)].copy()
    r["month"] = r.date.str[:7]
    r.groupby("month")[["fcr_eur_mw_h", "afrr_pos_avg", "afrr_pos_marg", "afrr_neg_avg", "afrr_neg_marg"]].mean() \
        .round(2).to_csv(OUT / "reserve_prices_monthly_mean.csv")
    p = pd.concat([da[d] for d in days])
    summary["da_price_stats"] = {"mean": round(float(p.mean()), 2), "std": round(float(p.std()), 2),
                                 "n_negative_qh": int((p < 0).sum()), "n_qh": int(len(p))}
    summary["reserve_price_means_eur_mw_h"] = {k: round(float(r[k].mean()), 2) for k in
                                               ["fcr_eur_mw_h", "afrr_pos_avg", "afrr_pos_marg", "afrr_neg_avg", "afrr_neg_marg"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
