"""Ex-ante bidding study: why the base ex-ante aFRR stack loses to DA + FCR, and whether
simple non-anticipative fixes recover it. Writes results/bid_study_*.csv and figure 04."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_backtest import load  # noqa: E402
from stack.bidding import (BASE_RULES, Rule, bid_table, reliability, run_rule,  # noqa: E402
                           selection_bias)
from stack.model import Battery  # noqa: E402

OUT, FIG = ROOT / "results", ROOT / "figures"
REF = "xa_da_fcr (base)"
N_BOOT, BLOCK_DAYS = 2000, 7


def job(args):
    dur, rule = args
    da, res, days = load()
    df, det = run_rule(da, res, Battery(duration_h=dur), days, rule)
    df["duration_h"] = dur
    det["duration_h"] = dur
    return df, det


def block_bootstrap_ci(diff: np.ndarray, rng, n=N_BOOT, block=BLOCK_DAYS):
    """95 % CI of the annual sum of daily differences, moving-block bootstrap (7-day blocks)."""
    m = len(diff)
    starts = np.arange(m - block + 1)
    k = int(np.ceil(m / block))
    sums = np.empty(n)
    for i in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, k)])[:m]
        sums[i] = diff[idx].sum()
    return np.percentile(sums, [2.5, 97.5])


def main():
    da, res, days = load()
    jobs = [(d, r) for d in (1.0, 2.0, 4.0) for r in BASE_RULES]
    with ProcessPoolExecutor(8) as ex:
        out = list(ex.map(job, jobs))
    daily = pd.concat([o[0] for o in out], ignore_index=True)
    detail = pd.concat([o[1] for o in out], ignore_index=True)
    daily.to_csv(OUT / "bid_study_daily.csv.gz", index=False, compression="gzip", float_format="%.3f")

    # ---- annual table with half-year split and bootstrap CI vs ex-ante DA + FCR --------------
    rng = np.random.default_rng(42)
    rows = []
    for (dur, rule), g in daily.groupby(["duration_h", "rule"], sort=False):
        ref = daily[(daily.duration_h == dur) & (daily.rule == REF)].set_index("date").total
        g = g.set_index("date")
        diff = (g.total - ref.reindex(g.index)).to_numpy()
        lo, hi = block_bootstrap_ci(diff, rng)
        h1 = g.index < "2026-05-01"
        rows.append(dict(duration_h=dur, rule=rule, total=g.total.sum(), da=g.da.sum(),
                         fcr=g.fcr.sum(), afrr=(g.afrr_pos + g.afrr_neg).sum(), deg=g.deg.sum(),
                         afrr_acceptance=g.afrr_accepted_mwh.sum() / g.afrr_offered_mwh.sum()
                         if g.afrr_offered_mwh.sum() > 0 else np.nan,
                         diff_vs_xa_da_fcr=diff.sum(), ci95_lo=lo, ci95_hi=hi,
                         diff_nov_apr=diff[h1].sum(), diff_may_oct=diff[~h1].sum()))
    ann = pd.DataFrame(rows)
    ann.to_csv(OUT / "bid_study_annual.csv", index=False, float_format="%.1f")
    print(ann.round(1).to_string(index=False))

    # ---- hypothesis 1: calibration of the 28-day acceptance probability (all eval days) -------
    rel = []
    for r in (Rule("optimal (base)"), Rule("quantile 80 %", afrr="quantile", accept_target=0.8),
              Rule("quantile 90 %", afrr="quantile", accept_target=0.9)):
        bt = bid_table(res, days, r)
        t = reliability(bt)
        t.insert(0, "rule", r.name)
        t["bin"] = t["bin"].astype(str)
        rel.append(t)
        rel.append(pd.DataFrame([dict(rule=r.name, bin="all", n=len(bt), predicted=bt.p_pred.mean(),
                                      realised=bt.accepted.mean())]))
    rel = pd.concat(rel, ignore_index=True)
    rel.to_csv(OUT / "bid_study_calibration.csv", index=False, float_format="%.3f")
    print(rel.round(3).to_string(index=False))

    # ---- hypothesis 2: selection bias of the FCR forecast in the base stack (2 h) ------------
    sb = selection_bias(detail[(detail.duration_h == 2.0) & (detail.rule == "xa_stack (base)")])
    sb.to_csv(OUT / "bid_study_selection_bias.csv", index=False, float_format="%.3f")
    print(sb.round(2).to_string(index=False))

    summary = json.loads((OUT / "summary.json").read_text())
    summary["bid_study"] = {
        "reference": REF, "bootstrap": f"{N_BOOT} moving-block resamples of {BLOCK_DAYS}-day blocks",
        "annual_2h": {r.rule: {"total": round(r.total, 0), "diff": round(r.diff_vs_xa_da_fcr, 0),
                               "ci95": [round(r.ci95_lo, 0), round(r.ci95_hi, 0)]}
                      for r in ann[ann.duration_h == 2.0].itertuples()},
        "calibration_all": {r.rule: {"predicted": round(r.predicted, 3), "realised": round(r.realised, 3)}
                            for r in rel[rel.bin == "all"].itertuples()},
        "selection_bias_2h": json.loads(sb.round(3).to_json(orient="records")),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.7, 1]})
    a2 = ann[ann.duration_h == 2.0].reset_index(drop=True)
    y = np.arange(len(a2))[::-1]
    ax = axes[0]
    col = ["#888" if "base" in r else "#1f5f8b" for r in a2.rule]
    ax.barh(y, a2.diff_vs_xa_da_fcr / 1e3, color=col)
    ax.errorbar(a2.diff_vs_xa_da_fcr / 1e3, y, xerr=[(a2.diff_vs_xa_da_fcr - a2.ci95_lo) / 1e3,
                                                      (a2.ci95_hi - a2.diff_vs_xa_da_fcr) / 1e3],
                fmt="none", ecolor="k", capsize=3, lw=1)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{r} ({t/1e3:.1f} k)" for r, t in zip(a2.rule, a2.total)],
                                         fontsize=8)
    ax.set_xlabel("difference vs ex-ante DA + FCR, k EUR/MW/yr (2 h, 95 % block-bootstrap CI)")
    ax.set_title("Ex-ante rules, 2 h battery, 1 Nov 2025 - 31 Oct 2026", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax = axes[1]
    for name, c in (("optimal (base)", "#d62728"), ("quantile 90 %", "#1f5f8b")):
        t = rel[(rel.rule == name) & (rel.bin != "all")]
        ax.plot(t.predicted, t.realised, marker="o", color=c, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("predicted P(accept), 28-day history"); ax.set_ylabel("realised acceptance")
    ax.set_title("aFRR acceptance calibration (all blocks, both directions)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "04_ex_ante_bidding_rules.png", dpi=140); plt.close()


if __name__ == "__main__":
    main()
