"""Out-of-sample validation of the bidding-rule choice (added 17 Nov 2026).

Reads only the committed ``results/bid_study_daily.csv.gz`` (10 ex-ante rules x 3 durations x
365 days), so it runs without the regelleistung.net files. Writes
``results/rule_validation_spa.csv``, ``results/rule_validation_walk_forward.csv``,
``results/rule_validation_picks.csv`` and ``figures/05_rule_validation.png``, and adds a
``rule_validation`` block to ``results/summary.json``.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_bid_study import block_bootstrap_ci  # noqa: E402
from stack.selection import single_pair_p, spa_test, walk_forward_select, wide_daily  # noqa: E402

OUT, FIG = ROOT / "results", ROOT / "figures"
REF = "xa_da_fcr (base)"
BEST = "stack, quantile 90 % + FCR 7-day + recal."
WINDOWS = (28, 56, 91, None)   # None = expanding
N_BOOT = 2000


def main():
    daily = pd.read_csv(OUT / "bid_study_daily.csv.gz")
    spa_rows, wf_rows, pick_rows = [], [], []
    for dur in (1.0, 2.0, 4.0):
        w = wide_daily(daily, dur)
        others = [c for c in w.columns if c != REF]
        diffs = w[others].to_numpy() - w[[REF]].to_numpy()
        for mb in (7.0, 14.0):
            s = spa_test(diffs, mean_block=mb, n_boot=N_BOOT, seed=0)
            best_col = others[int(np.argmax(diffs.mean(axis=0)))]
            spa_rows.append(dict(duration_h=dur, mean_block_days=mb, n_rules=len(others), best_rule=best_col,
                                 best_diff_eur=float(diffs.sum(axis=0).max()), t_spa=s["t_spa"],
                                 p_spa_c=s["p_spa_c"], p_spa_l=s["p_spa_l"], p_reality_check=s["p_spa_u"],
                                 p_best_alone_no_search=single_pair_p(
                                     (w[best_col] - w[REF]).to_numpy(), mb, N_BOOT, 0)))
        rng = np.random.default_rng(42)
        for lag in (1, 2):
            for win in WINDOWS:
                sel = walk_forward_select(w, win, default=REF, lag=lag)
                v = sel.set_index("date").total
                d_ref = (v - w[REF]).to_numpy()
                d_best = (v - w[BEST]).to_numpy()
                lo, hi = block_bootstrap_ci(d_ref, rng, n=N_BOOT)
                h1 = v.index < "2026-05-01"
                counts = sel.rule.value_counts()
                wf_rows.append(dict(duration_h=dur, lag_days=lag, window_days="expanding" if win is None else win,
                                    total=v.sum(), diff_vs_da_fcr=d_ref.sum(), ci95_lo=lo, ci95_hi=hi,
                                    diff_nov_apr=d_ref[h1].sum(), diff_may_oct=d_ref[~h1].sum(),
                                    diff_vs_insample_best=d_best.sum(), n_switches=int((sel.rule != sel.rule.shift()).sum() - 1),
                                    most_used_rule=counts.index[0], most_used_share=counts.iloc[0] / len(sel)))
                if lag == 1:
                    s2 = sel.copy(); s2["duration_h"] = dur; s2["window_days"] = "expanding" if win is None else win
                    pick_rows.append(s2)
        # reference rows: fixed rules
        for name in (REF, BEST):
            wf_rows.append(dict(duration_h=dur, lag_days=np.nan, window_days=f"fixed: {name}", total=w[name].sum(),
                                diff_vs_da_fcr=(w[name] - w[REF]).sum(), diff_vs_insample_best=(w[name] - w[BEST]).sum()))
    spa = pd.DataFrame(spa_rows)
    wf = pd.DataFrame(wf_rows)
    picks = pd.concat(pick_rows, ignore_index=True)
    spa.to_csv(OUT / "rule_validation_spa.csv", index=False, float_format="%.4f")
    wf.to_csv(OUT / "rule_validation_walk_forward.csv", index=False, float_format="%.1f")
    picks.to_csv(OUT / "rule_validation_picks.csv.gz", index=False, compression="gzip", float_format="%.3f")
    print(spa.round(3).to_string(index=False))
    print(wf.round(1).to_string(index=False))

    summary = json.loads((OUT / "summary.json").read_text())
    summary["rule_validation"] = {
        "reference": REF, "insample_best": BEST,
        "spa_mean_block_7": {str(r.duration_h): {"p_spa_c": round(r.p_spa_c, 4), "p_reality_check": round(r.p_reality_check, 4),
                                                  "p_best_alone_no_search": round(r.p_best_alone_no_search, 4)}
                             for r in spa[spa.mean_block_days == 7.0].itertuples()},
        "walk_forward_lag1": {f"{r.duration_h}|{r.window_days}": {"total": round(r.total, 0), "diff_vs_da_fcr": round(r.diff_vs_da_fcr, 0),
                                                                   "ci95": [round(r.ci95_lo, 0), round(r.ci95_hi, 0)]}
                              for r in wf[wf.lag_days == 1].itertuples()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure -------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    sub = wf[wf.lag_days == 1]
    labels = [str(x) for x in WINDOWS[:-1]] + ["expanding"]
    xs = np.arange(len(labels))
    for k, (dur, c) in enumerate(((1.0, "#9ecae1"), (2.0, "#1f5f8b"), (4.0, "#08306b"))):
        s = sub[sub.duration_h == dur].set_index("window_days").reindex([int(l) if l.isdigit() else l for l in labels])
        x = xs + (k - 1) * 0.26
        ax.bar(x, s.diff_vs_da_fcr / 1e3, width=0.25, color=c, label=f"{dur:g} h")
        ax.errorbar(x, s.diff_vs_da_fcr / 1e3, yerr=[(s.diff_vs_da_fcr - s.ci95_lo) / 1e3, (s.ci95_hi - s.diff_vs_da_fcr) / 1e3],
                    fmt="none", ecolor="k", capsize=2, lw=0.8)
        best = wf[(wf.duration_h == dur) & (wf.window_days == f"fixed: {BEST}")].diff_vs_da_fcr.iloc[0] / 1e3
        ax.hlines(best, xs[0] - 0.5, xs[-1] + 0.5, colors=c, linestyles=":", lw=1.2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([f"trailing {l} d" if l != "expanding" else l for l in labels], fontsize=8)
    ax.set_ylabel("k EUR/MW/yr vs ex-ante DA + FCR (95 % CI)")
    ax.set_title("Walk-forward rule selection (lag 1 d); dotted = in-sample best rule", fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    p2 = picks[(picks.duration_h == 2.0) & (picks.window_days == 56)].copy()
    p2["date"] = pd.to_datetime(p2.date)
    rules = sorted(daily.rule.unique(), key=lambda r: (r != REF, r))
    ymap = {r: i for i, r in enumerate(rules)}
    ax.scatter(p2.date, p2.rule.map(ymap), s=6, color="#1f5f8b")
    ax.set_yticks(range(len(rules))); ax.set_yticklabels(rules, fontsize=7)
    ax.set_title("Rule picked each day, 2 h, trailing 56 d", fontsize=9); ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(FIG / "05_rule_validation.png", dpi=140); plt.close()


if __name__ == "__main__":
    main()
