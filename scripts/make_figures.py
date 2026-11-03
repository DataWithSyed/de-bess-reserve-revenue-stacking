"""Figures from results/ (run scripts/run_backtest.py first)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures"
F.mkdir(exist_ok=True)
LABEL = {"xa_da": "DA only\n(ex-ante)", "pf_fcr": "FCR only\n(no forecast)",
         "xa_da_fcr": "DA + FCR\n(ex-ante)", "xa_stack": "DA + FCR + aFRR\n(ex-ante)", "pf_da": "DA only\n(PF)",
         "pf_da_fcr": "DA + FCR\n(PF)", "pf_stack_avg": "Full stack PF\n(aFRR @ avg)",
         "pf_stack": "Full stack PF\n(aFRR @ marginal)"}
ORDER = list(LABEL)
STREAMS = [("da", "Day-ahead arbitrage", "#4c72b0"), ("fcr", "FCR capacity", "#dd8452"),
           ("afrr_pos", "aFRR+ capacity", "#55a868"), ("afrr_neg", "aFRR- capacity", "#8172b3"),
           ("deg", "Degradation cost", "#c44e52")]


def headline():
    a = pd.read_csv(R / "annual_eur_per_mw.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)
    for ax, h in zip(axes, ("1", "2", "4")):
        g = a[a.case == f"base_{h}h"].drop(columns="case").set_index("strategy").loc[ORDER] / 1000
        x = np.arange(len(g))
        pos = np.zeros(len(g))
        for k, lab, c in STREAMS:
            v = g[k].to_numpy()
            if k == "deg":
                ax.bar(x, v, color=c, label=lab)
            else:
                ax.bar(x, v, bottom=pos, color=c, label=lab)
                pos += v
        for i, t in enumerate(g["total"]):
            ax.text(i, pos[i] + 4, f"{t:.0f}", ha="center", fontsize=8, fontweight="bold")
        ax.axvspan(-0.5, 3.5, color="0.93", zorder=-1)
        ax.set_xticks(x, [LABEL[s] for s in ORDER], rotation=90, fontsize=8)
        ax.set_title(f"1 MW / {float(h):g} MWh", fontsize=11)
        ax.axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("k EUR per MW per year (1 Nov 2025 - 31 Oct 2026)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("German BESS revenue stacking: realisable (shaded, non-anticipative) vs perfect-foresight bounds",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(F / "01_annual_revenue_by_strategy.png", dpi=140)
    plt.close(fig)


def monthly():
    d = pd.read_csv(R / "daily_results.csv.gz")
    d = d[d.case == "base_2h"].copy()
    d["month"] = d.date.str[:7]
    m = d.pivot_table(index="month", columns="strategy", values="total", aggfunc="sum") / 1000
    rp = pd.read_csv(R / "reserve_prices_monthly_mean.csv", index_col=0).loc[m.index]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for s, c in (("pf_stack", "0.5"), ("pf_stack_avg", "0.7"), ("xa_da_fcr", "#dd8452"), ("xa_stack", "#55a868"),
                 ("pf_fcr", "#c44e52"), ("xa_da", "#4c72b0")):
        a1.plot(m.index, m[s], marker="o", color=c, ls="--" if s.startswith("pf_stack") else "-",
                label=LABEL[s].replace("\n", " "))
    a1.set_ylabel("k EUR / MW / month (2 h)")
    a1.legend(fontsize=8, ncol=2)
    a1.grid(alpha=0.3)
    a2.plot(rp.index, rp.fcr_eur_mw_h, marker="o", color="#dd8452", label="FCR (pay-as-cleared)")
    a2.plot(rp.index, rp.afrr_pos_avg, marker="s", color="#55a868", label="aFRR+ avg awarded")
    a2.plot(rp.index, rp.afrr_neg_avg, marker="s", color="#8172b3", label="aFRR- avg awarded")
    a2.plot(rp.index, rp.afrr_pos_marg, ls=":", color="#55a868", label="aFRR+ marginal")
    a2.plot(rp.index, rp.afrr_neg_marg, ls=":", color="#8172b3", label="aFRR- marginal")
    a2.set_ylabel("Mean capacity price, EUR/MW/h (DE)")
    a2.legend(fontsize=8, ncol=2)
    a2.grid(alpha=0.3)
    plt.setp(a2.get_xticklabels(), rotation=45)
    fig.suptitle("Monthly revenue (1 MW / 2 MWh) and German reserve capacity prices")
    fig.tight_layout()
    fig.savefig(F / "02_monthly_revenue_and_reserve_prices.png", dpi=140)
    plt.close(fig)


def sensitivity():
    s = pd.read_csv(R / "sensitivity_2h.csv", index_col=0)
    order = ["base_2h", "e_afrr_0.25", "e_afrr_0.5", "e_fcr_0.5", "deg_0", "deg_25", "cycles_1", "cycles_1.5"]
    lab = {"base_2h": "base (e_aFRR 1 h, e_FCR 0.25 h,\ndeg 10 EUR/MWh, 2 cyc/d)",
           "e_afrr_0.25": "aFRR energy 0.25 h", "e_afrr_0.5": "aFRR energy 0.5 h", "e_fcr_0.5": "FCR energy 0.5 h",
           "deg_0": "deg cost 0", "deg_25": "deg cost 25 EUR/MWh", "cycles_1": "max 1 cycle/day",
           "cycles_1.5": "max 1.5 cycles/day"}
    s = s.loc[order] / 1000
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y + 0.27, s.pf_stack, 0.27, color="0.55", label="Full stack PF (aFRR @ marginal)")
    ax.barh(y, s.pf_stack_avg, 0.27, color="0.75", label="Full stack PF (aFRR @ avg)")
    ax.barh(y - 0.27, s.xa_stack, 0.27, color="#55a868", label="Full stack ex-ante")
    ax.set_yticks(y, [lab[i] for i in s.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("k EUR / MW / year (1 MW / 2 MWh)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Sensitivity of stacked revenue to technical / prequalification assumptions")
    fig.tight_layout()
    fig.savefig(F / "03_sensitivity_2h.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    headline()
    monthly()
    sensitivity()
    print("figures written")
