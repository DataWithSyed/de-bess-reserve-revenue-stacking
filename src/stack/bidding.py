"""Ex-ante reserve bidding: diagnostics of the base rule and alternative non-anticipative rules.

The base ex-ante stack (``backtest.run_day``, strategy ``xa_stack``) values aFRR at
bid x P(accept), both taken from the last 28 days of marginal prices (``model.optimal_bid``),
and forecasts FCR with the previous day's price. In the base backtest it earned *less* than
ex-ante DA + FCR. The README named two possible causes without testing them:

1. the 28-day acceptance probabilities are optimistic (poor calibration), and
2. aFRR is picked when the naive FCR forecast happens to be low (selection bias).

This module measures both and backtests rules that address them. Every rule uses only
information available before the D-1 auctions (prices and auction results up to D-1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import DT, day_arrays, naive_da_forecast
from .model import Battery, N_BLOCKS, optimal_bid, settle, solve_day


def quantile_bid(history: np.ndarray, accept_target: float) -> tuple[float, float]:
    """Bid at the (1 - target) quantile of past marginal prices: P_hist(marginal >= bid) ~ target."""
    h = np.sort(np.asarray(history, float)[np.isfinite(history)])
    if len(h) == 0:
        return 0.0, 0.0
    b = float(np.quantile(h, 1.0 - accept_target, method="lower"))
    return b, float((h >= b).mean())


@dataclass(frozen=True)
class Rule:
    """One ex-ante variant.

    fcr_fc:   "d1" previous day's block price, "mean7" mean of the last 7 days for that block
    afrr:     "none", "optimal" (base: max b*P on history), "quantile" (target acceptance)
    accept_target: for "quantile"
    calibrate: scale P(accept) by realised/predicted acceptance of the same rule over the
               previous ``calib_days`` days (non-anticipative recalibration)
    hurdle:   offer aFRR in stage 1 only at hurdle x expected value (risk margin vs FCR)
    """
    name: str
    fcr_fc: str = "d1"
    afrr: str = "optimal"
    accept_target: float = 0.8
    calibrate: bool = False
    calib_days: int = 28
    hurdle: float = 1.0


BASE_RULES = (
    Rule("xa_da_fcr (base)", afrr="none"),
    Rule("xa_stack (base)"),
    Rule("da_fcr, FCR fc = 7-day mean", fcr_fc="mean7", afrr="none"),
    Rule("stack, FCR fc = 7-day mean", fcr_fc="mean7"),
    Rule("stack, recalibrated P(accept)", calibrate=True),
    Rule("stack, recalibrated + FCR 7-day", fcr_fc="mean7", calibrate=True),
    Rule("stack, quantile bid 80 %", afrr="quantile", accept_target=0.8),
    Rule("stack, quantile bid 90 %", afrr="quantile", accept_target=0.9),
    Rule("stack, quantile 90 % + FCR 7-day + recal.", fcr_fc="mean7", afrr="quantile",
         accept_target=0.9, calibrate=True),
    Rule("stack, hurdle 1.5x + FCR 7-day + recal.", fcr_fc="mean7", calibrate=True, hurdle=1.5),
)


def bid_table(reserves: pd.DataFrame, days: list[str], rule: Rule, window: int = 28) -> pd.DataFrame:
    """Bid, predicted acceptance and realised acceptance per day x block x direction."""
    res = reserves.set_index(["date", "block"]).sort_index()
    rows = []
    for day in days:
        d0 = pd.Timestamp(day)
        hist_days = [(d0 - pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in range(1, window + 1)]
        hist = res.loc[res.index.get_level_values(0).isin(hist_days)]
        today = res.loc[day].reindex(range(N_BLOCKS))
        for direction, col in (("pos", "afrr_pos_marg"), ("neg", "afrr_neg_marg")):
            for b in range(N_BLOCKS):
                h = hist.xs(b, level=1)[col].to_numpy()
                if rule.afrr == "quantile":
                    bid, p = quantile_bid(h, rule.accept_target)
                else:
                    bid, p = optimal_bid(h)
                marg = float(today.loc[b, col])
                rows.append(dict(date=day, block=b, direction=direction, bid=bid, p_pred=p,
                                 marginal=marg, accepted=float(bid <= marg + 1e-9)))
    return pd.DataFrame(rows)


def add_calibration(bt: pd.DataFrame, calib_days: int) -> pd.DataFrame:
    """Add a non-anticipative calibration factor: realised / predicted acceptance of this rule's
    bids over the previous ``calib_days`` days (all blocks and directions pooled)."""
    daily = bt.groupby("date")[["accepted", "p_pred"]].sum().sort_index()
    roll = daily.rolling(calib_days, min_periods=7).sum().shift(1)   # up to D-1 only
    factor = (roll["accepted"] / roll["p_pred"]).clip(upper=1.0).fillna(1.0)
    out = bt.merge(factor.rename("calib").reset_index(), on="date", how="left")
    out["p_used"] = (out["p_pred"] * out["calib"]).clip(0, 1)
    return out


def fcr_forecast(res: pd.DataFrame, day: str, kind: str) -> np.ndarray:
    d0 = pd.Timestamp(day)
    if kind == "d1":
        prev = (d0 - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return res.loc[prev].reindex(range(N_BLOCKS))["fcr_eur_mw_h"].to_numpy()
    if kind == "mean7":
        ds = [(d0 - pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in range(1, 8)]
        h = res.loc[res.index.get_level_values(0).isin(ds)]
        return h.groupby(level=1)["fcr_eur_mw_h"].mean().reindex(range(N_BLOCKS)).to_numpy()
    raise ValueError(kind)


def run_rule(da_days: dict, reserves: pd.DataFrame, bat: Battery, days: list[str], rule: Rule,
             window: int = 28) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest one rule. Returns (daily revenue table, block-level stage-1 detail)."""
    res = reserves.set_index(["date", "block"]).sort_index()
    all_days = sorted(da_days)
    need = days
    if rule.calibrate:   # calibration needs bids for the days before the evaluation window
        first = pd.Timestamp(days[0])
        need = [(first - pd.Timedelta(days=k)).strftime("%Y-%m-%d")
                for k in range(rule.calib_days, 0, -1)] + list(days)
        need = [d for d in need if d in set(res.index.get_level_values(0))
                and len(res.loc[res.index.get_level_values(0) < d].index.get_level_values(0).unique()) >= window]
    bt = bid_table(reserves, need, rule, window) if rule.afrr != "none" else None
    if bt is not None:
        bt = add_calibration(bt, rule.calib_days) if rule.calibrate else bt.assign(p_used=bt.p_pred)
        bt = bt.set_index(["date", "direction", "block"]).sort_index()
    rows, detail = [], []
    z = np.zeros(N_BLOCKS)
    for day in days:
        g = da_days[day]
        price, block, slots = day_arrays(g)
        prev_day = all_days[all_days.index(day) - 1]
        fc = naive_da_forecast(da_days[prev_day], slots)
        ffc = fcr_forecast(res, day, rule.fcr_fc)
        rr = res.loc[day].reindex(range(N_BLOCKS))
        fcr = rr["fcr_eur_mw_h"].to_numpy()
        pos_m, neg_m = rr["afrr_pos_marg"].to_numpy(), rr["afrr_neg_marg"].to_numpy()
        hours = np.bincount(block, minlength=N_BLOCKS) * DT
        if bt is None:
            r1 = solve_day(fc, block, bat, DT, ffc, markets=("da", "fcr"))
            fixed = (r1.fcr, z, z)
            paid_p = paid_n = z
            acc_p = acc_n = np.zeros(N_BLOCKS, bool)
            bp = bn = pp = pn = z
        else:
            bp = bt.loc[(day, "pos")]["bid"].to_numpy(); pp = bt.loc[(day, "pos")]["p_used"].to_numpy()
            bn = bt.loc[(day, "neg")]["bid"].to_numpy(); pn = bt.loc[(day, "neg")]["p_used"].to_numpy()
            r1 = solve_day(fc, block, bat, DT, ffc, bp * pp / rule.hurdle, bn * pn / rule.hurdle,
                           markets=("da", "fcr", "afrr"))
            acc_p = (bp <= pos_m + 1e-9) & (r1.afrr_pos > 1e-6)
            acc_n = (bn <= neg_m + 1e-9) & (r1.afrr_neg > 1e-6)
            fixed = (r1.fcr, np.where(acc_p, r1.afrr_pos, 0.0), np.where(acc_n, r1.afrr_neg, 0.0))
            paid_p, paid_n = np.where(acc_p, bp, 0.0), np.where(acc_n, bn, 0.0)
        r2 = solve_day(fc, block, bat, DT, markets=("da",), fixed_reserves=fixed)
        out = settle(price, block, r2, bat, DT, fcr, paid_p, paid_n)
        out.update(date=day, rule=rule.name,
                   afrr_offered_mwh=float(np.sum((r1.afrr_pos + r1.afrr_neg) * hours)),
                   afrr_accepted_mwh=float(np.sum((fixed[1] + fixed[2]) * hours)))
        rows.append(out)
        for b in range(N_BLOCKS):
            detail.append(dict(date=day, block=b, rule=rule.name, fcr_mw=r1.fcr[b],
                               afrr_pos_mw=r1.afrr_pos[b], afrr_neg_mw=r1.afrr_neg[b],
                               fcr_fc=ffc[b], fcr_real=fcr[b], bid_pos=bp[b], bid_neg=bn[b],
                               p_pos=pp[b], p_neg=pn[b], acc_pos=bool(acc_p[b]),
                               acc_neg=bool(acc_n[b])))
    df = pd.DataFrame(rows)
    df["total"] = df[["da", "fcr", "afrr_pos", "afrr_neg", "deg"]].sum(axis=1)
    return df, pd.DataFrame(detail)


def reliability(bt: pd.DataFrame, bins=(0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001)) -> pd.DataFrame:
    """Predicted vs realised acceptance by predicted-probability bin."""
    b = pd.cut(bt["p_pred"], bins, right=False)
    g = bt.groupby(b, observed=True).agg(n=("accepted", "size"), predicted=("p_pred", "mean"),
                                         realised=("accepted", "mean"))
    return g.reset_index().rename(columns={"p_pred": "bin"})


def selection_bias(detail: pd.DataFrame) -> pd.DataFrame:
    """FCR forecast error on blocks where stage 1 chose aFRR over FCR vs blocks where it kept FCR."""
    d = detail.copy()
    d["chose_afrr"] = (d.afrr_pos_mw + d.afrr_neg_mw) > d.fcr_mw + 1e-6
    d["fcr_fc_error"] = d.fcr_fc - d.fcr_real       # negative = forecast too low
    g = d.groupby("chose_afrr").agg(n=("fcr_fc_error", "size"),
                                     mean_fcr_fc=("fcr_fc", "mean"),
                                     mean_fcr_real=("fcr_real", "mean"),
                                     mean_fc_error=("fcr_fc_error", "mean"))
    return g.reset_index()
