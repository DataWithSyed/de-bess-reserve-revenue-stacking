"""Daily LP for a BESS co-optimising day-ahead arbitrage with FCR and aFRR capacity.

Decision variables for one delivery day with ``n`` market time units (15 min) and 6 reserve
blocks (4 h, local time):

* ``c_t, d_t`` charge / discharge power on the day-ahead schedule [MW]
* ``s_t`` state of energy at the end of step t [MWh]
* ``f_b`` FCR (symmetric), ``u_b`` aFRR positive, ``w_b`` aFRR negative capacity [MW]

Constraints (all linear):

* power headroom   d_t + f_b + u_b <= P   and   c_t + f_b + w_b <= P
* energy headroom  s >= E_min + e_fcr f_b + e_afrr u_b   and   s <= E_max - e_fcr f_b - e_afrr w_b
  at both the start and the end of every step (reserve must be deliverable for e_* hours)
* energy balance   s_t = s_{t-1} + eta c_t dt - d_t dt / eta, with s_end = s_0
* throughput limit sum_t d_t dt <= max_cycles_per_day * E

Objective: day-ahead margin - degradation cost per MWh discharged + capacity payments.
Expected activation energy of FCR/aFRR is not modelled (assumed energy-neutral, see README).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

N_BLOCKS = 6


@dataclass(frozen=True)
class Battery:
    power_mw: float = 1.0
    duration_h: float = 2.0
    rte: float = 0.88               # AC round-trip efficiency
    soc_min: float = 0.05           # fraction of energy capacity
    soc_max: float = 0.95
    soc0: float = 0.5               # start and end of every day
    deg_cost_eur_mwh: float = 10.0  # cost per MWh discharged (cycle-ageing proxy)
    max_cycles_per_day: float = 2.0
    e_fcr_h: float = 0.25           # energy reserved per MW FCR in each direction (15-min criterion)
    e_afrr_h: float = 1.0           # energy reserved per MW aFRR (assumption, see README)

    @property
    def energy_mwh(self) -> float:
        return self.power_mw * self.duration_h

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.rte))

    def with_(self, **kw) -> "Battery":
        return replace(self, **kw)


@dataclass
class DayResult:
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    fcr: np.ndarray
    afrr_pos: np.ndarray
    afrr_neg: np.ndarray
    objective: float


def solve_day(price: np.ndarray, block: np.ndarray, bat: Battery, dt: float = 0.25,
              fcr_price: np.ndarray | None = None, afrr_pos_price: np.ndarray | None = None,
              afrr_neg_price: np.ndarray | None = None, markets=("da", "fcr", "afrr"),
              fixed_reserves: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> DayResult:
    """Solve one delivery day.

    price: DA price per step [EUR/MWh]; block: reserve block index (0..5) per step.
    Reserve prices are per block in EUR/MW/h. ``fixed_reserves`` = (f, u, w) pins the reserve
    volumes (stage-2 re-dispatch after the capacity auctions).
    """
    price = np.asarray(price, float)
    block = np.asarray(block, int)
    n = len(price)
    P, E = bat.power_mw, bat.energy_mwh
    emin, emax, s0 = bat.soc_min * E, bat.soc_max * E, bat.soc0 * E
    eta = bat.eta
    z6 = np.zeros(N_BLOCKS)
    fp = z6 if fcr_price is None else np.asarray(fcr_price, float)
    up = z6 if afrr_pos_price is None else np.asarray(afrr_pos_price, float)
    wp = z6 if afrr_neg_price is None else np.asarray(afrr_neg_price, float)
    hours = np.bincount(block, minlength=N_BLOCKS) * dt

    iC, iD, iS = 0, n, 2 * n
    iF, iU, iW = 3 * n, 3 * n + N_BLOCKS, 3 * n + 2 * N_BLOCKS
    nv = 3 * n + 3 * N_BLOCKS

    cost = np.zeros(nv)
    cost[iC:iC + n] = price * dt
    cost[iD:iD + n] = -(price - bat.deg_cost_eur_mwh) * dt
    cost[iF:iF + N_BLOCKS] = -fp * hours
    cost[iU:iU + N_BLOCKS] = -up * hours
    cost[iW:iW + N_BLOCKS] = -wp * hours

    da_ub = P if "da" in markets else 0.0
    bounds = [(0, da_ub)] * (2 * n) + [(emin, emax)] * n
    for key, idx in (("fcr", 0), ("afrr", 1), ("afrr", 2)):
        if fixed_reserves is not None:
            bounds += [(float(v), float(v)) for v in fixed_reserves[idx]]
        else:
            ub = P if key in markets else 0.0
            bounds += [(0, ub)] * N_BLOCKS

    rows, cols, vals, rhs = [], [], [], []
    r = 0

    def add(entries, b):
        nonlocal r
        for c_, v_ in entries:
            rows.append(r); cols.append(c_); vals.append(v_)
        rhs.append(b)
        r += 1

    ef, ea = bat.e_fcr_h, bat.e_afrr_h
    for t in range(n):
        b = block[t]
        add([(iD + t, 1), (iF + b, 1), (iU + b, 1)], P)
        add([(iC + t, 1), (iF + b, 1), (iW + b, 1)], P)
        # end-of-step energy headroom
        add([(iS + t, -1), (iF + b, ef), (iU + b, ea)], -emin)
        add([(iS + t, 1), (iF + b, ef), (iW + b, ea)], emax)
        # start-of-step energy headroom
        if t == 0:
            add([(iF + b, ef), (iU + b, ea)], s0 - emin)
            add([(iF + b, ef), (iW + b, ea)], emax - s0)
        else:
            add([(iS + t - 1, -1), (iF + b, ef), (iU + b, ea)], -emin)
            add([(iS + t - 1, 1), (iF + b, ef), (iW + b, ea)], emax)
    add([(iD + t, dt) for t in range(n)], bat.max_cycles_per_day * E)
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))

    rows, cols, vals, beq = [], [], [], []
    for t in range(n):
        rows += [t, t, t]
        cols += [iS + t, iC + t, iD + t]
        vals += [1.0, -eta * dt, dt / eta]
        if t > 0:
            rows.append(t); cols.append(iS + t - 1); vals.append(-1.0)
            beq.append(0.0)
        else:
            beq.append(s0)
    rows.append(n); cols.append(iS + n - 1); vals.append(1.0); beq.append(s0)
    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(n + 1, nv))

    res = linprog(cost, A_ub=A_ub, b_ub=rhs, A_eq=A_eq, b_eq=beq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x
    return DayResult(x[iC:iC + n], x[iD:iD + n], x[iS:iS + n], x[iF:iF + N_BLOCKS],
                     x[iU:iU + N_BLOCKS], x[iW:iW + N_BLOCKS], -res.fun)


def settle(price: np.ndarray, block: np.ndarray, res: DayResult, bat: Battery, dt: float,
           fcr_paid: np.ndarray, afrr_pos_paid: np.ndarray, afrr_neg_paid: np.ndarray) -> dict:
    """Realised revenue streams [EUR] of a schedule at actual prices."""
    hours = np.bincount(np.asarray(block, int), minlength=N_BLOCKS) * dt
    da = float(np.sum(np.asarray(price) * (res.discharge - res.charge)) * dt)
    thr = float(np.sum(res.discharge) * dt)
    return {
        "da": da,
        "fcr": float(np.sum(res.fcr * fcr_paid * hours)),
        "afrr_pos": float(np.sum(res.afrr_pos * afrr_pos_paid * hours)),
        "afrr_neg": float(np.sum(res.afrr_neg * afrr_neg_paid * hours)),
        "deg": -bat.deg_cost_eur_mwh * thr,
        "throughput_mwh": thr,
    }


def optimal_bid(history: np.ndarray) -> tuple[float, float]:
    """Pay-as-bid shading for a price-taker: choose bid b maximising b * P(marginal >= b)
    on the empirical distribution of past marginal prices. Returns (bid, acceptance prob)."""
    h = np.sort(np.asarray(history, float)[np.isfinite(history)])
    if len(h) == 0:
        return 0.0, 0.0
    p_acc = np.array([(h >= v).mean() for v in h])
    val = h * p_acc
    i = int(np.argmax(val))
    return float(h[i]), float(p_acc[i])
