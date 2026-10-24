"""Loading and cleaning an option chain, then turning it into implied vols.

Steps per expiry:
  1. drop quotes with no bid or a crossed / locked market
  2. back out the forward and discount factor from put call parity
     (C - P = df * (F - K)), using a robust Theil-Sen line so one stale
     quote doesn't drag the forward around
  3. keep out of the money options only (puts below F, calls above), since
     ITM quotes are wider and add nothing the OTM side doesn't
  4. implied vol from mid, plus from bid and ask for error bars
"""
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .iv import implied_vols


def load_chain(path):
    c = pd.read_csv(path, parse_dates=["expiry"])
    c["type"] = c["type"].str.upper().str[0]
    return c


def clean(chain, max_rel_spread=1.0):
    c = chain.copy()
    c = c[(c["bid"] > 0) & (c["ask"] > c["bid"])]
    c["mid"] = 0.5 * (c["bid"] + c["ask"])
    c = c[(c["ask"] - c["bid"]) / c["mid"] <= max_rel_spread]
    return c.reset_index(drop=True)


def implied_forward(exp_chain, near_pct=0.08):
    """Forward and discount factor for one expiry from put call parity."""
    piv = exp_chain.pivot_table(index="strike", columns="type", values="mid")
    piv = piv.dropna()
    if len(piv) < 3:
        raise ValueError("not enough strikes with both a call and a put")
    # rough forward first: where C - P crosses zero
    diff = piv["C"] - piv["P"]
    f_guess = float(np.interp(0.0, diff.to_numpy()[::-1], diff.index.to_numpy()[::-1]))
    near = piv[np.abs(piv.index / f_guess - 1) < near_pct]
    if len(near) < 3:
        near = piv
    slope, intercept, _, _ = theilslopes(near["C"] - near["P"], near.index.to_numpy())
    df = -slope
    return intercept / df, df


def to_vols(chain, asof):
    asof = pd.Timestamp(asof)
    out = []
    for exp, g in chain.groupby("expiry"):
        T = (exp - asof).days / 365
        F, df = implied_forward(g)
        otm = g[((g["type"] == "P") & (g["strike"] < F)) | ((g["type"] == "C") & (g["strike"] >= F))].copy()
        call = (otm["type"] == "C").to_numpy()
        K = otm["strike"].to_numpy(float)
        otm["iv"] = implied_vols(otm["mid"], F, K, T, df, call)
        otm["iv_bid"] = implied_vols(otm["bid"], F, K, T, df, call)
        otm["iv_ask"] = implied_vols(otm["ask"], F, K, T, df, call)
        otm["T"], otm["forward"], otm["df"] = T, F, df
        otm["k"] = np.log(K / F)
        out.append(otm)
    v = pd.concat(out, ignore_index=True)
    return v.dropna(subset=["iv"]).sort_values(["expiry", "strike"]).reset_index(drop=True)
