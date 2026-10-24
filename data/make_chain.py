"""Synthetic NKY style option chain with realistic quoting mess.

Prices come from a known SVI surface so the fit can be checked against the
truth. On top of that: bid/ask spreads that widen in the wings, mid noise,
zero bids on far OTM options and a couple of stale quotes.

    python data/make_chain.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from volsurface.iv import black  # noqa: E402
from volsurface.svi import SVI  # noqa: E402

rng = np.random.default_rng(42)
ASOF = pd.Timestamp("2026-10-23")
SPOT, R, Q, TICK = 38500.0, 0.004, 0.018, 125
EXPIRIES = pd.to_datetime(["2026-11-13", "2026-12-11", "2027-01-08", "2027-03-12", "2027-06-11", "2027-12-10"])

rows, truth = [], []
for exp in EXPIRIES:
    T = (exp - ASOF).days / 365
    F = SPOT * np.exp((R - Q) * T)
    df = np.exp(-R * T)
    atm_var = (0.19 + 0.02 * np.sqrt(T)) ** 2 * T        # upward sloping term structure
    b = 0.12 * np.sqrt(T)
    sigma = 0.10 + 0.10 * np.sqrt(T)
    rho = -0.70 + 0.10 * T
    m = 0.01 * np.sqrt(T)
    a = atm_var - b * (rho * -m + np.sqrt(m * m + sigma * sigma))
    svi = SVI(a, b, rho, m, sigma)
    truth.append([exp.date(), T, F, df, *svi.as_tuple()])

    width = 0.30 * np.sqrt(T) + 0.08
    strikes = np.arange(np.floor(F * np.exp(-2.2 * width) / TICK), np.ceil(F * np.exp(1.2 * width) / TICK)) * TICK
    for K in strikes:
        k = np.log(K / F)
        vol = svi.vol(k, T)
        for call in (True, False):
            theo = float(black(F, K, T, df, vol, call))
            spread = max(5.0, 0.02 * theo) * (1 + 3 * abs(k) / width)
            mid = theo + 0.15 * spread * rng.standard_normal()
            bid, ask = mid - spread / 2, mid + spread / 2
            bid = np.floor(bid / 5) * 5          # 5 yen price tick
            ask = np.ceil(ask / 5) * 5
            if bid < 5:
                bid = 0.0
            if rng.uniform() < 0.01:             # stale quote
                bid, ask = bid * 1.4 + 20, ask * 1.4 + 20
            rows.append([exp.date(), K, "C" if call else "P", bid, max(ask, 5.0)])

pd.DataFrame(rows, columns=["expiry", "strike", "type", "bid", "ask"]).to_csv("data/nky_chain_2026-10-23.csv", index=False)
pd.DataFrame(truth, columns=["expiry", "T", "forward", "df", "a", "b", "rho", "m", "sigma"]).to_csv(
    "data/true_svi.csv", index=False)
print(len(rows), "quotes")
