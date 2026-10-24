"""Static arbitrage checks on a fitted surface.

Butterfly (within an expiry): the implied density must be non negative.
For a total variance smile w(k) this is Durrleman's condition

    g(k) = (1 - k w' / (2w))^2 - (w'^2 / 4) (1/w + 1/4) + w'' / 2  >= 0

Calendar (across expiries): total variance must not decrease with T at
any fixed log moneyness, otherwise a calendar spread has negative value.
"""
import numpy as np
import pandas as pd


def durrleman_g(svi, k):
    w, dw, d2w = svi.w(k), svi.dw(k), svi.d2w(k)
    return (1 - k * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2


def butterfly_check(svi, k_grid=None, tol=-1e-8):
    k = np.linspace(-1.0, 1.0, 401) if k_grid is None else np.asarray(k_grid)
    g = durrleman_g(svi, k)
    bad = g < tol
    return {"ok": not bad.any(), "min_g": float(g.min()), "k_at_min": float(k[np.argmin(g)]),
            "bad_k": k[bad]}


def calendar_check(fits, k_grid=None, tol=1e-8):
    """fits: list of (T, SVI) sorted by T."""
    k = np.linspace(-0.6, 0.6, 241) if k_grid is None else np.asarray(k_grid)
    rows = []
    for (t1, s1), (t2, s2) in zip(fits[:-1], fits[1:]):
        diff = s2.w(k) - s1.w(k)
        bad = diff < -tol
        rows.append({"T1": t1, "T2": t2, "ok": not bad.any(), "min_dw": float(diff.min()),
                     "k_at_min": float(k[np.argmin(diff)])})
    return pd.DataFrame(rows)
