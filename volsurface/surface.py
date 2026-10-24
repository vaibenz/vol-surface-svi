"""Fit SVI to every expiry and put the slices together into a surface."""
import numpy as np
import pandas as pd

from .arbitrage import butterfly_check, calendar_check
from .svi import fit_svi


def fit_slice(sl, outlier_mult=3.0, arb_free=True, prev=None):
    """Fit one expiry. Weighted by 1 / bid ask width in vol so tight ATM
    quotes matter more than wide wings, then one pass of outlier removal
    (points more than outlier_mult x their own half spread away from the
    first fit) and a refit."""
    T = float(sl["T"].iloc[0])
    half = 0.5 * (sl["iv_ask"] - sl["iv_bid"])
    half = half.fillna(half.median()).clip(lower=0.002)
    w_vol = 1.0 / half
    weights = w_vol / (2 * sl["iv"] * T)          # residuals are in total variance, convert
    k, iv = sl["k"].to_numpy(), sl["iv"].to_numpy()

    svi, _ = fit_svi(k, iv, T, weights.to_numpy(), arb_free=arb_free, prev=prev)
    resid = np.abs(svi.vol(k, T) - iv)
    keep = resid <= outlier_mult * half.to_numpy()
    if (~keep).any():
        svi, _ = fit_svi(k[keep], iv[keep], T, weights.to_numpy()[keep], arb_free=arb_free, prev=prev)

    fitted = svi.vol(k, T)
    inside = (fitted >= sl["iv_bid"].fillna(-1)) & (fitted <= sl["iv_ask"].fillna(9))
    stats = {
        "expiry": sl["expiry"].iloc[0], "T": T, "forward": sl["forward"].iloc[0],
        "n_quotes": len(sl), "n_outliers": int((~keep).sum()),
        "rmse_volpts": float(np.sqrt(np.mean((fitted[keep] - iv[keep]) ** 2)) * 100),
        "pct_inside_bidask": float(inside[keep].mean() * 100),
        "atm_vol": float(svi.vol(0.0, T)),
        **dict(zip(["a", "b", "rho", "m", "sigma"], svi.as_tuple())),
    }
    return svi, stats, keep


def fit_surface(vols, arb_free=True):
    """Fits expiries shortest first so each one can be kept above the last
    in total variance when arb_free is on."""
    slices, rows, flags = [], [], []
    prev = None
    for _, sl in vols.groupby("expiry"):
        svi, stats, keep = fit_slice(sl, arb_free=arb_free, prev=prev if arb_free else None)
        prev = svi
        bf = butterfly_check(svi)
        stats["butterfly_ok"] = bf["ok"]
        stats["min_g"] = bf["min_g"]
        slices.append((stats["T"], svi))
        rows.append(stats)
        flags.append(pd.Series(~keep, index=sl.index))
    table = pd.DataFrame(rows).set_index("expiry")
    cal = calendar_check(slices)
    vols = vols.assign(outlier=pd.concat(flags))
    return Surface(slices), table, cal, vols


class Surface:
    """Vol lookup at any (k, T). Between expiries total variance is
    interpolated linearly in T at fixed k, which keeps the surface calendar
    arbitrage free as long as the slices are."""

    def __init__(self, slices):
        self.slices = sorted(slices, key=lambda s: s[0])
        self.Ts = np.array([t for t, _ in self.slices])

    def total_var(self, k, T):
        k = np.asarray(k, float)
        ws = np.array([s.w(k) for _, s in self.slices])
        if T <= self.Ts[0]:
            return ws[0] * T / self.Ts[0]            # scale down to zero at T = 0
        if T >= self.Ts[-1]:
            return ws[-1] * T / self.Ts[-1]
        j = np.searchsorted(self.Ts, T)
        t0, t1 = self.Ts[j - 1], self.Ts[j]
        x = (T - t0) / (t1 - t0)
        return (1 - x) * ws[j - 1] + x * ws[j]

    def vol(self, k, T):
        return np.sqrt(np.maximum(self.total_var(k, T), 0.0) / T)
