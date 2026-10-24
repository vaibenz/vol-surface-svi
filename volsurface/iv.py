"""Black (forward) pricing and implied vol inversion.

Working off the forward instead of spot means dividends and rates only
enter through F and the discount factor, which is how option chains are
usually quoted for index options anyway.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

VOL_LO, VOL_HI = 1e-4, 5.0


def black(F, K, T, df, vol, call):
    F, K, T, vol = (np.asarray(x, float) for x in (F, K, T, vol))
    sd = vol * np.sqrt(T)
    d1 = np.log(F / K) / sd + 0.5 * sd
    d2 = d1 - sd
    c = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    p = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(call, c, p)


def black_vega(F, K, T, df, vol):
    sd = vol * np.sqrt(T)
    d1 = np.log(F / K) / sd + 0.5 * sd
    return df * F * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, F, K, T, df, call):
    """Implied vol for one option. Returns nan if the price is outside the
    no arbitrage bounds (below intrinsic or above the max possible value)."""
    intrinsic = df * max(F - K, 0.0) if call else df * max(K - F, 0.0)
    upper = df * F if call else df * K
    if not (intrinsic < price < upper):
        return np.nan

    f = lambda v: float(black(F, K, T, df, v, call)) - price  # noqa: E731
    if f(VOL_LO) > 0 or f(VOL_HI) < 0:
        return np.nan
    return brentq(f, VOL_LO, VOL_HI, xtol=1e-10)


def implied_vols(prices, F, K, T, df, call):
    prices, K, call = np.broadcast_arrays(np.asarray(prices, float), np.asarray(K, float), np.asarray(call, bool))
    F = np.broadcast_to(F, prices.shape)
    T = np.broadcast_to(T, prices.shape)
    df = np.broadcast_to(df, prices.shape)
    return np.array([implied_vol(p, f, k, t, d, c) for p, f, k, t, d, c in
                     zip(prices.ravel(), F.ravel(), K.ravel(), T.ravel(), df.ravel(), call.ravel())]).reshape(prices.shape)
