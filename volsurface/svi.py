"""Raw SVI (Gatheral 2004) fitted one expiry at a time.

Total implied variance as a function of log moneyness k = ln(K / F):

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

with b >= 0, |rho| < 1, sigma > 0 and a + b * sigma * sqrt(1 - rho^2) >= 0
(the last one keeps the minimum variance non negative).
"""
from dataclasses import astuple, dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SVI:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def w(self, k):
        k = np.asarray(k, float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2))

    def vol(self, k, T):
        return np.sqrt(np.maximum(self.w(k), 0.0) / T)

    # first and second derivatives in k, needed for the butterfly check
    def dw(self, k):
        x = np.asarray(k, float) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.sigma**2))

    def d2w(self, k):
        x = np.asarray(k, float) - self.m
        return self.b * self.sigma**2 / (x * x + self.sigma**2) ** 1.5

    def min_variance(self):
        return self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)

    def as_tuple(self):
        return astuple(self)


def _residuals(p, k, w_mkt, weights):
    a, b, rho, m, sig = p
    model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig**2))
    res = (model - w_mkt) * weights
    # soft penalty for negative minimum variance
    pen = max(0.0, -(a + b * sig * np.sqrt(1 - rho**2))) * 1e3
    return np.append(res, pen)


def fit_svi(k, iv, T, weights=None, n_starts=8, seed=0):
    """Fit raw SVI to one smile.

    k:  log moneyness, iv: implied vols, T: year fraction.
    weights: per point weights (e.g. vega or 1 / bid ask spread). The fit is
    done in total variance space, so the default weights divide by 2*iv*T to
    make residuals roughly vol errors.
    Several random starts because SVI has local minima, best one kept.
    """
    k = np.asarray(k, float)
    iv = np.asarray(iv, float)
    w_mkt = iv**2 * T
    if weights is None:
        weights = 1.0 / (2 * iv * T)
    weights = np.asarray(weights, float)

    rng = np.random.default_rng(seed)
    w_atm = float(np.interp(0.0, k, w_mkt))
    lo = [-1.0, 1e-6, -0.999, 2 * k.min() - 0.1, 1e-4]
    hi = [max(w_mkt) * 2, 5.0, 0.999, 2 * k.max() + 0.1, 2.0]

    best = None
    for i in range(n_starts):
        if i == 0:
            x0 = [w_atm * 0.8, 0.1, -0.5, 0.0, 0.1]
        else:
            x0 = [rng.uniform(0, w_atm), rng.uniform(0.01, 0.5), rng.uniform(-0.9, 0.5),
                  rng.uniform(k.min(), k.max()), rng.uniform(0.01, 0.5)]
        x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        res = least_squares(_residuals, x0, bounds=(lo, hi), args=(k, w_mkt, weights), method="trf")
        if best is None or res.cost < best.cost:
            best = res

    params = SVI(*best.x)
    rmse_vol = float(np.sqrt(np.mean((params.vol(k, T) - iv) ** 2)))
    return params, rmse_vol
