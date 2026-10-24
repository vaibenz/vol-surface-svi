import numpy as np
import pytest

from volsurface.svi import SVI, fit_svi


def test_derivatives_vs_finite_diff():
    s = SVI(0.02, 0.15, -0.6, 0.02, 0.15)
    k, h = np.linspace(-0.4, 0.3, 15), 1e-5
    assert np.allclose(s.dw(k), (s.w(k + h) - s.w(k - h)) / (2 * h), atol=1e-6)
    assert np.allclose(s.d2w(k), (s.w(k + h) - 2 * s.w(k) + s.w(k - h)) / h**2, atol=1e-3)


def test_recovers_clean_smile():
    true = SVI(0.015, 0.12, -0.55, 0.03, 0.18)
    T = 0.5
    k = np.linspace(-0.35, 0.25, 25)
    fit, rmse = fit_svi(k, true.vol(k, T), T)
    assert rmse < 1e-4
    assert np.allclose(fit.vol(k, T), true.vol(k, T), atol=5e-4)


def test_noisy_smile_rmse_close_to_noise():
    true = SVI(0.03, 0.2, -0.4, 0.0, 0.25)
    T = 1.0
    k = np.linspace(-0.5, 0.4, 30)
    rng = np.random.default_rng(3)
    noise = 0.003
    iv = true.vol(k, T) + noise * rng.standard_normal(len(k))
    _, rmse = fit_svi(k, iv, T)
    assert rmse < 1.5 * noise


def test_min_variance_non_negative():
    true = SVI(0.0005, 0.05, -0.2, 0.0, 0.05)
    T = 0.08
    k = np.linspace(-0.15, 0.12, 15)
    fit, _ = fit_svi(k, true.vol(k, T), T)
    assert fit.min_variance() >= -1e-6


def test_arb_free_fit_removes_butterfly_arbitrage():
    from volsurface.arbitrage import butterfly_check

    bad = SVI(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)  # Axel Vogt params, has arbitrage
    T = 1.0
    k = np.linspace(-1.0, 1.0, 40)
    assert not butterfly_check(bad)["ok"]
    fit, rmse = fit_svi(k, bad.vol(k, T), T, arb_free=True)
    assert butterfly_check(fit)["ok"]
    assert rmse < 0.02  # still close to the quotes, just not exact


def test_calendar_penalty_keeps_slice_above_previous():
    short = SVI(0.04, 0.1, -0.5, 0.0, 0.2)
    T = 0.6
    k = np.linspace(-0.3, 0.3, 20)
    # quotes that sit slightly below the shorter expiry in total variance
    iv = np.sqrt(short.w(k) * 0.97 / T)
    fit, _ = fit_svi(k, iv, T, prev=short)
    grid = np.linspace(-1, 1, 161)
    assert (fit.w(grid) - short.w(grid)).min() > -1e-4
