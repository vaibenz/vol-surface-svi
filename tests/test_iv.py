import numpy as np
import pytest

from volsurface.iv import black, implied_vol, implied_vols


@pytest.mark.parametrize("K", [60, 90, 100, 110, 160])
@pytest.mark.parametrize("call", [True, False])
def test_roundtrip(K, call):
    F, T, df, vol = 100.0, 0.5, 0.99, 0.27
    p = float(black(F, K, T, df, vol, call))
    assert implied_vol(p, F, K, T, df, call) == pytest.approx(vol, abs=1e-7)


def test_below_intrinsic_is_nan():
    assert np.isnan(implied_vol(5.0, 110.0, 100.0, 1.0, 1.0, True))


def test_vectorised_matches_scalar():
    K = np.array([90.0, 100.0, 110.0])
    p = black(100.0, K, 1.0, 1.0, 0.2, True)
    assert np.allclose(implied_vols(p, 100.0, K, 1.0, 1.0, True), 0.2)
