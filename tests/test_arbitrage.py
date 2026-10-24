import numpy as np

from volsurface.arbitrage import butterfly_check, calendar_check
from volsurface.svi import SVI


def test_reasonable_smile_passes():
    assert butterfly_check(SVI(0.02, 0.1, -0.5, 0.0, 0.2))["ok"]


def test_axel_vogt_example_fails():
    # well known SVI parameter set from Gatheral and Jacquier (2014) that
    # has butterfly arbitrage
    s = SVI(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    r = butterfly_check(s, np.linspace(-1.5, 1.5, 601))
    assert not r["ok"]
    assert r["min_g"] < 0


def test_calendar_crossing_detected():
    short = SVI(0.02, 0.1, -0.5, 0.0, 0.2)
    long_ok = SVI(0.04, 0.12, -0.5, 0.0, 0.2)
    long_bad = SVI(0.01, 0.1, -0.5, 0.0, 0.2)
    good = calendar_check([(0.25, short), (0.5, long_ok)])
    bad = calendar_check([(0.25, short), (0.5, long_bad)])
    assert good["ok"].all()
    assert not bad["ok"].all()
