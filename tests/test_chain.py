import numpy as np
import pandas as pd

from volsurface.chain import clean, implied_forward, load_chain, to_vols
from volsurface.iv import black


def test_clean_drops_bad_quotes():
    c = pd.DataFrame({"expiry": pd.Timestamp("2027-01-01"), "strike": [1.0, 2.0, 3.0],
                      "type": "C", "bid": [0.0, 5.0, 4.0], "ask": [1.0, 4.0, 4.5]})
    out = clean(c)
    assert out["strike"].tolist() == [3.0]


def test_forward_from_parity():
    F, df, T = 100.0, 0.98, 0.5
    K = np.arange(80.0, 121.0, 2.5)
    rows = []
    for k in K:
        for typ in "CP":
            p = float(black(F, k, T, df, 0.25, typ == "C"))
            rows.append([k, typ, p])
    g = pd.DataFrame(rows, columns=["strike", "type", "mid"])
    f_hat, df_hat = implied_forward(g)
    assert abs(f_hat - F) < 1e-6 and abs(df_hat - df) < 1e-8


def test_sample_chain_forwards_reasonable():
    v = to_vols(clean(load_chain("data/nky_chain_2026-10-23.csv")), "2026-10-23")
    truth = pd.read_csv("data/true_svi.csv", parse_dates=["expiry"]).set_index("expiry")
    f = v.groupby("expiry")["forward"].first()
    assert (np.abs(f / truth["forward"] - 1) < 0.002).all()
