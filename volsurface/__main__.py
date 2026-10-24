"""Fit an SVI surface to an option chain.

    python -m volsurface data/nky_chain_2026-10-23.csv --asof 2026-10-23 --truth data/true_svi.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .chain import clean, load_chain, to_vols
from .plots import smiles, surface_and_term
from .surface import fit_surface
from .svi import SVI


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("chain")
    p.add_argument("--asof", required=True)
    p.add_argument("--truth", help="csv of true svi params, only for the synthetic data")
    p.add_argument("--no-arb-penalty", action="store_true")
    p.add_argument("--out", default="out")
    a = p.parse_args(argv)

    raw = load_chain(a.chain)
    c = clean(raw)
    vols = to_vols(c, a.asof)
    print(f"{len(raw)} quotes, {len(c)} after cleaning, {len(vols)} otm vols")

    surf, table, cal, vols = fit_surface(vols, arb_free=not a.no_arb_penalty)
    pd.set_option("display.width", 200)
    show = ["T", "forward", "n_quotes", "n_outliers", "atm_vol", "rmse_volpts", "pct_inside_bidask",
            "butterfly_ok", "min_g"]
    print("\nper expiry fit\n", table[show].round(4))
    print("\nsvi params\n", table[["a", "b", "rho", "m", "sigma"]].round(5))
    print("\ncalendar check\n", cal.round(5))

    truth = None
    if a.truth:
        t = pd.read_csv(a.truth, parse_dates=["expiry"]).set_index("expiry")
        truth = {e: SVI(r.a, r.b, r.rho, r.m, r.sigma) for e, r in t.iterrows()}
        errs = []
        for (T, s), e in zip(surf.slices, table.index):
            sl = vols[vols["expiry"] == e]
            k = np.linspace(sl["k"].min(), sl["k"].max(), 50)
            errs.append(np.abs(s.vol(k, T) - truth[e].vol(k, T)).max() * 100)
        print("\nmax abs error vs true surface (vol pts) over quoted range:",
              ", ".join(f"{x:.2f}" for x in errs))

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    smiles(vols, surf, table, out / "smiles.png", truth)
    surface_and_term(surf, table, out / "surface.png")
    table.to_csv(out / "svi_params.csv")
    print(f"\nplots and params written to {out}/")


if __name__ == "__main__":
    main()
