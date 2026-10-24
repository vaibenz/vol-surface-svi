import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def smiles(vols, surface, table, path, truth=None):
    exps = list(table.index)
    n = len(exps)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    for ax, (exp, (T, svi)) in zip(axes.ravel(), zip(exps, surface.slices)):
        sl = vols[vols["expiry"] == exp]
        good, bad = sl[~sl["outlier"]], sl[sl["outlier"]]
        ax.errorbar(good["k"], good["iv"] * 100,
                    yerr=[(good["iv"] - good["iv_bid"]).fillna(0) * 100, (good["iv_ask"] - good["iv"]).fillna(0) * 100],
                    fmt=".", ms=3, lw=0.6, color="0.4", label="bid / mid / ask")
        if len(bad):
            ax.plot(bad["k"], bad["iv"] * 100, "rx", ms=5, label="outlier")
        k = np.linspace(sl["k"].min() - 0.05, sl["k"].max() + 0.05, 200)
        ax.plot(k, svi.vol(k, T) * 100, color="C0", lw=1.5, label="svi fit")
        if truth is not None and exp in truth:
            ax.plot(k, truth[exp].vol(k, T) * 100, "--", color="C2", lw=1, label="true")
        r = table.loc[exp]
        ax.set_title(f"{exp:%d %b %y}  T={T:.2f}  rmse {r['rmse_volpts']:.2f}vp", fontsize=9)
        ax.set_xlabel("log moneyness ln(K/F)")
        ax.set_ylabel("implied vol %")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def surface_and_term(surface, table, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    Ts = np.linspace(surface.Ts[0], surface.Ts[-1], 60)
    k = np.linspace(-0.4, 0.25, 80)
    Z = np.array([surface.vol(k, T) for T in Ts]) * 100
    im = ax1.contourf(k, Ts, Z, levels=20, cmap="viridis")
    fig.colorbar(im, ax=ax1, label="implied vol %")
    ax1.set_xlabel("log moneyness")
    ax1.set_ylabel("T (years)")
    ax1.set_title("SVI surface (linear in total variance between expiries)")

    atm = table["atm_vol"] * 100
    # skew as vol difference between k = -0.1 and k = +0.1, scaled by sqrt(T)
    skew = [(s.vol(-0.1, T) - s.vol(0.1, T)) * 100 for T, s in surface.slices]
    ax2.plot(table["T"], atm, "o-", label="ATM vol")
    ax2.set_xlabel("T (years)")
    ax2.set_ylabel("ATM vol %")
    ax3 = ax2.twinx()
    ax3.plot(table["T"], skew, "s--", color="C3", label="90/110 skew")
    ax3.set_ylabel("vol(k=-0.1) - vol(k=+0.1), vol pts")
    ax2.set_title("Term structure")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
