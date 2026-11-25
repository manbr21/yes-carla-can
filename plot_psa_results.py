#!/usr/bin/env python3
"""Plot the lane keeping runs (benign vs. Perspective-Shift Attack) and print a summary table.

    python plot_psa_results.py benign.csv psa5.csv psa7.csv --out result.png

The labels are taken from the file names. All runs are cut to the window they have in common.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]      # categorical slots 1-3, in fixed order
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
LANE_HALF_WIDTH = 1.75                          # m (Town04 highway lane: 3.5 m)
SETTLE = (8.0, 13.0)                            # seconds used for the steady state value


def label_of(path):
    name = os.path.splitext(os.path.basename(path))[0]
    if name.startswith("psa"):
        return "Com PSA, câmera girada %s°" % name[3:]
    return "Sem ataque"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="psa_result.png")
    ap.add_argument("--t-max", type=float, default=None, help="end of the common window (default: shortest run)")
    args = ap.parse_args()

    runs = [(label_of(p), np.genfromtxt(p, delimiter=",", names=True)) for p in args.csv]
    t_max = args.t_max or min(d["t_s"][-1] for _, d in runs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=10)

    print("%-28s %12s %14s %10s" % ("rodada", "desvio (m)", "vs. sem ataque", "erro visto"))
    base = None
    for (label, d), color in zip(runs, COLORS):
        m = d["t_s"] <= t_max
        t = d["t_s"][m]
        ax1.plot(t, d["lateral_m"][m], color=color, linewidth=2, label=label)
        ax2.plot(t, d["perceived_error"][m], color=color, linewidth=2)
        s = (d["t_s"] >= SETTLE[0]) & (d["t_s"] <= min(SETTLE[1], t_max))
        lat = float(d["lateral_m"][s].mean())
        err = float(d["perceived_error"][s].mean())
        base = lat if base is None else base
        print("%-28s %+12.2f %+14.2f %+10.3f" % (label, lat, lat - base, err))
        ax1.text(t[-1] + 0.15, d["lateral_m"][m][-1], "%+.2f m" % lat, color=INK, fontsize=10, va="center")

    ax1.axhline(0, color=MUTED, linewidth=1)
    ax1.axhline(-LANE_HALF_WIDTH, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax1.text(t_max + 1.3, -LANE_HALF_WIDTH + 0.05, "limite da faixa", color=MUTED, fontsize=9, va="bottom", ha="right")
    ax1.set_ylim(-1.95, 0.35)
    ax1.set_ylabel("Desvio lateral do centro da faixa (m)\n(negativo = esquerda)", color=INK, fontsize=10)
    ax1.set_title("O ataque desloca o carro e o controlador não percebe", color=INK, fontsize=13,
                  loc="left", fontweight="bold")
    ax1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.1), fontsize=10, labelcolor=INK)

    ax2.axhline(0, color=MUTED, linewidth=1)
    ax2.set_ylim(-0.12, 0.12)
    ax2.set_ylabel("Erro que a câmera\nacha que tem", color=INK, fontsize=10)
    ax2.set_xlabel("Tempo de simulação (s)", color=INK, fontsize=10)
    ax2.set_xlim(3, t_max + 1.4)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("figura:", args.out)


if __name__ == "__main__":
    main()
