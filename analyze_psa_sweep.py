#!/usr/bin/env python3
"""Summarise a folder of lane keeping runs (yawNN_rM.csv, made by run_psa_sweep.sh): table and plot.

    python analyze_psa_sweep.py psa_sweep_20261125_0100
"""
import glob
import os
import re
import sys

import numpy as np

LANE_WIDTH = 3.5
SETTLE = (8.0, 13.0)          # seconds of simulation used for the steady state


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    lat = d["lateral_m"].copy()
    # The ground truth switches to the neighbouring lane when the car crosses a marking (jump of ~one lane):
    # remove the jump so the offset is always measured from the lane where the run started.
    for i in np.where(np.abs(np.diff(lat)) > 0.6 * LANE_WIDTH)[0]:
        lat[i + 1:] -= np.round((lat[i + 1] - lat[i]) / LANE_WIDTH) * LANE_WIDTH
    return d, lat


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = {}
    for path in sorted(glob.glob(os.path.join(folder, "yaw*_r*.csv"))):
        m = re.search(r"yaw(\d+)_r(\d+)", path)
        if m and os.path.getsize(path) > 100:
            runs[(int(m.group(1)), int(m.group(2)))] = load(path)
    if not runs:
        print("no runs found in", folder)
        return

    print("%-5s %-4s %12s %12s %10s %9s %8s %9s" % ("yaw", "rep", "desvio (m)", "max |lat| (m)", "erro visto", "fora faixa", "km/h", "alarme"))
    steady = {}
    for (yaw, rep), (d, lat) in sorted(runs.items()):
        t = d["t_s"]
        s = (t >= SETTLE[0]) & (t <= SETTLE[1])
        ss = float(lat[s].mean()) if s.any() else float("nan")
        steady.setdefault(yaw, []).append(ss)
        alarm = "-"
        if "alarm" in d.dtype.names and (d["alarm"] > 0).any():
            alarm = "%.1f s" % t[d["alarm"] > 0][0]
        print("%-5d %-4d %+12.2f %12.2f %+10.3f %9.0f%% %8.1f %9s" % (
            yaw, rep, ss, np.abs(lat).max(), d["perceived_error"][s].mean() if s.any() else float("nan"),
            100 * d["off_lane"].mean(), d["speed_kmh"][s].mean() if s.any() else float("nan"), alarm))

    print("\nPor yaw (média e variação entre as repetições, desvio em m, negativo = esquerda):")
    base = np.mean(steady[0]) if 0 in steady else 0.0
    for yaw in sorted(steady):
        v = np.array(steady[yaw])
        print("  yaw %2d: média %+.2f  (min %+.2f, max %+.2f, n=%d)  efeito vs. sem ataque %+.2f" % (
            yaw, v.mean(), v.min(), v.max(), len(v), v.mean() - base))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: no plot)")
        return
    ink, muted, grid = "#0b0b0b", "#52514e", "#e3e2de"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="#fcfcfb")
    yaws = sorted(steady)
    for yaw in yaws:
        a1.scatter([yaw] * len(steady[yaw]), steady[yaw], color="#2a78d6", s=36, zorder=3)
    a1.plot(yaws, [np.mean(steady[y]) for y in yaws], color="#2a78d6", linewidth=2, zorder=2)
    a1.axhline(-LANE_WIDTH / 2, color=muted, linestyle=(0, (4, 3)), linewidth=1)
    a1.text(yaws[-1], -LANE_WIDTH / 2 + 0.06, "limite da faixa", color=muted, fontsize=9, ha="right", va="bottom")
    a1.set_xlabel("Giro da câmera (graus, 0 = sem ataque)", color=ink)
    a1.set_ylabel("Desvio lateral estável (m)", color=ink)
    a1.set_title("Desvio causado pelo ataque", loc="left", color=ink, fontweight="bold")
    ramp = plt.cm.Blues(np.linspace(0.35, 0.95, len(yaws)))
    for color, yaw in zip(ramp, yaws):
        key = (yaw, min(r for (y, r) in runs if y == yaw))
        d, lat = runs[key]
        a2.plot(d["t_s"], lat, color=color, linewidth=2, label="%d°" % yaw if yaw else "sem ataque")
    a2.axhline(-LANE_WIDTH / 2, color=muted, linestyle=(0, (4, 3)), linewidth=1)
    a2.set_xlabel("Tempo de simulação (s)", color=ink)
    a2.set_title("Desvio ao longo do tempo (1ª repetição)", loc="left", color=ink, fontweight="bold")
    a2.legend(frameon=False, fontsize=9, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 0.1))
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)
    fig.tight_layout()
    out = os.path.join(folder, "sweep.png")
    fig.savefig(out, dpi=180, facecolor="#fcfcfb")
    print("\nfigura:", out)


if __name__ == "__main__":
    main()
