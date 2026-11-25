#!/usr/bin/env python3
"""Compare the lane keeping runs with the hood defence off and on (made by run_hood_defense.sh).

    python analyze_hood_defense.py hood_defense_20261125_1300
"""
import glob
import os
import re
import sys

import numpy as np

LANE_WIDTH = 3.5
FOCAL_PX = 240.0              # 480 px wide image, 90 degrees of field of view
SETTLE = (8.0, 13.0)


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    lat = d["lateral_m"].copy()
    for i in np.where(np.abs(np.diff(lat)) > 0.6 * LANE_WIDTH)[0]:
        lat[i + 1:] -= np.round((lat[i + 1] - lat[i]) / LANE_WIDTH) * LANE_WIDTH
    return d, lat


def collect(folder):
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "yaw*_r*.csv"))):
        m = re.search(r"yaw(\d+)_r(\d+)", path)
        if m and os.path.getsize(path) > 100:
            out.setdefault(int(m.group(1)), []).append(load(path))
    return out


def steady(runs):
    vals = []
    for d, lat in runs:
        s = (d["t_s"] >= SETTLE[0]) & (d["t_s"] <= SETTLE[1])
        vals.append(float(lat[s].mean()))
    return float(np.mean(vals))


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    off, on = collect(os.path.join(folder, "off")), collect(os.path.join(folder, "compensate"))
    if not off and not on:
        print("no runs found in", folder)
        return
    yaws = sorted(set(off) | set(on))
    print("%-5s %14s %14s %12s %14s %16s" % ("yaw", "sem defesa (m)", "com defesa (m)", "alarme", "capô medido", "capô previsto"))
    rows = []
    for yaw in yaws:
        a = steady(off[yaw]) if yaw in off else float("nan")
        b = steady(on[yaw]) if yaw in on else float("nan")
        alarm, hood = "-", float("nan")
        src = (off.get(yaw) or on.get(yaw))[0][0]
        if "alarm" in src.dtype.names:
            if (src["alarm"] > 0).any():
                alarm = "%.1f s" % src["t_s"][src["alarm"] > 0][0]
            s = src["t_s"] >= 5
            hood = float(np.nanmean(src["hood_shift_px"][s]))
        pred = -FOCAL_PX * np.tan(np.radians(yaw))
        rows.append((yaw, a, b, hood, pred))
        print("%-5d %+14.2f %+14.2f %12s %+11.1f px %+13.1f px" % (yaw, a, b, alarm, hood, pred))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: no plot)")
        return
    ink, muted, grid = "#0b0b0b", "#52514e", "#e3e2de"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="#fcfcfb")
    ys = [r[0] for r in rows]
    a1.plot(ys, [r[1] for r in rows], color="#2a78d6", linewidth=2, marker="o", label="sem defesa")
    a1.plot(ys, [r[2] for r in rows], color="#eb6834", linewidth=2, marker="o", label="com defesa do capô")
    a1.axhline(0, color=muted, linewidth=0.8)
    a1.axhline(-LANE_WIDTH / 2, color=muted, linestyle=(0, (4, 3)), linewidth=1)
    a1.text(ys[-1], -LANE_WIDTH / 2 + 0.05, "limite da faixa", color=muted, fontsize=9, ha="right", va="bottom")
    a1.set_xlabel("Giro da câmera (graus, 0 = sem ataque)", color=ink)
    a1.set_ylabel("Desvio lateral estável (m)\n(negativo = esquerda)", color=ink)
    a1.set_title("A defesa devolve o carro ao centro?", loc="left", color=ink, fontweight="bold")
    a1.legend(frameon=False, fontsize=10, loc="center left")
    a2.scatter([r[4] for r in rows], [r[3] for r in rows], color="#2a78d6", s=42, zorder=3)
    lim = max(abs(r[4]) for r in rows) * 1.1 + 1
    a2.plot([-lim, lim], [-lim, lim], color=muted, linestyle=(0, (1, 2)), linewidth=1.5, label="previsto: f x tan(ângulo)")
    for r in rows:
        a2.annotate("%d°" % r[0], (r[4], r[3]), textcoords="offset points", xytext=(6, -10), fontsize=9, color=ink)
    a2.set_xlabel("Deslocamento previsto do capô (px)", color=ink)
    a2.set_ylabel("Deslocamento medido do capô (px)", color=ink)
    a2.set_title("O capô mede o deslocamento da imagem", loc="left", color=ink, fontweight="bold")
    a2.legend(frameon=False, fontsize=9, loc="upper left")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)
    fig.tight_layout()
    path = os.path.join(folder, "hood_defense.png")
    fig.savefig(path, dpi=180, facecolor="#fcfcfb")
    print("\nfigura:", path)


if __name__ == "__main__":
    main()
