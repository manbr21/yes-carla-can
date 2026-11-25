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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor="#fcfcfb")
    yaws = sorted(steady)
    half = LANE_WIDTH / 2
    # Simple geometric model: the controller looks at the road about Z metres ahead, so a camera turned by theta sees the lane
    # Z*tan(theta) to the side; the car moves until it looks centred again. The estimator only sees the lane position modulo
    # one lane width, so beyond half a lane the answer wraps around to the other side. Z is fitted on the small angles only.
    small = [y for y in yaws if 0 < y <= 8]
    z_eff = np.mean([abs(np.mean(steady[y]) - base) / np.tan(np.radians(y)) for y in small]) if small else None
    if z_eff:
        grid_deg = np.linspace(0, max(yaws) + 1, 400)
        raw = -z_eff * np.tan(np.radians(grid_deg))
        wrapped = (raw + half) % LANE_WIDTH - half + base
        wrapped[np.abs(np.diff(wrapped, prepend=wrapped[0])) > half] = np.nan      # no line across the wrap
        a1.plot(grid_deg, wrapped, color=muted, linewidth=1.5, linestyle=(0, (1, 2)), zorder=1,
                label="modelo: %.1f m x tan(ângulo), com volta de faixa" % z_eff)
    for yaw in yaws:
        a1.scatter([yaw] * len(steady[yaw]), steady[yaw], color="#2a78d6", s=42, zorder=3,
                   label="medido (cada ponto é uma rodada)" if yaw == yaws[0] else None)
    for edge in (-half, half):
        a1.axhline(edge, color=muted, linestyle=(0, (4, 3)), linewidth=1)
    a1.text(0.2, half + 0.04, "limite da faixa", color=muted, fontsize=9, va="bottom")
    a1.axhline(0, color=muted, linewidth=0.8)
    a1.set_ylim(-2.1, 2.1)
    a1.set_xlabel("Giro da câmera (graus, 0 = sem ataque)", color=ink)
    a1.set_ylabel("Desvio lateral estável (m)\n(negativo = esquerda)", color=ink)
    a1.set_title("Desvio causado pelo ataque", loc="left", color=ink, fontweight="bold")
    a1.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    ramp = plt.cm.Blues(np.linspace(0.35, 0.95, len(yaws)))
    t_end = 0
    for color, yaw in zip(ramp, yaws):
        key = (yaw, min(r for (y, r) in runs if y == yaw))
        d, lat = runs[key]
        a2.plot(d["t_s"], lat, color=color, linewidth=2)
        t_end = max(t_end, d["t_s"][-1])
        a2.text(d["t_s"][-1] + 0.2, lat[-1], "%d°" % yaw if yaw else "sem ataque", color=ink, fontsize=9, va="center")
    for edge in (-half, half):
        a2.axhline(edge, color=muted, linestyle=(0, (4, 3)), linewidth=1)
    a2.set_ylim(-2.1, 2.1)
    a2.set_xlim(3, t_end + 2.2)
    a2.set_xlabel("Tempo de simulação (s)", color=ink)
    a2.set_title("Desvio ao longo do tempo (1ª repetição)", loc="left", color=ink, fontweight="bold")
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
