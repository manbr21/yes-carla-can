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
        m = re.search(r"yaw(m?)(\d+)_r(\d+)", path)
        if m and os.path.getsize(path) > 100:
            yaw = int(m.group(2)) * (-1 if m.group(1) else 1)
            out.setdefault(yaw, []).append(load(path))
    return out


def steady(runs):
    vals = []
    for d, lat in runs:
        s = (d["t_s"] >= SETTLE[0]) & (d["t_s"] <= SETTLE[1])
        vals.append(float(lat[s].mean()))
    return float(np.mean(vals))


def main():
    folders = sys.argv[1:] or ["."]            # several folders are merged: the same yaw in two folders = two executions
    folder = folders[-1]                       # the plot is saved in the last one
    off, on = {}, {}
    for f in folders:
        for dst, sub_dir in ((off, "off"), (on, "compensate")):
            for yaw, runs in collect(os.path.join(f, sub_dir)).items():
                dst.setdefault(yaw, []).extend(runs)
    if not off and not on:
        print("no runs found in", folder)
        return
    yaws = sorted(set(off) | set(on))
    print("%-5s %14s %14s %11s %9s %10s %9s %13s %13s" % ("yaw", "sem defesa (m)", "com defesa (m)", "alarme", "alarme", "b (artigo)", "delta b", "capô medido", "capô previsto"))
    print("%-5s %14s %14s %11s %9s   (o artigo alarma com delta b < -0,1346)" % ("", "", "", "(centroide)", "(artigo)"))
    rows, extra, bs = [], [], {}
    for yaw in yaws:
        a = steady(off[yaw]) if yaw in off else float("nan")
        b = steady(on[yaw]) if yaw in on else float("nan")
        alarm, alarm_p, hood, bmean = "-", "-", float("nan"), float("nan")
        src = (off.get(yaw) or on.get(yaw))[0][0]
        if "alarm" in src.dtype.names:
            if (src["alarm"] > 0).any():
                alarm = "%.1f s" % src["t_s"][src["alarm"] > 0][0]
            s = src["t_s"] >= 5
            hood = float(np.nanmean(src["hood_shift_px"][s]))
        if "alarm_paper" in src.dtype.names:
            if (src["alarm_paper"] > 0).any():
                alarm_p = "%.1f s" % src["t_s"][src["alarm_paper"] > 0][0]
            s = src["t_s"] >= 5
            bmean = float(np.nanmean(src["hood_b"][s]))
        pred = -FOCAL_PX * np.tan(np.radians(yaw))
        rows.append((yaw, a, b, hood, pred))
        bs[yaw] = bmean
        extra.append((yaw, a, b, alarm, alarm_p, bmean, hood, pred))

    repeated = [y for y in yaws if len(off.get(y, [])) > 1 or len(on.get(y, [])) > 1]
    if repeated:
        print("\nReprodutibilidade (desvio estável de cada execução, em m):")
        for y in repeated:
            fmt = lambda runs: ", ".join("%+.2f" % steady([r]) for r in runs) if runs else "-"
            print("  yaw %3d: sem defesa [%s]   com defesa [%s]" % (y, fmt(off.get(y, [])), fmt(on.get(y, []))))
    b0 = bs.get(0, float("nan"))
    for yaw, a, b, alarm, alarm_p, bmean, hood, pred in extra:
        print("%-5d %+14.2f %+14.2f %11s %9s %+10.4f %+9.4f %+10.1f px %+10.1f px" % (yaw, a, b, alarm, alarm_p, bmean, bmean - b0, hood, pred))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: no plot)")
        return
    ink, muted, grid = "#0b0b0b", "#52514e", "#e3e2de"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor="#fcfcfb")
    ys = [r[0] for r in rows]
    half = LANE_WIDTH / 2
    base = next((r[1] for r in rows if r[0] == 0), 0.0)
    # Geometric model (see analyze_psa_sweep.py): the controller looks Z metres ahead, so a shift of the picture moves the car by
    # Z*tan(angle); the lane estimator only sees the position modulo one lane width, so beyond half a lane it wraps around.
    small = [r for r in rows if 0 < r[0] <= 10]
    z_eff = np.mean([abs(r[1] - base) / np.tan(np.radians(r[0])) for r in small]) if small else None
    if z_eff:
        deg = np.linspace(min(ys) - 1, max(ys) + 1, 600)
        wrapped = (-z_eff * np.tan(np.radians(deg)) + half) % LANE_WIDTH - half + base
        wrapped[np.abs(np.diff(wrapped, prepend=wrapped[0])) > half] = np.nan
        a1.plot(deg, wrapped, color=muted, linewidth=1.5, linestyle=(0, (1, 2)), zorder=1,
                label="modelo: %.1f m x tan(ângulo), com volta de faixa" % z_eff)
    a1.scatter(ys, [r[1] for r in rows], color="#2a78d6", s=48, zorder=3, label="sem defesa")
    a1.plot(ys, [r[2] for r in rows], color="#eb6834", linewidth=2, marker="o", zorder=2, label="com defesa do capô")
    a1.axhline(0, color=muted, linewidth=0.8)
    for edge in (-half, half):
        a1.axhline(edge, color=muted, linestyle=(0, (4, 3)), linewidth=1)
    a1.text(min(ys), half + 0.04, "limite da faixa", color=muted, fontsize=9, va="bottom")
    a1.set_ylim(-2.1, 2.1)
    a1.set_xlabel("Deslocamento da imagem (graus, 0 = sem ataque)", color=ink)
    a1.set_ylabel("Desvio lateral estável (m)\n(negativo = esquerda)", color=ink)
    a1.set_title("A defesa devolve o carro ao centro", loc="left", color=ink, fontweight="bold")
    a1.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
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
