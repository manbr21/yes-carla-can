"""Shared bits between analyze_psa_sweep.py and analyze_hood_defense.py: loading a run's CSV and plot styling."""
import numpy as np

LANE_WIDTH = 3.5
SETTLE = (8.0, 13.0)          # seconds of simulation used for the steady state

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    lat = d["lateral_m"].copy()
    # The ground truth switches to the neighbouring lane when the car crosses a marking (jump of ~one lane):
    # remove the jump so the offset is always measured from the lane where the run started.
    for i in np.where(np.abs(np.diff(lat)) > 0.6 * LANE_WIDTH)[0]:
        lat[i + 1:] -= np.round((lat[i + 1] - lat[i]) / LANE_WIDTH) * LANE_WIDTH
    return d, lat


def style_axes(*axes):
    """Recessive grid/axes, in the same look used by both analysis plots."""
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)
