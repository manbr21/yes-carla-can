"""Hood reference defence against the Perspective-Shift Attack: the ego car's own hood is part of the car
body, so a camera mounted inside the car always sees it in the same place. If the whole image is shifted
(a film in front of the lens, or any other cause), the hood moves in the image by about the same amount
as the rest of the scene, so it works as a free, always-available reference. See README.md in this folder.

Two independent detectors, both reading the same semantic-segmentation frame:

- Our own method, ``hood_shift``: the centroid of the car-body pixels at the bottom of the image. Used by
  sensors/lane_keeping.py both to raise an alarm (``--defense detect``/``stop``) and, in
  ``--defense compensate``, to measure and undo the shift before estimating the lane.
- The paper's method, ``hood_edge_fit``: a quadratic fit of the top edge of the car body
  (PSA/Defense/Camera/engineHood_Detection.py in the authors' own repository), logged for comparison but
  not used to correct anything here.

Like the attack module, this one only looks at a single segmentation frame; it has no notion of CAN,
control, or CARLA.
"""
import numpy as np

HOOD_TAGS = (14,)         # semantic label of the ego car body (CARLA 0.9.15: 14 = Car)
HOOD_ROWS = (0.80, 1.0)   # the hood shows up in the bottom of the image
MIN_HOOD_PIXELS = 150
HOOD_TOL_PX = 8.0         # hood moved more than this from the image centre: the camera is not where it was mounted
ALARM_FRAMES = 5          # ... for this many frames in a row (0.25 s at the camera's 0.05 s tick)

# Camera defence of the paper (PSA/Defense/Camera/engineHood_Detection.py): take the top edge of the car body in the
# segmentation, keep the points between 70% and 95% of the image height, fit y = a*x^2 + b*x + c and flag the image
# when b < -0.15. On their benign example b = -0.0154, so the threshold is 0.1346 below the benign value: it is
# applied relative to the benign b measured on our own camera (--hood-b-ref); with the default reference it is the
# paper's absolute -0.15.
PAPER_B_THRESHOLD = -0.15
PAPER_B_BASELINE = -0.0154
HOOD_EDGE_ROWS = (0.70, 0.95)
MIN_EDGE_POINTS = 30


def hood_mask(tags):
    """Boolean mask of the ego car body (tag 14), shared by hood_shift and hood_edge_fit so a frame pays for it once."""
    return np.isin(tags, HOOD_TAGS)


def hood_shift(tags, mask=None):
    """How far (pixels, + = to the right) the ego car body is from the image centre; None if it is not visible."""
    height, width = tags.shape
    mask = hood_mask(tags) if mask is None else mask
    _, xs = np.nonzero(mask[int(HOOD_ROWS[0] * height):int(HOOD_ROWS[1] * height)])
    if xs.size < MIN_HOOD_PIXELS:
        return None, int(xs.size)
    return float(xs.mean() + 0.5 - width / 2.0), int(xs.size)


def hood_edge_fit(tags, mask=None):
    """The paper's hood feature: quadratic fit y = a*x^2 + b*x + c of the top edge of the car body.

    For every column, the edge is the top of the contiguous run of car pixels that reaches the bottom of the
    column (so a stray car pixel higher up, with a gap below it, is not picked up). Returns (a, b, c, number
    of points), or None when too few columns have the hood edge inside the expected band of the image.
    """
    mask = hood_mask(tags) if mask is None else mask
    height, width = mask.shape
    row = np.arange(height)[:, None]
    have = mask.any(axis=0)
    last = height - 1 - np.argmax(mask[::-1], axis=0)                       # bottommost car row per column
    # Nearest non-car row at or before each row, carried forward; -1 where every row above is car.
    last_gap = np.maximum.accumulate(np.where(mask, -1, row), axis=0)
    last_gap_before = np.vstack([np.full((1, width), -1), last_gap[:-1]])
    top = last_gap_before[last, np.arange(width)] + 1                       # top of the run ending at `last`
    sel = have & (top > HOOD_EDGE_ROWS[0] * height) & (top <= HOOD_EDGE_ROWS[1] * height)
    if sel.sum() < MIN_EDGE_POINTS:
        return None
    xs, ys = np.flatnonzero(sel), top[sel]
    a, b, c = np.polyfit(xs, ys, 2)
    return float(a), float(b), float(c), int(xs.size)
