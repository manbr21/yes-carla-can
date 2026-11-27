"""Perspective-Shift Attack (PSA): a transparent film placed in front of a camera lens bends the light by
a fixed angle, so the whole image the sensor captures is shifted sideways by a fixed number of pixels.
Nothing in the scene moved; the camera's own field of view did. See README.md in this folder.

The attack is a pure image transformation: it has no notion of CAN, control, or CARLA. It is used by
sensors/lane_keeping.py, which applies it to every frame of the camera-based lane keeper before that node
estimates the lane and steers, so the effect on a real control loop can be measured.
"""
import numpy as np


def shift_px_for_yaw(yaw_deg, focal_px):
    """Pixel shift equivalent to a film that bends light by yaw_deg, for a camera with focal length
    focal_px (in pixels). CARLA's yaw convention is positive = view turned to the right, which is why a
    positive yaw gives a negative (leftward) shift: turning your head right moves what was in front of
    you towards the left edge of your new field of view."""
    return int(round(-focal_px * np.tan(np.radians(yaw_deg))))


def shift_tags(tags, dx):
    """Move the whole image dx pixels to the right (dx < 0: to the left).

    This is the attack itself: it does not matter whether `tags` holds semantic labels, raw pixels or
    anything else, every column is simply moved. The columns that come in from outside the original frame
    are left empty (label/value 0), because a single shifted frame has no information about what the
    sensor would have seen there; a real film would show the same kind of edge loss.
    """
    if dx == 0:
        return tags
    out = np.zeros_like(tags)
    if dx > 0:
        out[:, dx:] = tags[:, :-dx]
    else:
        out[:, :dx] = tags[:, -dx:]
    return out
