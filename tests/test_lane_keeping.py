"""Unit tests for sensors/lane_keeping.py (no CARLA server needed)."""
import importlib.util
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

_path = os.path.join(os.path.dirname(__file__), "..", "sensors", "lane_keeping.py")
_spec = importlib.util.spec_from_file_location("lane_keeping", _path)   # load by path: the sensors package pulls pygame/av/scapy
lk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lk)

H, W = 200, 480


def tags_with_lines(left_x, right_x):
    """Synthetic semantic image with a vertical road marking at each given column (None = absent)."""
    tags = np.zeros((H, W), dtype=np.uint8)
    for x in (left_x, right_x):
        if x is not None:
            tags[:, x - 2:x + 3] = lk.ROAD_LINE_TAG
    return tags


def test_centered_lane_gives_zero_error():
    error, width = lk.lane_center_error(tags_with_lines(120, 360))
    assert error == pytest.approx(0.0, abs=0.01)
    assert width == pytest.approx(240, abs=6)


def test_lane_centre_to_the_right_gives_positive_error():
    error, _ = lk.lane_center_error(tags_with_lines(150, 390))      # centre at 270 -> 30 px right of 240
    assert error == pytest.approx(30 / 240, abs=0.02)


def test_single_marking_uses_last_known_lane_width():
    error, _ = lk.lane_center_error(tags_with_lines(120, None), lane_width_px=240)
    assert error == pytest.approx(0.0, abs=0.02)


def test_no_marking_returns_none():
    error, width = lk.lane_center_error(tags_with_lines(None, None), lane_width_px=200)
    assert error is None and width == 200


def test_steering_law_sign_and_saturation():
    assert lk.steering_law(0.2, 0.0, 0.5, 0.05, 0.5) > 0          # lane to the right -> steer right
    assert lk.steering_law(-0.2, 0.0, 0.5, 0.05, 0.5) < 0
    assert lk.steering_law(5.0, 0.0, 0.5, 0.05, 0.5) == 0.5


def _node(mode="drive", run_seconds=0.0, log_path=None):
    """LaneKeepingNode wired to mocks."""
    parent = MagicMock()
    parent.bounding_box.extent.x, parent.bounding_box.extent.z = 2.3, 0.7
    parent.get_velocity.return_value = SimpleNamespace(x=2.0, y=0.0, z=0.0)
    parent.get_location.return_value = SimpleNamespace(x=0.0, y=0.5, z=0.0)
    wp = MagicMock()
    wp.transform.location = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    wp.transform.get_right_vector.return_value = SimpleNamespace(x=0.0, y=1.0, z=0.0)
    wp.road_id, wp.lane_id, wp.lane_width = 1, -1, 3.5
    parent.get_world.return_value.get_map.return_value.get_waypoint.return_value = wp
    can = MagicMock()
    node = lk.LaneKeepingNode(parent, can, mode=mode, run_seconds=run_seconds, log_path=log_path)
    return node, can


def _image(tags, t):
    bgra = np.zeros((H, W, 4), dtype=np.uint8)
    bgra[:, :, 2] = tags
    return SimpleNamespace(raw_data=bgra.tobytes(), height=H, width=W, timestamp=t)


def test_node_sends_can_commands_after_warmup(tmp_path):
    node, can = _node(log_path=str(tmp_path / "run.csv"))
    tags = tags_with_lines(150, 390)                               # lane centre right of the image centre
    lk.LaneKeepingNode._on_image(lambda: node, _image(tags, 0.0))  # warm-up frame: no command
    can.send_steer_msg.assert_not_called()
    lk.LaneKeepingNode._on_image(lambda: node, _image(tags, 4.0))
    controls = can.send_steer_msg.call_args[0][0]
    assert controls is not None and can.send_throttle_msg.called
    node.destroy()
    assert (tmp_path / "run.csv").read_text().count("\n") >= 2      # header + one row


def test_log_mode_never_sends_can(tmp_path):
    node, can = _node(mode="log")
    tags = tags_with_lines(120, 360)
    lk.LaneKeepingNode._on_image(lambda: node, _image(tags, 0.0))
    lk.LaneKeepingNode._on_image(lambda: node, _image(tags, 5.0))   # past warm-up: it measures, but does not send
    assert len(node._rows) == 1
    can.send_steer_msg.assert_not_called()
    can.send_throttle_msg.assert_not_called()
    node.destroy()


def test_run_ends_with_brake():
    node, can = _node(run_seconds=2.0)
    tags = tags_with_lines(120, 360)
    lk.LaneKeepingNode._on_image(lambda: node, _image(tags, 0.0))
    lk.LaneKeepingNode._on_image(lambda: node, _image(tags, 4.0))
    lk.LaneKeepingNode._on_image(lambda: node, _image(tags, 8.0))   # past warm-up (3 s) + run (2 s)
    can.send_brake_msg.assert_called_once()
