"""Camera-based lane keeping node for YES, CARLA CAN.

The node reads a semantic-segmentation camera mounted like the RGB camera, estimates where the centre of
the ego lane is in the image and, in "drive" mode, publishes THROTTLE and STEER frames on the CAN bus
(the same frames the keyboard sender produces).

The camera can be mounted with a yaw offset (``psa_yaw``). This models a Perspective-Shift Attack: a film
that bends the light by a fixed angle. The controller still assumes a fixed mount, so a shifted view makes
it steer towards the wrong place.

Modes
    "log"    only measures and logs (drive with the keyboard sender or the autopilot)
    "drive"  also sends THROTTLE and STEER over CAN (do not run vehicle_controls_module at the same time)
"""
import csv
import os
import time
import weakref

import numpy as np
import carla

ROAD_LINE_TAG = 24   # semantic label of road markings in CARLA 0.9.15 (CityScapes palette)


LANE_K = 2.2        # lane width in pixels per image row below the horizon (measured on Town04 highway frames)
REF_DY = 35.0       # image row (pixels below the horizon) where the returned error is expressed
HOOD_TAGS = (14,)    # semantic label of the ego car body (CARLA 0.9.15: 14 = Car)
HOOD_ROWS = (0.80, 1.0)   # the hood shows up in the bottom of the image
MIN_HOOD_PIXELS = 150
HOOD_TOL_PX = 8.0    # hood moved more than this from the image centre: the camera is not where it was mounted
ALARM_FRAMES = 5     # ... for this many frames in a row (0.25 s)
# Camera defence of the paper (PSA/Defense/Camera/engineHood_Detection.py): take the top edge of the car body in the segmentation,
# keep the points between 70% and 95% of the image height, fit y = a*x^2 + b*x + c and flag the image when b < -0.15.
# On their benign example b = -0.0154, so the threshold is 0.1346 below the benign value: it is applied relative to
# the benign b measured on our own camera (--hood-b-ref); with the default reference it is the paper's absolute -0.15.
PAPER_B_THRESHOLD = -0.15
PAPER_B_BASELINE = -0.0154
HOOD_EDGE_ROWS = (0.70, 0.95)
MIN_EDGE_POINTS = 30
MIN_COHERENCE = 0.25   # below this the frame is ignored (the last estimate is kept); under a PSA the grid is distorted and coherence drops to ~0.3-0.5


def lane_center_error(tags, lane_k=LANE_K, rows=(0.65, 0.80), line_tag=ROAD_LINE_TAG, shift_px=0.0):
    """Estimate where the centre of the nearest lane is, relative to the image centre.

    The road has several lanes and the markings are dashed, so in a given frame only a few lines are
    visible and they may belong to different lanes. But all markings sit on a regular grid: by perspective
    the lines are spaced lane_k * dy pixels apart at dy pixels below the horizon. So for every marking
    pixel we take its position inside that grid ("phase", in lane widths) and average the phases. A single
    visible line is enough, and it does not matter which lane it belongs to. The camera looks straight
    ahead, so the horizon is the middle row.

    Returns (error, coherence). error: lane centre minus image centre, in image half-widths at row REF_DY
    (+ = lane centre to the right); it lives in [-0.5, 0.5) lane widths, so it wraps when the car leaves
    its lane. coherence in [0, 1] tells how well the lines agree with the grid (low = do not trust).
    (None, 0) when there are too few marking pixels. shift_px: image shift to remove (see hood_shift).
    """
    height, width = tags.shape
    cx, horizon = width / 2.0, height / 2.0
    y0, y1 = int(rows[0] * height), int(rows[1] * height)
    ys, xs = np.nonzero(tags[y0:y1] == line_tag)
    if ys.size < 5:
        return None, 0.0
    dy = ys + y0 - horizon
    angle = 2.0 * np.pi * (xs + 0.5 - cx - shift_px) / (lane_k * dy)
    c, s = np.cos(angle).mean(), np.sin(angle).mean()
    coherence = float(np.hypot(c, s))
    line_phase = np.arctan2(s, c) / (2.0 * np.pi)            # where the lines are, in lane widths
    centre = (line_phase + 1.0) % 1.0 - 0.5                  # the lane centre is half a lane away from a line
    return float(centre * lane_k * REF_DY / cx), coherence


def hood_shift(tags, hood_tags=HOOD_TAGS, rows=HOOD_ROWS, min_pixels=MIN_HOOD_PIXELS):
    """How far (pixels, + = to the right) the ego car body is from the image centre; None if it is not visible.

    The hood is part of the car, so a camera mounted in the middle of the car always sees it centred. If a film
    (or anything else) shifts the view, the hood moves in the image by about the same amount as the whole scene.
    That makes the hood a free reference: it tells how much the image was shifted.
    """
    height, width = tags.shape
    sub = tags[int(rows[0] * height):int(rows[1] * height)]
    _, xs = np.nonzero(np.isin(sub, hood_tags))
    if xs.size < min_pixels:
        return None, int(xs.size)
    return float(xs.mean() + 0.5 - width / 2.0), int(xs.size)


def hood_edge_fit(tags, car_tags=HOOD_TAGS, rows=HOOD_EDGE_ROWS, min_points=MIN_EDGE_POINTS):
    """The paper's hood feature: quadratic fit y = a*x^2 + b*x + c of the top edge of the car body.

    For every column, starting at the bottom, find the first car pixel and walk up to the first pixel that is not car: that is the
    edge. Returns (a, b, c, number of points) or None when the hood edge is not visible.
    """
    mask = np.isin(tags, car_tags)
    height, width = mask.shape
    xs, ys = [], []
    for x in range(width):
        idx = np.flatnonzero(mask[:, x])
        if not idx.size:
            continue
        last = idx[-1]
        run = mask[:last + 1, x][::-1]
        top = last - (run.size if run.all() else int(np.argmin(run))) + 1
        if rows[0] * height < top <= rows[1] * height:
            xs.append(x)
            ys.append(top)
    if len(xs) < min_points:
        return None
    a, b, c = np.polyfit(xs, ys, 2)
    return float(a), float(b), float(c), len(xs)


def shift_tags(tags, dx):
    """Move the whole image dx pixels to the right (dx < 0: to the left). The columns that come in are left empty (label 0)."""
    if dx == 0:
        return tags
    out = np.zeros_like(tags)
    if dx > 0:
        out[:, dx:] = tags[:, :-dx]
    else:
        out[:, :dx] = tags[:, -dx:]
    return out


def steering_law(error, d_error, kp, kd, max_steer):
    """PD controller. CARLA steer: positive turns right, range [-1, 1]."""
    return float(np.clip(kp * error + kd * d_error, -max_steer, max_steer))


class LaneKeepingNode(object):
    IMAGE_WIDTH = 480
    IMAGE_HEIGHT = 200
    FOV = 90
    FOCAL_PX = IMAGE_WIDTH / 2.0 / np.tan(np.radians(FOV / 2.0))
    WARMUP_S = 3.0   # let the car settle on the road before steering

    def __init__(self, parent_actor, can_net, mode="drive", psa_yaw=0.0, target_speed_kmh=30.0,
                 run_seconds=0.0, log_path=None, kp=0.5, kd=0.05, max_steer=0.5, bias=0.0, dump_dir=None, camera_x=None, defense="off", hood_ref=0.0, hood_b_ref=PAPER_B_BASELINE, psa_mode="rotate"):
        self.sensor = None
        self._parent = parent_actor
        self._can = can_net
        self.mode = mode
        self.psa_yaw = psa_yaw
        # "rotate": the sensor is turned by psa_yaw inside the car (what the camera sees of the car itself changes).
        # "shift": the whole picture, hood included, is moved sideways by f*tan(psa_yaw) pixels, as a film in front of the lens
        # would do (the paper's example shows the hood moved together with the scene).
        self._psa_shift_px = 0 if psa_mode == "rotate" else int(round(-self.FOCAL_PX * np.tan(np.radians(psa_yaw))))
        self._target_speed = target_speed_kmh / 3.6
        self._run_seconds = run_seconds
        self._kp, self._kd, self._max_steer = kp, kd, max_steer
        self._dump_dir, self._last_dump = dump_dir, None
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        self._hood_b_ref = hood_b_ref    # benign value of the paper's linear coefficient on this camera
        self._paper_count = 0
        self._paper_alarm_t = None
        self._hood_ref = hood_ref        # where the hood sits when nothing is wrong (measured on a run without attack)
        self.defense = defense           # "off", "detect" (log + alarm), "stop" (alarm + brake), "compensate" (undo the shift)
        self._hood_ema = None
        self._alarm_count = 0
        self._alarm_t = None
        self._bias = bias                # calibration: error measured on the benign run at the start

        world = parent_actor.get_world()
        self._map = world.get_map()
        self._t0 = None
        self._last_t = None
        self._last_error = 0.0
        self._ref_lane = None            # (road_id, lane_id) of the lane the car started in
        self._rows = []                  # (t, error, steer, speed_kmh, lateral_m, off_lane, coherence)
        self._finished = False
        self._reported_first = False
        self._reported_hood = False
        self._reported_edge = False
        self._log = open(log_path, "w", newline="") if log_path else None
        self._writer = csv.writer(self._log) if self._log else None
        if self._writer:
            self._writer.writerow(["t_s", "perceived_error", "steer", "speed_kmh", "lateral_m", "off_lane", "coherence", "hood_shift_px", "alarm",
                                  "hood_a", "hood_b", "alarm_paper"])

        # Same mount as the RGB camera (front of the car), plus the PSA yaw.
        bound_x = 0.5 + parent_actor.bounding_box.extent.x
        bound_z = 0.5 + parent_actor.bounding_box.extent.z
        cam_height = bound_z + 0.1
        lane_width = self._map.get_waypoint(parent_actor.get_location(), lane_type=carla.LaneType.Driving).lane_width
        # By perspective a lane of width W seen from height h is (W / h) pixels wide per image row below the horizon.
        self._lane_k = lane_width / cam_height
        self._vehicle_id = parent_actor.type_id
        bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        bp.set_attribute("image_size_x", str(self.IMAGE_WIDTH))
        bp.set_attribute("image_size_y", str(self.IMAGE_HEIGHT))
        bp.set_attribute("fov", str(self.FOV))
        bp.set_attribute("sensor_tick", "0.05")
        # By default the camera is in front of the car (it never sees the hood). With camera_x (metres from the car
        # centre, e.g. 0.6) it sits inside the cabin and the hood is visible, which the hood defence needs.
        cam_x = bound_x + 0.3 if camera_x is None else camera_x
        transform = carla.Transform(
            carla.Location(x=cam_x, z=bound_z + 0.1),
            carla.Rotation(yaw=psa_yaw if psa_mode == "rotate" else 0.0),
        )
        self.sensor = world.spawn_actor(bp, transform, attach_to=parent_actor,
                                        attachment_type=carla.AttachmentType.Rigid)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: LaneKeepingNode._on_image(weak_self, image))
        print(f"[LK] lane keeping '{mode}' started (PSA yaw {psa_yaw:+.1f} deg, target {target_speed_kmh:.0f} km/h)")
        print(f"[LK] PSA mode '{psa_mode}'" + (f": the image is shifted by {self._psa_shift_px:+d} px" if psa_mode != "rotate" else ""))
        print(f"[LK] camera x = {cam_x:.2f} m from the car centre, hood defence '{defense}'")
        print(f"[LK] vehicle {self._vehicle_id}, camera height {cam_height:.2f} m, lane width {lane_width:.2f} m -> k = {self._lane_k:.2f}")

    # ------------------------------------------------------------------ ground truth for the metrics
    def _lateral_offset(self):
        """Signed distance (m) from the centre of the nearest driving lane (+ = right) and lane-change flag."""
        loc = self._parent.get_location()
        wp = self._map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        right = wp.transform.get_right_vector()
        lateral = (loc.x - wp.transform.location.x) * right.x + (loc.y - wp.transform.location.y) * right.y
        if self._ref_lane is None or self._ref_lane[0] != wp.road_id:
            self._ref_lane = (wp.road_id, wp.lane_id)
        changed_lane = wp.lane_id != self._ref_lane[1]
        return lateral, abs(lateral) > wp.lane_width / 2.0 or changed_lane

    # ------------------------------------------------------------------ main loop (runs on every camera frame)
    @staticmethod
    def _on_image(weak_self, image):
        self = weak_self()
        if self is None or self._finished:
            return
        if self._t0 is None:
            self._t0 = image.timestamp
        t = image.timestamp - self._t0
        if t < self.WARMUP_S:
            return

        tags = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, 2]
        if self._psa_shift_px:
            tags = shift_tags(tags, self._psa_shift_px)
        if not self._reported_first:
            self._reported_first = True
            n_lines = int((tags == ROAD_LINE_TAG).sum())
            fits = [(lane_center_error(tags, k)[1], k) for k in np.arange(1.5, 3.31, 0.05)]
            best_coh, best_k = max(fits)
            print(f"[LK] first frame: best-fit k = {best_k:.2f} (coherence {best_coh:.2f}); k in use {self._lane_k:.2f}")
            print(f"[LK] first frame: {n_lines} road-marking pixels (tag {ROAD_LINE_TAG})"
                  + ("" if n_lines else " -> none found: check ROAD_LINE_TAG or the road/spawn point"))
        hood, n_hood = hood_shift(tags)
        if not self._reported_hood:
            self._reported_hood = True
            band = tags[int(HOOD_ROWS[0] * tags.shape[0]):]
            labels, counts = np.unique(band, return_counts=True)
            top = ", ".join("tag %d: %d px" % (l, c) for c, l in sorted(zip(counts, labels), reverse=True)[:3])
            print(f"[LK] bottom of the image: {top}; hood {'visible' if hood is not None else 'NOT visible'}"
                  + (f" ({n_hood} px, {hood:+.1f} px from the centre)" if hood is not None else ""))
        if hood is not None:
            hood -= self._hood_ref
            self._hood_ema = hood if self._hood_ema is None else 0.8 * self._hood_ema + 0.2 * hood
            self._alarm_count = self._alarm_count + 1 if abs(self._hood_ema) > HOOD_TOL_PX else 0
        alarm = self._alarm_count >= ALARM_FRAMES
        fit = hood_edge_fit(tags)
        if fit is not None:
            if not self._reported_edge:
                self._reported_edge = True
                print(f"[LK] hood edge: {fit[3]} points, a={fit[0]:.3e}, b={fit[1]:.4f} "
                      f"(paper rule: alarm if b < {self._hood_b_ref + PAPER_B_THRESHOLD - PAPER_B_BASELINE:.4f})")
            flagged = fit[1] - self._hood_b_ref < PAPER_B_THRESHOLD - PAPER_B_BASELINE
            self._paper_count = self._paper_count + 1 if flagged else 0
        paper_alarm = self._paper_count >= ALARM_FRAMES
        if paper_alarm and self._paper_alarm_t is None:
            self._paper_alarm_t = t
        if alarm and self._alarm_t is None:
            self._alarm_t = t
            if self.defense != "off":
                print(f"[LK] DEFENCE ALARM at {t:.1f} s: the hood is {self._hood_ema:+.1f} px from the image centre (camera shifted)")
        if alarm and self.defense == "stop" and not self._finished:
            self._stop()
            return
        shift = self._hood_ema if (self.defense == "compensate" and self._hood_ema is not None) else 0.0
        if self._dump_dir and (self._last_dump is None or t - self._last_dump >= 2.0):   # for debugging the estimator
            self._last_dump = t
            image.save_to_disk(os.path.join(self._dump_dir, "seg_%05.2f.png" % t), carla.ColorConverter.CityScapesPalette)
        error, coherence = lane_center_error(tags, self._lane_k, shift_px=shift)
        if error is None or coherence < MIN_COHERENCE:   # nothing reliable in this frame: keep the last estimate
            error = self._last_error
        else:
            error -= self._bias
        if self._last_t is None:               # first frame: no derivative yet (avoids a kick at the start)
            self._last_error = error
        dt = (t - self._last_t) if self._last_t is not None else 0.05
        d_error = (error - self._last_error) / dt if dt > 0 else 0.0
        self._last_error, self._last_t = error, t

        steer = steering_law(error, d_error, self._kp, self._kd, self._max_steer)
        v = self._parent.get_velocity()
        speed = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
        throttle = float(np.clip(0.5 * (self._target_speed - speed), 0.0, 0.7))

        if self._run_seconds and t > self.WARMUP_S + self._run_seconds:
            self._stop()
            return
        if self.mode == "drive":
            controls = carla.VehicleControl(throttle=throttle, steer=steer)
            self._can.send_throttle_msg(controls)
            self._can.send_steer_msg(controls)

        lateral, off_lane = self._lateral_offset()
        row = (t, error, steer, speed * 3.6, lateral, int(off_lane), coherence,
               hood if hood is not None else float("nan"), int(alarm),
               fit[0] if fit else float("nan"), fit[1] if fit else float("nan"), int(paper_alarm))
        self._rows.append(row)
        if self._writer:
            self._writer.writerow([f"{x:.3f}" if isinstance(x, float) else x for x in row])

    # ------------------------------------------------------------------ end of run
    def _stop(self):
        self._finished = True
        if self.mode == "drive":
            controls = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            self._can.send_throttle_msg(controls)
            self._can.send_steer_msg(controls)
            self._can.send_brake_msg(controls)
        self.print_summary()
        if self._log:
            self._log.flush()

    def print_summary(self):
        if not self._rows:
            print("[LK] no data recorded")
            return
        data = np.array(self._rows)
        lateral, off = data[:, 4], data[:, 5]
        first_off = data[off > 0, 0][:1]
        print("[LK] summary: %.0f s, max |lateral| %.2f m, off-lane %.0f%% of the time, first off-lane at %s"
              % (data[-1, 0] - data[0, 0], np.abs(lateral).max(), 100 * off.mean(),
                 ("%.1f s" % first_off[0]) if first_off.size else "never"))
        if self._hood_ema is not None:
            print("[LK] hood detectors: centroid rule alarm %s; paper rule alarm %s" % (
                ("at %.1f s" % self._alarm_t) if self._alarm_t is not None else "never",
                ("at %.1f s" % self._paper_alarm_t) if self._paper_alarm_t is not None else "never"))
        if self.defense != "off":
            print("[LK] defence '%s': alarm %s" % (self.defense, ("at %.1f s" % self._alarm_t) if self._alarm_t is not None else "never raised"))

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.stop()
                self.sensor.destroy()
            except Exception as e:
                print(f"[LK] error destroying the sensor: {e}")
            self.sensor = None
        if not self._finished:
            self._finished = True
            self.print_summary()
        if self._log:
            self._log.close()
            self._log = None
