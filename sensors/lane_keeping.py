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
import time
import weakref

import numpy as np
import carla

ROAD_LINE_TAG = 24   # semantic label of road markings in CARLA 0.9.15 (CityScapes palette)


def lane_center_error(tags, lane_width_px=None, rows=(0.60, 0.75), line_tag=ROAD_LINE_TAG):
    """Estimate where the ego-lane centre is in the image.

    tags          (H, W) array of semantic labels
    lane_width_px last known lane width in pixels, used when only one marking is visible
    Returns (error, lane_width_px). error is in [-1, 1] (+ = lane centre to the right of the image centre),
    or None when no marking is visible.
    """
    height, width = tags.shape
    cx = width / 2.0
    centers, widths = [], []
    for y in range(int(rows[0] * height), int(rows[1] * height), 2):
        xs = np.flatnonzero(tags[y] == line_tag)
        left, right = xs[xs < cx], xs[xs >= cx]
        if left.size and right.size:
            centers.append((left.max() + right.min()) / 2.0)
            widths.append(right.min() - left.max())
        elif lane_width_px and left.size:
            centers.append(left.max() + lane_width_px / 2.0)
        elif lane_width_px and right.size:
            centers.append(right.min() - lane_width_px / 2.0)
    if not centers:
        return None, lane_width_px
    if widths:
        w = float(np.median(widths))
        lane_width_px = w if not lane_width_px else 0.8 * lane_width_px + 0.2 * w
    return float((np.median(centers) - cx) / cx), lane_width_px


def steering_law(error, d_error, kp, kd, max_steer):
    """PD controller. CARLA steer: positive turns right, range [-1, 1]."""
    return float(np.clip(kp * error + kd * d_error, -max_steer, max_steer))


class LaneKeepingNode(object):
    IMAGE_WIDTH = 480
    IMAGE_HEIGHT = 200
    FOV = 90
    WARMUP_S = 3.0   # let the car settle on the road before steering

    def __init__(self, parent_actor, can_net, mode="drive", psa_yaw=0.0, target_speed_kmh=30.0,
                 run_seconds=0.0, log_path=None, kp=0.5, kd=0.05, max_steer=0.5):
        self.sensor = None
        self._parent = parent_actor
        self._can = can_net
        self.mode = mode
        self.psa_yaw = psa_yaw
        self._target_speed = target_speed_kmh / 3.6
        self._run_seconds = run_seconds
        self._kp, self._kd, self._max_steer = kp, kd, max_steer

        world = parent_actor.get_world()
        self._map = world.get_map()
        self._t0 = None
        self._last_t = None
        self._last_error = 0.0
        self._lane_width_px = None
        self._ref_lane = None            # (road_id, lane_id) of the lane the car started in
        self._rows = []                  # (t, error, steer, speed_kmh, lateral_m, off_lane)
        self._finished = False
        self._reported_first = False
        self._log = open(log_path, "w", newline="") if log_path else None
        self._writer = csv.writer(self._log) if self._log else None
        if self._writer:
            self._writer.writerow(["t_s", "perceived_error", "steer", "speed_kmh", "lateral_m", "off_lane"])

        # Same mount as the RGB camera (front of the car), plus the PSA yaw.
        bound_x = 0.5 + parent_actor.bounding_box.extent.x
        bound_z = 0.5 + parent_actor.bounding_box.extent.z
        bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        bp.set_attribute("image_size_x", str(self.IMAGE_WIDTH))
        bp.set_attribute("image_size_y", str(self.IMAGE_HEIGHT))
        bp.set_attribute("fov", str(self.FOV))
        bp.set_attribute("sensor_tick", "0.05")
        transform = carla.Transform(
            carla.Location(x=bound_x + 0.3, z=bound_z + 0.1),
            carla.Rotation(yaw=psa_yaw),
        )
        self.sensor = world.spawn_actor(bp, transform, attach_to=parent_actor,
                                        attachment_type=carla.AttachmentType.Rigid)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: LaneKeepingNode._on_image(weak_self, image))
        print(f"[LK] lane keeping '{mode}' started (PSA yaw {psa_yaw:+.1f} deg, target {target_speed_kmh:.0f} km/h)")

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
        if not self._reported_first:
            self._reported_first = True
            n_lines = int((tags == ROAD_LINE_TAG).sum())
            print(f"[LK] first frame: {n_lines} road-marking pixels (tag {ROAD_LINE_TAG})"
                  + ("" if n_lines else " -> none found: check ROAD_LINE_TAG or the road/spawn point"))
        error, self._lane_width_px = lane_center_error(tags, self._lane_width_px)
        if error is None:                      # no marking visible: keep the last estimate
            error = self._last_error
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
        row = (t, error, steer, speed * 3.6, lateral, int(off_lane))
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
