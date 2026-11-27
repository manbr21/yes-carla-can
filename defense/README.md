# Hood Reference Defence

> This folder also has `id_time_intrusion_detection.py` and `ids_training/`, a CAN-bus intrusion detector
> that predates this one; it is documented in the main `README.md` (Experiment #3), not here.

A defence against the Perspective-Shift Attack (see `../attacks/`). It never touches `vcan0` either: it
looks at a single camera frame and decides whether the image has been shifted.

## Files

- `hood_reference_defense.py` — the defence itself. No dependency on CAN, CARLA or control, just a
  semantic-segmentation frame in, a measurement out. Two independent detectors:
  - `hood_shift(tags)` — our own method: the centroid of the car-body pixels at the bottom of the image.
  - `hood_edge_fit(tags)` — the paper's method: a quadratic fit `y = a·x² + b·x + c` of the top edge of
    the car body (ported from the authors' own `PSA/Defense/Camera/engineHood_Detection.py`).
- `../sensors/lane_keeping.py` — the consumer. Runs both detectors on every camera frame when
  `--defense` is not `off`, and, in `--defense compensate` mode, uses `hood_shift`'s measurement to undo
  the attack's shift before estimating the lane.

## The idea

The car's own hood is part of the car, so a camera mounted inside the cabin always sees it in the same
place. If the whole image is shifted (the Perspective-Shift Attack, or any other cause), the hood moves
in the image by about the same amount as the rest of the scene, because it was shifted too. That makes
the hood a reference that needs no calibration against the outside world: compare where the hood is now
to where it should be, and the difference is exactly how much the image was shifted, and in which
direction. `--defense compensate` subtracts that difference from the lane estimate before it reaches the
controller, so the attack's effect on steering is undone without ever knowing the attack happened.

`--defense detect` only raises an alarm (useful on its own, or to flag the car for a human); `--defense
stop` also brakes the car.

## Running it

Needs the camera positioned where the hood is visible (`--camera-x`, metres from the car centre; the
default camera position, in front of the car, never sees the hood):

```bash
cd ..
./1_up_environment.sh --map Town04 --lane-keeping drive --spawn-index <N> --run-seconds 15 \
  --camera-x 0.6 --psa-yaw 10 --defense compensate --log-csv psa10_defended.csv
```

`./run_hood_defense.sh` automates the whole thing: it first finds a camera position where the hood shows
up, then runs a set of angles with the defence off and on, and plots the comparison
(`analyze_hood_defense.py`). See the main `README.md` for the full set of options.

## Result (one run per angle, Town04, same car and spawn point)

| Camera shift | Without defence | With `compensate` |
|---|---|---|
| 5° | ~0.7 m off centre | ~0.08 m |
| 7° | ~1.0 m off centre | ~0.08 m |
| 10° | ~1.4 m off centre | ~0.07 m |
| 20° | ~0.6 m off centre | ~0.20 m (worst case) |

The hood's measured shift matches the geometric prediction (`focal_px · tan(angle)`) to within a few
pixels, up to the point where the shift pushes the hood mostly out of frame. The paper's `hood_edge_fit`
rule, with its original threshold, never fired on any of these angles: it was tuned for a much larger
shift in the authors' own setup. See the main `README.md` for the full numbers and the limits (a single
run per angle, and the defence assumes the whole image, hood included, was shifted together — a film
that only covered part of the lens would not be caught this way).
