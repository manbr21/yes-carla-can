# Perspective-Shift Attack (PSA)

> This folder also has `denial_of_service.py` and `reverse_engineering.py`, two CAN-bus attacks that
> predate this one; they are documented in the main `README.md` (Experiments #1 and #2), not here.

A sensor attack, not a CAN attack: it never touches `vcan0`. A transparent film placed in front of a
camera lens bends the light by a fixed angle, so the whole image the sensor captures is shifted sideways.
Nothing in the real scene moved; the camera's field of view did. This follows the
[Perspective-Shift Attack](https://github.com/tum-esi/PSA) paper (a copy of the authors' own repository
is checked out at `../PSA` in this project).

## Files

- `perspective_shift_attack.py` — the attack itself. Two functions, no dependency on CAN, CARLA or
  control: `shift_px_for_yaw(yaw_deg, focal_px)` converts an angle into a pixel shift, and
  `shift_tags(tags, dx)` moves a whole image (of any kind of per-pixel value, here semantic labels) `dx`
  pixels sideways, leaving the edge that comes from outside the original frame empty.
- `../sensors/lane_keeping.py` — the victim. A camera-based lane keeper that reads a semantic-segmentation
  camera, estimates the lane centre, and drives the car over the project's virtual CAN bus (`vcan0`),
  exactly like the keyboard sender would. It applies `shift_tags` to every camera frame before estimating
  the lane, so the effect of the attack on a real, closed control loop can be measured, not just on the
  raw sensor reading.

## Why it fools the lane keeper

The road markings are dashed, but they repeat at a regular spacing in the image (by perspective, wider
lower in the frame). The lane keeper finds the lane centre from that repeating pattern, not from any single
mark, so a missing dash does not confuse it. Shifting the whole image preserves the pattern, just moved
sideways: the lane keeper still finds a perfectly good, perfectly regular pattern, and reads the lane
centre as being wherever that shifted pattern puts it. The attack is invisible to the lane keeper's own
confidence measure; it only shows up once something external (ground truth, or the hood reference defence
in `../defense/`) is compared against it.

## Running it

```bash
cd ..
./1_up_environment.sh --map Town04 --lane-keeping drive --spawn-index <N> --run-seconds 40 --log-csv psa20.csv --psa-yaw 20
```

`--psa-yaw` is in degrees (positive turns the simulated view to the right). `--psa-mode shift` (the
default) is this attack; `--psa-mode rotate` turns the camera sensor itself instead, which is a different,
cruder approximation kept only for the first sweep of this experiment (see the main `README.md`,
Experiment #5, for the full set of options and for `run_psa_sweep.sh`, which runs a whole sweep of angles
automatically).

## Result (one run per angle, Town04, same car and spawn point)

| Camera shift | Lateral offset once the car settles |
|---|---|
| 0° (no attack) | ~0.1 m |
| 5° | ~0.7 m |
| 7° | ~1.0 m |
| 10° | ~1.4 m |

The perceived error the lane keeper reports stays near zero throughout: the car is measurably off the
lane centre, but the controller never notices. See the main `README.md` for the full results, the plots,
and the limits of this setup (a single run per angle, one simple PD controller, not the neural lane
detector of the original paper).
