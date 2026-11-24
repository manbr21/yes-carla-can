#!/usr/bin/env python3
"""List spawn points that sit at the start of a long straight stretch of road.

Useful to pick --spawn-index for the lane keeping experiment (a straight road makes the Perspective-Shift
Attack easy to see). Needs a running CARLA server.

    python find_straight_spawns.py --map Town04
"""
import argparse

import carla


def heading_change(wp, length_m, step_m=5.0):
    """Largest heading difference (degrees) along the next length_m metres of the lane."""
    h0, worst, cur = wp.transform.rotation.yaw, 0.0, wp
    for _ in range(int(length_m / step_m)):
        nxt = cur.next(step_m)
        if not nxt:
            return None
        cur = nxt[0]
        worst = max(worst, abs((cur.transform.rotation.yaw - h0 + 180) % 360 - 180))
    return worst


def main():
    parser = argparse.ArgumentParser(description="Find straight spawn points")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default=None, help="map to load (e.g. Town04); default: the current one")
    parser.add_argument("--length", type=float, default=150.0, help="straight length required, in metres")
    parser.add_argument("--max-turn", type=float, default=3.0, help="maximum heading change, in degrees")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.load_world(args.map) if args.map else client.get_world()
    carla_map = world.get_map()
    print(f"map: {carla_map.name}")
    for index, spawn in enumerate(carla_map.get_spawn_points()):
        wp = carla_map.get_waypoint(spawn.location, lane_type=carla.LaneType.Driving)
        turn = heading_change(wp, args.length)
        if turn is not None and turn <= args.max_turn:
            print(f"  spawn {index:3d}: straight for {args.length:.0f} m (max turn {turn:.1f} deg), "
                  f"road {wp.road_id} lane {wp.lane_id}, lane width {wp.lane_width:.1f} m")


if __name__ == "__main__":
    main()
