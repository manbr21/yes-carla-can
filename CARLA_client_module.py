#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import argparse
import glob
import logging
import os
import signal
import sys

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import tkinter as tk

import carla
import pygame

from can_network.network import CAN_Network, VCAN_CHANNEL
from gui import CANTrafficDisplay, HUD, KeyboardControl, World


def game_loop(args):
    pygame.init()
    pygame.font.init()

    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    print(f"width: {width}, height: {height}")

    world = None
    original_settings = None
    can_bus = CAN_Network(channel=args.vcan)
    can_display = CANTrafficDisplay(channel=args.vcan)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        # Load the requested map (if any) before grabbing the world.
        if args.map:
            logging.info("Loading map %s ...", args.map)
            sim_world = client.load_world(args.map)
        else:
            sim_world = client.get_world()
            
        # Disable rendering and set fixed time step

        world_settings = sim_world.get_settings()
        world_settings.no_rendering_mode = False
        # fps = 30
        # world_settings.fixed_delta_seconds = round(1/fps, 2) # Set FPS
        if args.lane_keeping != "off" and not args.sync:
            # Fixed physics step: on a slow machine a variable step (~0.25 s) makes the controller unstable.
            world_settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(world_settings)

        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print(
                "WARNING: You are currently in asynchronous mode and could "
                "experience some issues with the traffic simulation"
            )

        display = pygame.display.set_mode(
            (args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args, can_bus)
        controller = KeyboardControl(world, args.autopilot)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync, can_bus):
                return
            world.tick(clock)
            world.render(display)
            can_display.render(display)
            # Render PiP last so it sits above all other overlays.
            
            if world.rgb_camera_sensor is not None:

                disp_w, disp_h = display.get_size()
              
                pip_w = world.rgb_camera_sensor.DISPLAY_WIDTH
                pip_h = world.rgb_camera_sensor.DISPLAY_HEIGHT
                world.rgb_camera_sensor.render(
                    display, pos=(disp_w - pip_w - 10, disp_h - pip_h - 10)
                )
            pygame.display.flip()

    finally:
        # Stop CARLA sensor streams first so the server can close sessions cleanly
        # before any other teardown that might talk to the server or tear down the
        # CAN interface.
        if world is not None:
            try:
                world.destroy()
            except Exception:
                pass

        can_display.stop()
        can_bus.bus.shutdown()

        try:
            if original_settings:
                sim_world.apply_settings(original_settings)
        except Exception:
            pass

        try:
            if world and world.recording_enabled:
                client.stop_recorder()
        except Exception:
            pass

        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Receiver Client")
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="print debug information",
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "-a", "--autopilot", action="store_true", help="enable autopilot"
    )
    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="1280x720",
        help="window resolution (default: 1280x720)",
    )
    argparser.add_argument(
        "--filter",
        metavar="PATTERN",
        default="vehicle.*",
        help='actor filter (default: "vehicle.*")',
    )
    argparser.add_argument(
        "--generation",
        metavar="G",
        default="2",
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")',
    )
    argparser.add_argument(
        "--rolename",
        metavar="NAME",
        default="hero",
        help='actor role name (default: "hero")',
    )
    argparser.add_argument(
        "--gamma",
        default=2.2,
        type=float,
        help="Gamma correction of the camera (default: 2.2)",
    )
    argparser.add_argument(
        "--sync", action="store_true", help="Activate synchronous mode execution"
    )
    argparser.add_argument(
        "--vcan",
        default=VCAN_CHANNEL,
        help=f"Virtual CAN interface name (default: {VCAN_CHANNEL})",
    )
    argparser.add_argument(
        "--map",
        metavar="NAME",
        default=None,
        help='CARLA map to load (e.g. "Town01", "Town10HD"). '
             'If not set, the current world is used.',
    )
    argparser.add_argument(
        "--psa-yaw",
        type=float,
        default=0.0,
        metavar="DEG",
        help="Perspective-Shift Attack: yaw offset (degrees) applied to the front cameras. "
             "Positive turns the view to the right (CARLA convention). Default: 0 (no attack).",
    )
    argparser.add_argument(
        "--lane-keeping",
        choices=["off", "log", "drive"],
        default="off",
        help="Camera-based lane keeping node: 'log' only measures, 'drive' also sends STEER/THROTTLE over CAN.",
    )
    argparser.add_argument("--lk-gains", type=float, nargs=2, default=[0.5, 0.05], metavar=("KP", "KD"),
                           help="Lane keeping PD gains (default: 0.5 0.05).")
    argparser.add_argument("--camera-x", type=float, default=None, metavar="X",
                           help="Lane keeping camera position, metres from the car centre (default: in front of the car; 0.6 shows the hood).")
    argparser.add_argument("--hood-ref", type=float, default=0.0, metavar="PX",
                           help="Hood position (px from the image centre) measured without attack; the defence works on the difference.")
    argparser.add_argument("--psa-mode", choices=["rotate", "shift"], default="shift",
                           help="shift (default): shift the whole image, hood included, by the angle of --psa-yaw; rotate: turn the camera instead.")
    argparser.add_argument("--hood-b-ref", type=float, default=-0.0154, metavar="B",
                           help="Benign value of the paper's linear coefficient b on this camera (default: the paper's own example).")
    argparser.add_argument("--defense", choices=["off", "detect", "stop", "compensate"], default="off",
                           help="Hood defence: detect = alarm only, stop = alarm and brake, compensate = undo the shift.")
    argparser.add_argument("--lk-dump", default=None, metavar="DIR", help="Save a segmentation image every 2 s into DIR (debug).")
    argparser.add_argument("--target-speed", type=float, default=30.0, metavar="KMH", help="Lane keeping target speed (default: 30).")
    argparser.add_argument("--run-seconds", type=float, default=0.0, metavar="S",
                           help="Stop the lane keeping run after S seconds of driving (0 = never).")
    argparser.add_argument("--log-csv", default=None, metavar="FILE", help="Write lane keeping measurements to this CSV file.")
    argparser.add_argument("--spawn-index", type=int, default=None, metavar="N",
                           help="Use spawn point N of the map instead of a random one (repeatable runs).")
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split("x")]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)
    logging.info("listening to server %s:%s", args.host, args.port)

    print(__doc__)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")


if __name__ == "__main__":
    main()
