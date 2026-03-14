# import libraries
import time
import yaml

# import functions from serial_io
from serial_io.steering import SteeringSerial
from serial_io.throttle import ThrottleSerial
from serial_io.brake import BrakeSerial

# import functions from navigation
from navigation.cone_frame import FrameConfig
from navigation.cone_target import ConeNavConfig, compute_target_from_cones
from navigation.pure_pursuit import steer_deg_from_target, steer_percent_from_deg
from navigation.speed_rules import SpeedConfig, compute_speed_commands

# import function from sensors
from sensors.cone_source_file import FileConeSource

def load_cfg(path: str) -> dict:
    # Load YAML config file into a Python dictionary.
    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)

def main() -> None:
    # Load configuration and derive loop timing.
    config = load_cfg("config.yaml")
    dry_run = bool(config.get("dry_run", False))

    loop_hz = float(config.get("loop_hz", 20))
    loop_period_s = 1.0 / max(1.0, loop_hz)

    # Configure how LiDAR cone coordinates map into vehicle control frame (right, forward).
    frame_cfg = FrameConfig(forward_axis=str(config["frame"]["forward_axis"]),
                            forward_sign=int(config["frame"]["forward_sign"]),
                            right_sign=int(config["frame"]["right_sign"]),)

    # Cone navigation settings: windowing, corridor assumptions, and target selection.
    cone_nav_cfg = ConeNavConfig(fwd_min_m=float(config["cone_nav"]["fwd_min_m"]),
                                 fwd_max_m=float(config["cone_nav"]["fwd_max_m"]),
                                 corridor_width_m=float(config["cone_nav"]["corridor_width_m"]),
                                 target_fwd_m=float(config["cone_nav"]["target_fwd_m"]),
                                 use_nearest_n=int(config["cone_nav"]["use_nearest_n"]),)

    lookahead_m = float(config["pure_pursuit"]["lookahead_m"])

    # Steering mapping settings: steering degrees mapped to 0..100 percent.
    max_wheel_deg = float(config["steering"]["max_wheel_deg"])
    steering_invert = bool(config["steering"]["invert"])

    # Speed/braking rule settings: open-loop throttle plus safety overrides.
    speed_cfg = SpeedConfig(base_throttle_pct=int(config["speed"]["base_throttle_pct"]),
                            min_throttle_pct=int(config["speed"]["min_throttle_pct"]),
                            max_throttle_pct=int(config["speed"]["max_throttle_pct"]),
                            steering_slow_gain=float(config["speed"]["steering_slow_gain"]),
                            enable_emergency_stop=bool(config["speed"]["enable_emergency_stop"]),
                            estop_corridor_half_width_m=float(
                                config["speed"]["estop_corridor_half_width_m"]),
                            estop_distance_m=float(config["speed"]["estop_distance_m"]),
                            slow_distance_m=float(config["speed"]["slow_distance_m"]),
                            brake_pct_estop=int(config["speed"]["brake_pct_estop"]),
                            brake_pct_slow=int(config["speed"]["brake_pct_slow"]),)

    # Select cone source implementation. For v1, we support file-based cones for demos/tests.
    cone_source_type = str(config["cone_source"]["type"])
    if cone_source_type == "file":
        cone_source = FileConeSource(str(config["cone_source"]["file_path"]))
    else:
        raise ValueError(f"Unknown cone_source.type: {cone_source_type}")

    # Initialize subsystem serial interfaces unless in dry_run mode.
    steering_serial = None
    throttle_serial = None
    brake_serial = None

    if not dry_run:
        baud_rate = int(config["serial"]["baud"])
        steering_port = str(config["serial"]["steering_port"])
        throttle_port = str(config["serial"]["throttle_port"])
        brake_port = str(config["serial"]["brake_port"])

        steering_serial = SteeringSerial(steering_port, baud_rate)
        throttle_serial = ThrottleSerial(throttle_port, baud_rate)
        brake_serial = BrakeSerial(brake_port, baud_rate)

        # Safe initialization: center steering, zero throttle, zero brake.
        steering_serial.set_percent(50)
        throttle_serial.set_percent(0)
        brake_serial.set_percent(0)

        print("autonav_v1 running (LIVE SERIAL). Ctrl+C to stop safely.")
    else:
        print("autonav_v1 running (DRY RUN - no serial). Ctrl+C to stop.")

    last_print_time_s = 0.0

    try:
        while True:
            # Mark loop start time for rate control.
            loop_start_time_s = time.time()

            # Read the current cone detections (xc, yc, zc, w, d, h).
            cone_boxes = cone_source.get_cones()

            # Compute a target point (right_m, fwd_m) based on cones and corridor assumptions.
            target_point = compute_target_from_cones(cone_boxes, frame_cfg, cone_nav_cfg,)

            if target_point is None:
                # Fail-safe behavior when cones are not visible: stop and apply moderate braking.
                steering_percent = 50
                throttle_percent = 0
                brake_percent = 60
                steering_degrees = None
            else:
                # Use pure pursuit toward target point to compute desired steering.
                target_right_m, target_fwd_m = target_point
                steering_degrees = steer_deg_from_target(target_right_m, target_fwd_m, lookahead_m,)
                steering_percent = steer_percent_from_deg(steering_degrees,
                                                          max_wheel_deg,
                                                          steering_invert,)

                # Compute throttle and brake commands using steering demand and safety corridor checks.
                throttle_percent, brake_percent = compute_speed_commands(steer_deg=steering_degrees,
                                                                         cones_xyzwdh=cone_boxes,
                                                                         frame_cfg=frame_cfg,
                                                                         spd_cfg=speed_cfg,)

            # Send actuator commands only if serial is enabled.
            if not dry_run:
                steering_serial.set_percent(steering_percent)
                throttle_serial.set_percent(throttle_percent)
                brake_serial.set_percent(brake_percent)

            # Print status periodically so the console remains readable.
            now_time_s = time.time()
            if now_time_s - last_print_time_s > 0.5:
                last_print_time_s = now_time_s

                if steering_degrees is None:
                    print("cones={} target={} steer={} thr={} brk={}".format(len(cone_boxes),
                                                                            target_point,
                                                                            steering_percent,
                                                                            throttle_percent,
                                                                            brake_percent,))
                else:
                    print(("cones={} target={} steer_deg={:.2f} steer={} thr={} brk={}").format(
                        len(cone_boxes),
                        target_point,
                        steering_degrees,
                        steering_percent,
                        throttle_percent,
                        brake_percent,))

            # Sleep the remaining loop time to maintain loop_hz.
            elapsed_s = time.time() - loop_start_time_s
            if elapsed_s < loop_period_s:
                time.sleep(loop_period_s - elapsed_s)

    except KeyboardInterrupt:
        # On Ctrl+C, bring the system to a safe state (if live) or just exit (if dry run).
        if not dry_run:
            print("\nStopping: throttle STOP, brake=100, steer center.")

            try:
                throttle_serial.stop()
            except Exception:
                pass

            try:
                throttle_serial.set_percent(0)
            except Exception:
                pass

            try:
                brake_serial.set_percent(100)
            except Exception:
                pass

            try:
                steering_serial.set_percent(50)
            except Exception:
                pass
        else:
            print("\nStopped (dry run).")

    finally:
        # Close serial resources if opened.
        if steering_serial is not None:
            try:
                steering_serial.close()
            except Exception:
                pass

        if throttle_serial is not None:
            try:
                throttle_serial.close()
            except Exception:
                pass

        if brake_serial is not None:
            try:
                brake_serial.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()