# Import libraries
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Serial IO (not used in push_test, but kept for bench/road later)
from serial_io.brake import BrakeSerial
from serial_io.steering import SteeringSerial
from serial_io.throttle import ThrottleSerial

# Navigation / Control logic
from navigation.cone_frame import FrameConfig
from navigation.cone_target import ConeNavConfig, compute_target_from_cones
from navigation.pure_pursuit import steer_deg_from_target, steer_percent_from_deg
from navigation.speed_rules import SpeedConfig, compute_speed_commands

# New sensor sources (camera / lidar / fusion)
from sensors.cone_source_camera import CameraConeSource, CameraConeConfig
from sensors.cone_source_lidar import LidarConeSource, LidarConeConfig
from sensors.cone_source_fusion import FusionConeSource, FusionConfig


JsonDict = Dict[str, Any]
ConeList = List[Any]


def load_config(config_path: str) -> JsonDict:
    # Load YAML so behavior can be tuned without touching code
    with open(config_path, "r", encoding="utf-8") as file_handle:
        cfg = yaml.safe_load(file_handle)

    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must load to a dictionary at the top level.")

    return cfg


class JsonLinesLogger:
    # Write one JSON object per line so logs survive crashes and stream cleanly
    def __init__(self, log_path: str) -> None:
        logs_folder = os.path.dirname(log_path)
        if logs_folder:
            os.makedirs(logs_folder, exist_ok=True)

        # Overwrite each run so the current test is always easy to find
        self._file_handle = open(log_path, "w", encoding="utf-8")

    def write(self, record: JsonDict) -> None:
        self._file_handle.write(json.dumps(record) + "\n")
        self._file_handle.flush()

    def close(self) -> None:
        try:
            self._file_handle.close()
        except Exception:
            pass


def now_wall_time_s() -> float:
    # Wall time is useful for timestamps humans will read later
    return time.time()


def now_monotonic_s() -> float:
    # Monotonic time is stable for dt measurements and loop timing
    return time.monotonic()


def read_cones_with_metadata(
    cone_source: Any,
) -> Tuple[ConeList, Optional[float], Optional[str]]:
    # Try to read richer live metadata first so we can enforce freshness rules
    if hasattr(cone_source, "get_frame"):
        try:
            cone_boxes, frame_time_s, frame_id = cone_source.get_frame()
            return cone_boxes, frame_time_s, None if frame_id is None else str(frame_id)
        except Exception:
            # If a live read fails, fall back to basic reads so the system keeps running
            pass

    # Fall back to basic cone reads when metadata is not available
    try:
        cone_boxes = cone_source.get_cones()
    except Exception:
        cone_boxes = []
    return cone_boxes, None, None

def serialize_cones(cones: ConeList) -> List[JsonDict]:
    # Convert cones into a consistent shape so logs are easy to parse later
    serialized_cones: List[JsonDict] = []

    for cone_box in cones:
        if isinstance(cone_box, (list, tuple)) and len(cone_box) >= 6:
            serialized_cones.append(
                {
                    "xc": float(cone_box[0]),
                    "yc": float(cone_box[1]),
                    "zc": float(cone_box[2]),
                    "w": float(cone_box[3]),
                    "d": float(cone_box[4]),
                    "h": float(cone_box[5]),
                }
            )
            continue

        if hasattr(cone_box, "__dict__"):
            serialized_cones.append(dict(cone_box.__dict__))
            continue

        if isinstance(cone_box, dict):
            serialized_cones.append(dict(cone_box))
            continue

        serialized_cones.append({"raw": str(cone_box)})

    return serialized_cones


def main() -> None:
    # Always resolve config relative to this file so running from other CWDs works
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")

    config = load_config(config_path)

    mode = str(config.get("mode", "bench")).lower().strip()
    dry_run_from_config = bool(config.get("dry_run", False))

    # In push_test we always disable actuation even if someone edits config incorrectly
    dry_run = True if mode == "push_test" else dry_run_from_config

    loop_hz = float(config.get("loop_hz", 20))
    loop_period_s = 1.0 / max(1.0, loop_hz)

    logging_config = config.get("logging", {})
    if not isinstance(logging_config, dict):
        logging_config = {}

    logging_enabled = bool(logging_config.get("enable", mode == "push_test"))
    log_path = str(logging_config.get("path", "logs/push_test.jsonl"))
    log_path = os.path.join(base_dir, log_path) if not os.path.isabs(log_path) else log_path

    write_every_n = int(logging_config.get("write_every_n", 1))
    write_every_n = max(1, write_every_n)

    logger: Optional[JsonLinesLogger] = JsonLinesLogger(log_path) if logging_enabled else None

    health_config = config.get("health", {})
    if not isinstance(health_config, dict):
        health_config = {}

    sensor_timeout_ms = int(health_config.get("sensor_timeout_ms", 200))

    max_cones = int(health_config.get("max_cones", 50))
    max_cones = max(1, max_cones)

    # Validate required nested sections exist before accessing them
    for required_key in ("frame", "cone_nav", "pure_pursuit", "steering", "speed", "cone_source"):
        if required_key not in config or not isinstance(config[required_key], dict):
            raise ValueError(f"config.yaml missing required section: {required_key}")

    frame_config = FrameConfig(
        forward_axis=str(config["frame"]["forward_axis"]),
        forward_sign=int(config["frame"]["forward_sign"]),
        right_sign=int(config["frame"]["right_sign"]),
    )

    cone_nav_config = ConeNavConfig(
        fwd_min_m=float(config["cone_nav"]["fwd_min_m"]),
        fwd_max_m=float(config["cone_nav"]["fwd_max_m"]),
        corridor_width_m=float(config["cone_nav"]["corridor_width_m"]),
        target_fwd_m=float(config["cone_nav"]["target_fwd_m"]),
        use_nearest_n=int(config["cone_nav"]["use_nearest_n"]),
    )

    lookahead_m = float(config["pure_pursuit"]["lookahead_m"])

    max_wheel_deg = float(config["steering"]["max_wheel_deg"])
    steering_invert = bool(config["steering"]["invert"])

    speed_config = SpeedConfig(
        base_throttle_pct=int(config["speed"]["base_throttle_pct"]),
        min_throttle_pct=int(config["speed"]["min_throttle_pct"]),
        max_throttle_pct=int(config["speed"]["max_throttle_pct"]),
        steering_slow_gain=float(config["speed"]["steering_slow_gain"]),
        enable_emergency_stop=bool(config["speed"]["enable_emergency_stop"]),
        estop_corridor_half_width_m=float(config["speed"]["estop_corridor_half_width_m"]),
        estop_distance_m=float(config["speed"]["estop_distance_m"]),
        slow_distance_m=float(config["speed"]["slow_distance_m"]),
        brake_pct_estop=int(config["speed"]["brake_pct_estop"]),
        brake_pct_slow=int(config["speed"]["brake_pct_slow"]),
    )

    cone_source_config = config.get("cone_source", {})
    if not isinstance(cone_source_config, dict):
        cone_source_config = {}

    cone_source_type = str(cone_source_config.get("type", "file")).lower().strip()

    # ---------
    # Cone source selection
    # ---------
    if cone_source_type == "file":
        # Import only if you actually use file mode
        try:
            from sensors.cone_source_file import FileConeSource  # type: ignore
        except Exception as exc:
            raise ImportError(
                "cone_source.type is 'file' but sensors/cone_source_file.py is missing "
                "or has an import error."
            ) from exc

        file_path = str(cone_source_config.get("file_path", "cone_sample.json"))
        file_path = os.path.join(base_dir, file_path) if not os.path.isabs(file_path) else file_path
        cone_source = FileConeSource(file_path)

    elif cone_source_type == "live":
        # Import only if you actually use live mode
        try:
            from sensors.cone_source_live import LiveConeSource  # type: ignore
        except Exception as exc:
            raise ImportError(
                "cone_source.type is 'live' but sensors/cone_source_live.py is missing "
                "or has an import error."
            ) from exc

        live_config = cone_source_config.get("live", {})
        if not isinstance(live_config, dict):
            live_config = {}

        cone_source = LiveConeSource(live_config)

    elif cone_source_type == "camera":
        camera_raw = cone_source_config.get("camera", {})
        if not isinstance(camera_raw, dict):
            camera_raw = {}

        camera_cfg = CameraConeConfig(
            camera_height_m=float(camera_raw.get("camera_height_m", 0.8)),
            camera_pitch_deg=float(camera_raw.get("camera_pitch_deg", 0.0)),
            horizontal_fov_deg=float(camera_raw.get("horizontal_fov_deg", 70.0)),
            vertical_fov_deg=float(camera_raw.get("vertical_fov_deg", 55.0)),
            assumed_cone_w_m=float(camera_raw.get("assumed_cone_w_m", 0.15)),
            assumed_cone_d_m=float(camera_raw.get("assumed_cone_d_m", 0.15)),
            assumed_cone_h_m=float(camera_raw.get("assumed_cone_h_m", 0.0)),
            min_forward_m=float(camera_raw.get("min_forward_m", 0.25)),
            max_forward_m=float(camera_raw.get("max_forward_m", 30.0)),
        )

        model_path = camera_raw.get("model_path", None)
        if isinstance(model_path, str) and model_path and not os.path.isabs(model_path):
            model_path = os.path.join(base_dir, model_path)

        rgb = bool(camera_raw.get("rgb", True))
        cone_source = CameraConeSource(model_path=model_path, rgb=rgb, config=camera_cfg)

    elif cone_source_type == "lidar":
        lidar_raw = cone_source_config.get("lidar", {})
        if not isinstance(lidar_raw, dict):
            lidar_raw = {}

        lidar_cfg = LidarConeConfig(
            port=str(lidar_raw.get("port", "/dev/serial0")),
            baudrate=int(lidar_raw.get("baudrate", 230400)),
            min_cluster_size=int(lidar_raw.get("min_cluster_size", 5)),
            cluster_distance_threshold=float(lidar_raw.get("cluster_distance_threshold", 0.08)),
            min_range_m=float(lidar_raw.get("min_range_m", 0.20)),
            max_range_m=float(lidar_raw.get("max_range_m", 30.0)),
            max_box_width_m=float(lidar_raw.get("max_box_width_m", 1.5)),
            max_box_depth_m=float(lidar_raw.get("max_box_depth_m", 1.5)),
        )

        cone_source = LidarConeSource(config=lidar_cfg)

        print(f"LiDAR config: port={lidar_cfg.port} baud={lidar_cfg.baudrate}")

    elif cone_source_type == "fusion":
        fusion_raw = cone_source_config.get("fusion", {})
        if not isinstance(fusion_raw, dict):
            fusion_raw = {}

        fusion_cfg = FusionConfig(
            lidar_port=str(fusion_raw.get("lidar_port", "/dev/serial0")),
            cones_only=bool(fusion_raw.get("cones_only", True)),
            min_range_m=float(fusion_raw.get("min_range_m", 0.20)),
            max_range_m=float(fusion_raw.get("max_range_m", 30.0)),
            max_cones=int(fusion_raw.get("max_cones", 50)),
        )

        cone_source = FusionConeSource(config=fusion_cfg)

        print(f"Fusion config: lidar_port={fusion_cfg.lidar_port}")

    else:
        raise ValueError(f"Unknown cone_source.type: {cone_source_type}")

    # Prepare serial interfaces for later, but keep them unused during push_test
    steering_serial: Optional[SteeringSerial] = None
    throttle_serial: Optional[ThrottleSerial] = None
    brake_serial: Optional[BrakeSerial] = None

    if not dry_run:
        baud_rate = int(config["serial"]["baud"])
        steering_port = str(config["serial"]["steering_port"])
        throttle_port = str(config["serial"]["throttle_port"])
        brake_port = str(config["serial"]["brake_port"])

        steering_serial = SteeringSerial(steering_port, baud_rate)
        throttle_serial = ThrottleSerial(throttle_port, baud_rate)
        brake_serial = BrakeSerial(brake_port, baud_rate)

        steering_serial.set_percent(50)
        throttle_serial.set_percent(0)
        brake_serial.set_percent(0)

        print(f"autonav_v2 RUNNING (LIVE SERIAL) mode={mode} source={cone_source_type}")
    else:
        print(f"autonav_v2 RUNNING (PUSH TEST / DRY RUN) mode={mode} source={cone_source_type}")

    print(f"Logging: {'ON' if logger else 'OFF'} -> {log_path}")

    if mode == "push_test":
        print("Push-test safety: ACTUATION DISABLED (throttle/brake sent = 0)")

    last_console_print_time_s = 0.0
    loop_count = 0
    previous_loop_time_s = now_monotonic_s()

    try:
        while True:
            loop_start_time_s = now_monotonic_s()
            wall_time_s = now_wall_time_s()

            cone_boxes, frame_time_s, frame_id = read_cones_with_metadata(cone_source)

            if cone_boxes is None:
                cone_boxes = []

            if len(cone_boxes) > max_cones:
                cone_boxes = cone_boxes[:max_cones]

            sensor_age_ms: Optional[int] = None
            sensor_frame_ok = True

            if frame_time_s is not None:
                sensor_age_ms = int((wall_time_s - float(frame_time_s)) * 1000.0)
                if sensor_age_ms > sensor_timeout_ms:
                    sensor_frame_ok = False

            if not sensor_frame_ok:
                cone_boxes = []

            target_point_m = compute_target_from_cones(
                cone_boxes,
                frame_config,
                cone_nav_config,
            )

            if target_point_m is None:
                steering_deg: Optional[float] = None
                steering_pct = 50
                throttle_pct_would = 0
                brake_pct_would = 60
            else:
                target_right_m, target_forward_m = target_point_m

                steering_deg = steer_deg_from_target(
                    target_right_m,
                    target_forward_m,
                    lookahead_m,
                )

                steering_pct = steer_percent_from_deg(
                    steering_deg,
                    max_wheel_deg,
                    steering_invert,
                )

                throttle_pct_would, brake_pct_would = compute_speed_commands(
                    steer_deg=steering_deg,
                    cones_xyzwdh=cone_boxes,
                    frame_cfg=frame_config,
                    spd_cfg=speed_config,
                )

            if mode == "push_test":
                throttle_pct_sent = 0
                brake_pct_sent = 0
            else:
                throttle_pct_sent = int(throttle_pct_would)
                brake_pct_sent = int(brake_pct_would)

            if not dry_run:
                steering_serial.set_percent(int(steering_pct))
                throttle_serial.set_percent(int(throttle_pct_sent))
                brake_serial.set_percent(int(brake_pct_sent))

            current_loop_time_s = now_monotonic_s()
            dt_s = current_loop_time_s - previous_loop_time_s
            previous_loop_time_s = current_loop_time_s

            if logger is not None and (loop_count % write_every_n == 0):
                record: JsonDict = {
                    "t_wall_s": wall_time_s,
                    "dt_s": dt_s,
                    "loop_hz_meas": (1.0 / dt_s) if dt_s > 1e-6 else None,
                    "mode": mode,
                    "dry_run": dry_run,
                    "sensor": {
                        "source": cone_source_type,
                        "frame_id": frame_id,
                        "age_ms": sensor_age_ms,
                        "ok": sensor_frame_ok,
                    },
                    "cones_raw": serialize_cones(cone_boxes),
                    "nav": {
                        "target_point_m": None if target_point_m is None else {
                            "right": float(target_point_m[0]),
                            "fwd": float(target_point_m[1]),
                        },
                        "steer_deg": None if steering_deg is None else float(steering_deg),
                        "steer_pct": int(steering_pct),
                    },
                    "cmd": {
                        "sent": {
                            "steer_pct": int(steering_pct),
                            "throttle_pct": int(throttle_pct_sent),
                            "brake_pct": int(brake_pct_sent),
                        },
                        "would": {
                            "throttle_pct": int(throttle_pct_would),
                            "brake_pct": int(brake_pct_would),
                        },
                    },
                }
                logger.write(record)

            if (loop_start_time_s - last_console_print_time_s) > 0.5:
                last_console_print_time_s = loop_start_time_s

                if steering_deg is None:
                    print(
                        "cones={} age_ms={} ok={} target=None steer={} "
                        "would_thr={} would_brk={}".format(
                            len(cone_boxes),
                            sensor_age_ms,
                            sensor_frame_ok,
                            steering_pct,
                            throttle_pct_would,
                            brake_pct_would,
                        )
                    )
                else:
                    print(
                        "cones={} age_ms={} ok={} target={} steer_deg={:.2f} "
                        "steer={} would_thr={} would_brk={}".format(
                            len(cone_boxes),
                            sensor_age_ms,
                            sensor_frame_ok,
                            target_point_m,
                            steering_deg,
                            steering_pct,
                            throttle_pct_would,
                            brake_pct_would,
                        )
                    )

            elapsed_s = now_monotonic_s() - loop_start_time_s
            if elapsed_s < loop_period_s:
                time.sleep(loop_period_s - elapsed_s)

            loop_count += 1

    except KeyboardInterrupt:
        if not dry_run:
            print("\nStopping: throttle STOP, brake=100, steer center")
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
            print("\nStopped (push test / dry run)")

    finally:
        if logger is not None:
            logger.close()

        if hasattr(cone_source, "close"):
            try:
                cone_source.close()
            except Exception:
                pass

        for serial_dev in (steering_serial, throttle_serial, brake_serial):
            if serial_dev is not None:
                try:
                    serial_dev.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()