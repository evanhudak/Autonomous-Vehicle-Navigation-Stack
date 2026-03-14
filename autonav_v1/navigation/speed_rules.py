# import function from dataclass
from dataclasses import dataclass

# import functions from typing
from typing import Iterable, Tuple

# import functions from cone_fram
from .cone_frame import FrameConfig, to_vehicle_frame

ConeBox = Tuple[float, float, float, float, float, float]

@dataclass
class SpeedConfig:

    """Configuration for throttle/brake behavior based on steering and nearby obstacles."""

    base_throttle_pct: int
    min_throttle_pct: int
    max_throttle_pct: int
    steering_slow_gain: float

    enable_emergency_stop: bool
    estop_corridor_half_width_m: float
    estop_distance_m: float
    slow_distance_m: float
    brake_pct_estop: int
    brake_pct_slow: int

def compute_speed_commands(steer_deg: float,
                           cones_xyzwdh: Iterable[ConeBox],
                           frame_cfg: FrameConfig,
                           spd_cfg: SpeedConfig) -> Tuple[int, int]:

    """
    Compute throttle and brake commands based on steering demand and cone proximity.

    Args:
        steer_deg: Current steering demand (degrees).
        cones_xyzwdh: Iterable of cone detections (xc, yc, zc, w, d, h).
        frame_cfg: Mapping from sensor coordinates into vehicle (right, forward).
        spd_cfg: Speed/brake tuning parameters.

    Returns:
        (throttle_pct, brake_pct) as integers in [0, 100].
    """

    # Start with a baseline throttle and reduce it proportionally with steering magnitude.
    throttle_cmd = float(spd_cfg.base_throttle_pct)
    throttle_cmd -= float(spd_cfg.steering_slow_gain) * abs(float(steer_deg))

    # Clamp throttle into the configured operating range.
    min_throttle = float(spd_cfg.min_throttle_pct)
    max_throttle = float(spd_cfg.max_throttle_pct)
    throttle_cmd = max(min_throttle, min(max_throttle, throttle_cmd))

    brake_cmd = 0

    # If emergency stop logic is disabled, return only turn-slowed throttle.
    if not spd_cfg.enable_emergency_stop:
        return int(round(throttle_cmd)), brake_cmd

    # Scan cone detections for a close obstacle in a forward corridor.
    # "Corridor" means: forward of vehicle and within +/- corridor half-width laterally.
    should_stop = False
    should_slow = False

    for (xc, yc, _zc, _w, _d, _h) in cones_xyzwdh:
        right_m, fwd_m = to_vehicle_frame(xc, yc, frame_cfg)

        # Ignore cones behind the vehicle.
        if fwd_m <= 0.0:
            continue

        # Ignore cones outside the lateral corridor around the vehicle centerline.
        if abs(right_m) > float(spd_cfg.estop_corridor_half_width_m):
            continue

        # If any cone is within the emergency-stop distance, stop immediately.
        if fwd_m < float(spd_cfg.estop_distance_m):
            should_stop = True
            break

        # If a cone is within the slow distance (but not stop distance), request slowing.
        if fwd_m < float(spd_cfg.slow_distance_m):
            should_slow = True

    if should_stop:
        return 0, int(spd_cfg.brake_pct_estop)

    if should_slow:
        # Reduce throttle to minimum and apply a moderate brake value.
        throttle_cmd = min(throttle_cmd, min_throttle)
        brake_cmd = int(spd_cfg.brake_pct_slow)

    return int(round(throttle_cmd)), brake_cmd