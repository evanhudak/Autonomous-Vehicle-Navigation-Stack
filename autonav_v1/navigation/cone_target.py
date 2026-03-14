# import function from dataclass
from dataclasses import dataclass

# import functions from typing
from typing import Iterable, List, Optional, Tuple

# import functions from cone_frame
from .cone_frame import FrameConfig, to_vehicle_frame

@dataclass
class ConeNavConfig:

    """Configuration for converting cone detections into a single lookahead target point."""
    
    fwd_min_m: float
    fwd_max_m: float
    corridor_width_m: float
    target_fwd_m: float
    use_nearest_n: int

ConeBox = Tuple[float, float, float, float, float, float]
Point2D = Tuple[float, float]

def compute_target_from_cones(cones_xyzwdh: Iterable[ConeBox],
                              frame_cfg: FrameConfig,
                              nav_cfg: ConeNavConfig) -> Optional[Point2D]:

    """
    Convert detected cones into a local navigation target in the vehicle frame.

    Inputs:
        cones_xyzwdh: Iterable of (xc, yc, zc, w, d, h) cone detections from LiDAR/perception.
        frame_cfg: Mapping from sensor axes to vehicle frame (right, forward).
        nav_cfg: Forward window, corridor assumptions, and target selection parameters.

    Output:
        (right_target_m, fwd_target_m) target point, or None if no valid cones exist.
    """

    # Partition cones by side based on sign of right coordinate in vehicle frame.
    left_cones: List[Point2D] = []
    right_cones: List[Point2D] = []

    for (xc, yc, _zc, _w, _d, _h) in cones_xyzwdh:
        right_m, fwd_m = to_vehicle_frame(xc, yc, frame_cfg)

        # Ignore cones outside the forward window (behind, too close, or too far).
        if not (nav_cfg.fwd_min_m <= fwd_m <= nav_cfg.fwd_max_m):
            continue

        # Negative right = left side, positive right = right side.
        if right_m < 0.0:
            left_cones.append((right_m, fwd_m))
        elif right_m > 0.0:
            right_cones.append((right_m, fwd_m))

    # Sort by forward distance so the closest cones ahead are first.
    left_cones.sort(key=lambda point: point[1])
    right_cones.sort(key=lambda point: point[1])

    # Keep only the nearest N cones on each side to reduce noise from far detections.
    nearest_count = max(1, int(nav_cfg.use_nearest_n))
    left_cones = left_cones[:nearest_count]
    right_cones = right_cones[:nearest_count]

    # If nothing remains after filtering, we cannot compute a target.
    if not left_cones and not right_cones:
        return None

    # Choose a forward lookahead location where we want the lateral target to be evaluated.
    target_fwd_m = float(nav_cfg.target_fwd_m)

    def get_right_at_target_fwd(side_points: List[Point2D]) -> Optional[float]:

        """
        Estimate lateral position (right_m) of a corridor boundary at target_fwd_m.

        For v1: choose the cone whose forward distance is closest to target_fwd_m.
        """

        if not side_points:
            return None

        closest_point = min(side_points, key=lambda point: abs(point[1] - target_fwd_m))
        return float(closest_point[0])

    left_right_m = get_right_at_target_fwd(left_cones)
    right_right_m = get_right_at_target_fwd(right_cones)

    half_corridor_width_m = float(nav_cfg.corridor_width_m) / 2.0

    # If both corridor boundaries are visible, aim for the midpoint between them.
    if left_right_m is not None and right_right_m is not None:
        target_right_m = 0.5 * (left_right_m + right_right_m)
        return (target_right_m, target_fwd_m)

    # If only left boundary is visible, assume center is half corridor width to the right.
    if left_right_m is not None and right_right_m is None:
        return (left_right_m + half_corridor_width_m, target_fwd_m)

    # If only right boundary is visible, assume center is half corridor width to the left.
    if right_right_m is not None and left_right_m is None:
        return (right_right_m - half_corridor_width_m, target_fwd_m)

    return None