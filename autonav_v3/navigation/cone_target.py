# Import libraries
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Any

# Import local modules (robust to different run styles)
try:
    from .cone_frame import FrameConfig, to_vehicle_frame
except Exception:
    from navigation.cone_frame import FrameConfig, to_vehicle_frame  # type: ignore


@dataclass
class ConeNavConfig:
    # Settings that control how cones become one "aim point" for steering
    # We filter cones in a forward window, split them into left/right, then aim for the corridor center

    fwd_min_m: float
    fwd_max_m: float
    corridor_width_m: float
    target_fwd_m: float
    use_nearest_n: int


ConeBox = Tuple[float, float, float, float, float, float]
Point2D = Tuple[float, float]


def _validate_nav_config(navigation_config: ConeNavConfig) -> None:
    # Fail fast so we don't silently produce garbage targets
    if float(navigation_config.fwd_max_m) <= float(navigation_config.fwd_min_m):
        raise ValueError("cone_nav.fwd_max_m must be > cone_nav.fwd_min_m")

    if float(navigation_config.corridor_width_m) <= 0.0:
        raise ValueError("cone_nav.corridor_width_m must be > 0")

    if int(navigation_config.use_nearest_n) < 1:
        raise ValueError("cone_nav.use_nearest_n must be >= 1")


def _safe_cone_xy(cone: Any) -> Optional[Tuple[float, float]]:
    # Extract (xc, yc) from a cone tuple/list without crashing on malformed entries
    if not isinstance(cone, (list, tuple)) or len(cone) < 2:
        return None

    try:
        xc = float(cone[0])
        yc = float(cone[1])
        return xc, yc
    except Exception:
        return None


def compute_target_from_cones(
    cones_xyzwdh: Iterable[ConeBox],
    frame_config: FrameConfig,
    navigation_config: ConeNavConfig,
) -> Optional[Point2D]:
    # Validate configuration up front so bugs show up immediately
    _validate_nav_config(navigation_config)

    # Split cones into left and right boundaries so we can estimate the corridor center
    left_boundary_points: List[Point2D] = []
    right_boundary_points: List[Point2D] = []

    fwd_min = float(navigation_config.fwd_min_m)
    fwd_max = float(navigation_config.fwd_max_m)

    # Walk each cone and convert sensor coordinates into our vehicle frame (right, forward)
    for cone in cones_xyzwdh:
        xy = _safe_cone_xy(cone)
        if xy is None:
            continue

        xc, yc = xy
        right_m, forward_m = to_vehicle_frame(xc, yc, frame_config)

        # Ignore cones that are too close, behind us, or too far to be useful
        if not (fwd_min <= forward_m <= fwd_max):
            continue

        # Use the sign of right_m to decide which boundary this cone belongs to.
        # If right_m is exactly 0, treat it as "unknown side" and ignore it.
        # (Centerline detections can happen with noisy fusion; ignoring them avoids bias.)
        if right_m < 0.0:
            left_boundary_points.append((right_m, forward_m))
        elif right_m > 0.0:
            right_boundary_points.append((right_m, forward_m))

    # Sort by forward distance so "closest ahead" cones show up first
    left_boundary_points.sort(key=lambda point: point[1])
    right_boundary_points.sort(key=lambda point: point[1])

    # Keep only the nearest N cones per side to reduce noise from distant detections
    nearest_cone_count = int(navigation_config.use_nearest_n)
    left_boundary_points = left_boundary_points[:nearest_cone_count]
    right_boundary_points = right_boundary_points[:nearest_cone_count]

    # If we have no usable cones at all, we cannot compute a target
    if not left_boundary_points and not right_boundary_points:
        return None

    # Choose the forward lookahead location where we want to aim the steering
    target_forward_m = float(navigation_config.target_fwd_m)

    def estimate_right_at_target_forward(side_points: List[Point2D]) -> Optional[float]:
        # If a side has no cones, we cannot estimate its corridor boundary
        if not side_points:
            return None

        # Pick the cone whose forward distance is closest to our target lookahead
        closest_point = min(
            side_points,
            key=lambda point: abs(point[1] - target_forward_m),
        )
        return float(closest_point[0])

    # Estimate the corridor boundaries at the lookahead distance
    left_right_m = estimate_right_at_target_forward(left_boundary_points)
    right_right_m = estimate_right_at_target_forward(right_boundary_points)

    half_corridor_width_m = float(navigation_config.corridor_width_m) / 2.0

    # If both sides are visible, aim at the midpoint between left and right boundaries
    if left_right_m is not None and right_right_m is not None:
        target_right_m = 0.5 * (left_right_m + right_right_m)
        return (target_right_m, target_forward_m)

    # If we only see the left boundary, assume the corridor center is to the right by half width
    if left_right_m is not None and right_right_m is None:
        target_right_m = left_right_m + half_corridor_width_m
        return (target_right_m, target_forward_m)

    # If we only see the right boundary, assume the corridor center is to the left by half width
    if right_right_m is not None and left_right_m is None:
        target_right_m = right_right_m - half_corridor_width_m
        return (target_right_m, target_forward_m)

    # Fallback in case both estimates are None for any reason
    return None