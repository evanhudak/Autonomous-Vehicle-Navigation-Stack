# Import libraries
import math
from typing import Optional


def _is_finite(value: float) -> bool:
    # Helper so weird sensor numbers can't break steering math
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def steer_deg_from_target(
    right_m: float,
    fwd_m: float,
    lookahead_m: float,
) -> float:
    # Clamp lookahead so we do not blow up on tiny values
    lookahead_dist_m = max(0.25, float(lookahead_m))

    # If inputs are bad, default to straight (safe)
    if not _is_finite(right_m) or not _is_finite(fwd_m) or not _is_finite(lookahead_dist_m):
        return 0.0

    # Compute distance to the target so we can estimate curvature safely
    target_dist_m = math.hypot(float(right_m), float(fwd_m))

    # Protect against divide-by-zero when the target is extremely close
    target_dist_m = max(1e-3, target_dist_m)

    # Estimate curvature using a simple pure pursuit approximation
    # Larger lateral offset or closer target should create a stronger steering response
    curvature = 2.0 * float(right_m) / (target_dist_m * target_dist_m)

    # Convert curvature into a stable steering angle proxy using lookahead
    steering_rad = math.atan(curvature * lookahead_dist_m)

    # Output degrees so it is easier to reason about and easier to map to hardware
    steering_deg = math.degrees(steering_rad)

    # Final defense against NaN/inf
    if not _is_finite(steering_deg):
        return 0.0

    return float(steering_deg)


def steer_percent_from_deg(
    steering_deg: float,
    max_wheel_deg: float,
    invert: bool,
) -> int:
    # Clamp max wheel angle so we never divide by zero
    max_allowed_deg = float(max_wheel_deg)

    if not _is_finite(max_allowed_deg) or max_allowed_deg <= 0.0:
        raise ValueError("steering.max_wheel_deg must be > 0")

    # If input is bad, default to centered
    if not _is_finite(steering_deg):
        return 50

    # Clamp requested steering to what the hardware can actually do
    clamped_steering_deg = max(
        -max_allowed_deg,
        min(max_allowed_deg, float(steering_deg)),
    )

    # Map [-max_deg, +max_deg] into [0, 100] with 50 as centered
    steering_percent = 50.0 + (clamped_steering_deg / max_allowed_deg) * 50.0

    # Allow inversion so wiring differences do not require changing math logic
    if bool(invert):
        steering_percent = 100.0 - steering_percent

    # Clamp one more time in case rounding pushes us out of bounds
    steering_percent = max(0.0, min(100.0, steering_percent))

    # Return an integer command because serial protocols are usually integer-based
    return int(round(steering_percent))