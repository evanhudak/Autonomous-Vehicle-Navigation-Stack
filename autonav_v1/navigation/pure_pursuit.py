# import library
import math

def steer_deg_from_target(right_m: float,
                          fwd_m: float,
                          lookahead_m: float) -> float:

    """
    Compute a steering angle (degrees) using a pure-pursuit-like formulation.

    Args:
        right_m: Lateral offset of the target point (+right is vehicle right).
        fwd_m: Forward distance to the target point (+forward is ahead).
        lookahead_m: Desired lookahead distance for steering stability.

    Returns:
        Steering angle in degrees. Sign convention matches right_m.
    """

    # Ensure lookahead distance is not too small to avoid numerical instability.
    lookahead_dist_m = max(0.25, float(lookahead_m))

    # Compute distance to target point; clamp to avoid division by zero.
    target_dist_m = math.hypot(right_m, fwd_m)
    target_dist_m = max(1e-3, target_dist_m)

    # Approximate curvature for a pure pursuit controller in the vehicle frame.
    # kappa ≈ 2 * lateral_offset / distance^2
    curvature = 2.0 * right_m / (target_dist_m * target_dist_m)

    # Convert curvature into a steering angle proxy.
    # Without an explicit wheelbase, atan(kappa * lookahead) is a stable substitute.
    steering_rad = math.atan(curvature * lookahead_dist_m)

    return math.degrees(steering_rad)

def steer_percent_from_deg(steering_deg: float,
                           max_wheel_deg: float,
                           invert: bool) -> int:

    """
    Map a steering angle in degrees to a normalized percentage command [0, 100].

    Args:
        steering_deg: Desired steering angle in degrees.
        max_wheel_deg: Maximum steering angle magnitude supported by hardware.
        invert: If True, invert left/right mapping (useful for wiring differences).

    Returns:
        Integer steering command percentage:
            0   -> full left
            50  -> centered
            100 -> full right
    """

    # Clamp maximum steering angle to a reasonable minimum.
    max_allowed_deg = max(1e-3, float(max_wheel_deg))

    # Clamp steering request to physical steering limits.
    clamped_deg = max(-max_allowed_deg,
                      min(max_allowed_deg, float(steering_deg)))

    # Linearly map steering range [-max, +max] to [0, 100].
    steering_percent = 50.0 + (clamped_deg / max_allowed_deg) * 50.0

    # Optionally invert steering direction if required by hardware orientation.
    if invert:
        steering_percent = 100.0 - steering_percent

    # Final clamp and integer rounding for serial transmission.
    steering_percent = max(0.0, min(100.0, steering_percent))
    return int(round(steering_percent))