# import function from dataclass
from dataclasses import dataclass

# import function from typing
from typing import Tuple

@dataclass
class FrameConfig:

    """
    Configuration for mapping sensor-frame coordinates into the vehicle frame.

    forward_axis:
        Axis in the sensor frame that represents forward motion ("x" or "y").

    forward_sign:
        Sign applied to the forward axis (+1 or -1) to correct orientation.

    right_sign:
        Sign applied to the lateral axis (+1 or -1) so that +right is vehicle right.
    """
    forward_axis: str
    forward_sign: int
    right_sign: int

def to_vehicle_frame(xc: float,
                     yc: float,
                     frame_cfg: FrameConfig) -> Tuple[float, float]:

    """
    Convert sensor-frame cone coordinates into vehicle-frame coordinates.

    Args:
        xc: Cone x-position in the sensor frame (meters).
        yc: Cone y-position in the sensor frame (meters).
        frame_cfg: Mapping configuration describing axis and sign conventions.

    Returns:
        (right_m, fwd_m):
            right_m > 0 is to the vehicle's right
            fwd_m > 0 is forward of the vehicle
    """

    # Validate forward axis configuration early to catch misconfiguration.
    if frame_cfg.forward_axis not in ("x", "y"):
        raise ValueError("forward_axis must be 'x' or 'y'")

    # Map sensor coordinates into forward/right based on configured axis.
    if frame_cfg.forward_axis == "y":
        forward_m = frame_cfg.forward_sign * yc
        right_m = frame_cfg.right_sign * xc
    else:
        forward_m = frame_cfg.forward_sign * xc
        right_m = frame_cfg.right_sign * yc

    return right_m, forward_m