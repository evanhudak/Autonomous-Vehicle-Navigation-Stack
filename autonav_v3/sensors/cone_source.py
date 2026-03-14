# Import libraries
from typing import List, Protocol, Tuple, Optional


# A cone detection is represented as:
# (xc, yc, zc, w, d, h)
#
# xc, yc, zc are the cone center position in the sensor frame (meters)
# w, d, h are the bounding box dimensions (meters), if available
ConeBox = Tuple[float, float, float, float, float, float]


class ConeSource(Protocol):
    # Common interface for anything that provides cones to navigation:
    # file replay, LiDAR pipelines, camera pipelines, and fused pipelines.

    def get_cones(self) -> List[ConeBox]:
        # Return the current list of detected cones as ConeBox tuples
        ...

    # Optional richer interface some sources provide (camera/lidar/fusion)
    def get_frame(self) -> Tuple[List[ConeBox], float, Optional[str]]:
        # Return (cones, frame_time_s, frame_id)
        ...