# import functions form typing
from typing import List, Protocol, Tuple

# Cone detection tuple: (xc, yc, zc, w, d, h)
#   xc, yc, zc: cone center position in the sensor frame (meters)
#   w, d, h:    bounding box dimensions (meters), if available
ConeBox = Tuple[float, float, float, float, float, float]


class ConeSource(Protocol):
    """Interface for any component that provides cone detections to the navigation stack."""

    def get_cones(self) -> List[ConeBox]:
        """Return the current list of detected cones as ConeBox tuples."""
        ...