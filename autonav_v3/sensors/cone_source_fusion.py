# Import libraries
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Cone data type
from sensors.cone_source import ConeBox


# Try to make SensorIntegration importable without manual PYTHONPATH edits
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
SENSOR_INTEGRATION_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "SensorIntegration"))

if os.path.isdir(SENSOR_INTEGRATION_DIR) and SENSOR_INTEGRATION_DIR not in sys.path:
    sys.path.insert(0, SENSOR_INTEGRATION_DIR)

# Sensor fusion pipeline (camera + LiDAR)
from Sensors_Main import Sensors  # type: ignore


@dataclass
class FusionConfig:
    # LD14 port on Pi:
    #   GPIO UART typically uses "/dev/serial0"
    lidar_port: str = "/dev/serial0"

    # Only output obstacles tagged as cones by fusion matching
    cones_only: bool = True

    # Optional filters to keep junk out of nav
    min_range_m: float = 0.20
    max_range_m: float = 30.0

    # If you want to cap how many cones go into nav/log
    max_cones: int = 50


class FusionConeSource:
    # Fusion cone source for push testing (camera labels + LiDAR geometry)
    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.cfg = config or FusionConfig()

        # Initialize the fusion pipeline (opens LiDAR serial + camera/YOLO internally)
        self.sensors = Sensors(self.cfg.lidar_port)

        # Track frame counters and timestamps for debug / logging
        self._frame_id = 0
        self._last_frame_time_s = 0.0

    def close(self) -> None:
        # Clean up subcomponents so the camera and serial port are released between runs.
        # Sensors_Main doesn't define a close() method, so we safely close parts if present.

        # Close YOLO camera if present
        try:
            yolo = getattr(self.sensors, "yolo", None)
            if yolo is not None and hasattr(yolo, "close"):
                yolo.close()
        except Exception:
            pass

        # Close LiDAR serial if possible
        try:
            lidar = getattr(self.sensors, "lidar", None)
            if lidar is not None and hasattr(lidar, "close"):
                lidar.close()
                return
        except Exception:
            pass

        # SensorIntegration LiDAR path: lidar.driver.ser exists
        try:
            lidar = getattr(self.sensors, "lidar", None)
            driver = getattr(lidar, "driver", None)
            ser = getattr(driver, "ser", None)
            if ser is not None:
                ser.close()
        except Exception:
            pass

    @staticmethod
    def _bbox_to_conebox(box: Dict[str, Any]) -> Optional[ConeBox]:
        # Convert a fused LiDAR bounding box into a ConeBox tuple.
        # Return None if required keys are missing or non-numeric.
        required = ("min_x", "max_x", "min_y", "max_y")
        for key in required:
            if key not in box:
                return None

        try:
            min_x = float(box["min_x"])
            max_x = float(box["max_x"])
            min_y = float(box["min_y"])
            max_y = float(box["max_y"])
        except Exception:
            return None

        # Compute box center (meters)
        xc = 0.5 * (min_x + max_x)
        yc = 0.5 * (min_y + max_y)

        # Compute box extents (meters)
        w = max_x - min_x
        d = max_y - min_y

        # Package into ConeBox: (xc, yc, zc, w, d, h)
        return (xc, yc, 0.0, w, d, 0.0)

    def get_frame(self) -> Tuple[List[ConeBox], float, str]:
        # Run one fusion step (YOLO + LiDAR) and return cone boxes with metadata.
        # Sensors_Main returns (timestamp, obstacles).

        try:
            timestamp, obstacles = self.sensors.GetObstacles(useCamera=True)
        except Exception:
            # If the fusion pipeline throws, treat as "no cones" so main stays safe
            timestamp = None
            obstacles = []

        # Use sensor timestamp if provided; otherwise fall back to wall time
        try:
            frame_time_s = float(timestamp) if timestamp is not None else time.time()
        except Exception:
            frame_time_s = time.time()

        if not isinstance(obstacles, list):
            obstacles = []

        cones: List[ConeBox] = []

        for raw in obstacles:
            if not isinstance(raw, dict):
                continue

            # Optionally filter to cones only (as labeled by camera matching)
            is_cone = bool(raw.get("is_cone", False))
            if self.cfg.cones_only and not is_cone:
                continue

            conebox = self._bbox_to_conebox(raw)
            if conebox is None:
                continue

            # Range filter on center distance to reduce junk
            xc, yc = conebox[0], conebox[1]
            rng = (xc * xc + yc * yc) ** 0.5
            if rng < float(self.cfg.min_range_m) or rng > float(self.cfg.max_range_m):
                continue

            cones.append(conebox)

            # Cap cone output so a bad frame cannot explode logs or computations
            if len(cones) >= int(self.cfg.max_cones):
                break

        self._frame_id += 1
        self._last_frame_time_s = frame_time_s
        return cones, frame_time_s, f"fusion_{self._frame_id}"

    def get_cones(self) -> List[ConeBox]:
        # Compatibility helper for code paths that only expect a cone list
        cones, _t, _id = self.get_frame()
        return cones