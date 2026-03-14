# Import libraries
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Any

# Cone data type
from sensors.cone_source import ConeBox

# Ensure SensorIntegration is importable without manual PYTHONPATH changes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
SENSOR_INTEGRATION_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "SensorIntegration"))

if os.path.isdir(SENSOR_INTEGRATION_DIR) and SENSOR_INTEGRATION_DIR not in sys.path:
    sys.path.insert(0, SENSOR_INTEGRATION_DIR)

# LiDAR clustering / fallback detector (LD14 -> clustered bounding boxes)
from LIDAR_Fallback import LidarObjectDetector


@dataclass
class LidarConeConfig:
    # Serial port to the LD14 on the Pi.
    # For GPIO UART, "/dev/serial0" is the usual choice.
    port: str = "/dev/serial0"
    baudrate: int = 230400

    # Cluster tuning (used only if the detector supports it)
    min_cluster_size: int = 5
    cluster_distance_threshold: float = 0.08

    # Filter objects by center distance (meters)
    min_range_m: float = 0.20
    max_range_m: float = 30.0

    # Optional additional filters for “reasonable” boxes
    max_box_width_m: float = 1.5
    max_box_depth_m: float = 1.5


class LidarConeSource:
    # LiDAR-only cone/obstacle source for push testing
    def __init__(self, config: Optional[LidarConeConfig] = None) -> None:
        self.cfg = config or LidarConeConfig()

        # Support both detector APIs:
        #  1) New-style: LidarObjectDetector(port=..., baudrate=..., min_cluster_size=..., cluster_distance_threshold=...)
        #  2) Current SensorIntegration: LidarObjectDetector(port) and detect_objects() yields (timestamp, boxes)
        try:
            self.detector = LidarObjectDetector(
                port=self.cfg.port,
                baudrate=self.cfg.baudrate,
                min_cluster_size=self.cfg.min_cluster_size,
                cluster_distance_threshold=self.cfg.cluster_distance_threshold,
            )
            self._detector_style: str = "kwargs"
        except TypeError:
            self.detector = LidarObjectDetector(self.cfg.port)
            self._detector_style = "port_only"

        self._frame_id = 0
        self._last_frame_time_s = 0.0

    def close(self) -> None:
        # Clean up serial resources so the device is released between runs.
        # Some detector versions do not expose close(), so we try a few safe options.
        try:
            if hasattr(self.detector, "close"):
                self.detector.close()
                return
        except Exception:
            pass

        # SensorIntegration path: detector.driver.ser is a pyserial object
        try:
            driver = getattr(self.detector, "driver", None)
            ser = getattr(driver, "ser", None)
            if ser is not None:
                ser.close()
        except Exception:
            pass

    def _read_boxes_once(self) -> Tuple[float, List[dict]]:
        # Return (timestamp_s, list_of_boxes_dict)

        if self._detector_style == "port_only":
            # detect_objects() is a generator yielding (timestamp, bounding_boxes)
            gen = self.detector.detect_objects()

            try:
                timestamp, boxes = next(gen)
            except StopIteration:
                return time.time(), []
            except Exception:
                return time.time(), []

            return float(timestamp), list(boxes) if boxes is not None else []

        # kwargs style: detector.detect_objects() returns either:
        #   - an iterable of dict boxes
        #   - OR a tuple (timestamp, iterable_of_boxes)
        timestamp = time.time()
        boxes_any = self.detector.detect_objects()
        boxes: List[dict] = []

        # If detector returns (timestamp, boxes), unpack it safely
        if isinstance(boxes_any, tuple) and len(boxes_any) == 2:
            maybe_timestamp, maybe_boxes = boxes_any

            try:
                timestamp = float(maybe_timestamp)
            except Exception:
                timestamp = time.time()

            boxes_any = maybe_boxes

        # Collect only dict boxes
        if isinstance(boxes_any, Iterable):
            for item in boxes_any:
                if isinstance(item, dict):
                    boxes.append(item)

        return timestamp, boxes

    def get_frame(self) -> Tuple[List[ConeBox], float, str]:
        # Read one full LiDAR detection cycle and return cone boxes with metadata
        timestamp_s, boxes = self._read_boxes_once()
        frame_time_s = float(timestamp_s) if timestamp_s is not None else time.time()

        cones: List[ConeBox] = []

        for box in boxes:
            if not isinstance(box, dict):
                continue

            # Extract bounding box edges
            try:
                min_x = float(box["min_x"])
                max_x = float(box["max_x"])
                min_y = float(box["min_y"])
                max_y = float(box["max_y"])
            except Exception:
                continue

            # Compute box center (meters)
            xc = 0.5 * (min_x + max_x)
            yc = 0.5 * (min_y + max_y)

            # Compute box extents (meters)
            w = abs(max_x - min_x)
            d = abs(max_y - min_y)

            # Reject NaNs/Infs defensively
            if not (math.isfinite(xc) and math.isfinite(yc) and math.isfinite(w) and math.isfinite(d)):
                continue

            # Reject objects that are too close or too far away
            rng = math.hypot(xc, yc)
            if rng < float(self.cfg.min_range_m) or rng > float(self.cfg.max_range_m):
                continue

            # Reject boxes that are unrealistically large for our use case
            if w > float(self.cfg.max_box_width_m) or d > float(self.cfg.max_box_depth_m):
                continue

            cones.append((xc, yc, 0.0, w, d, 0.0))

        self._frame_id += 1
        self._last_frame_time_s = frame_time_s
        return cones, frame_time_s, f"lidar_{self._frame_id}"

    def get_cones(self) -> List[ConeBox]:
        cones, _t, _id = self.get_frame()
        return cones