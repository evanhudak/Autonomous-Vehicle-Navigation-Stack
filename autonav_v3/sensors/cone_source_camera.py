# Import libraries
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# Cone data type
from sensors.cone_source import ConeBox

# Ensure SensorIntegration is importable without manual PYTHONPATH changes
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))                 # ...\autonav_v2\sensors
AUTONAV_DIR = os.path.normpath(os.path.join(THIS_FILE_DIR, ".."))         # ...\autonav_v2
PARENT_DIR = os.path.normpath(os.path.join(AUTONAV_DIR, ".."))            # ...\Test_current
SENSOR_INTEGRATION_DIR = os.path.normpath(os.path.join(PARENT_DIR, "SensorIntegration"))

if SENSOR_INTEGRATION_DIR not in sys.path and os.path.isdir(SENSOR_INTEGRATION_DIR):
    sys.path.insert(0, SENSOR_INTEGRATION_DIR)

# YOLO wrapper (camera + model inference)
from yolo import YOLODetector

@dataclass
class CameraConeConfig:
    # Camera geometry for simple ground-plane projection
    camera_height_m: float = 0.8
    camera_pitch_deg: float = 0.0
    horizontal_fov_deg: float = 70.0
    vertical_fov_deg: float = 55.0

    # Placeholder cone dimensions so downstream code has consistent fields
    assumed_cone_w_m: float = 0.15
    assumed_cone_d_m: float = 0.15
    assumed_cone_h_m: float = 0.0

    # Reject detections that project unrealistically close/far
    min_forward_m: float = 0.25
    max_forward_m: float = 30.0


class CameraConeSource:
    # Camera-only cone source for push testing
    def __init__(
        self,
        model_path: Optional[str] = None,
        rgb: bool = True,
        config: Optional[CameraConeConfig] = None,
    ) -> None:
        # Store config so projection behavior can be tuned from config.yaml
        self.cfg = config or CameraConeConfig()

        # Initialize YOLO detector (opens the camera internally)
        self.detector = YOLODetector(model_path=model_path, rgb=rgb)

        # Track frame counters and timestamps for debug / logging
        self._frame_id = 0
        self._last_frame_time_s = 0.0

    def close(self) -> None:
        # Clean up camera resources so the device is released between runs
        self.detector.close()

    def _image_to_ground(
        self,
        px: float,
        py: float,
        img_w: int,
        img_h: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        # Convert an image pixel into an approximate ground-plane (forward, right) point
        if img_w <= 0 or img_h <= 0:
            return None, None

        h_fov = math.radians(float(self.cfg.horizontal_fov_deg))
        v_fov = math.radians(float(self.cfg.vertical_fov_deg))
        pitch = math.radians(float(self.cfg.camera_pitch_deg))

        # Normalize pixel coordinates to [-1, 1]
        nx = (float(px) / float(img_w)) * 2.0 - 1.0
        ny = 1.0 - (float(py) / float(img_h)) * 2.0

        # Convert normalized coordinates into ray angles
        theta_x = nx * (h_fov / 2.0)
        theta_y = ny * (v_fov / 2.0)

        # Combine camera pitch with vertical ray angle
        total_vertical_angle = pitch + theta_y

        # If the ray never intersects the ground in front of the camera, reject it
        if total_vertical_angle <= 1e-6:
            return None, None

        # Project ray onto ground plane assuming a flat surface
        forward_m = float(self.cfg.camera_height_m) / math.tan(total_vertical_angle)
        right_m = forward_m * math.tan(theta_x)

        # Reject projections that are outside our useful range
        if not (float(self.cfg.min_forward_m) <= forward_m <= float(self.cfg.max_forward_m)):
            return None, None

        # Reject NaNs/infs defensively
        if not math.isfinite(forward_m) or not math.isfinite(right_m):
            return None, None

        return forward_m, right_m

    def get_frame(self) -> Tuple[List[ConeBox], float, str]:
        # Run one camera + YOLO inference step and return cone boxes with metadata
        frame_time_s = time.time()

        try:
            frame, _annotated, results = self.detector.detect()
        except Exception:
            # If YOLO/camera fails, return empty cones but keep metadata stable
            self._frame_id += 1
            self._last_frame_time_s = frame_time_s
            return [], frame_time_s, f"cam_{self._frame_id}"

        # Validate frame shape before using it
        img_h = getattr(frame, "shape", [0, 0])[0] if frame is not None else 0
        img_w = getattr(frame, "shape", [0, 0])[1] if frame is not None else 0
        if not isinstance(img_h, int) or not isinstance(img_w, int):
            img_h, img_w = 0, 0

        cone_boxes: List[ConeBox] = []

        # Defensive: results could be empty or not indexable in edge cases
        if not results or len(results) < 1:
            self._frame_id += 1
            self._last_frame_time_s = frame_time_s
            return [], frame_time_s, f"cam_{self._frame_id}"

        r0 = results[0]
        boxes_obj = getattr(r0, "boxes", None)
        if boxes_obj is None:
            self._frame_id += 1
            self._last_frame_time_s = frame_time_s
            return [], frame_time_s, f"cam_{self._frame_id}"

        # Walk each detection box from YOLO
        for box in boxes_obj:
            # Class id extraction can differ depending on ultralytics version
            try:
                cls_raw = box.cls[0]
                cls_id = int(cls_raw)
            except Exception:
                continue

            # Resolve label name safely
            try:
                names = getattr(self.detector.model, "names", {})
                label = str(names.get(cls_id, ""))
            except Exception:
                label = ""

            if label.lower() != "cone":
                continue

            # Extract the bounding box corners (x1,y1,x2,y2)
            try:
                x1, y1, x2, y2 = box.xyxy[0]
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
            except Exception:
                continue

            # Need a valid image size for projection math
            if img_w <= 0 or img_h <= 0:
                continue

            # Use bottom-center as a better ground contact point than bbox center
            cx_px = (x1 + x2) / 2.0
            cy_px = y2

            forward_m, right_m = self._image_to_ground(cx_px, cy_px, img_w, img_h)
            if forward_m is None or right_m is None:
                continue

            # Package into ConeBox: (xc, yc, zc, w, d, h)
            # NOTE: we treat xc as right_m and yc as forward_m to match autonav_v2 downstream logic
            cone_boxes.append(
                (
                    float(right_m),
                    float(forward_m),
                    0.0,
                    float(self.cfg.assumed_cone_w_m),
                    float(self.cfg.assumed_cone_d_m),
                    float(self.cfg.assumed_cone_h_m),
                )
            )

        self._frame_id += 1
        self._last_frame_time_s = frame_time_s
        return cone_boxes, frame_time_s, f"cam_{self._frame_id}"

    def get_cones(self) -> List[ConeBox]:
        # Compatibility helper for code paths that only expect a cone list
        cones, _t, _id = self.get_frame()
        return cones