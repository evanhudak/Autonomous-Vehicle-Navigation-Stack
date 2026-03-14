# import libraries
import json
import time

# import functions from typing
from typing import Any, Dict, List, Union

# import function from cone_source
from .cone_source import ConeBox

JsonList = List[Dict[str, Any]]
JsonDict = Dict[str, Any]
JsonData = Union[JsonList, JsonDict]

class FileConeSource:

    """
    Load cone detections from a JSON file.

    Supported file formats:

    A) Static list format (always returns the same cones):
        [
          {"xc":..., "yc":..., "zc":..., "w":..., "d":..., "h":...},
          ...
        ]

    B) Multi-frame format (cycles through frames at a fixed rate):
        {
          "fps": 2,
          "frames": [
             [ {cone}, {cone}, ... ],
             [ {cone}, {cone}, ... ],
             ...
          ]
        }
    """

    def __init__(self, path: str):
        self.path = path

        # Track frame index and last-advance time for multi-frame playback.
        self.last_frame_advance_s = time.time()
        self.frame_index = 0

    def _load_json(self) -> JsonData:
        # Read and parse JSON from disk each call (simple and reliable for v1 demos).
        with open(self.path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    @staticmethod
    def _to_cone_boxes(items: JsonList) -> List[ConeBox]:
        # Convert list of dicts into typed cone tuples: (xc, yc, zc, w, d, h).
        cone_boxes: List[ConeBox] = []

        for item in items:
            cone_boxes.append((float(item["xc"]),
                               float(item["yc"]),
                               float(item.get("zc", 0.0)),
                               float(item.get("w", 0.0)),
                               float(item.get("d", 0.0)),
                               float(item.get("h", 0.0)),))

        return cone_boxes

    def _get_cones_static(self, cone_list: JsonList) -> List[ConeBox]:
        # Static format: return all cones as-is.
        return self._to_cone_boxes(cone_list)

    def _get_cones_multiframe(self, cone_dict: JsonDict) -> List[ConeBox]:
        # Multi-frame format: advance frames based on fps and return current frame's cones.
        frames = cone_dict.get("frames", [])
        if not isinstance(frames, list) or not frames:
            return []

        fps = float(cone_dict.get("fps", 2.0))
        fps = max(0.1, fps)
        frame_period_s = 1.0 / fps

        now_s = time.time()
        if (now_s - self.last_frame_advance_s) >= frame_period_s:
            self.frame_index = (self.frame_index + 1) % len(frames)
            self.last_frame_advance_s = now_s

        frame_items = frames[self.frame_index]
        if not isinstance(frame_items, list):
            return []

        return self._to_cone_boxes(frame_items)

    def get_cones(self) -> List[ConeBox]:
        # Load JSON and dispatch based on detected format.
        json_data = self._load_json()

        if isinstance(json_data, list):
            return self._get_cones_static(json_data)

        if isinstance(json_data, dict) and "frames" in json_data:
            return self._get_cones_multiframe(json_data)

        # Unknown format or invalid JSON shape.
        return []