# Import libraries
import json
import time
from typing import Any, Dict, List, Union

# Import local modules
from .cone_source import ConeBox


JsonList = List[Dict[str, Any]]
JsonDict = Dict[str, Any]
JsonData = Union[JsonList, JsonDict]


class FileConeSource:

    # Provide cone detections from a JSON file so we can test navigation without hardware
    # Supports a static list of cones or a multi-frame replay format with FPS control

    def __init__(self, path: str) -> None:
        # Store path so this source can be swapped just by changing config
        self.path = path

        # Track which frame we are on so we can replay multi-frame files
        self.last_frame_advance_s = time.time()
        self.frame_index = 0

    def _load_json(self) -> JsonData:
        # Read and parse JSON each call so edits to the file take effect immediately
        with open(self.path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    @staticmethod
    def _to_cone_boxes(items: JsonList) -> List[ConeBox]:
        # Convert dict items into ConeBox tuples so the rest of the code sees one format
        cone_boxes: List[ConeBox] = []

        # Convert each JSON entry into (xc, yc, zc, w, d, h)
        for item in items:
            cone_boxes.append(
                (
                    float(item["xc"]),
                    float(item["yc"]),
                    float(item.get("zc", 0.0)),
                    float(item.get("w", 0.0)),
                    float(item.get("d", 0.0)),
                    float(item.get("h", 0.0)),
                )
            )

        return cone_boxes

    def _get_cones_static(self, cone_list: JsonList) -> List[ConeBox]:
        # Static format returns the same cones every time
        return self._to_cone_boxes(cone_list)

    def _get_cones_multiframe(self, cone_dict: JsonDict) -> List[ConeBox]:
        # Multi-frame format cycles through frames based on fps
        frames_object = cone_dict.get("frames", [])

        # If frames are missing or invalid, return empty so nav falls back safely
        if not isinstance(frames_object, list) or not frames_object:
            return []

        # Use fps to decide when to advance to the next frame
        fps = float(cone_dict.get("fps", 2.0))
        fps = max(0.1, fps)
        frame_period_s = 1.0 / fps

        # Advance frame index when enough time has passed
        now_s = time.time()
        if (now_s - self.last_frame_advance_s) >= frame_period_s:
            self.frame_index = (self.frame_index + 1) % len(frames_object)
            self.last_frame_advance_s = now_s

        # Pull the current frame and convert its items into ConeBox tuples
        frame_items = frames_object[self.frame_index]
        if not isinstance(frame_items, list):
            return []

        return self._to_cone_boxes(frame_items)

    def get_cones(self) -> List[ConeBox]:
        # Load the JSON and decide which supported format we are dealing with
        json_data = self._load_json()

        # Static format is just a list of cone dicts
        if isinstance(json_data, list):
            return self._get_cones_static(json_data)

        # Multi-frame format is a dict that contains "frames"
        if isinstance(json_data, dict) and "frames" in json_data:
            return self._get_cones_multiframe(json_data)

        # Unknown format means we return empty so navigation falls back safely
        return []