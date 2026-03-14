# Import libraries
import json
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

# Try to import pyserial only if we need serial_json transport
try:
    import serial
except Exception:
    serial = None  # type: ignore

# Import local modules
from .cone_source import ConeBox


def now_wall_time_s() -> float:
    # Use wall time so timestamps match the rest of the system logs
    return time.time()


def parse_cones_from_payload(payload: Any) -> List[ConeBox]:
    # Accept a few payload formats so different sensor pipelines can plug in easily
    cones_object = payload

    # If payload is a frame dict, look for a "cones" field
    if isinstance(payload, dict) and "cones" in payload:
        cones_object = payload["cones"]

    # If cones are not a list, treat it as invalid
    if not isinstance(cones_object, list):
        return []

    parsed_cones: List[ConeBox] = []

    # Convert each cone entry into a ConeBox tuple (xc, yc, zc, w, d, h)
    for cone_entry in cones_object:
        if isinstance(cone_entry, dict):
            # Support both x/y and xc/yc naming so we do not fight key conventions
            xc = float(cone_entry.get("xc", cone_entry.get("x", 0.0)))
            yc = float(cone_entry.get("yc", cone_entry.get("y", 0.0)))
            zc = float(cone_entry.get("zc", cone_entry.get("z", 0.0)))

            w = float(cone_entry.get("w", 0.2))
            d = float(cone_entry.get("d", 0.2))
            h = float(cone_entry.get("h", 0.3))

            parsed_cones.append((xc, yc, zc, w, d, h))
            continue

        # Support list/tuple formats like [x, y] or [x, y, z, w, d, h]
        if isinstance(cone_entry, (list, tuple)) and len(cone_entry) >= 2:
            xc = float(cone_entry[0])
            yc = float(cone_entry[1])

            zc = float(cone_entry[2]) if len(cone_entry) > 2 else 0.0
            w = float(cone_entry[3]) if len(cone_entry) > 3 else 0.2
            d = float(cone_entry[4]) if len(cone_entry) > 4 else 0.2
            h = float(cone_entry[5]) if len(cone_entry) > 5 else 0.3

            parsed_cones.append((xc, yc, zc, w, d, h))

    return parsed_cones


class LiveConeSource:

    # Cone source that reads from a real sensor pipeline
    # This is a "bridge" layer so main.py does not care how detections arrive

    # Supported transports:
    # udp_json    = one JSON frame per UDP packet
    # serial_json = one JSON frame per serial line
    # file_tail   = one JSON frame per line in a growing .jsonl file

    def __init__(self, live_config: Dict[str, Any]) -> None:
        # Choose how we receive frames so we can swap transports without code changes
        self.transport = str(live_config.get("transport", "udp_json")).lower().strip()

        # Keep the last known frame so nav always has something to read
        self._latest_cones: List[ConeBox] = []
        self._latest_frame_time_s: Optional[float] = None
        self._latest_frame_id: Optional[int] = None

        # Set up transport-specific resources
        self._udp_socket: Optional[socket.socket] = None
        self._serial_port = None
        self._tail_file_handle = None

        if self.transport == "udp_json":
            # Bind UDP so we can receive one-frame JSON packets
            udp_ip = str(live_config.get("udp_ip", "0.0.0.0"))
            udp_port = int(live_config.get("udp_port", 5005))

            self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_socket.bind((udp_ip, udp_port))

            # Short timeout keeps get_frame non-blocking
            self._udp_socket.settimeout(0.001)

        elif self.transport == "serial_json":
            # Use serial input when a device streams JSON lines directly
            if serial is None:
                raise RuntimeError(
                    "pyserial is not installed, but transport=serial_json was selected"
                )

            serial_port_name = str(live_config.get("port", "/dev/ttyUSB0"))
            serial_baud_rate = int(live_config.get("baud", 115200))
            serial_timeout_s = float(live_config.get("timeout_s", 0.001))

            self._serial_port = serial.Serial(
                serial_port_name,
                baudrate=serial_baud_rate,
                timeout=serial_timeout_s,
            )

        elif self.transport == "file_tail":
            # Tail a JSONL file so another process can append frames
            jsonl_path = str(live_config.get("path", "logs/live_cones.jsonl"))
            self._tail_file_handle = open(jsonl_path, "r", encoding="utf-8")

            # Seek to end so we only read new frames as they appear
            self._tail_file_handle.seek(0, 2)

        else:
            raise ValueError(f"Unknown live transport: {self.transport}")

    def close(self) -> None:
        # Close whatever transport we opened so ports and files are released cleanly
        if self._udp_socket is not None:
            try:
                self._udp_socket.close()
            except Exception:
                pass

        if self._serial_port is not None:
            try:
                self._serial_port.close()
            except Exception:
                pass

        if self._tail_file_handle is not None:
            try:
                self._tail_file_handle.close()
            except Exception:
                pass

    def _consume_payload(self, payload: Any) -> None:
        # Pull metadata so main.py can enforce freshness and correlate frames
        frame_time_s: Optional[float] = None
        frame_id: Optional[int] = None

        if isinstance(payload, dict):
            # If timestamp exists, use it so the producer controls time
            if "t" in payload:
                try:
                    frame_time_s = float(payload["t"])
                except Exception:
                    frame_time_s = None

            # If a frame counter exists, keep it for debugging and log correlation
            if "frame_id" in payload:
                try:
                    frame_id = int(payload["frame_id"])
                except Exception:
                    frame_id = None

        # Parse cones into our standard ConeBox tuple format
        parsed_cones = parse_cones_from_payload(payload)

        # Always update latest state so main.py sees the newest valid frame
        self._latest_cones = parsed_cones
        self._latest_frame_time_s = (
            frame_time_s if frame_time_s is not None else now_wall_time_s()
        )
        self._latest_frame_id = frame_id

    def _poll_udp_once(self) -> None:
        # Try to read one UDP packet without blocking
        if self._udp_socket is None:
            return

        try:
            data, _addr = self._udp_socket.recvfrom(65535)
        except (socket.timeout, BlockingIOError):
            return
        except Exception:
            return

        try:
            payload = json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            return

        self._consume_payload(payload)

    def _poll_serial_once(self) -> None:
        # Try to read one serial line without blocking
        if self._serial_port is None:
            return

        try:
            line_bytes = self._serial_port.readline()
            if not line_bytes:
                return
        except Exception:
            return

        try:
            payload = json.loads(line_bytes.decode("utf-8", errors="ignore"))
        except Exception:
            return

        self._consume_payload(payload)

    def _poll_file_tail_once(self) -> None:
        # Try to read one new JSONL line without blocking
        if self._tail_file_handle is None:
            return

        try:
            line_text = self._tail_file_handle.readline()
            if not line_text:
                return
        except Exception:
            return

        try:
            payload = json.loads(line_text)
        except Exception:
            return

        self._consume_payload(payload)

    def get_frame(self) -> Tuple[List[ConeBox], Optional[float], Optional[int]]:
        # Poll once so each main loop tick has a chance to pull the newest sensor frame
        if self.transport == "udp_json":
            self._poll_udp_once()

        elif self.transport == "serial_json":
            self._poll_serial_once()

        elif self.transport == "file_tail":
            self._poll_file_tail_once()

        # Return whatever the latest valid frame is
        return (
            self._latest_cones,
            self._latest_frame_time_s,
            self._latest_frame_id,
        )

    def get_cones(self) -> List[ConeBox]:
        # Compatibility wrapper for code that only asks for cones
        cones, _frame_time_s, _frame_id = self.get_frame()
        return cones