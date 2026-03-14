import json
import math
import os
from typing import Any, Dict, List, Tuple


def steer_pct_from_deg(steer_deg: float, max_wheel_deg: float = 45.0, invert: bool = False) -> int:
    steer_deg = max(-max_wheel_deg, min(max_wheel_deg, steer_deg))
    pct = int(round(50.0 + (steer_deg / max_wheel_deg) * 50.0))
    pct = max(0, min(100, pct))
    if invert:
        pct = 100 - pct
    return pct


def rot_world_to_vehicle(dx: float, dy: float, psi: float) -> Tuple[float, float]:
    """
    World frame: X right, Y up
    Vehicle frame: right, forward
    psi=0 means vehicle forward aligns with +Y
    """
    right = dx * math.cos(psi) - dy * math.sin(psi)
    fwd = dx * math.sin(psi) + dy * math.cos(psi)
    return right, fwd


def build_centerline_points(step_m: float = 0.25) -> List[Tuple[float, float]]:
    """
    Build a longer, more interesting centerline in world coordinates (X,Y).
    Shape: straight -> left arc -> straight -> right arc -> straight
    """
    pts: List[Tuple[float, float]] = []

    # Segment A: straight up
    y = 0.0
    while y <= 15.0:
        pts.append((0.0, y))
        y += step_m

    # Segment B: left arc (quarter-ish turn)
    # Arc center at (-6, 15), radius 6, sweep from 0 to +60 degrees
    cx, cy, r = -6.0, 15.0, 6.0
    # angle 0 means point at (cx+r, cy) = (0, 15)
    for deg in range(0, 61, 2):
        th = math.radians(deg)
        x = cx + r * math.cos(th)
        y = cy + r * math.sin(th)
        pts.append((x, y))

    # Segment C: straight along new heading (roughly left/up)
    # Continue from last point in the arc in direction of tangent
    x0, y0 = pts[-1]
    # approximate heading from last two points
    x1, y1 = pts[-2]
    hdg = math.atan2(y0 - y1, x0 - x1)
    length = 14.0
    dist = step_m
    while dist <= length:
        pts.append((x0 + dist * math.cos(hdg), y0 + dist * math.sin(hdg)))
        dist += step_m

    # Segment D: right arc to bend back (S curve)
    # Build an arc that turns right ~70 degrees
    # Pick center offset to the "right" of current heading
    x_start, y_start = pts[-1]
    # define arc radius
    r2 = 7.0
    # Right-normal from heading
    nx = math.cos(hdg - math.pi / 2.0)
    ny = math.sin(hdg - math.pi / 2.0)
    c2x = x_start + nx * r2
    c2y = y_start + ny * r2

    # Determine start angle around circle
    start_ang = math.atan2(y_start - c2y, x_start - c2x)
    # sweep negative for right turn
    for i in range(1, 71):
        th = start_ang - math.radians(i)
        pts.append((c2x + r2 * math.cos(th), c2y + r2 * math.sin(th)))

    # Segment E: straight again
    x0, y0 = pts[-1]
    x1, y1 = pts[-2]
    hdg2 = math.atan2(y0 - y1, x0 - x1)
    length2 = 18.0
    dist = step_m
    while dist <= length2:
        pts.append((x0 + dist * math.cos(hdg2), y0 + dist * math.sin(hdg2)))
        dist += step_m

    return pts


def build_cone_boundaries(
    centerline: List[Tuple[float, float]],
    corridor_half_width_m: float = 1.0,
    cone_spacing_m: float = 1.0,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Generate left/right cone positions in world coordinates from a centerline.
    """
    left: List[Tuple[float, float]] = []
    right: List[Tuple[float, float]] = []

    if len(centerline) < 3:
        return left, right

    # Walk along centerline and drop cones every cone_spacing_m
    accum = 0.0
    last = centerline[0]
    last_cone_at = 0.0
    total_s = 0.0

    # precompute segment distances and headings
    for i in range(1, len(centerline)):
        x0, y0 = centerline[i - 1]
        x1, y1 = centerline[i]
        dx = x1 - x0
        dy = y1 - y0
        seg = math.hypot(dx, dy)
        if seg < 1e-9:
            continue

        hdg = math.atan2(dy, dx)
        # unit normal pointing left of heading (dx,dy)
        lx = -math.sin(hdg)
        ly = math.cos(hdg)

        # step along this segment and place cones at spacing
        s = 0.0
        while s < seg:
            if total_s - last_cone_at >= cone_spacing_m:
                px = x0 + (s / seg) * dx
                py = y0 + (s / seg) * dy

                left.append((px + lx * corridor_half_width_m, py + ly * corridor_half_width_m))
                right.append((px - lx * corridor_half_width_m, py - ly * corridor_half_width_m))
                last_cone_at = total_s

            ds = 0.1
            s += ds
            total_s += ds

    return left, right


def interp_path_pose(centerline: List[Tuple[float, float]], s_m: float) -> Tuple[float, float, float]:
    """
    Return (x,y,psi) on the path at distance s_m from start.
    psi is heading in world: psi=0 means facing +Y in your sim convention,
    but for rotation we just use the world heading directly.
    """
    if s_m <= 0.0:
        x0, y0 = centerline[0]
        x1, y1 = centerline[1]
        hdg = math.atan2(y1 - y0, x1 - x0)
        # Convert world heading (atan2(dy,dx)) to your psi where 0 faces +Y:
        # world heading 90deg (pi/2) means +Y, so psi = (pi/2 - hdg)
        psi = (math.pi / 2.0) - hdg
        return x0, y0, psi

    remaining = s_m
    for i in range(1, len(centerline)):
        x0, y0 = centerline[i - 1]
        x1, y1 = centerline[i]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-9:
            continue
        if remaining <= seg:
            t = remaining / seg
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            hdg = math.atan2(y1 - y0, x1 - x0)
            psi = (math.pi / 2.0) - hdg
            return x, y, psi
        remaining -= seg

    # End of path
    x0, y0 = centerline[-2]
    x1, y1 = centerline[-1]
    hdg = math.atan2(y1 - y0, x1 - x0)
    psi = (math.pi / 2.0) - hdg
    return centerline[-1][0], centerline[-1][1], psi


def write_testbench(
    path: str,
    seconds: float = 90.0,
    hz: float = 20.0,
    walk_speed_mps: float = 1.4,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    dt = 1.0 / hz
    t = 0.0
    frame_id = 0

    # Build a curvy track in WORLD coordinates
    centerline = build_centerline_points(step_m=0.25)
    left_world, right_world = build_cone_boundaries(
        centerline=centerline,
        corridor_half_width_m=1.0,
        cone_spacing_m=1.0,
    )

    # Detection window (in vehicle frame)
    min_fwd = 0.2
    max_fwd = 8.0
    max_right = 4.0

    with open(path, "w", encoding="utf-8") as f:
        while t <= seconds:
            frame_id += 1

            # Move along the path by distance = v*t
            s = walk_speed_mps * t
            x, y, psi = interp_path_pose(centerline, s_m=s)

            # Convert world cones into vehicle detections for this frame
            cones_raw: List[Dict[str, Any]] = []
            for (cx, cy) in left_world + right_world:
                dx = cx - x
                dy = cy - y
                right_m, fwd_m = rot_world_to_vehicle(dx, dy, psi)

                # Only include cones that would be "visible" in front of the rig
                if fwd_m < min_fwd or fwd_m > max_fwd:
                    continue
                if abs(right_m) > max_right:
                    continue

                cones_raw.append(
                    {"xc": right_m, "yc": fwd_m, "zc": 0.0, "w": 0.15, "d": 0.15, "h": 0.0}
                )

            # Target: centerline lookahead in vehicle frame
            lookahead = 3.0
            x2, y2, _psi2 = interp_path_pose(centerline, s_m=s + lookahead)
            dx_t = x2 - x
            dy_t = y2 - y
            target_right, target_fwd = rot_world_to_vehicle(dx_t, dy_t, psi)

            # Add a small wiggle for visual interest (optional)
            target_right += 0.20 * math.sin(0.35 * t)

            # Steering "tries" to reduce right error (simple proportional demo)
            steer_deg = clamp(20.0 * (target_right / 1.0), -35.0, 35.0)
            steer_pct = steer_pct_from_deg(steer_deg)

            # Throttle/brake "would" (for HUD + vectors)
            throttle_would = 25
            brake_would = 0
            # Add occasional slow-down zones so you see red accel vectors sometimes
            if (t % 18.0) > 15.0:
                throttle_would = 0
                brake_would = 40

            record: Dict[str, Any] = {
                "t_wall_s": 1700000000.0 + t,
                "dt_s": dt,
                "loop_hz_meas": hz,
                "mode": "push_test",
                "dry_run": True,
                "sensor": {
                    "source": "testbench_curvy",
                    "frame_id": f"tb_{frame_id}",
                    "age_ms": 20,
                    "ok": True,
                },
                "cones_raw": cones_raw,
                "nav": {
                    "target_point_m": {"right": float(target_right), "fwd": float(target_fwd)},
                    "steer_deg": float(steer_deg),
                    "steer_pct": int(steer_pct),
                },
                "cmd": {
                    "sent": {"steer_pct": int(steer_pct), "throttle_pct": 0, "brake_pct": 0},
                    "would": {"throttle_pct": int(throttle_would), "brake_pct": int(brake_would)},
                },
            }

            f.write(json.dumps(record) + "\n")
            t += dt


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


if __name__ == "__main__":
    out_path = "logs/testbench.jsonl"
    write_testbench(out_path, seconds=90.0, hz=20.0, walk_speed_mps=1.4)
    print(f"Wrote {out_path}")