import os

# Hide the pygame support prompt so demo runs look clean
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import pygame

# Alias for the json lines format used by push_test.jsonl
JsonDict = Dict[str, Any]

# -----------------------------
# KART PHYSICAL CONSTANTS
# -----------------------------

# Convert inches -> meters for correct environment scaling
KART_LENGTH_M = 67.3 * 0.0254
KART_WIDTH_M = 44.0 * 0.0254

# Convert mph -> m/s for speed scaling
KART_MAX_SPEED_MPS = 18.0 * 0.44704

# Wheelbase affects how quickly heading changes for a given steering angle
WHEELBASE_M = 49.61 * 0.0254

# Shrink the kart drawing slightly so it looks better in a larger environment
KART_DRAW_SCALE = 0.75

# -----------------------------
# DYNAMICS TUNING
# -----------------------------

# Maximum forward acceleration at 100% throttle (only used in "kart" mode)
MAX_ACCEL_MPS2 = 3.0

# Maximum braking deceleration at 100% brake (only used in "kart" mode)
MAX_BRAKE_MPS2 = 6.0

# Simple linear drag term to keep speed stable and realistic
DRAG_GAIN = 0.15

# -----------------------------
# CONE MAP + FADE SETTINGS
# -----------------------------

# If a new cone appears within this radius of an existing cone, update the existing one
CONE_ASSOCIATE_RADIUS_M = 0.35

# Cone markers fade out over this many seconds after last seen
CONE_FADE_SECONDS = 2.5

# Cone alpha range during fade (helps cones disappear smoothly)
CONE_ALPHA_MIN = 25
CONE_ALPHA_MAX = 255

# How long to keep cones in memory before deleting them (useful for long demos)
CONE_KEEP_SECONDS = 30.0

# -----------------------------
# VIEW SETTINGS
# -----------------------------

# Visible simulation window (meters). Kart stays fixed; environment moves around it.
VIEW_SIZE_M = 8.0

# Grid spacing (meters). This is optional but helpful for scale.
GRID_STEP_M = 1.0

# -----------------------------
# TRACK / ASPHALT RENDERING
# -----------------------------

# Track polygon uses cones out to this forward distance (vehicle frame)
TRACK_LOOKAHEAD_M = 8.0

# Limit cone count per boundary to keep polygon stable
TRACK_MAX_CONES_PER_SIDE = 12

# Require at least this many cones per side to draw track
TRACK_MIN_CONES_PER_SIDE = 2

# Small inflation to make asphalt look nicer even with sparse cones
TRACK_EDGE_INFLATE_M = 0.10


def clamp(value: float,
          low: float,
          high: float,
          ) -> float:
    # Keep a value within [low, high] so math stays stable and safe
    return max(low, min(high, value))


def load_jsonl(log_path: str,
              ) -> List[JsonDict]:
    # Load a JSONL file where each line is one JSON object
    records: List[JsonDict] = []

    with open(log_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue

            # Skip malformed lines so demos don't crash from partial writes
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    return records

def parse_cones_vehicle(record: JsonDict,
                        ) -> List[Tuple[float, float]]:
    # Extract (right_m, fwd_m) cone centers from the log record
    cones: List[Tuple[float, float]] = []

    raw_cones = record.get("cones_raw", [])
    if not isinstance(raw_cones, list):
        return cones

    for cone_entry in raw_cones:
        # Each cone is expected to look like {"xc":..., "yc":...}
        if not isinstance(cone_entry, dict):
            continue
        if "xc" not in cone_entry or "yc" not in cone_entry:
            continue

        # Convert to floats and append, skipping bad values
        try:
            right_m = float(cone_entry["xc"])
            forward_m = float(cone_entry["yc"])
        except Exception:
            continue

        cones.append((right_m, forward_m))

    return cones

def parse_target_vehicle(record: JsonDict,
                         ) -> Optional[Tuple[float, float]]:
    # Extract the target point (right_m, fwd_m) from nav.target_point_m
    nav = record.get("nav", {})
    if not isinstance(nav, dict):
        return None

    target = nav.get("target_point_m", None)
    if target is None or not isinstance(target, dict):
        return None

    try:
        right_m = float(target["right"])
        forward_m = float(target["fwd"])
        return (right_m, forward_m)
    except Exception:
        return None

def parse_controls(record: JsonDict,
                   ) -> Tuple[float, float, float, float]:
    # Read the dt and the "would" commands (throttle/brake) so HUD/vectors match logs

    # dt is logged by main.py and represents how long the control loop step was
    dt_s_raw = record.get("dt_s", 0.05)
    try:
        dt_s = float(dt_s_raw)
    except Exception:
        dt_s = 0.05

    # Clamp dt to avoid crazy values breaking the simulation
    dt_s = clamp(dt_s, 1e-3, 0.2)

    # Steering comes from nav.steer_deg
    nav = record.get("nav", {})
    steering_deg = 0.0

    if isinstance(nav, dict) and nav.get("steer_deg", None) is not None:
        try:
            steering_deg = float(nav["steer_deg"])
        except Exception:
            steering_deg = 0.0

    # Throttle and brake come from cmd.would (push test never actuates, but we log intent)
    cmd = record.get("cmd", {})
    throttle_pct = 0.0
    brake_pct = 0.0

    if isinstance(cmd, dict):
        would_cmd = cmd.get("would", {})
        if isinstance(would_cmd, dict):
            try:
                throttle_pct = float(would_cmd.get("throttle_pct", 0.0))
            except Exception:
                throttle_pct = 0.0

            try:
                brake_pct = float(would_cmd.get("brake_pct", 0.0))
            except Exception:
                brake_pct = 0.0

    # Clamp to 0..100 so vectors and dynamics do not explode
    throttle_pct = clamp(throttle_pct, 0.0, 100.0)
    brake_pct = clamp(brake_pct, 0.0, 100.0)

    return dt_s, steering_deg, throttle_pct, brake_pct

def vehicle_to_world(right_m: float,
                     forward_m: float,
                     world_x_m: float,
                     world_y_m: float,
                     heading_rad: float,
                     ) -> Tuple[float, float]:
    # Convert a (right, forward) vehicle-frame point into world coordinates
    world_x = world_x_m + right_m * math.cos(heading_rad) + forward_m * math.sin(heading_rad)
    world_y = world_y_m - right_m * math.sin(heading_rad) + forward_m * math.cos(heading_rad)
    return world_x, world_y

def world_to_vehicle(world_x: float,
                     world_y: float,
                     world_x_m: float,
                     world_y_m: float,
                     heading_rad: float,
                     ) -> Tuple[float, float]:
    # Convert a world point into the current vehicle frame (right, forward)
    dx = world_x - world_x_m
    dy = world_y - world_y_m

    right_m = dx * math.cos(heading_rad) - dy * math.sin(heading_rad)
    forward_m = dx * math.sin(heading_rad) + dy * math.cos(heading_rad)

    return right_m, forward_m

def associate_or_add_cone(cone_map_world: List[Dict[str, float]],
                          world_x: float,
                          world_y: float,
                          sim_time_s: float,
                          ) -> None:
    # Update an existing cone if it is close, otherwise add a new cone to the map
    for cone in cone_map_world:
        existing_x = cone["x"]
        existing_y = cone["y"]

        # If the detection is near an existing cone, update it instead of adding a new one
        dist2 = (existing_x - world_x) ** 2 + (existing_y - world_y) ** 2
        if dist2 <= CONE_ASSOCIATE_RADIUS_M ** 2:
            cone["last_seen_s"] = sim_time_s

            # Light smoothing so cones don't jitter when detections shift slightly
            cone["x"] = 0.7 * existing_x + 0.3 * world_x
            cone["y"] = 0.7 * existing_y + 0.3 * world_y
            return

    # Otherwise add a brand new cone entry
    cone_map_world.append({"x": world_x, "y": world_y, "last_seen_s": sim_time_s})

def prune_old_cones(cone_map_world: List[Dict[str, float]],
                    sim_time_s: float,
                    ) -> None:
    # Remove cones that have not been seen recently so the map doesn't grow forever
    keep: List[Dict[str, float]] = []

    for cone in cone_map_world:
        age_s = sim_time_s - cone["last_seen_s"]
        if age_s <= CONE_KEEP_SECONDS:
            keep.append(cone)

    cone_map_world[:] = keep

def cone_alpha_from_age(age_s: float,
                        ) -> int:
    # Convert "seconds since last seen" into an alpha value for smooth fade-out
    if age_s <= 0.0:
        return CONE_ALPHA_MAX
    if age_s >= CONE_FADE_SECONDS:
        return 0

    # Fade from max alpha down to 0 over CONE_FADE_SECONDS
    fade_t = 1.0 - (age_s / CONE_FADE_SECONDS)
    alpha = int(CONE_ALPHA_MIN + fade_t * (CONE_ALPHA_MAX - CONE_ALPHA_MIN))
    return int(clamp(alpha, 0, 255))

def draw_circle_alpha(surface: pygame.Surface,
                      color_rgb: Tuple[int, int, int],
                      center_px: Tuple[int, int],
                      radius_px: int,
                      alpha: int,
                      ) -> None:
    # Draw a circle with alpha by rendering onto a temporary surface
    if alpha <= 0:
        return

    temp = pygame.Surface((radius_px * 2 + 2, radius_px * 2 + 2), pygame.SRCALPHA)
    pygame.draw.circle(temp, (*color_rgb, int(alpha)), (radius_px + 1, radius_px + 1), radius_px)
    surface.blit(temp, (center_px[0] - (radius_px + 1), center_px[1] - (radius_px + 1)))

def draw_text_left(surface: pygame.Surface,
                   font: pygame.font.Font,
                   x_px: int,
                   y_px: int,
                   lines: List[str],
                   ) -> None:
    # Render a list of text lines aligned to the left
    current_y = y_px
    for line in lines:
        text_img = font.render(line, True, (240, 240, 240))
        surface.blit(text_img, (x_px, current_y))
        current_y += text_img.get_height() + 2

def draw_text_right(surface: pygame.Surface,
                    font: pygame.font.Font,
                    right_x_px: int,
                    y_px: int,
                    lines: List[str],
                    ) -> None:
    # Render a list of text lines aligned to the right edge
    current_y = y_px
    for line in lines:
        text_img = font.render(line, True, (240, 240, 240))
        surface.blit(text_img, (right_x_px - text_img.get_width(), current_y))
        current_y += text_img.get_height() + 2

def vehicle_to_screen_px(right_m: float,
                         forward_m: float,
                         kart_center_x_px: int,
                         kart_center_y_px: int,
                         pixels_per_meter: float,
                         ) -> Tuple[int, int]:
    # Convert vehicle (right, forward) coordinates into screen pixels
    screen_x = kart_center_x_px + int(right_m * pixels_per_meter)
    screen_y = kart_center_y_px - int(forward_m * pixels_per_meter)
    return screen_x, screen_y

def build_track_polygon_vehicle(cones_vehicle: List[Tuple[float, float]],
                                ) -> Optional[List[Tuple[float, float]]]:
    # Build a corridor polygon from cone detections in vehicle coordinates
    left_boundary = [
        (right_m, forward_m)
        for (right_m, forward_m) in cones_vehicle
        if right_m < 0.0 and 0.0 <= forward_m <= TRACK_LOOKAHEAD_M
    ]

    right_boundary = [
        (right_m, forward_m)
        for (right_m, forward_m) in cones_vehicle
        if right_m > 0.0 and 0.0 <= forward_m <= TRACK_LOOKAHEAD_M
    ]

    # Sort cones by forward distance to form a stable boundary path
    left_boundary.sort(key=lambda p: p[1])
    right_boundary.sort(key=lambda p: p[1])

    # Cap cone count so a noisy frame doesn't create a crazy polygon
    left_boundary = left_boundary[:TRACK_MAX_CONES_PER_SIDE]
    right_boundary = right_boundary[:TRACK_MAX_CONES_PER_SIDE]

    # Require both boundaries to draw asphalt
    if len(left_boundary) < TRACK_MIN_CONES_PER_SIDE or len(right_boundary) < TRACK_MIN_CONES_PER_SIDE:
        return None

    # Inflate the corridor slightly so the asphalt looks nicer with sparse cones
    left_edge = [(right_m - TRACK_EDGE_INFLATE_M, forward_m) for (right_m, forward_m) in left_boundary]
    right_edge = [(right_m + TRACK_EDGE_INFLATE_M, forward_m) for (right_m, forward_m) in right_boundary]

    return right_edge + list(reversed(left_edge))

def build_track_polygon_from_world(cone_map_world: List[Dict[str, float]],
                                   world_x_m: float,
                                   world_y_m: float,
                                   heading_rad: float,
                                   ) -> Optional[List[Tuple[float, float]]]:
    # Convert persistent world cones into vehicle frame and reuse the same polygon logic
    cones_vehicle: List[Tuple[float, float]] = []

    for cone in cone_map_world:
        right_m, forward_m = world_to_vehicle(cone["x"], cone["y"], world_x_m, world_y_m, heading_rad)
        cones_vehicle.append((right_m, forward_m))

    return build_track_polygon_vehicle(cones_vehicle)

def main() -> None:
    # Parse command line options so we can replay any log file without editing code
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/push_test.jsonl", help="Path to push_test.jsonl")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier (1.0 = real-time-ish)")
    parser.add_argument("--mode", choices=["kart", "walk"], default="walk")
    parser.add_argument("--walk_speed", type=float, default=0.025)
    parser.add_argument("--walk_speed_min", type=float, default=0.025)
    parser.add_argument("--walk_speed_max", type=float, default=2.5)
    args = parser.parse_args()

    # Load records from jsonl file (each line is one control loop step)
    records = load_jsonl(args.log)
    if not records:
        raise RuntimeError("No valid JSONL records found.")

    # Initialize pygame window and utilities
    pygame.init()
    window_w_px, window_h_px = 1000, 800
    screen = pygame.display.set_mode((window_w_px, window_h_px))
    pygame.display.set_caption("autonav_v2 Push Test Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # Pixels per meter based on view size (bigger view -> smaller objects)
    pixels_per_meter = window_w_px / VIEW_SIZE_M

    # Kart stays fixed on screen; world moves around it
    kart_center_x_px = window_w_px // 2
    kart_center_y_px = int(window_h_px * 0.82)

    # -----------------------------
    # SIMULATION STATE
    # -----------------------------

    # World pose of the virtual rig/kart
    world_x_m = 0.0
    world_y_m = 0.0
    heading_rad = 0.0

    # Speed and acceleration (used for HUD)
    speed_mps = 0.0
    accel_mps2 = 0.0

    # Simulation time used for fade logic
    sim_time_s = 0.0

    # Persistent cone map in world coordinates
    cone_map_world: List[Dict[str, float]] = []

    # Trail shows where the rig has been in world coordinates
    trail_world: List[Tuple[float, float]] = []

    # Playback state
    record_index = 0
    paused = False
    replay_speed = max(0.1, float(args.speed))

    # Mode state
    sim_mode = args.mode
    walk_speed_mps = float(args.walk_speed)
    walk_speed_min = float(args.walk_speed_min)
    walk_speed_max = float(args.walk_speed_max)

    while True:
        # -----------------------------
        # HANDLE INPUT EVENTS
        # -----------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type != pygame.KEYDOWN:
                continue

            # Quit
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                pygame.quit()
                return

            # Pause
            if event.key == pygame.K_SPACE:
                paused = not paused

            # Reset playback + sim
            if event.key == pygame.K_r:
                world_x_m = 0.0
                world_y_m = 0.0
                heading_rad = 0.0
                speed_mps = 0.0
                accel_mps2 = 0.0
                sim_time_s = 0.0
                trail_world.clear()
                cone_map_world.clear()
                record_index = 0

            # Replay speed controls
            if event.key == pygame.K_LEFT:
                replay_speed = max(0.1, replay_speed * 0.8)
            if event.key == pygame.K_RIGHT:
                replay_speed = min(10.0, replay_speed * 1.25)

            # Toggle mode between walking and kart dynamics
            if event.key == pygame.K_m:
                sim_mode = "kart" if sim_mode == "walk" else "walk"

            # Adjust walking speed (Up/Down; Shift = bigger step)
            if event.key == pygame.K_UP:
                step = 0.25 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 0.1
                walk_speed_mps = clamp(walk_speed_mps + step, walk_speed_min, walk_speed_max)

            if event.key == pygame.K_DOWN:
                step = 0.25 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 0.1
                walk_speed_mps = clamp(walk_speed_mps - step, walk_speed_min, walk_speed_max)

        # -----------------------------
        # CLAMP PLAYBACK INDEX
        # -----------------------------
        if record_index >= len(records):
            record_index = len(records) - 1
            paused = True

        record = records[record_index]

        # -----------------------------
        # READ LOG VALUES FOR THIS STEP
        # -----------------------------
        dt_s, steer_deg, throttle_pct, brake_pct = parse_controls(record)

        # Replay speed changes how fast we advance through the log visually
        dt_s = dt_s * replay_speed

        # -----------------------------
        # UPDATE PERSISTENT CONE MAP
        # -----------------------------
        cones_vehicle = parse_cones_vehicle(record)
        for (right_m, forward_m) in cones_vehicle:
            cone_world_x, cone_world_y = vehicle_to_world(right_m,
                                                          forward_m,
                                                          world_x_m,
                                                          world_y_m,
                                                          heading_rad,
                                                          )
            associate_or_add_cone(cone_map_world, cone_world_x, cone_world_y, sim_time_s)

        prune_old_cones(cone_map_world, sim_time_s)

        # -----------------------------
        # ADVANCE SIMULATION STATE
        # -----------------------------
        if not paused:
            sim_time_s += dt_s

            # Convert steering to radians and clamp to realistic range
            steer_rad = math.radians(clamp(steer_deg, -45.0, 45.0))

            # Update speed based on selected mode
            if sim_mode == "kart":
                throttle_accel = (throttle_pct / 100.0) * MAX_ACCEL_MPS2
                brake_decel = (brake_pct / 100.0) * MAX_BRAKE_MPS2
                drag_decel = DRAG_GAIN * speed_mps

                # Net acceleration (+ forward, - braking)
                accel_mps2 = throttle_accel - brake_decel - drag_decel

                # Integrate speed, clamp to physical max
                speed_mps = clamp(speed_mps + accel_mps2 * dt_s, 0.0, KART_MAX_SPEED_MPS)
            else:
                # Walking mode ignores throttle/brake and uses a human-controlled constant speed
                speed_mps = clamp(walk_speed_mps, 0.0, walk_speed_max)
                accel_mps2 = 0.0

            # Update heading based on bicycle model yaw rate
            if abs(WHEELBASE_M) > 1e-6:
                yaw_rate = (speed_mps / WHEELBASE_M) * math.tan(steer_rad)
            else:
                yaw_rate = 0.0

            heading_rad = heading_rad + yaw_rate * dt_s

            # Integrate position in world frame
            vel_x = speed_mps * math.sin(heading_rad)
            vel_y = speed_mps * math.cos(heading_rad)

            world_x_m = world_x_m + vel_x * dt_s
            world_y_m = world_y_m + vel_y * dt_s

            # Append trail point for visualization
            trail_world.append((world_x_m, world_y_m))
            if len(trail_world) > 800:
                trail_world.pop(0)

            # Advance to next log record
            record_index += 1
        else:
            # Even if paused, we still need steer_rad for drawing vectors
            steer_rad = math.radians(clamp(steer_deg, -45.0, 45.0))

        # -----------------------------
        # RENDER: BACKGROUND + TRACK
        # -----------------------------
        grass_rgb = (34, 86, 48)
        asphalt_rgb = (38, 38, 42)
        grid_rgb = (28, 28, 32)

        screen.fill(grass_rgb)

        half_view_m = VIEW_SIZE_M / 2.0

        # Draw track polygon from persistent cone map so track stays visible
        track_poly_vehicle = build_track_polygon_from_world(cone_map_world,
                                                            world_x_m,
                                                            world_y_m,
                                                            heading_rad,
                                                            )

        if track_poly_vehicle is not None:
            polygon_px: List[Tuple[int, int]] = []
            for (right_m, forward_m) in track_poly_vehicle:
                # Keep polygon points in a slightly expanded view so the polygon is stable
                if abs(right_m) <= half_view_m * 1.5 and -half_view_m <= forward_m <= half_view_m * 2.0:
                    polygon_px.append(vehicle_to_screen_px(right_m,
                                                           forward_m,
                                                           kart_center_x_px,
                                                           kart_center_y_px,
                                                           pixels_per_meter,
                                                           ))

            if len(polygon_px) >= 3:
                pygame.draw.polygon(screen, asphalt_rgb, polygon_px)

        # -----------------------------
        # RENDER: OPTIONAL GRID
        # -----------------------------
        grid_min = -half_view_m
        grid_max = half_view_m

        r = math.floor(grid_min / GRID_STEP_M) * GRID_STEP_M
        while r <= grid_max + 1e-6:
            x1 = kart_center_x_px + int(r * pixels_per_meter)
            y1 = kart_center_y_px - int(grid_min * pixels_per_meter)
            x2 = kart_center_x_px + int(r * pixels_per_meter)
            y2 = kart_center_y_px - int(grid_max * pixels_per_meter)
            pygame.draw.line(screen, grid_rgb, (x1, y1), (x2, y2), 1)
            r += GRID_STEP_M

        f = math.floor(grid_min / GRID_STEP_M) * GRID_STEP_M
        while f <= grid_max + 1e-6:
            x1 = kart_center_x_px + int(grid_min * pixels_per_meter)
            y1 = kart_center_y_px - int(f * pixels_per_meter)
            x2 = kart_center_x_px + int(grid_max * pixels_per_meter)
            y2 = kart_center_y_px - int(f * pixels_per_meter)
            pygame.draw.line(screen, grid_rgb, (x1, y1), (x2, y2), 1)
            f += GRID_STEP_M

        # -----------------------------
        # RENDER: TRAIL
        # -----------------------------
        if len(trail_world) >= 2:
            trail_px: List[Tuple[int, int]] = []
            for (trail_x, trail_y) in trail_world:
                right_m, forward_m = world_to_vehicle(trail_x,
                                                      trail_y,
                                                      world_x_m,
                                                      world_y_m,
                                                      heading_rad,
                                                      )
                trail_px.append(vehicle_to_screen_px(right_m,
                                                     forward_m,
                                                     kart_center_x_px,
                                                     kart_center_y_px,
                                                     pixels_per_meter,
                                                     ))
            pygame.draw.lines(screen, (90, 90, 120), False, trail_px, 2)

        # -----------------------------
        # RENDER: CONES
        # -----------------------------
        for cone in cone_map_world:
            right_m, forward_m = world_to_vehicle(cone["x"],
                                                  cone["y"],
                                                  world_x_m,
                                                  world_y_m,
                                                  heading_rad,
                                                  )

            # Only draw cones that are inside the view window
            if abs(right_m) > half_view_m or forward_m < -half_view_m or forward_m > half_view_m:
                continue

            age_s = sim_time_s - cone["last_seen_s"]
            alpha = cone_alpha_from_age(age_s)

            cone_px = vehicle_to_screen_px(right_m,
                                           forward_m,
                                           kart_center_x_px,
                                           kart_center_y_px,
                                           pixels_per_meter,
                                           )
            draw_circle_alpha(screen, (255, 140, 0), cone_px, 7, alpha)

        # -----------------------------
        # RENDER: TARGET
        # -----------------------------
        target = parse_target_vehicle(record)
        if target is not None:
            target_right_m, target_forward_m = target
            if abs(target_right_m) <= half_view_m and -half_view_m <= target_forward_m <= half_view_m:
                target_px = vehicle_to_screen_px(target_right_m,
                                                 target_forward_m,
                                                 kart_center_x_px,
                                                 kart_center_y_px,
                                                 pixels_per_meter,
                                                 )
                pygame.draw.circle(screen, (0, 220, 120), target_px, 7, 2)

        # -----------------------------
        # RENDER: KART (FIXED)
        # -----------------------------
        kart_len_px = int(KART_LENGTH_M * pixels_per_meter * KART_DRAW_SCALE)
        kart_w_px = int(KART_WIDTH_M * pixels_per_meter * KART_DRAW_SCALE)

        kart_rect = pygame.Rect(0, 0, kart_w_px, kart_len_px)
        kart_rect.center = (kart_center_x_px, kart_center_y_px)

        pygame.draw.rect(screen, (230, 230, 235), kart_rect, 2)

        # Draw forward direction line for the kart
        pygame.draw.line(screen,
                         (230, 230, 235),
                         (kart_center_x_px, kart_center_y_px),
                         (kart_center_x_px, kart_center_y_px - kart_len_px // 2),
                         2,
                         )

        # -----------------------------
        # RENDER: VECTORS
        # -----------------------------

        # Velocity vector: blue, points in steering direction, length scales with speed
        vel_vec_max_px = int(1.8 * pixels_per_meter)
        vel_vec_len_px = int((speed_mps / max(KART_MAX_SPEED_MPS, 1e-6)) * vel_vec_max_px)
        vel_vec_len_px = max(0, vel_vec_len_px)

        vel_dx = int(math.sin(steer_rad) * vel_vec_len_px)
        vel_dy = int(math.cos(steer_rad) * vel_vec_len_px)

        pygame.draw.line(screen,
                         (60, 140, 255),
                         (kart_center_x_px, kart_center_y_px),
                         (kart_center_x_px + vel_dx, kart_center_y_px - vel_dy),
                         4,
                         )

        # Throttle vector: green forward, length scales with throttle percent
        accel_vec_max_px = int(1.2 * pixels_per_meter)
        throttle_vec_len_px = int((throttle_pct / 100.0) * accel_vec_max_px)

        if throttle_vec_len_px > 0:
            pygame.draw.line(screen,
                             (40, 220, 90),
                             (kart_center_x_px - 14, kart_center_y_px),
                             (kart_center_x_px - 14, kart_center_y_px - throttle_vec_len_px),
                             4,
                             )

        # Brake vector: red backward, length scales with brake percent
        brake_vec_len_px = int((brake_pct / 100.0) * accel_vec_max_px)

        if brake_vec_len_px > 0:
            pygame.draw.line(screen,
                             (240, 60, 60),
                             (kart_center_x_px - 28, kart_center_y_px),
                             (kart_center_x_px - 28, kart_center_y_px + brake_vec_len_px),
                             4,
                             )

        # -----------------------------
        # HUD (TOP LEFT + TOP RIGHT)
        # -----------------------------
        nav = record.get("nav", {})
        if not isinstance(nav, dict):
            nav = {}

        steer_pct_value = nav.get("steer_pct", None)

        left_lines = [
            f"Log: {os.path.basename(args.log)}  Frame: {record_index}/{len(records) - 1}",
            f"Steering Degree: {steer_deg:.2f}°",
            f"Steering Percent: {steer_pct_value}%",
            f"Throttle Percent: {throttle_pct:.0f}%",
            f"Brake Percent: {brake_pct:.0f}%",
            f"Velocity: {speed_mps:.2f} m/s",
            f"Acceleration: {accel_mps2:.2f} m/s^2",
        ]

        right_lines = [
            "Toggle Mode: M",
            "Adjust Speed: Up/Down",
            "Big Step: Shift",
            "Pause: Space",
            "Replay Speed: Left/Right",
            "Reset: R",
            "Quit: Q",
        ]

        draw_text_left(screen, font, 12, 10, left_lines)
        draw_text_right(screen, font, window_w_px - 12, 10, right_lines)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()