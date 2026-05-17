import os
import csv
import math
import glob
import random
import uuid
import numpy as np
import imageio
from typing import Optional, Set, List, Tuple

import habitat_sim
import magnum as mn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# ===================== Paths & Params =====================

DATA_PATH = "/cvhci/temp/kpeng/habitat/"  # <--- change to yours


HM3D_DATASET_CFG = "/media/kpeng/Elements/DATA/HM3D/versioned_data/hm3d-0.2/hm3d/train/hm3d_annotated_train_basis.scene_dataset_config.json"
SCENES_ROOT = "/media/kpeng/Elements/DATA/HM3D/versioned_data/hm3d-0.2/hm3d/train"
OUTPUT_ROOT = "/media/kpeng/Elements/DATA/data_collection/three_robot"

NUM_ROBOTS = 3
NUM_EXPLORATIONS_PER_SCENE = 15
NUM_STEPS = 1000
DT = 0.1

# Motion
FORWARD_SPEED = 0.32
ROT_SPEED = math.radians(28)

# Camera mount
CAM_HEIGHT = -1.2

# ===================== SPEED PRESET =====================

# Increase this if you want better RGB quality, but it will be slower.
SENSOR_RESOLUTION = [256, 256]  # [H, W]

SAVE_VIDEO = True
SAVE_EVERY = 2

BEV_ENABLE = True
BEV_EVERY = 10

VISIBLE_DOWNSAMPLE_STRIDE = 8

# ===================== Recording =====================

SAVE_SEMANTIC_RAW_PNG = False
SAVE_DEPTH_RAW_NPY = False
DEPTH_VIZ_MAX_M = 5.0

# Nearby objects
NEARBY_RADIUS_M = 3.0
NEARBY_TOPK = 10
NEARBY_USE_XZ_ONLY = True

NEARBY_REQUIRE_VISIBLE = True
VISIBLE_MIN_PIXELS = 10

# Semantic video
SEMANTIC_VIDEO_MODE = "idx_rgb"  # "idx_rgb" or "hash"

# Exploration ID
EXPLORATION_ID_MODE = "seq"

# ===================== Start: same region + same floor =====================

START_MUST_SAME_REGION = True
START_REGION_MAX_TRIES = 800
START_POINT_MAX_TRIES = 800

# Stronger same-floor constraint
SAME_FLOOR_Y_TOL = 0.35
START_P2_Y_TRIES = 800

# Start yaw aligned-ish
START_YAW_ALIGN_DEG = 8.0
START_YAW_JITTER_DEG = 10.0

# ===================== Coverage bias =====================

COV_CELL_M = 1.25
COV_REJECT_VISITED_PROB = 0.75
COV_KEEP_FAR_PROB = 0.35

WAYPOINT_MIN_STEPS = 35
WAYPOINT_MAX_STEPS = 130
WAYPOINT_MIN_DIST_M = 2.0
WAYPOINT_MAX_DIST_M = 30.0

# ===================== Randomness =====================

WP_YAW_GAIN = 1.35
WP_FORWARD_BASE = 0.95

NOISE_YAW_STD_RAD = math.radians(1.8)
NOISE_V_STD = 0.04

WIGGLE_PROB_PER_STEP = 0.018
WIGGLE_MIN_STEPS = 6
WIGGLE_MAX_STEPS = 16
WIGGLE_TURN_MAG = math.radians(30)
WIGGLE_FORWARD_SCALE = 0.78

# ===================== Wall-avoid using depth =====================

WALL_AVOID_ENABLED = True
WALL_FRONT_TH_M = 0.75

DEP_STRIDE = 6
FRONT_BAND = (0.40, 0.60)

# ===================== Overlap guarantee =====================

MEET_EVERY_STEPS = 280
MEET_DURATION_STEPS = 90
MEET_TARGET_RADIUS_M = 1.8

TETHER_ENABLED = True
TETHER_MAX_DIST = 18.0
TETHER_PULL_PROB = 0.65
TETHER_PULL_DIST = 8.0

# ===================== Robust nav attempts =====================

NAV_STEP_TRIES = 8
NAV_TURN_DELTA_DEG = 18.0
NAV_FWD_SCALE_MIN = 0.35
NAV_BACKUP_DIST = 0.10
NAV_SIDE_DIST = 0.12

# Stuck handling
STUCK_EPS_MOVED_SQ = 8e-5
STUCK_SOFT_LIMIT = 30
STUCK_HARD_LIMIT = 140
HARD_STUCK_TELEPORT = True

# ===================== Camera stabilization =====================

CAM_USE_VELOCITY_HEADING = True
CAM_HEADING_LOOKBACK = 3

CAM_YAW_EMA_ALPHA = 0.93
CAM_MAX_YAW_RATE_DEG_PER_S = 60.0

CAM_SMOOTH_POS = False
CAM_POS_EMA_ALPHA = 0.85

# ===================== Already explored =====================

EXPLORED_LIST = [
    "1S7LAXRdDqK", "8B43pG641ff", "CthA7sQNTPK", "DsEJeNPcZtE", "FnDDfrBZPhh",
    "gjhYih4upQ9", "GtM3JtRvvvR", "HZ2iMMBsBQ9", "KjZrPggnHm8", "nS8T59Aw3sf",
    "qgZhhx1MpTi", "SgkmkWjjmDJ", "u5atqC7vRCY", "W9YAR9qcuvN", "YHmAkqgwe2p",
    "yX5efd48dLf",
    "3CBBjsNkhqW", "9h5JJxM6E5S", "DBBESbk4Y3k", "E1NrAhMoqvB", "fRZhp6vWGw7",
    "GPyDUnjwZQy", "HfMobPm86Xn", "iePHCSf119p", "LVgQNuK8vtv", "oahi4u45xMf",
    "R9fYpvCUkV7", "TSJmdttd2GV", "W16Bm4ysK8v", "XiJhRLvpKpX", "YJDUB7hWg9h",
    "Z2DQddYp1fn",
    "6YtDG3FhNvx", "aRKASs4e8j1", "DNWbUAJYsPy", "ENiCjXWB6aQ", "ggNAcMh8JPT",
    "GsQBY83r3hb", "hWDDQnSDMXb", "JptJPosx1Z6", "mt9H8KcxRKD", "ooq3SnvC79d",
    "RTV2n6fXB2w", "TYDavTf8oyy", "w8GiikYuFRk", "XVSZJAtHKdi", "YmWinf3mhb5",
    "zUG6FL9TYeR"
]


# ===================== Utils =====================

def wrap_angle(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi


def se2_to_mat(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]], dtype=np.float32)


def mat_to_se2(T):
    x, y = T[0, 2], T[1, 2]
    yaw = math.atan2(T[1, 0], T[0, 0])
    return x, y, yaw


def relative_pose_2d(x_ref, y_ref, yaw_ref, x_other, y_other, yaw_other):
    T_ref = se2_to_mat(x_ref, y_ref, yaw_ref)
    T_other = se2_to_mat(x_other, y_other, yaw_other)
    T_rel = np.linalg.inv(T_ref) @ T_other
    return mat_to_se2(T_rel)


def _id_to_color(u: np.ndarray) -> np.ndarray:
    x = u.astype(np.uint32)
    r = (x * 1664525 + 1013904223) & 0xFF
    g = (x * 22695477 + 1) & 0xFF
    b = (x * 1103515245 + 12345) & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def semantic_to_rgb_idx(sem_mask: np.ndarray) -> np.ndarray:
    u8 = (sem_mask.astype(np.uint32) % 256).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def colorize_semantic(sem_mask: np.ndarray) -> np.ndarray:
    if SEMANTIC_VIDEO_MODE == "idx_rgb":
        return semantic_to_rgb_idx(sem_mask)
    return _id_to_color(sem_mask)


def depth_to_uint8(depth_m: np.ndarray, max_m: float = 10) -> np.ndarray:
    d = np.array(depth_m, dtype=np.float32)
    d = np.nan_to_num(d, nan=max_m, posinf=max_m, neginf=max_m)
    d = np.clip(d, 0.0, max_m)
    d = 1.0 - (d / max_m)
    img = (255.0 * d).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def load_semantic_txt_map(path: str):
    m = {}
    if not path or (not os.path.exists(path)):
        return m
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                sid = int(parts[0])
            except Exception:
                continue
            label = " ".join(parts[1:]) if len(parts) > 1 else "unknown"
            m[sid] = label
    return m


def vec3_to_xyz(v):
    if hasattr(v, "x"):
        return float(v.x), float(v.y), float(v.z)
    return float(v[0]), float(v[1]), float(v[2])


def obj_to_semantic_instance_id(obj):
    if obj is None:
        return None
    if hasattr(obj, "semantic_id"):
        try:
            return int(getattr(obj, "semantic_id"))
        except Exception:
            pass
    try:
        s = str(obj.id)
        if "_" in s:
            tail = s.split("_")[-1]
            if tail.isdigit() or (tail.startswith("-") and tail[1:].isdigit()):
                return int(tail)
        return int(s)
    except Exception:
        return None


def build_semantic_id_to_category(sim: habitat_sim.Simulator):
    m = {}
    if sim.semantic_scene is None:
        return m
    for obj in sim.semantic_scene.objects:
        if obj is None:
            continue
        sid = obj_to_semantic_instance_id(obj)
        if sid is None:
            continue
        try:
            cat = obj.category.name() if obj.category else "unknown"
        except Exception:
            cat = "unknown"
        m[int(sid)] = cat
    return m


def get_visible_object_ids(sem_mask: np.ndarray, min_pixels: int = 10, downsample_stride: int = 8):
    m = sem_mask
    if downsample_stride and downsample_stride > 1:
        m = m[::downsample_stride, ::downsample_stride]
    flat = m.reshape(-1)
    ids, counts = np.unique(flat, return_counts=True)
    visible = set()
    for sid, c in zip(ids, counts):
        if int(sid) == 0:
            continue
        if int(c) >= int(min_pixels):
            visible.add(int(sid))
    return visible


def ema_angle(prev, new, alpha):
    if prev is None:
        return new
    err = wrap_angle(new - prev)
    return wrap_angle(prev + (1.0 - alpha) * err)


def limit_angle_rate(prev, new, max_rate_rad_per_s, dt):
    if prev is None:
        return new
    max_step = max_rate_rad_per_s * dt
    err = wrap_angle(new - prev)
    err = max(-max_step, min(max_step, err))
    return wrap_angle(prev + err)


def ema_vec3(prev, new, alpha):
    if prev is None:
        return new
    return alpha * prev + (1.0 - alpha) * new


def cov_cell_key(x: float, z: float, cell: float) -> Tuple[int, int]:
    return (int(math.floor(x / cell)), int(math.floor(z / cell)))


# ===================== Region helpers =====================

def _aabb_min_max(aabb):
    if hasattr(aabb, "min") and hasattr(aabb, "max"):
        return (
            np.array(vec3_to_xyz(aabb.min), dtype=np.float32),
            np.array(vec3_to_xyz(aabb.max), dtype=np.float32),
        )
    if hasattr(aabb, "min_corner") and hasattr(aabb, "max_corner"):
        return (
            np.array(vec3_to_xyz(aabb.min_corner), dtype=np.float32),
            np.array(vec3_to_xyz(aabb.max_corner), dtype=np.float32),
        )
    if hasattr(aabb, "back_bottom_left") and hasattr(aabb, "front_top_right"):
        return (
            np.array(vec3_to_xyz(aabb.back_bottom_left), dtype=np.float32),
            np.array(vec3_to_xyz(aabb.front_top_right), dtype=np.float32),
        )
    if hasattr(aabb, "center") and (hasattr(aabb, "sizes") or hasattr(aabb, "size") or hasattr(aabb, "extent")):
        c = np.array(vec3_to_xyz(aabb.center), dtype=np.float32)
        if hasattr(aabb, "sizes"):
            s = np.array(vec3_to_xyz(aabb.sizes), dtype=np.float32)
        elif hasattr(aabb, "size"):
            s = np.array(vec3_to_xyz(aabb.size), dtype=np.float32)
        else:
            s = np.array(vec3_to_xyz(aabb.extent), dtype=np.float32)
        half = 0.5 * s
        return c - half, c + half
    raise AttributeError(f"Unknown AABB/BBox type: {type(aabb)}")


def _in_aabb_xz(p: mn.Vector3, aabb) -> bool:
    mn_xyz, mx_xyz = _aabb_min_max(aabb)
    x, z = float(p.x), float(p.z)
    return (mn_xyz[0] <= x <= mx_xyz[0]) and (mn_xyz[2] <= z <= mx_xyz[2])


def sample_navigable_point_in_region(pf, region, max_tries=600):
    if region is None or getattr(region, "aabb", None) is None:
        return None
    for _ in range(max_tries):
        p = pf.get_random_navigable_point()
        p = mn.Vector3(float(p[0]), float(p[1]), float(p[2]))
        if _in_aabb_xz(p, region.aabb):
            return np.array([p.x, p.y, p.z], dtype=np.float32)
    return None


def sample_navigable_point_same_floor(pf, ref_y: float, max_tries: int = 800, y_tol: float = SAME_FLOOR_Y_TOL):
    for _ in range(max_tries):
        p = np.array(pf.get_random_navigable_point(), dtype=np.float32)
        if abs(float(p[1]) - float(ref_y)) <= float(y_tol):
            return p
    return None


def sample_navigable_point_in_region_same_floor(pf, region, ref_y: float, max_tries=800, y_tol: float = SAME_FLOOR_Y_TOL):
    if region is None or getattr(region, "aabb", None) is None:
        return None
    for _ in range(max_tries):
        p = pf.get_random_navigable_point()
        p = mn.Vector3(float(p[0]), float(p[1]), float(p[2]))
        if _in_aabb_xz(p, region.aabb) and abs(float(p.y) - float(ref_y)) <= float(y_tol):
            return np.array([p.x, p.y, p.z], dtype=np.float32)
    return None


def get_all_valid_regions(sim) -> List:
    sem = sim.semantic_scene
    if sem is None or getattr(sem, "regions", None) is None:
        return []
    regs = []
    for r in sem.regions:
        if r is None:
            continue
        if getattr(r, "aabb", None) is None:
            continue
        regs.append(r)
    return regs


def choose_random_region_with_nav(sim, pf, rng: random.Random, max_trials=120):
    regs = get_all_valid_regions(sim)
    if len(regs) == 0:
        return None
    idxs = list(range(len(regs)))
    rng.shuffle(idxs)
    for i in idxs[:max_trials]:
        r = regs[i]
        p = sample_navigable_point_in_region(pf, r, max_tries=200)
        if p is not None:
            return r
    return None


# ===================== Scene discovery =====================

def find_scenes_with_semantics(root_dir: str):
    scenes = []
    for scene_dir in sorted(glob.glob(os.path.join(root_dir, "*"))):
        if not os.path.isdir(scene_dir):
            continue
        sem_glbs = glob.glob(os.path.join(scene_dir, "*.semantic.glb"))
        if len(sem_glbs) == 0:
            continue

        semantic_glb = sem_glbs[0]
        scene_name = os.path.basename(semantic_glb).replace(".semantic.glb", "")

        basis_glb = os.path.join(scene_dir, f"{scene_name}.basis.glb")
        if not os.path.exists(basis_glb):
            basis_candidates = glob.glob(os.path.join(scene_dir, "*.basis.glb"))
            if len(basis_candidates) == 0:
                continue
            basis_glb = basis_candidates[0]
            scene_name = os.path.basename(basis_glb).replace(".basis.glb", "")

        semantic_txt = os.path.join(scene_dir, f"{scene_name}.semantic.txt")
        if not os.path.exists(semantic_txt):
            semantic_txt = ""

        scenes.append({
            "scene_dir": scene_dir,
            "scene_name": scene_name,
            "basis_glb": basis_glb,
            "semantic_glb": semantic_glb,
            "semantic_txt": semantic_txt,
        })
    return scenes


# ===================== Simulator =====================

def make_sim(scene_path: str, num_robots: int = NUM_ROBOTS):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = HM3D_DATASET_CFG
    sim_cfg.scene_id = scene_path
    sim_cfg.load_semantic_mesh = True
    sim_cfg.enable_physics = True

    def make_sensors(prefix: str):
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = f"{prefix}_rgb"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        rgb.resolution = SENSOR_RESOLUTION
        rgb.position = [0.0, CAM_HEIGHT, 0.0]

        sem = habitat_sim.CameraSensorSpec()
        sem.uuid = f"{prefix}_sem"
        sem.sensor_type = habitat_sim.SensorType.SEMANTIC
        sem.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sem.resolution = SENSOR_RESOLUTION
        sem.position = [0.0, CAM_HEIGHT, 0.0]

        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = f"{prefix}_depth"
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        depth.resolution = SENSOR_RESOLUTION
        depth.position = [0.0, CAM_HEIGHT, 0.0]

        return [rgb, sem, depth]

    agent_cfgs = []
    for i in range(num_robots):
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = make_sensors(f"agent{i}")
        agent_cfgs.append(agent_cfg)

    cfg = habitat_sim.Configuration(sim_cfg, agent_cfgs)
    return habitat_sim.Simulator(cfg)


# ===================== Nearby objects =====================

def get_nearby_semantic_objects(
    sim: habitat_sim.Simulator,
    agent_pos_xyz: np.ndarray,
    agent_yaw: float,
    radius_m: float = 3.0,
    topk: int = 10,
    use_xz_only: bool = True,
    visible_ids: Optional[Set[int]] = None,
):
    sem_scene = sim.semantic_scene
    if sem_scene is None:
        return []

    ax, ay, az = float(agent_pos_xyz[0]), float(agent_pos_xyz[1]), float(agent_pos_xyz[2])
    results = []

    for obj in sem_scene.objects:
        if obj is None:
            continue

        sid = obj_to_semantic_instance_id(obj)
        if sid is None:
            continue
        sid = int(sid)

        if visible_ids is not None and sid not in visible_ids:
            continue

        try:
            cat_name = obj.category.name() if obj.category is not None else "unknown"
        except Exception:
            cat_name = "unknown"

        try:
            cx0, cy0, cz0 = vec3_to_xyz(obj.aabb.center)
        except Exception:
            continue

        try:
            if hasattr(obj.aabb, "min") and hasattr(obj.aabb, "max"):
                xmin, ymin, zmin = vec3_to_xyz(obj.aabb.min)
                xmax, ymax, zmax = vec3_to_xyz(obj.aabb.max)
            else:
                xmin, ymin, zmin = vec3_to_xyz(obj.aabb.min_corner)
                xmax, ymax, zmax = vec3_to_xyz(obj.aabb.max_corner)
        except Exception:
            xmin = xmax = cx0
            ymin = ymax = cy0
            zmin = zmax = cz0

        px = min(max(ax, xmin), xmax)
        py = min(max(ay, ymin), ymax)
        pz = min(max(az, zmin), zmax)

        if use_xz_only:
            dx, dz = px - ax, pz - az
            dist = math.sqrt(dx * dx + dz * dz)
            obj_dir_world = math.atan2(dz, dx)
        else:
            dx, dy, dz = px - ax, py - ay, pz - az
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            obj_dir_world = math.atan2(dz, dx)

        if dist <= radius_m:
            bearing_rad = wrap_angle(obj_dir_world - agent_yaw)
            bearing_deg = math.degrees(bearing_rad)
            results.append({
                "semantic_instance_id": sid,
                "object_key": str(obj.id),
                "category": cat_name,
                "dist_m": float(dist),
                "bearing_rad": float(bearing_rad),
                "bearing_deg": float(bearing_deg),
                "center_x": cx0,
                "center_y": cy0,
                "center_z": cz0,
            })

    results.sort(key=lambda d: d["dist_m"])
    return results[:topk]


# ===================== Waypoint sampling =====================

def sample_waypoint_global(pf, rng: random.Random, origin_xyz: np.ndarray, visited_cells: Set[Tuple[int, int]]):
    allow_far = (rng.random() < COV_KEEP_FAR_PROB)

    ref_y = float(origin_xyz[1])

    for _ in range(900):
        p = np.array(pf.get_random_navigable_point(), dtype=np.float32)

        # keep waypoint roughly on same floor
        if abs(float(p[1]) - ref_y) > SAME_FLOOR_Y_TOL:
            continue

        dx = float(p[0] - origin_xyz[0])
        dz = float(p[2] - origin_xyz[2])
        d = math.sqrt(dx * dx + dz * dz)

        if d < WAYPOINT_MIN_DIST_M:
            continue

        if (not allow_far) and d > WAYPOINT_MAX_DIST_M:
            if rng.random() < 0.85:
                continue

        key = cov_cell_key(float(p[0]), float(p[2]), COV_CELL_M)
        if key in visited_cells and rng.random() < COV_REJECT_VISITED_PROB:
            continue

        return p

    # fallback same floor
    p = sample_navigable_point_same_floor(pf, ref_y=ref_y, max_tries=600, y_tol=SAME_FLOOR_Y_TOL)
    if p is not None:
        return p

    return np.array(origin_xyz, dtype=np.float32)


def sample_waypoint_near_point(pf, rng: random.Random, center_xyz: np.ndarray, radius: float):
    cy = float(center_xyz[1])
    for _ in range(450):
        ang = rng.uniform(-math.pi, math.pi)
        r = rng.uniform(0.3, radius)
        x = float(center_xyz[0]) + r * math.cos(ang)
        z = float(center_xyz[2]) + r * math.sin(ang)
        snapped = pf.snap_point(mn.Vector3(x, cy, z))
        if pf.is_navigable(snapped):
            return np.array([float(snapped.x), float(snapped.y), float(snapped.z)], dtype=np.float32)
    return np.array(center_xyz, dtype=np.float32)


def sample_waypoint_near_point_same_floor(
    pf,
    rng: random.Random,
    center_xyz: np.ndarray,
    radius: float,
    y_tol: float = SAME_FLOOR_Y_TOL,
):
    cy = float(center_xyz[1])
    for _ in range(600):
        ang = rng.uniform(-math.pi, math.pi)
        r = rng.uniform(0.3, radius)
        x = float(center_xyz[0]) + r * math.cos(ang)
        z = float(center_xyz[2]) + r * math.sin(ang)

        snapped = pf.snap_point(mn.Vector3(x, cy, z))
        if pf.is_navigable(snapped) and abs(float(snapped.y) - cy) <= y_tol:
            return np.array([float(snapped.x), float(snapped.y), float(snapped.z)], dtype=np.float32)
    return None


def compute_cmd_to_waypoint(x, y, z, yaw, wp_xyz: np.ndarray, rng: random.Random):
    dx = float(wp_xyz[0] - x)
    dz = float(wp_xyz[2] - z)
    target_yaw = math.atan2(dz, dx)
    yaw_err = wrap_angle(target_yaw - yaw)

    w = WP_YAW_GAIN * yaw_err / DT
    w = max(-ROT_SPEED, min(ROT_SPEED, w))

    yaw_scale = max(0.30, 1.0 - abs(yaw_err) / math.radians(90))
    v = FORWARD_SPEED * WP_FORWARD_BASE * yaw_scale

    w += rng.gauss(0.0, NOISE_YAW_STD_RAD) / DT
    v += rng.gauss(0.0, NOISE_V_STD)

    v = max(0.0, min(FORWARD_SPEED, v))
    w = max(-ROT_SPEED, min(ROT_SPEED, w))
    return v, w


# ===================== Done markers =====================

def exploration_done_marker_path(out_dir: str) -> str:
    return os.path.join(out_dir, "DONE")


def get_done_count_for_scene(scene_name: str) -> int:
    scene_dir = os.path.join(OUTPUT_ROOT, scene_name)
    if not os.path.isdir(scene_dir):
        return 0
    done = 0
    for exp_dir in glob.glob(os.path.join(scene_dir, "explore_*")):
        if os.path.exists(os.path.join(exp_dir, "DONE")):
            done += 1
    return done


def make_exploration_id(scene_name: str, explore_idx: int):
    if EXPLORATION_ID_MODE.lower() == "uuid":
        return f"explore_{uuid.uuid4().hex[:8]}"
    return f"explore_{explore_idx:03d}"


def velocity_heading(pos_hist, fallback_yaw):
    if len(pos_hist) < (CAM_HEADING_LOOKBACK + 1):
        return fallback_yaw
    x_old, z_old = pos_hist[0]
    x_new, z_new = pos_hist[-1]
    dx = x_new - x_old
    dz = z_new - z_old
    if dx * dx + dz * dz < 1e-8:
        return fallback_yaw
    return math.atan2(dz, dx)


# ===================== Simulator runner =====================

def run_one_exploration(scene_info: dict, explore_idx: int):
    scene_name = scene_info["scene_name"]
    scene_path = scene_info["basis_glb"]
    sem_txt_path = scene_info["semantic_txt"]

    exploration_id = make_exploration_id(scene_name, explore_idx)
    out_dir = os.path.join(OUTPUT_ROOT, scene_name, exploration_id)
    os.makedirs(out_dir, exist_ok=True)

    if EXPLORATION_ID_MODE.lower() == "seq":
        if os.path.exists(os.path.join(out_dir, "DONE")):
            print(f"    [SKIP] already DONE: {scene_name}/{exploration_id}")
            return True

    id_to_label_txt = load_semantic_txt_map(sem_txt_path)

    sim = make_sim(scene_path, NUM_ROBOTS)
    if sim.semantic_scene is None or sim.semantic_scene.objects is None or len(sim.semantic_scene.objects) == 0:
        sim.close()
        print(f"[SKIP] {scene_name}: semantic_scene missing/empty.")
        return False

    id_to_category = build_semantic_id_to_category(sim)
    agents = [sim.get_agent(i) for i in range(NUM_ROBOTS)]

    pf = sim.pathfinder
    if not pf.is_loaded:
        nav_settings = habitat_sim.NavMeshSettings()
        nav_settings.set_defaults()
        sim.recompute_navmesh(pf, nav_settings, include_static_objects=True)

    nav_min, nav_max = pf.get_bounds()
    xmin, ymin, zmin = float(nav_min[0]), float(nav_min[1]), float(nav_min[2])
    xmax, ymax, zmax = float(nav_max[0]), float(nav_max[1]), float(nav_max[2])

    obj_mgr = sim.get_rigid_object_manager()
    obj_templates_mgr = sim.get_object_template_manager()
    obj_templates_mgr.load_configs(os.path.join(DATA_PATH, "objects/locobot_merged"))
    all_handles = obj_templates_mgr.get_template_handles("")
    locobot_handles = [h for h in all_handles if "locobot" in h.lower()]
    if len(locobot_handles) == 0:
        raise RuntimeError("No template handle containing 'locobot'. Check objects/locobot_merged")
    locobot_handle = locobot_handles[0]

    locobots = []
    for _ in range(NUM_ROBOTS):
        bot = obj_mgr.add_object_by_template_handle(locobot_handle)
        bot.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        locobots.append(bot)

    seed_base = (hash(scene_name) ^ (explore_idx * 1000003) ^ 0x9E3779B9) & 0xFFFFFFFF
    rng_global = random.Random(seed_base)
    rngs = [random.Random((seed_base + 101 * (i + 1)) & 0xFFFFFFFF) for i in range(NUM_ROBOTS)]

    # ===================== Start =====================

    regions = get_all_valid_regions(sim)
    start_region = None

    if START_MUST_SAME_REGION and len(regions) > 0:
        for _ in range(START_REGION_MAX_TRIES):
            rr = choose_random_region_with_nav(sim, pf, rng_global)
            if rr is None:
                continue
            ptest = sample_navigable_point_in_region(pf, rr, max_tries=200)
            if ptest is not None:
                start_region = rr
                break

    if START_MUST_SAME_REGION and start_region is None:
        print("    [WARN] cannot enforce same region start. fallback to same-floor random nav points.")

    start_points = []

    p0 = sample_navigable_point_in_region(pf, start_region, max_tries=START_POINT_MAX_TRIES) if start_region is not None else None
    if p0 is None:
        p0 = np.array(pf.get_random_navigable_point(), dtype=np.float32)

    start_points.append(p0)
    base_y = float(p0[1])

    for rid in range(1, NUM_ROBOTS):
        pi = None

        if start_region is not None:
            pi = sample_navigable_point_in_region_same_floor(
                pf, start_region, ref_y=base_y, max_tries=START_P2_Y_TRIES, y_tol=SAME_FLOOR_Y_TOL
            )

        if pi is None:
            pi = sample_navigable_point_same_floor(
                pf, ref_y=base_y, max_tries=1200, y_tol=SAME_FLOOR_Y_TOL
            )

        if pi is None:
            pi = sample_waypoint_near_point_same_floor(
                pf, rngs[rid], center_xyz=p0, radius=6.0, y_tol=SAME_FLOOR_Y_TOL
            )

        if pi is None:
            pi = p0.copy()
            pi[0] += rngs[rid].uniform(-1.0, 1.0)
            pi[2] += rngs[rid].uniform(-1.0, 1.0)
            snapped = pf.snap_point(mn.Vector3(float(pi[0]), float(base_y), float(pi[2])))
            pi = np.array([float(snapped.x), float(snapped.y), float(snapped.z)], dtype=np.float32)

        start_points.append(pi)

    positions = [np.array(p, dtype=np.float32) for p in start_points]

    yaws = []
    yaw0 = rngs[0].uniform(-math.pi, math.pi)
    yaw0 = wrap_angle(yaw0 + math.radians(rngs[0].uniform(-START_YAW_JITTER_DEG, START_YAW_JITTER_DEG)))
    yaws.append(yaw0)
    for i in range(1, NUM_ROBOTS):
        yi = wrap_angle(yaw0 + math.radians(rngs[i].uniform(-START_YAW_ALIGN_DEG, START_YAW_ALIGN_DEG)))
        yaws.append(yi)

    fps = int(1.0 / DT)
    save_fps = max(1, fps // max(1, SAVE_EVERY))
    bev_fps = max(1, fps // max(1, BEV_EVERY))

    rgb_writers, sem_writers, depth_writers = [], [], []
    if SAVE_VIDEO:
        for i in range(NUM_ROBOTS):
            rgb_writers.append(imageio.get_writer(os.path.join(out_dir, f"locobot{i+1}_cam.mp4"), fps=save_fps))
            sem_writers.append(imageio.get_writer(os.path.join(out_dir, f"locobot{i+1}_sem.mp4"), fps=save_fps))
            depth_writers.append(imageio.get_writer(os.path.join(out_dir, f"locobot{i+1}_depth.mp4"), fps=save_fps))
    else:
        rgb_writers = [None] * NUM_ROBOTS
        sem_writers = [None] * NUM_ROBOTS
        depth_writers = [None] * NUM_ROBOTS

    bev_writer = imageio.get_writer(os.path.join(out_dir, "bev_trajectories.mp4"), fps=bev_fps) if BEV_ENABLE else None

    csv_path = os.path.join(out_dir, "locobot_pairwise_trajectory_and_cmds.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "t", "ref_robot_id", "other_robot_id",
        "ref_x", "ref_y", "ref_z", "ref_yaw",
        "other_rel_x", "other_rel_y", "other_rel_z", "other_rel_yaw",
        "v_cmd", "omega_cmd",
    ])

    nearby_csv_path = os.path.join(out_dir, "nearby_objects.csv")
    nearby_file = open(nearby_csv_path, "w", newline="")
    nearby_writer = csv.writer(nearby_file)
    nearby_writer.writerow([
        "t", "robot_id",
        "semantic_instance_id",
        "object_key",
        "category_name",
        "label_from_txt",
        "dist_m",
        "bearing_rad",
        "bearing_deg",
        "center_x", "center_y", "center_z",
    ])

    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write(f"scene_name: {scene_name}\n")
        f.write(f"scene_path: {scene_path}\n")
        f.write(f"exploration_id: {exploration_id}\n")
        f.write(f"num_robots: {NUM_ROBOTS}\n")
        f.write(f"start_must_same_region: {START_MUST_SAME_REGION}\n")
        f.write(f"same_floor_y_tol: {SAME_FLOOR_Y_TOL}\n")
        f.write(f"sensor_resolution: {SENSOR_RESOLUTION}\n")
        f.write(f"save_video: {SAVE_VIDEO}\n")
        f.write(f"save_every: {SAVE_EVERY}\n")
        f.write(f"bev_enable: {BEV_ENABLE}\n")
        f.write(f"bev_every: {BEV_EVERY}\n")
        f.write(f"meet_every_steps: {MEET_EVERY_STEPS}\n")
        f.write(f"meet_duration_steps: {MEET_DURATION_STEPS}\n")
        f.write(f"wall_avoid_enabled: {WALL_AVOID_ENABLED}\n")
        f.write(f"coverage_cell_m: {COV_CELL_M}\n")

    def nav_constrained_step(x, y, z, yaw, v, w):
        yaw_base = wrap_angle(yaw + w * DT)

        for k in range(NAV_STEP_TRIES):
            if k == 0:
                yaw_try = yaw_base
            else:
                m = (k + 1) // 2
                sign = +1 if (k % 2 == 1) else -1
                yaw_try = wrap_angle(yaw_base + sign * m * math.radians(NAV_TURN_DELTA_DEG))

            scale = max(NAV_FWD_SCALE_MIN, 1.0 - 0.12 * k)
            v_try = v * scale

            x_new = x + v_try * math.cos(yaw_try) * DT
            z_new = z + v_try * math.sin(yaw_try) * DT
            snapped = pf.snap_point(mn.Vector3(x_new, y, z_new))
            if pf.is_navigable(snapped):
                return float(snapped.x), float(snapped.y), float(snapped.z), yaw_try

        for sign in (+1, -1):
            side_yaw = wrap_angle(yaw_base + sign * math.pi / 2.0)
            x_new = x + NAV_SIDE_DIST * math.cos(side_yaw)
            z_new = z + NAV_SIDE_DIST * math.sin(side_yaw)
            snapped = pf.snap_point(mn.Vector3(x_new, y, z_new))
            if pf.is_navigable(snapped):
                return float(snapped.x), float(snapped.y), float(snapped.z), yaw_base

        x_back = x - NAV_BACKUP_DIST * math.cos(yaw_base)
        z_back = z - NAV_BACKUP_DIST * math.sin(yaw_base)
        snapped = pf.snap_point(mn.Vector3(x_back, y, z_back))
        if pf.is_navigable(snapped):
            return float(snapped.x), float(snapped.y), float(snapped.z), yaw_base

        return x, y, z, yaw_base

    def maybe_shift_waypoint_from_wall(dep, rng, cur_xyz, wp, visited):
        dd = dep[::DEP_STRIDE, ::DEP_STRIDE].astype(np.float32)
        dd = np.nan_to_num(dd, nan=DEPTH_VIZ_MAX_M, posinf=DEPTH_VIZ_MAX_M, neginf=DEPTH_VIZ_MAX_M)
        dd = np.clip(dd, 0.0, DEPTH_VIZ_MAX_M)
        H, W = dd.shape
        x0 = int(FRONT_BAND[0] * W)
        x1 = int(FRONT_BAND[1] * W)
        front_med = float(np.median(dd[:, x0:x1]))
        if front_med >= WALL_FRONT_TH_M:
            return wp

        wp_new = sample_waypoint_near_point_same_floor(
            pf, rng, center_xyz=cur_xyz, radius=6.0, y_tol=SAME_FLOOR_Y_TOL
        )
        if wp_new is None:
            wp_new = sample_waypoint_global(pf, rng, cur_xyz, visited)

        key = cov_cell_key(float(wp_new[0]), float(wp_new[2]), COV_CELL_M)
        if key in visited and rng.random() < 0.6:
            wp_new = sample_waypoint_global(pf, rng, cur_xyz, visited)

        return wp_new

    visited = [set() for _ in range(NUM_ROBOTS)]
    waypoints = [sample_waypoint_global(pf, rngs[i], positions[i], visited[i]) for i in range(NUM_ROBOTS)]
    wp_timers = [rngs[i].randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS) for i in range(NUM_ROBOTS)]
    wiggles = [0 for _ in range(NUM_ROBOTS)]
    stucks = [0 for _ in range(NUM_ROBOTS)]
    prev_positions_xz = [(float(positions[i][0]), float(positions[i][2])) for i in range(NUM_ROBOTS)]

    meet_timer = MEET_EVERY_STEPS
    meet_active = 0
    meet_wp = None

    pos_hists = [[(float(positions[i][0]), float(positions[i][2]))] for i in range(NUM_ROBOTS)]
    cam_yaws = list(yaws)
    cam_positions = [positions[i].copy() for i in range(NUM_ROBOTS)]
    trajectories = [[] for _ in range(NUM_ROBOTS)]

    try:
        for t in range(NUM_STEPS):
            for i in range(NUM_ROBOTS):
                visited[i].add(cov_cell_key(float(positions[i][0]), float(positions[i][2]), COV_CELL_M))

            if meet_active <= 0:
                meet_timer -= 1
                if meet_timer <= 0:
                    meet_active = MEET_DURATION_STEPS
                    meet_timer = MEET_EVERY_STEPS
                    mid = np.mean(np.stack(positions, axis=0), axis=0)
                    meet_wp = sample_waypoint_near_point_same_floor(
                        pf, rng_global, mid, radius=MEET_TARGET_RADIUS_M * 2.5, y_tol=SAME_FLOOR_Y_TOL
                    )
                    if meet_wp is None:
                        meet_wp = sample_waypoint_near_point(pf, rng_global, mid, radius=MEET_TARGET_RADIUS_M * 2.5)

            tether_flags = [False] * NUM_ROBOTS
            if TETHER_ENABLED:
                center = np.mean(np.stack(positions, axis=0), axis=0)
                center_y = float(center[1])
                for i in range(NUM_ROBOTS):
                    dx = float(positions[i][0] - center[0])
                    dz = float(positions[i][2] - center[2])
                    dist_to_center = math.sqrt(dx * dx + dz * dz)
                    same_floor = abs(float(positions[i][1]) - center_y) <= SAME_FLOOR_Y_TOL
                    if dist_to_center > TETHER_MAX_DIST and same_floor and rng_global.random() < TETHER_PULL_PROB:
                        tether_flags[i] = True

            if meet_active > 0 and meet_wp is not None:
                for i in range(NUM_ROBOTS):
                    if abs(float(positions[i][1]) - float(meet_wp[1])) <= SAME_FLOOR_Y_TOL:
                        waypoints[i] = meet_wp
                        wp_timers[i] = 10
                meet_active -= 1
            else:
                center = np.mean(np.stack(positions, axis=0), axis=0)
                for i in range(NUM_ROBOTS):
                    wp_timers[i] -= 1
                    if wp_timers[i] <= 0 or tether_flags[i]:
                        if tether_flags[i]:
                            wp_pull = sample_waypoint_near_point_same_floor(
                                pf, rngs[i], center, radius=TETHER_PULL_DIST, y_tol=SAME_FLOOR_Y_TOL
                            )
                            if wp_pull is None:
                                wp_pull = sample_waypoint_global(pf, rngs[i], positions[i], visited[i])
                            waypoints[i] = wp_pull
                        else:
                            waypoints[i] = sample_waypoint_global(pf, rngs[i], positions[i], visited[i])
                        wp_timers[i] = rngs[i].randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS)

            v_cmds = []
            w_cmds = []
            for i in range(NUM_ROBOTS):
                x, y, z = positions[i]
                yaw = yaws[i]
                v, w = compute_cmd_to_waypoint(x, y, z, yaw, waypoints[i], rngs[i])

                if wiggles[i] <= 0 and rngs[i].random() < WIGGLE_PROB_PER_STEP:
                    wiggles[i] = rngs[i].randint(WIGGLE_MIN_STEPS, WIGGLE_MAX_STEPS)

                if wiggles[i] > 0:
                    w += rngs[i].uniform(-WIGGLE_TURN_MAG, WIGGLE_TURN_MAG) / DT
                    v *= WIGGLE_FORWARD_SCALE
                    wiggles[i] -= 1

                v = max(0.0, min(FORWARD_SPEED, v))
                w = max(-ROT_SPEED, min(ROT_SPEED, w))

                v_cmds.append(v)
                w_cmds.append(w)

            for i in range(NUM_ROBOTS):
                x, y, z = positions[i]
                yaw = yaws[i]
                x, y, z, yaw = nav_constrained_step(x, y, z, yaw, v_cmds[i], w_cmds[i])

                snapped_pose = pf.snap_point(mn.Vector3(float(x), float(y), float(z)))
                x = float(snapped_pose.x)
                y = float(snapped_pose.y)
                z = float(snapped_pose.z)

                positions[i] = np.array([x, y, z], dtype=np.float32)
                yaws[i] = yaw

            for i in range(NUM_ROBOTS):
                x, _, z = positions[i]
                prev_x, prev_z = prev_positions_xz[i]

                if (x - prev_x) ** 2 + (z - prev_z) ** 2 < STUCK_EPS_MOVED_SQ:
                    stucks[i] += 1
                else:
                    stucks[i] = 0

                prev_positions_xz[i] = (x, z)

                if stucks[i] > STUCK_SOFT_LIMIT:
                    waypoints[i] = sample_waypoint_global(pf, rngs[i], positions[i], visited[i])
                    wp_timers[i] = rngs[i].randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS)
                    wiggles[i] = rngs[i].randint(WIGGLE_MIN_STEPS, WIGGLE_MAX_STEPS)
                    stucks[i] = 0

                if HARD_STUCK_TELEPORT and stucks[i] > STUCK_HARD_LIMIT:
                    cur_y = float(positions[i][1])

                    p = sample_navigable_point_same_floor(
                        pf, ref_y=cur_y, max_tries=1200, y_tol=SAME_FLOOR_Y_TOL
                    )

                    if p is None:
                        p = sample_waypoint_near_point_same_floor(
                            pf, rngs[i], center_xyz=positions[i], radius=8.0, y_tol=SAME_FLOOR_Y_TOL
                        )

                    if p is None:
                        p = positions[i].copy()

                    snapped_pose = pf.snap_point(mn.Vector3(float(p[0]), float(p[1]), float(p[2])))
                    p = np.array([float(snapped_pose.x), float(snapped_pose.y), float(snapped_pose.z)], dtype=np.float32)

                    positions[i] = p
                    yaws[i] = rngs[i].uniform(-math.pi, math.pi)

                    wp_same_floor = sample_waypoint_near_point_same_floor(
                        pf, rngs[i], center_xyz=positions[i], radius=10.0, y_tol=SAME_FLOOR_Y_TOL
                    )
                    if wp_same_floor is None:
                        wp_same_floor = sample_waypoint_global(pf, rngs[i], positions[i], visited[i])

                    waypoints[i] = wp_same_floor
                    wp_timers[i] = rngs[i].randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS)
                    stucks[i] = 0

            for i in range(NUM_ROBOTS):
                x, y, z = positions[i]
                yaw = yaws[i]

                snapped_pose = pf.snap_point(mn.Vector3(float(x), float(y), float(z)))
                x = float(snapped_pose.x)
                y = float(snapped_pose.y)
                z = float(snapped_pose.z)
                positions[i] = np.array([x, y, z], dtype=np.float32)

                locobots[i].translation = mn.Vector3(x, y, z)
                locobots[i].rotation = mn.Quaternion.rotation(mn.Rad(float(yaw)), mn.Vector3(0, 1, 0))

            for i in range(NUM_ROBOTS):
                x, y, z = positions[i]
                pos_hists[i].append((float(x), float(z)))
                if len(pos_hists[i]) > (CAM_HEADING_LOOKBACK + 2):
                    pos_hists[i].pop(0)

                desired_cam_yaw = yaws[i]
                if CAM_USE_VELOCITY_HEADING:
                    desired_cam_yaw = velocity_heading(pos_hists[i], fallback_yaw=yaws[i])

                cam_yaws[i] = limit_angle_rate(
                    cam_yaws[i], desired_cam_yaw,
                    math.radians(CAM_MAX_YAW_RATE_DEG_PER_S), DT
                )
                cam_yaws[i] = ema_angle(cam_yaws[i], desired_cam_yaw, CAM_YAW_EMA_ALPHA)

                if CAM_SMOOTH_POS:
                    cam_positions[i] = ema_vec3(cam_positions[i], positions[i], CAM_POS_EMA_ALPHA)
                    pcam = cam_positions[i]
                else:
                    pcam = positions[i]

                state = agents[i].get_state()
                state.position = pcam
                half = cam_yaws[i] / 2.0
                state.rotation = np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float32)
                agents[i].set_state(state)

            sim.step_physics(DT)

            rgb_frames = []
            sem_frames = []
            dep_frames = []
            for i in range(NUM_ROBOTS):
                obs = sim.get_sensor_observations(i)
                rgb = obs[f"agent{i}_rgb"]
                sem = obs[f"agent{i}_sem"]
                dep = obs[f"agent{i}_depth"]

                if rgb.shape[2] == 4:
                    rgb = rgb[:, :, :3]

                rgb = rgb[::-1, ...]
                sem = sem[::-1, ...]
                dep = dep[::-1, ...]

                rgb_frames.append(rgb)
                sem_frames.append(sem)
                dep_frames.append(dep)

            if WALL_AVOID_ENABLED:
                for i in range(NUM_ROBOTS):
                    waypoints[i] = maybe_shift_waypoint_from_wall(
                        dep_frames[i], rngs[i], positions[i], waypoints[i], visited[i]
                    )

            if SAVE_VIDEO and (t % max(1, SAVE_EVERY) == 0):
                for i in range(NUM_ROBOTS):
                    rgb_writers[i].append_data(rgb_frames[i])
                    sem_writers[i].append_data(colorize_semantic(sem_frames[i]))
                    depth_writers[i].append_data(depth_to_uint8(dep_frames[i], DEPTH_VIZ_MAX_M))

            for i in range(NUM_ROBOTS):
                xi, yi, zi = positions[i]
                yawi = yaws[i]
                for j in range(NUM_ROBOTS):
                    if i == j:
                        continue
                    xj, yj, zj = positions[j]
                    yawj = yaws[j]
                    rel_x, rel_z, rel_yaw = relative_pose_2d(xi, zi, yawi, xj, zj, yawj)
                    csv_writer.writerow([
                        t, i + 1, j + 1,
                        xi, yi, zi, yawi,
                        rel_x, 0.0, rel_z, rel_yaw,
                        v_cmds[i], w_cmds[i]
                    ])

            for i in range(NUM_ROBOTS):
                visible_ids = None
                if NEARBY_REQUIRE_VISIBLE:
                    visible_ids = get_visible_object_ids(
                        sem_frames[i], VISIBLE_MIN_PIXELS, VISIBLE_DOWNSAMPLE_STRIDE
                    )

                near_items = get_nearby_semantic_objects(
                    sim, positions[i], yaws[i],
                    NEARBY_RADIUS_M, NEARBY_TOPK, NEARBY_USE_XZ_ONLY, visible_ids
                )

                for item in near_items:
                    sid = int(item["semantic_instance_id"])
                    cat = id_to_category.get(sid, item["category"])
                    lbl = id_to_label_txt.get(sid, "unknown")
                    nearby_writer.writerow([
                        t, i + 1, sid, item["object_key"], cat, lbl,
                        item["dist_m"], item["bearing_rad"], item["bearing_deg"],
                        item["center_x"], item["center_y"], item["center_z"]
                    ])

            for i in range(NUM_ROBOTS):
                x, _, z = positions[i]
                trajectories[i].append((x, z))

            if BEV_ENABLE and (t % max(1, BEV_EVERY) == 0):
                fig = plt.figure(figsize=(6, 6), dpi=150)
                ax = fig.add_subplot(111)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(zmin, zmax)
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"BEV Trajectories (x-z)\n{scene_name} | {exploration_id}")

                for i in range(NUM_ROBOTS):
                    arr = np.array(trajectories[i], dtype=np.float32)
                    if len(arr) > 0:
                        ax.plot(arr[:, 0], arr[:, 1], linewidth=2, label=f"robot{i+1}")
                        ax.scatter([positions[i][0]], [positions[i][2]], s=30)

                ax.legend()
                canvas = FigureCanvas(fig)
                canvas.draw()
                wfig, hfig = fig.canvas.get_width_height()
                bev_frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(hfig, wfig, 3)
                plt.close(fig)
                bev_writer.append_data(bev_frame)

            if (t + 1) % 200 == 0:
                center = np.mean(np.stack(positions, axis=0), axis=0)
                dists = []
                for i in range(NUM_ROBOTS):
                    dx = float(positions[i][0] - center[0])
                    dz = float(positions[i][2] - center[2])
                    dists.append(math.sqrt(dx * dx + dz * dz))
                vis_counts = [len(v) for v in visited]
                ys = [round(float(p[1]), 3) for p in positions]
                print(
                    f"    [{t+1}/{NUM_STEPS}] done | "
                    f"meet_active={meet_active} | "
                    f"dist_to_center={[round(d, 2) for d in dists]} | "
                    f"visited={vis_counts} | "
                    f"ys={ys}"
                )

        with open(os.path.join(out_dir, "DONE"), "w") as f:
            f.write("ok\n")
        return True

    finally:
        try:
            for w in rgb_writers:
                if w is not None:
                    w.close()
            for w in sem_writers:
                if w is not None:
                    w.close()
            for w in depth_writers:
                if w is not None:
                    w.close()
            if bev_writer is not None:
                bev_writer.close()
        except Exception:
            pass

        csv_file.close()
        nearby_file.close()
        sim.close()


# ===================== Main =====================

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    scenes = find_scenes_with_semantics(SCENES_ROOT)
    print(f"Found {len(scenes)} scenes with *.semantic.glb under: {SCENES_ROOT}")
    if len(scenes) == 0:
        print("Nothing to do.")
        return

    for sidx, scene_info in enumerate(scenes):
        scene_name = scene_info["scene_name"]
        done_count = get_done_count_for_scene(scene_name)

        if done_count >= NUM_EXPLORATIONS_PER_SCENE + 100:
            print(f"\n[{sidx+1}/{len(scenes)}] Scene: {scene_name}  [SKIP] finished")
            continue

        if scene_name in EXPLORED_LIST:
            continue

        print(f"\n[{sidx+1}/{len(scenes)}] Scene: {scene_name}")
        print(f"  basis: {scene_info['basis_glb']}")
        print(f"  done : {done_count}/{NUM_EXPLORATIONS_PER_SCENE}")

        start_idx = 100
        for explore_idx in range(start_idx, NUM_EXPLORATIONS_PER_SCENE + 100):
            print(f"  -> exploration {explore_idx:03d}")
            success = run_one_exploration(scene_info, explore_idx)
            if not success:
                print(f"  [STOP] {scene_name} semantic load failed.")
                break

    print("\nDone! Outputs in:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()