import os
import csv
import math
import glob
import random
import uuid
import numpy as np
import imageio
from typing import Set
import habitat_sim
import magnum as mn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from habitat.config.default import get_config
from habitat.core.dataset import Dataset
from typing import Optional, Set, List, Tuple


# ===================== Paths & Params =====================

DATA_PATH = "/cvhci/temp/kpeng/habitat/"  # <--- change to yours

HM3D_DATASET_CFG = "/media/kpeng/Elements/DATA/HM3D/versioned_data/hm3d-0.2/hm3d/train/hm3d_annotated_train_basis.scene_dataset_config.json"
SCENES_ROOT = "/media/kpeng/Elements/DATA/HM3D/versioned_data/hm3d-0.2/hm3d/train"
OUTPUT_ROOT = "/media/kpeng/Elements/DATA/data_collection/two_robot_retry"

NUM_EXPLORATIONS_PER_SCENE = 15
NUM_STEPS = 1000
DT = 0.1

# Motion (稍快一点更容易走开)
FORWARD_SPEED = 0.32
ROT_SPEED = math.radians(28)

# Camera mount
CAM_HEIGHT = -1.2

# ===================== SPEED PRESET (提速核心) =====================

# 传感器分辨率（强烈建议：比 480x640 快很多）
SENSOR_RESOLUTION = [1080, 1080]  # H, W

# 视频写入：每 N 步写一帧（例如 SAVE_EVERY=2 => 10Hz -> 5Hz）
SAVE_VIDEO = True
SAVE_EVERY = 2

# BEV：每 N 步画一帧（Matplotlib 很慢，稀疏画可以数量级提速）
BEV_ENABLE = True
BEV_EVERY = 10

# Nearby visible：下采样更粗，降低 np.unique 成本
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


# ===================== Start: MUST same region + same floor =====================

START_MUST_SAME_REGION = True
START_REGION_MAX_TRIES = 800
START_POINT_MAX_TRIES = 800

# “同层”约束：用 y 高度接近判断（低成本）
SAME_FLOOR_Y_TOL = 0.6   # meters
START_P2_Y_TRIES = 600

# 起始朝向更一致 => 起始视野更重叠
START_YAW_ALIGN_DEG = 8.0     # robot2 yaw = robot1 yaw + jitter
START_YAW_JITTER_DEG = 10.0   # robot1 yaw jitter itself


# ===================== Coverage bias (explore whole scene) =====================

COV_CELL_M = 1.25
COV_REJECT_VISITED_PROB = 0.75
COV_KEEP_FAR_PROB = 0.35

WAYPOINT_MIN_STEPS = 35
WAYPOINT_MAX_STEPS = 130
WAYPOINT_MIN_DIST_M = 2.0
WAYPOINT_MAX_DIST_M = 30.0


# ===================== Randomness (keep random but less jitter) =====================

WP_YAW_GAIN = 1.35
WP_FORWARD_BASE = 0.95

NOISE_YAW_STD_RAD = math.radians(1.8)
NOISE_V_STD = 0.04

WIGGLE_PROB_PER_STEP = 0.018
WIGGLE_MIN_STEPS = 6
WIGGLE_MAX_STEPS = 16
WIGGLE_TURN_MAG = math.radians(30)
WIGGLE_FORWARD_SCALE = 0.78


# ===================== Wall-avoid using depth (reduce facing wall) =====================

WALL_AVOID_ENABLED = True
WALL_FRONT_TH_M = 0.75
WALL_FRONT_TH_M2 = 1.10

DEP_STRIDE = 6
FRONT_BAND = (0.40, 0.60)
SIDE_BAND = (0.15, 0.35)

WALL_TURN_BOOST = math.radians(18)
WALL_SLOWDOWN = 0.75
WALL_MIN_MOVE = 0.08


# ===================== Overlap guarantee (two robots FOV overlap sometimes) =====================

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


# ===================== Camera stabilization (reduce 左右摇晃) =====================

CAM_USE_VELOCITY_HEADING = True
CAM_HEADING_LOOKBACK = 3

CAM_YAW_EMA_ALPHA = 0.93
CAM_MAX_YAW_RATE_DEG_PER_S = 60.0

CAM_SMOOTH_POS = True
CAM_POS_EMA_ALPHA = 0.85



EXPLORED_SCENE_IDS: Set[str] = {
    "1UnKg1rAb8A","5Kw4nGdqYtS","ACZZiU6BXLz","DoSbsoo4EAg","g7hUFVNac26","h6nwVLpAKQz",
    "iLDo95ZbDJq","kJxT5qssH4H","NGyoyh91xXJ","PE6kVEtrxtj","RaYrxWt5pR1","v7DzfFFEpsD",
    "wPLokgvCnuk","XYyR54sxe6b","1xGrZPxG1Hz","6HRFAUDqpTb","b3WpMbPFB6q","DqJKU7YU7dA",
    "g8Xrdbe9fir","H8rQCnvBgo6","ixTj1aTMup2","L5QEsaVqwrY","NPHxDe6VeCc","PPTLa8SkUfo",
    "S7uMvxjBVZq","vDfkYo5VqEQ","wsAYBFtQaL7","yHLr6bvWsVm","226REUyJh2K","6imZUJGRUq4",
    "bB6nKqfsb1z","dQrLTxHvLXU","GGBvSFddQgs","HeSYRw7eMtG","j2EJhFEQGCL","LcAd9dhvVwh",
    "NtnvZSMK3en","qk9eeNeR4vw","sX9xad6ULKc","vLpv2VX547B","X6Pct1msZv5","YMNvYDhK8mB",
    "2Pc8W48bu21","741Fdj7NLF9","bdp1XNEdvmW","erXNfWVjqZ8","gmuS7Wgsbrx","HkseAnWCgqk",
    "j6fHrce9pHR","MVVzj944atG","oEPjPNSPmzL","QN2dRqwd84J","U3oQjwTuMX8","VoVGtfYrpuQ",
    "xAHnY3QzFUN","YY8rqV6L6rf","3XYAD64HpDr","77mMEyxhs44","bHKTDQFJxTw","fK2vEV32Lag",
    "gQ3xxshDiCz","HxmXPBbFCkH","Jfyvj3xn2aJ","nACV8wLu1u5","oPj9qMxrDEa","QVAA6zecMHu",
    "u9rPN5cHWBg","VSxVP19Cdyw","XfUxBGTFQQb","zepmXAdrpjR","4vwGX7U38Ux","8wJuSPJ9FXG",
    "ceJTwFNjqCt","FRQ75PjD278","gQgtJ9Stk5s","iigzG1rtanx","JNiWU5TZLtt","NEVASPhcrxR",
    "oStKKWkQ1id","qz3829g1Lzf","URjpCob8MGw","WhNyDTnd9g5","xgLmjqzoAzF","ZNanfzgCdm3",
    "5biL7VEkByM","92vYG1q49FY","CQWES1bawee","fxbzYAGkrtm","GTV2Y73Sn5t","iKFn6fzyRqs",
    "kA2nG18hCAr","nGhNxKrgBPb","pcpn6mFqFCg","qZ4B7U6XE5Y","UuwwmrTsfBN","Wo6kuutE9i7",
    "xWvSkKiWQpC",
}

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


def scene_id_from_episode(ep) -> str:
    """
    Habitat episodes often store full paths like:
      data/scene_datasets/mp3d/<scene_id>/<scene_id>.glb
    We extract the <scene_id> robustly.
    """
    sid = getattr(ep, "scene_id", "")
    # handle paths ending with .../<scene_id>/<scene_id>.glb
    parts = sid.replace("\\", "/").split("/")
    # try to find the folder name that matches the basename
    if len(parts) >= 2:
        # common: .../<scene_id>/<scene_id>.glb
        candidate = parts[-2]
        if candidate:
            return candidate
    # fallback: filename without extension
    base = parts[-1]
    return base.split(".")[0] if base else sid

def filter_dataset_episodes(dataset: Dataset, excluded_scene_ids: Set[str]) -> Dataset:
    kept = []
    for ep in dataset.episodes:
        sid = scene_id_from_episode(ep)
        if sid not in excluded_scene_ids:
            kept.append(ep)
    dataset.episodes = kept
    return dataset
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
    else:
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


# ===================== Region helpers (robust AABB) =====================

def _aabb_min_max(aabb):
    if hasattr(aabb, "min") and hasattr(aabb, "max"):
        return (np.array(vec3_to_xyz(aabb.min), dtype=np.float32),
                np.array(vec3_to_xyz(aabb.max), dtype=np.float32))
    if hasattr(aabb, "min_corner") and hasattr(aabb, "max_corner"):
        return (np.array(vec3_to_xyz(aabb.min_corner), dtype=np.float32),
                np.array(vec3_to_xyz(aabb.max_corner), dtype=np.float32))
    if hasattr(aabb, "back_bottom_left") and hasattr(aabb, "front_top_right"):
        return (np.array(vec3_to_xyz(aabb.back_bottom_left), dtype=np.float32),
                np.array(vec3_to_xyz(aabb.front_top_right), dtype=np.float32))
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

def make_sim(scene_path: str):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = HM3D_DATASET_CFG
    sim_cfg.scene_id = scene_path
    sim_cfg.load_semantic_mesh = True
    sim_cfg.enable_physics = True

    agent_cfg_0 = habitat_sim.agent.AgentConfiguration()
    agent_cfg_1 = habitat_sim.agent.AgentConfiguration()

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

    agent_cfg_0.sensor_specifications = make_sensors("agent0")
    agent_cfg_1.sensor_specifications = make_sensors("agent1")
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg_0, agent_cfg_1])
    return habitat_sim.Simulator(cfg)


# ===================== Nearby objects =====================

def get_nearby_semantic_objects(sim: habitat_sim.Simulator,
                               agent_pos_xyz: np.ndarray,
                               agent_yaw: float,
                               radius_m: float = 3.0,
                               topk: int = 10,
                               use_xz_only: bool = True,
                               visible_ids: Optional[Set[int]] = None):
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


# ===================== Waypoint sampling (global + coverage bias) =====================

def sample_waypoint_global(pf,
                           rng: random.Random,
                           origin_xyz: np.ndarray,
                           visited_cells: Set[Tuple[int, int]]):
    allow_far = (rng.random() < COV_KEEP_FAR_PROB)

    for _ in range(900):
        p = np.array(pf.get_random_navigable_point(), dtype=np.float32)

        dx = float(p[0] - origin_xyz[0])
        dz = float(p[2] - origin_xyz[2])
        d = math.sqrt(dx*dx + dz*dz)

        if d < WAYPOINT_MIN_DIST_M:
            continue

        if (not allow_far) and d > WAYPOINT_MAX_DIST_M:
            if rng.random() < 0.85:
                continue

        key = cov_cell_key(float(p[0]), float(p[2]), COV_CELL_M)
        if key in visited_cells and rng.random() < COV_REJECT_VISITED_PROB:
            continue

        return p

    return np.array(pf.get_random_navigable_point(), dtype=np.float32)

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
    return np.array(pf.get_random_navigable_point(), dtype=np.float32)

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

def first_missing_seq_exploration(scene_name: str) -> int:
    for i in range(NUM_EXPLORATIONS_PER_SCENE):
        exploration_id = f"explore_{i:03d}"
        out_dir = os.path.join(OUTPUT_ROOT, scene_name, exploration_id)
        if not os.path.exists(exploration_done_marker_path(out_dir)):
            return i
    return NUM_EXPLORATIONS_PER_SCENE


# ===================== Simulator runner =====================

def make_exploration_id(scene_name: str, explore_idx: int):
    if EXPLORATION_ID_MODE.lower() == "uuid":
        return f"explore_{uuid.uuid4().hex[:8]}"
    return f"explore_{explore_idx:03d}"

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

    sim = make_sim(scene_path)
    if sim.semantic_scene is None or sim.semantic_scene.objects is None or len(sim.semantic_scene.objects) == 0:
        sim.close()
        print(f"[SKIP] {scene_name}: semantic_scene missing/empty.")
        return False

    id_to_category = build_semantic_id_to_category(sim)

    agent0 = sim.get_agent(0)
    agent1 = sim.get_agent(1)

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

    locobot1 = obj_mgr.add_object_by_template_handle(locobot_handle)
    locobot2 = obj_mgr.add_object_by_template_handle(locobot_handle)
    locobot1.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    locobot2.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    # RNG streams (保证每次 exploration 起点不同)
    seed_base = (hash(scene_name) ^ (explore_idx * 1000003) ^ 0x9E3779B9) & 0xFFFFFFFF
    rng_global = random.Random(seed_base)
    rng1 = random.Random((seed_base + 101) & 0xFFFFFFFF)
    rng2 = random.Random((seed_base + 202) & 0xFFFFFFFF)

    # ===================== Start: MUST same region + same floor =====================
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
        print("    [WARN] cannot enforce same region start (no valid regions). fallback to random nav points.")

    # p1 in start_region
    p1 = sample_navigable_point_in_region(pf, start_region, max_tries=START_POINT_MAX_TRIES) if start_region is not None else None
    if p1 is None:
        p1 = np.array(pf.get_random_navigable_point(), dtype=np.float32)

    p1y = float(p1[1])

    # p2 ALSO in same start_region, AND same floor by y-close
    p2 = None
    if start_region is not None:
        for _ in range(START_P2_Y_TRIES):
            cand = sample_navigable_point_in_region(pf, start_region, max_tries=80)
            if cand is None:
                continue
            if abs(float(cand[1]) - p1y) <= SAME_FLOOR_Y_TOL:
                p2 = cand
                break

        if p2 is None:
            # fallback: still in same region (no y constraint)
            p2 = sample_navigable_point_in_region(pf, start_region, max_tries=START_POINT_MAX_TRIES)

    if p2 is None:
        # last fallback
        p2 = np.array(pf.get_random_navigable_point(), dtype=np.float32)

    # yaw: keep aligned-ish for initial overlap
    yaw1 = rng1.uniform(-math.pi, math.pi)
    yaw1 = wrap_angle(yaw1 + math.radians(rng1.uniform(-START_YAW_JITTER_DEG, START_YAW_JITTER_DEG)))
    yaw2 = wrap_angle(yaw1 + math.radians(rng2.uniform(-START_YAW_ALIGN_DEG, START_YAW_ALIGN_DEG)))

    x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2])
    x2, y2, z2 = float(p2[0]), float(p2[1]), float(p2[2])

    # ===================== Writers (fps matched to subsampling) =====================
    fps = int(1.0 / DT)

    save_fps = max(1, fps // max(1, SAVE_EVERY))
    bev_fps = max(1, fps // max(1, BEV_EVERY))

    # video writers
    if SAVE_VIDEO:
        video_writer_1 = imageio.get_writer(os.path.join(out_dir, "locobot1_cam.mp4"), fps=save_fps)
        video_writer_2 = imageio.get_writer(os.path.join(out_dir, "locobot2_cam.mp4"), fps=save_fps)
        sem_writer_1 = imageio.get_writer(os.path.join(out_dir, "locobot1_sem.mp4"), fps=save_fps)
        sem_writer_2 = imageio.get_writer(os.path.join(out_dir, "locobot2_sem.mp4"), fps=save_fps)
        depth_writer_1 = imageio.get_writer(os.path.join(out_dir, "locobot1_depth.mp4"), fps=save_fps)
        depth_writer_2 = imageio.get_writer(os.path.join(out_dir, "locobot2_depth.mp4"), fps=save_fps)
    else:
        video_writer_1 = video_writer_2 = None
        sem_writer_1 = sem_writer_2 = None
        depth_writer_1 = depth_writer_2 = None

    # BEV writer
    if BEV_ENABLE:
        bev_writer = imageio.get_writer(os.path.join(out_dir, "bev_trajectories.mp4"), fps=bev_fps)
    else:
        bev_writer = None

    # CSV pose/cmd
    csv_path = os.path.join(out_dir, "locobot_trajectory_and_cmds.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "t", "robot_id",
        "x", "y", "z", "yaw",
        "rel_x", "rel_y", "rel_z", "rel_yaw",
        "v_cmd", "omega_cmd",
    ])

    # CSV nearby
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

    # ===================== Nav step =====================
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

        # side-step
        for sign in (+1, -1):
            side_yaw = wrap_angle(yaw_base + sign * math.pi / 2.0)
            x_new = x + NAV_SIDE_DIST * math.cos(side_yaw)
            z_new = z + NAV_SIDE_DIST * math.sin(side_yaw)
            snapped = pf.snap_point(mn.Vector3(x_new, y, z_new))
            if pf.is_navigable(snapped):
                return float(snapped.x), float(snapped.y), float(snapped.z), yaw_base

        # small backup
        x_back = x - NAV_BACKUP_DIST * math.cos(yaw_base)
        z_back = z - NAV_BACKUP_DIST * math.sin(yaw_base)
        snapped = pf.snap_point(mn.Vector3(x_back, y, z_back))
        if pf.is_navigable(snapped):
            return float(snapped.x), float(snapped.y), float(snapped.z), yaw_base

        return x, y, z, yaw_base

    # ===================== Policy state =====================

    visited1: Set[Tuple[int, int]] = set()
    visited2: Set[Tuple[int, int]] = set()

    wp1 = sample_waypoint_global(pf, rng1, np.array([x1, y1, z1], dtype=np.float32), visited1)
    wp2 = sample_waypoint_global(pf, rng2, np.array([x2, y2, z2], dtype=np.float32), visited2)
    wp_timer1 = rng1.randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS)
    wp_timer2 = rng2.randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS)

    wiggle1 = 0
    wiggle2 = 0

    stuck1 = 0
    stuck2 = 0
    prev_x1, prev_z1 = x1, z1
    prev_x2, prev_z2 = x2, z2

    # meet / overlap schedule
    meet_timer = MEET_EVERY_STEPS
    meet_active = 0
    meet_wp = None

    # camera stabilization vars
    pos_hist1 = [(x1, z1)]
    pos_hist2 = [(x2, z2)]
    cam_yaw1 = yaw1
    cam_yaw2 = yaw2
    cam_pos1 = np.array([x1, y1, z1], dtype=np.float32)
    cam_pos2 = np.array([x2, y2, z2], dtype=np.float32)

    traj1, traj2 = [], []

    try:
        for t in range(NUM_STEPS):
            # record coverage
            visited1.add(cov_cell_key(x1, z1, COV_CELL_M))
            visited2.add(cov_cell_key(x2, z2, COV_CELL_M))

            # ===================== overlap rendezvous scheduler =====================
            if meet_active <= 0:
                meet_timer -= 1
                if meet_timer <= 0:
                    meet_active = MEET_DURATION_STEPS
                    meet_timer = MEET_EVERY_STEPS
                    mid = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, (z1 + z2) * 0.5], dtype=np.float32)
                    meet_wp = sample_waypoint_near_point(pf, rng_global, mid, radius=MEET_TARGET_RADIUS_M * 2.5)

            # ===================== waypoint updates =====================
            wp_timer1 -= 1
            wp_timer2 -= 1

            dx12 = x2 - x1
            dz12 = z2 - z1
            dist12 = math.sqrt(dx12*dx12 + dz12*dz12)

            tether1 = False
            tether2 = False
            if TETHER_ENABLED and dist12 > TETHER_MAX_DIST:
                if rng_global.random() < TETHER_PULL_PROB:
                    if rng_global.random() < 0.5:
                        tether1 = True
                    else:
                        tether2 = True

            if meet_active > 0 and meet_wp is not None:
                wp1 = meet_wp
                wp2 = meet_wp
                wp_timer1 = 10
                wp_timer2 = 10
                meet_active -= 1
            else:
                if wp_timer1 <= 0 or tether1:
                    if tether1:
                        wp1 = sample_waypoint_near_point(pf, rng1, np.array([x2, y2, z2], dtype=np.float32), radius=TETHER_PULL_DIST)
                    else:
                        wp1 = sample_waypoint_global(pf, rng1, np.array([x1, y1, z1], dtype=np.float32), visited1)
                    wp_timer1 = rng1.randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS)

                if wp_timer2 <= 0 or tether2:
                    if tether2:
                        wp2 = sample_waypoint_near_point(pf, rng2, np.array([x1, y1, z1], dtype=np.float32), radius=TETHER_PULL_DIST)
                    else:
                        wp2 = sample_waypoint_global(pf, rng2, np.array([x2, y2, z2], dtype=np.float32), visited2)
                    wp_timer2 = rng2.randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS)

            # ===================== wiggle episodes =====================
            if wiggle1 <= 0 and rng1.random() < WIGGLE_PROB_PER_STEP:
                wiggle1 = rng1.randint(WIGGLE_MIN_STEPS, WIGGLE_MAX_STEPS)
            if wiggle2 <= 0 and rng2.random() < WIGGLE_PROB_PER_STEP:
                wiggle2 = rng2.randint(WIGGLE_MIN_STEPS, WIGGLE_MAX_STEPS)

            # ===================== compute cmds =====================
            v1, w1 = compute_cmd_to_waypoint(x1, y1, z1, yaw1, wp1, rng1)
            v2, w2 = compute_cmd_to_waypoint(x2, y2, z2, yaw2, wp2, rng2)

            if wiggle1 > 0:
                w1 += rng1.uniform(-WIGGLE_TURN_MAG, WIGGLE_TURN_MAG) / DT
                v1 *= WIGGLE_FORWARD_SCALE
                wiggle1 -= 1
            if wiggle2 > 0:
                w2 += rng2.uniform(-WIGGLE_TURN_MAG, WIGGLE_TURN_MAG) / DT
                v2 *= WIGGLE_FORWARD_SCALE
                wiggle2 -= 1

            v1 = max(0.0, min(FORWARD_SPEED, v1))
            v2 = max(0.0, min(FORWARD_SPEED, v2))
            w1 = max(-ROT_SPEED, min(ROT_SPEED, w1))
            w2 = max(-ROT_SPEED, min(ROT_SPEED, w2))

            # ===================== nav update =====================
            x1, y1, z1, yaw1 = nav_constrained_step(x1, y1, z1, yaw1, v1, w1)
            x2, y2, z2, yaw2 = nav_constrained_step(x2, y2, z2, yaw2, v2, w2)

            # stuck detect
            if (x1 - prev_x1) ** 2 + (z1 - prev_z1) ** 2 < STUCK_EPS_MOVED_SQ:
                stuck1 += 1
            else:
                stuck1 = 0
            if (x2 - prev_x2) ** 2 + (z2 - prev_z2) ** 2 < STUCK_EPS_MOVED_SQ:
                stuck2 += 1
            else:
                stuck2 = 0

            prev_x1, prev_z1 = x1, z1
            prev_x2, prev_z2 = x2, z2

            # soft stuck: resample global waypoint
            if stuck1 > STUCK_SOFT_LIMIT:
                wp1 = sample_waypoint_global(pf, rng1, np.array([x1, y1, z1], dtype=np.float32), visited1)
                wp_timer1 = rng1.randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS)
                wiggle1 = rng1.randint(WIGGLE_MIN_STEPS, WIGGLE_MAX_STEPS)
                stuck1 = 0

            if stuck2 > STUCK_SOFT_LIMIT:
                wp2 = sample_waypoint_global(pf, rng2, np.array([x2, y2, z2], dtype=np.float32), visited2)
                wp_timer2 = rng2.randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS)
                wiggle2 = rng2.randint(WIGGLE_MIN_STEPS, WIGGLE_MAX_STEPS)
                stuck2 = 0

            # hard stuck teleport
            if HARD_STUCK_TELEPORT and stuck1 > STUCK_HARD_LIMIT:
                p = np.array(pf.get_random_navigable_point(), dtype=np.float32)
                x1, y1, z1 = float(p[0]), float(p[1]), float(p[2])
                yaw1 = rng1.uniform(-math.pi, math.pi)
                wp1 = sample_waypoint_global(pf, rng1, np.array([x1, y1, z1], dtype=np.float32), visited1)
                wp_timer1 = rng1.randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS)
                stuck1 = 0

            if HARD_STUCK_TELEPORT and stuck2 > STUCK_HARD_LIMIT:
                p = np.array(pf.get_random_navigable_point(), dtype=np.float32)
                x2, y2, z2 = float(p[0]), float(p[1]), float(p[2])
                yaw2 = rng2.uniform(-math.pi, math.pi)
                wp2 = sample_waypoint_global(pf, rng2, np.array([x2, y2, z2], dtype=np.float32), visited2)
                wp_timer2 = rng2.randint(WAYPOINT_MIN_STEPS, WAYPOINT_MAX_STEPS)
                stuck2 = 0

            # write object pose
            locobot1.translation = mn.Vector3(x1, y1, z1)
            locobot1.rotation = mn.Quaternion.rotation(mn.Rad(yaw1), mn.Vector3(0, 1, 0))
            locobot2.translation = mn.Vector3(x2, y2, z2)
            locobot2.rotation = mn.Quaternion.rotation(mn.Rad(yaw2), mn.Vector3(0, 1, 0))

            # ===================== cameras: stabilized yaw =====================
            pos_hist1.append((x1, z1))
            pos_hist2.append((x2, z2))
            if len(pos_hist1) > (CAM_HEADING_LOOKBACK + 2):
                pos_hist1.pop(0)
            if len(pos_hist2) > (CAM_HEADING_LOOKBACK + 2):
                pos_hist2.pop(0)

            def velocity_heading(pos_hist, fallback_yaw):
                if len(pos_hist) < (CAM_HEADING_LOOKBACK + 1):
                    return fallback_yaw
                x_old, z_old = pos_hist[0]
                x_new, z_new = pos_hist[-1]
                dx = x_new - x_old
                dz = z_new - z_old
                if dx*dx + dz*dz < 1e-8:
                    return fallback_yaw
                return math.atan2(dz, dx)

            desired_cam_yaw1 = yaw1
            desired_cam_yaw2 = yaw2
            if CAM_USE_VELOCITY_HEADING:
                desired_cam_yaw1 = velocity_heading(pos_hist1, fallback_yaw=yaw1)
                desired_cam_yaw2 = velocity_heading(pos_hist2, fallback_yaw=yaw2)

            cam_yaw1 = limit_angle_rate(cam_yaw1, desired_cam_yaw1, math.radians(CAM_MAX_YAW_RATE_DEG_PER_S), DT)
            cam_yaw2 = limit_angle_rate(cam_yaw2, desired_cam_yaw2, math.radians(CAM_MAX_YAW_RATE_DEG_PER_S), DT)
            cam_yaw1 = ema_angle(cam_yaw1, desired_cam_yaw1, CAM_YAW_EMA_ALPHA)
            cam_yaw2 = ema_angle(cam_yaw2, desired_cam_yaw2, CAM_YAW_EMA_ALPHA)

            if CAM_SMOOTH_POS:
                cam_pos1 = ema_vec3(cam_pos1, np.array([x1, y1, z1], dtype=np.float32), CAM_POS_EMA_ALPHA)
                cam_pos2 = ema_vec3(cam_pos2, np.array([x2, y2, z2], dtype=np.float32), CAM_POS_EMA_ALPHA)
                pcam1 = cam_pos1
                pcam2 = cam_pos2
            else:
                pcam1 = np.array([x1, y1, z1], dtype=np.float32)
                pcam2 = np.array([x2, y2, z2], dtype=np.float32)

            state0 = agent0.get_state()
            state0.position = pcam1
            half = cam_yaw1 / 2.0
            state0.rotation = np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float32)
            agent0.set_state(state0)

            state1 = agent1.get_state()
            state1.position = pcam2
            half = cam_yaw2 / 2.0
            state1.rotation = np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float32)
            agent1.set_state(state1)

            # step physics
            sim.step_physics(DT)

            # observations
            obs0 = sim.get_sensor_observations(0)
            rgb1 = obs0["agent0_rgb"]
            sem1 = obs0["agent0_sem"]
            dep1 = obs0["agent0_depth"]

            obs1 = sim.get_sensor_observations(1)
            rgb2 = obs1["agent1_rgb"]
            sem2 = obs1["agent1_sem"]
            dep2 = obs1["agent1_depth"]

            # flip + strip alpha
            if rgb1.shape[2] == 4:
                rgb1 = rgb1[:, :, :3]
            if rgb2.shape[2] == 4:
                rgb2 = rgb2[:, :, :3]

            rgb1 = rgb1[::-1, ...]
            rgb2 = rgb2[::-1, ...]
            sem1 = sem1[::-1, ...]
            sem2 = sem2[::-1, ...]
            dep1 = dep1[::-1, ...]
            dep2 = dep2[::-1, ...]

            # ===================== Wall-avoid (cheap waypoint shift) =====================
            if WALL_AVOID_ENABLED:
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

                    # 在当前位置附近采一个点（更快摆脱墙）
                    center = np.array([cur_xyz[0], cur_xyz[1], cur_xyz[2]], dtype=np.float32)
                    wp_new = sample_waypoint_near_point(pf, rng, center, radius=6.0)

                    key = cov_cell_key(float(wp_new[0]), float(wp_new[2]), COV_CELL_M)
                    if key in visited and rng.random() < 0.6:
                        wp_new = sample_waypoint_global(pf, rng, center, visited)

                    return wp_new

                wp1 = maybe_shift_waypoint_from_wall(dep1, rng1, np.array([x1, y1, z1], dtype=np.float32), wp1, visited1)
                wp2 = maybe_shift_waypoint_from_wall(dep2, rng2, np.array([x2, y2, z2], dtype=np.float32), wp2, visited2)

            # ===================== Write videos (subsample) =====================
            if SAVE_VIDEO and (t % max(1, SAVE_EVERY) == 0):
                video_writer_1.append_data(rgb1)
                video_writer_2.append_data(rgb2)
                sem_writer_1.append_data(colorize_semantic(sem1))
                sem_writer_2.append_data(colorize_semantic(sem2))
                depth_writer_1.append_data(depth_to_uint8(dep1, DEPTH_VIZ_MAX_M))
                depth_writer_2.append_data(depth_to_uint8(dep2, DEPTH_VIZ_MAX_M))

            # CSV relative pose (keep every step)
            rel_x_21, rel_z_21, rel_yaw_21 = relative_pose_2d(x1, z1, yaw1, x2, z2, yaw2)
            rel_x_12, rel_z_12, rel_yaw_12 = relative_pose_2d(x2, z2, yaw2, x1, z1, yaw1)
            csv_writer.writerow([t, 1, x1, y1, z1, yaw1, rel_x_21, 0.0, rel_z_21, rel_yaw_21, v1, w1])
            csv_writer.writerow([t, 2, x2, y2, z2, yaw2, rel_x_12, 0.0, rel_z_12, rel_yaw_12, v2, w2])

            # nearby objects visible-only
            visible1 = None
            visible2 = None
            if NEARBY_REQUIRE_VISIBLE:
                visible1 = get_visible_object_ids(sem1, VISIBLE_MIN_PIXELS, VISIBLE_DOWNSAMPLE_STRIDE)
                visible2 = get_visible_object_ids(sem2, VISIBLE_MIN_PIXELS, VISIBLE_DOWNSAMPLE_STRIDE)

            near1 = get_nearby_semantic_objects(sim, np.array([x1, y1, z1], dtype=np.float32), yaw1,
                                                NEARBY_RADIUS_M, NEARBY_TOPK, NEARBY_USE_XZ_ONLY, visible1)
            near2 = get_nearby_semantic_objects(sim, np.array([x2, y2, z2], dtype=np.float32), yaw2,
                                                NEARBY_RADIUS_M, NEARBY_TOPK, NEARBY_USE_XZ_ONLY, visible2)

            for item in near1:
                sid = int(item["semantic_instance_id"])
                cat = id_to_category.get(sid, item["category"])
                lbl = id_to_label_txt.get(sid, "unknown")
                nearby_writer.writerow([t, 1, sid, item["object_key"], cat, lbl,
                                        item["dist_m"], item["bearing_rad"], item["bearing_deg"],
                                        item["center_x"], item["center_y"], item["center_z"]])

            for item in near2:
                sid = int(item["semantic_instance_id"])
                cat = id_to_category.get(sid, item["category"])
                lbl = id_to_label_txt.get(sid, "unknown")
                nearby_writer.writerow([t, 2, sid, item["object_key"], cat, lbl,
                                        item["dist_m"], item["bearing_rad"], item["bearing_deg"],
                                        item["center_x"], item["center_y"], item["center_z"]])

            # BEV (subsample)
            traj1.append((x1, z1))
            traj2.append((x2, z2))
            if BEV_ENABLE and (t % max(1, BEV_EVERY) == 0):
                fig = plt.figure(figsize=(6, 6), dpi=150)
                ax = fig.add_subplot(111)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(zmin, zmax)
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"BEV Trajectories (x-z)\n{scene_name} | {exploration_id}")

                t1_arr = np.array(traj1, dtype=np.float32)
                t2_arr = np.array(traj2, dtype=np.float32)
                ax.plot(t1_arr[:, 0], t1_arr[:, 1], linewidth=2)
                ax.plot(t2_arr[:, 0], t2_arr[:, 1], linewidth=2)
                ax.scatter([x1], [z1], s=30)
                ax.scatter([x2], [z2], s=30)

                canvas = FigureCanvas(fig)
                canvas.draw()
                wfig, hfig = fig.canvas.get_width_height()
                bev_frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(hfig, wfig, 3)
                plt.close(fig)
                bev_writer.append_data(bev_frame)

            if (t + 1) % 200 == 0:
                print(f"    [{t+1}/{NUM_STEPS}] done | dist12={dist12:.2f} | meet_active={meet_active} | visited1={len(visited1)} visited2={len(visited2)}")

        with open(os.path.join(out_dir, "DONE"), "w") as f:
            f.write("ok\n")
        return True

    finally:
        # close writers
        try:
            if video_writer_1 is not None: video_writer_1.close()
            if video_writer_2 is not None: video_writer_2.close()
            if sem_writer_1 is not None: sem_writer_1.close()
            if sem_writer_2 is not None: sem_writer_2.close()
            if depth_writer_1 is not None: depth_writer_1.close()
            if depth_writer_2 is not None: depth_writer_2.close()
            if bev_writer is not None: bev_writer.close()
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
        if done_count >= NUM_EXPLORATIONS_PER_SCENE+100:
            print(f"\n[{sidx+1}/{len(scenes)}] Scene: {scene_name}  [SKIP] finished")
            continue
        if scene_name in EXPLORED_LIST:
            continue
        print(f"\n[{sidx+1}/{len(scenes)}] Scene: {scene_name}")
        print(f"  basis: {scene_info['basis_glb']}")
        print(f"  done : {done_count}/{NUM_EXPLORATIONS_PER_SCENE}")

        #start_idx = first_missing_seq_exploration(scene_name) if EXPLORATION_ID_MODE.lower() == "seq" else 0
        start_idx = 100
        for explore_idx in range(start_idx, NUM_EXPLORATIONS_PER_SCENE + 100):
            #explore_idx = explore_idx
            print(f"  -> exploration {explore_idx:03d}")
            success = run_one_exploration(scene_info, explore_idx)
            if not success:
                print(f"  [STOP] {scene_name} semantic load failed.")
                break

    print("\nDone! Outputs in:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
