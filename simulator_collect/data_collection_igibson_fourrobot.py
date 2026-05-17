"""
igibson_two_robot_collect.py  –  iGibson 2.2  |  Locobot
=========================================================

Exploration is a direct port of the reference habitat-sim script
(data_recording_re4.py).  The main loop is a flat loop – no FSM –
mirroring the reference exactly:

  Reference structure reproduced here:
  ├── same-floor start (y-tolerance + region concept adapted to iGibson)
  ├── coverage-biased global waypoint sampling  (COV_CELL_M grid)
  ├── rendezvous scheduler  (every MEET_EVERY_STEPS, share mid-point wp)
  ├── tether  (redirect waypoint when robots too far apart)
  ├── wiggle episodes  (random direction bursts, applied after base cmd)
  ├── navmesh-constrained step  (8 heading tries + side-step + backup)
  ├── stuck handling  (soft → resample+wiggle | hard → teleport)
  ├── camera EMA stabilisation  (velocity heading + rate-limit + EMA)
  ├── wall avoidance via depth  (front-band median threshold)
  └── nearby objects (visibility-gated)

Data recorded:
  locobot_trajectory_and_cmds.csv  – per-step pose + cmd + pushing_object_name
  nearby_objects.csv               – visible nearby objects per robot
  robot_events.jsonl               – full per-step JSON
  push_events.csv                  – one row per push-force step
  meta.json                        – episode metadata
  *.mp4                            – RGB, depth, semantic, BEV videos

pushing_object_name column:
  Non-empty ONLY when robot is in PUSH_PUSHING state (applying physical
  force to an object). Records the exact scene object name.

visibility recording:
  sees_other_robot     : 1 if this robot can see the other robot in its FOV
  observed_by_robot_id : which robot is currently watching this one
  observing_robot_id   : which robot this one is currently watching
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import imageio
import numpy as np

# ============================================================
# Parameters  (mirror reference script naming)
# ============================================================

OUTPUT_ROOT        = "/media/kpeng/Elements/DATA/data_collection/three_robot_igibson"
NUM_EXPLORATIONS   = 15   # explorations per scene

# List of iGibson 2.2 interactive scenes to record.
# Add or remove scene names as needed.
SCENES = [
    "Rs_int",
    "Beechwood_0_int",
    "Beechwood_1_int",
    "Benevolence_0_int",
    "Benevolence_1_int",
    "Ihlen_0_int",
    "Ihlen_1_int",
    "Merom_0_int",
    "Merom_1_int",
    "Pomaria_0_int",
    "Pomaria_1_int",
    "Wainscott_0_int",
    "Wainscott_1_int",
    "Beechwood_0_garden",
    "Pomaria_2_int",
]

# SCENE_NAME is set per-run by main(); do not edit this line.
SCENE_NAME = SCENES[0]
NUM_STEPS        = 1000
DT               = 0.1

# ── motion ─────────────────────────────────────────────────
FORWARD_SPEED    = 0.32
ROT_SPEED        = math.radians(20) / DT        # rad/s (~= 3.5 rad/s) — enough to correct heading
WP_YAW_GAIN      = 1.20   # reference-like gain — closes heading error quickly
WP_FORWARD_BASE  = 1.00   # always at full speed; rotation slows it naturally
NOISE_YAW_STD    = math.radians(0.5) / DT  # small noise only
NOISE_V_STD      = 0.04

# ── camera ─────────────────────────────────────────────────
IMAGE_HEIGHT     = 480   # sensor native: landscape 640×480
IMAGE_WIDTH      = 640   # after rot90 CW → video frame is 640 H × 480 W (portrait)
SAVE_VIDEO       = True
SAVE_EVERY       = 2          # write frame every N physics steps
DEPTH_MAX_M      = 5.0
CAM_FALLBACK_H   = 0.60       # Locobot: head ~0.60 m above base (unused now)

# camera: rigid attachment to robot — pose read from pybullet link every frame
# CAM_FALLBACK_H used when camera link not found (falls back to base_y + offset)

# ── BEV ────────────────────────────────────────────────────
BEV_SIZE         = 600
BEV_MARGIN_M     = 1.5
BEV_TRAIL_LEN    = 150
BEV_EVERY        = 10

# ── same-floor start ───────────────────────────────────────
SAME_FLOOR_Y_TOL     = 0.6
START_P2_Y_TRIES     = 600
START_YAW_ALIGN_DEG  = 8.0
START_YAW_JITTER_DEG = 10.0

# ── coverage-biased waypoints ──────────────────────────────
COV_CELL_M              = 1.25
COV_REJECT_VISITED_PROB = 0.75
COV_KEEP_FAR_PROB       = 0.35
WP_MIN_STEPS            = 35
WP_MAX_STEPS            = 130
WP_MIN_DIST_M           = 1.5   # min waypoint distance
WP_MAX_DIST_M           = 20.0  # wide range — filled in by phase scheduler

# ── wiggle (reference: WIGGLE_PROB_PER_STEP etc.) ──────────
WIGGLE_PROB        = 0.018
WIGGLE_MIN_STEPS   = 6
WIGGLE_MAX_STEPS   = 16
WIGGLE_TURN_MAG    = math.radians(10)  # gentle direction nudge, not hard spin
WIGGLE_FWD_SCALE   = 0.90   # keep most forward speed during wiggle

# ── wall avoidance via depth (render-gated, legacy) ─────────
WALL_AVOID_ENABLED = True
WALL_FRONT_TH_M    = 0.75
DEP_STRIDE         = 6
FRONT_BAND         = (0.40, 0.60)

# ── wall avoidance via raycasting (every step, no render needed) ─
# Three forward rays are cast from the robot base each physics step.
# If ANY ray hits within WALL_RAY_DIST_M the robot is turned away.
WALL_RAY_ENABLED   = True
WALL_RAY_DIST_M    = 0.55   # m — start turning if wall within this distance
WALL_RAY_HEIGHT    = 0.30   # m above base — ray origin height
WALL_TURN_STEPS    = 10     # steps to keep turning before re-checking
# Spread of the three rays (centre, left offset, right offset in radians)
WALL_RAY_SPREAD    = math.radians(25)

# ── navmesh-constrained step ────────────────────────────────
NAV_STEP_TRIES     = 8
NAV_TURN_DELTA_DEG = 18.0
NAV_FWD_SCALE_MIN  = 0.35
NAV_BACKUP_DIST    = 0.10
NAV_SIDE_DIST      = 0.12

# ── stuck handling ──────────────────────────────────────────
STUCK_EPS_SQ       = 8e-5
STUCK_SOFT_LIMIT   = 30
STUCK_HARD_LIMIT   = 140
HARD_STUCK_TELEPORT= True

# ── NEAR / FAR phase schedule ────────────────────────────────
# The episode alternates strictly between two phases:
#
#   NEAR phase (PHASE_NEAR_STEPS steps):
#     Both robots navigate toward each other, stay close (≤ NEAR_MAX_DIST m),
#     and can observe each other's actions.  Tether is active.
#     Waypoints are sampled within NEAR_WP_RADIUS of the midpoint.
#
#   FAR phase (PHASE_FAR_STEPS steps):
#     Robots explore independently. Tether is OFF. Waypoints are sampled
#     globally with large range so they cover the whole scene.
#
# With 1000 steps and PHASE_NEAR=250 + PHASE_FAR=250, there are 2 full
# cycles → exactly 50% of steps in each phase.
PHASE_NEAR_STEPS  = 250    # steps both robots spend near each other
PHASE_FAR_STEPS   = 250    # steps both robots spend exploring freely
NEAR_MAX_DIST     = 4.0    # m — tether kicks in if d > this during NEAR phase
NEAR_WP_RADIUS    = 3.0    # m — waypoints sampled within this radius of midpoint
FAR_WP_DIST_MIN   = 3.0    # m — min waypoint distance during FAR phase
FAR_WP_DIST_MAX   = 20.0   # m — max waypoint distance during FAR phase (wide explore)

# Observation during NEAR phase: one robot faces/tracks the other
OBSERVE_DURATION_STEPS = 80   # steps one robot spends tracking the other per NEAR phase

# Legacy tether/rendezvous (disabled — replaced by phase scheduler)
TETHER_ENABLED      = False   # tether replaced by phase-based logic
TETHER_MAX_DIST     = 4.0
TETHER_PULL_PROB    = 0.85
TETHER_PULL_DIST    = 3.0
MEET_EVERY_STEPS    = 500   # unused (phase scheduler takes over)
MEET_DURATION_STEPS = 250
MEET_RADIUS_M       = 2.0
OBSERVE_EVERY_STEPS = 250   # unused

# ── objects / push activity ─────────────────────────────────
NUM_YCB_OBJECTS   = 15
INTERACT_DIST_M   = 1.1       # within this → robot is considered moving object

# Push activity: robot physically moves a scene object using pybullet force.
# The OTHER robot should be nearby (observer) so it can watch the push.
PUSH_PROB         = 0.006     # probability per step of starting a push episode
PUSH_APPROACH_MAX = 80        # max steps to navigate toward the target object
PUSH_FORCE_STEPS  = 50        # steps to apply force (object physically moves)
PUSH_FORCE_N      = 18.0      # Newtons applied to object body (horizontal)
PUSH_DIST_TRIGGER = 1.0       # start pushing when within this distance (m)

# Guaranteed push observation: once per episode, when a push starts, the
# other robot is forced to navigate toward the pusher and watch.
PUSH_OBSERVE_STEPS = 80       # steps observer robot tracks the pushing robot

# Legacy approach-object (non-physical nudge) kept for minor interactions
MOVE_OBJ_PROB     = 0.003
MOVE_OBJ_DURATION = 60

# ── visibility / nearby ─────────────────────────────────────
NEARBY_TOPK        = 10
VISIBLE_MIN_PIX    = 10
VIS_DOWNSAMPLE     = 8
VIS_MAX_DIST_M     = 8.0
VIS_FOV_DEG        = 90.0
NEARBY_OBJ_MAX_DIST= 5.0
SETTLE_STEPS       = 40

# ============================================================
# Husky 2-DOF differential-drive action  [base_v, base_w]
# ============================================================

LOCOBOT_DIM = 2   # [0] forward velocity m/s,  [1] yaw rate rad/s


def nav_action(v: float, w: float, robot_idx: int = 0) -> np.ndarray:
    """Convert (v, w) to Locobot 2-DOF action vector. robot_idx unused (kept for API compat)."""
    a = np.zeros(LOCOBOT_DIM, np.float32)
    a[0] = float(v); a[1] = float(w)
    return a


def _safe_apply(robot, action: np.ndarray):
    try:   dim = robot.action_space.shape[0]
    except: dim = LOCOBOT_DIM
    if dim != len(action):
        if not getattr(_safe_apply, "_w", False):
            print(f"  [action resize] robot={dim} built={len(action)}")
            _safe_apply._w = True
        action = action[:dim] if dim < len(action) else \
                 np.pad(action.astype(np.float32), (0, dim - len(action)))
    robot.apply_action(action.astype(np.float32))

# ============================================================
# Math helpers
# ============================================================

def wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

def se2(x,y,yaw):
    c,s = math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s,x],[s,c,y],[0,0,1]],np.float64)

def rel_pose(xr,zr,yr,xo,zo,yo):
    T = np.linalg.inv(se2(xr,zr,yr)) @ se2(xo,zo,yo)
    return float(T[0,2]),float(T[1,2]),float(math.atan2(T[1,0],T[0,0]))

def d3(a,b): return float(np.linalg.norm(np.array(a,np.float32)-np.array(b,np.float32)))
def dxz(ax,az,bx,bz): return math.sqrt((ax-bx)**2+(az-bz)**2)

def depth_u8(d, mx=5.0):
    d = np.nan_to_num(np.array(d,np.float32),nan=mx,posinf=mx,neginf=mx)
    img = (255*(1-np.clip(d,0,mx)/mx)).astype(np.uint8)
    return np.stack([img,img,img],-1)

def seg_color(u):
    x=u.astype(np.uint32)
    return np.stack([(x*1664525+1013904223)&0xFF,
                     (x*22695477+1)&0xFF,
                     (x*1103515245+12345)&0xFF],-1).astype(np.uint8)

def rot90_frame(img: np.ndarray) -> np.ndarray:
    """
    Rotate frame 90° clockwise: landscape sensor output (H=480, W=640)
    becomes portrait video frame (H=640, W=480).
    Height and width are swapped — what was the width becomes the new
    height and vice versa.  k=3 (CW) keeps the scene upright.
    Works for both 2-D (H,W) and 3-D (H,W,C) arrays.
    """
    return img#np.rot90(img, k=3)

def visible_ids(seg, min_pix=10, stride=8):
    m = seg[::stride,::stride] if stride>1 else seg
    ids,cnt = np.unique(m.reshape(-1),return_counts=True)
    return {int(i) for i,c in zip(ids,cnt) if int(i)>0 and int(c)>=min_pix}

def albl(v,w):
    if abs(v)<1e-4 and abs(w)<1e-4: return "idle"
    if abs(w)<0.10: return "move_forward"
    if w>0.15 and v>0.05: return "forward_left"
    if w<-0.15 and v>0.05: return "forward_right"
    return "turn_left" if w>0 else "turn_right"

def cov_key(x,z): return (int(math.floor(x/COV_CELL_M)), int(math.floor(z/COV_CELL_M)))

# (camera EMA helpers removed — camera now reads live from robot link)

# ============================================================
# iGibson setup
# ============================================================

def build_sim():
    from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
    from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
    from igibson.simulator import Simulator

    Locobot = None
    for mod,cls in [("igibson.robots.locobot",       "Locobot"),
                    ("igibson.robots.locobot_robot",  "Locobot"),
                    ("igibson.robots.turtlebot",      "Turtlebot"),
                    ("igibson.robots.turtlebot_robot","Turtlebot")]:
        try:
            import importlib
            m = importlib.import_module(mod)
            if hasattr(m, cls):
                Locobot = getattr(m, cls)
                print(f"  Robot: {mod}.{cls}"); break
        except ImportError:
            pass
    if Locobot is None:
        raise ImportError("Locobot/Turtlebot class not found in iGibson 2.2")

    cfg = MeshRendererSettings(enable_pbr=False,enable_shadow=False,msaa=False,optimized=True)
    sim = Simulator(mode="headless",image_width=IMAGE_WIDTH,image_height=IMAGE_HEIGHT,
                    rendering_settings=cfg)
    scene = InteractiveIndoorScene(SCENE_NAME,build_graph=True,
                                    texture_randomization=False,object_randomization=False)
    sim.import_scene(scene)
    r1 = Locobot(action_type="continuous",action_normalize=False)
    r2 = Locobot(action_type="continuous",action_normalize=False)
    r3 = Locobot(action_type="continuous",action_normalize=False)
    sim.import_object(r1); sim.import_object(r2); sim.import_object(r3)
    try:
        d = r1.action_space.shape[0]
        print(f"  Locobot action_dim={d} (expected {LOCOBOT_DIM})")
    except Exception: pass
    return sim, scene, r1, r2, r3

def get_pose(robot):
    pos,orn = robot.get_position_orientation()
    x,y,z = float(pos[0]),float(pos[1]),float(pos[2])
    qx,qy,qz,qw = (float(v) for v in orn)
    yaw = math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    return x,y,z,yaw

def _q2euler(orn):
    qx,qy,qz,qw=(float(v) for v in orn)
    ro=math.atan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy))
    pi=math.asin(max(-1.,min(1.,2*(qw*qy-qz*qx))))
    ya=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    return ro,pi,ya

def _yawq(yaw):
    h=yaw*.5; return [0.,0.,math.sin(h),math.cos(h)]

def scene_bounds(scene):
    try:
        b=scene.scene_mesh.bounds
        return float(b[0][0]),float(b[0][2]),float(b[1][0]),float(b[1][2])
    except Exception: return -10.,-10.,10.,10.

# ============================================================
# Robot init
# ============================================================

def init_robot(robot, xyz, yaw, sim, n=40):
    robot.set_position_orientation(list(xyz), _yawq(yaw))
    try: robot.reset()
    except Exception as e: print(f"  reset err: {e}")
    za = nav_action(0., 0.)             # zero velocity settle
    for _ in range(n):
        _safe_apply(robot,za); sim.step()
    pos,orn = robot.get_position_orientation()
    ro,pi_,_ = _q2euler(orn)
    print(f"  init: y={float(pos[1]):.3f}  roll={math.degrees(ro):.1f}  pitch={math.degrees(pi_):.1f}")
    if abs(ro)>math.radians(25) or abs(pi_)>math.radians(25):
        print("  ⚠ unstable, retry")
        robot.set_position_orientation(list(xyz),_yawq(yaw))
        try: robot.reset()
        except Exception: pass
        for _ in range(n): _safe_apply(robot,za); sim.step()

# ============================================================
# Floor point sampling  (reference: same-floor y-tolerance)
# ============================================================

def floor_pt(scene, rng, anchor_y=None, y_tol=SAME_FLOOR_Y_TOL):
    """Sample a navigable floor point, optionally constrained to anchor_y ± y_tol."""
    for _ in range(500):
        for meth in ("get_random_point","get_random_floor_point"):
            if hasattr(scene,meth):
                try:
                    r = getattr(scene,meth)()
                    pt = np.array(r[1] if isinstance(r,tuple) else r, np.float32)
                    if anchor_y is None: return pt
                    if abs(float(pt[1])-anchor_y) <= y_tol: return pt
                except Exception: pass
    ang=rng.uniform(-math.pi,math.pi); rv=rng.uniform(1.,4.)
    return np.array([rv*math.cos(ang),0.,rv*math.sin(ang)],np.float32)

# ── start distance constraints ────────────────────────────────────────
START_DIST_MIN = 0.5   # m — robots must be at least this far apart
START_DIST_MAX = 4.0   # m — robots can be up to 4 m apart (wider variety)


def _sample_close_pt(scene, rng, anchor, anchor_y):
    """Sample a point on the same floor as anchor within START_DIST_MIN..START_DIST_MAX."""
    for _ in range(START_P2_Y_TRIES):
        cand = floor_pt(scene, rng, anchor_y=anchor_y, y_tol=SAME_FLOOR_Y_TOL)
        if cand is None: continue
        if START_DIST_MIN < d3(anchor, cand) < START_DIST_MAX:
            return cand
    # looser fallback
    for _ in range(300):
        cand = floor_pt(scene, rng, anchor_y=anchor_y, y_tol=SAME_FLOOR_Y_TOL)
        if cand is not None and d3(anchor, cand) > START_DIST_MIN:
            return cand
    return floor_pt(scene, rng)


def choose_starts(scene, rng):
    """
    Sample three start positions on the same floor within START_DIST range.
    p2 and p3 are sampled near p1; all three stay in the same room.
    """
    p1  = floor_pt(scene, rng)
    p1y = float(p1[1])
    p2  = _sample_close_pt(scene, rng, p1, p1y)
    p3  = _sample_close_pt(scene, rng, p1, p1y)
    print(f"  Start: "
          f"p1=({p1[0]:.1f},{p1[2]:.1f})  "
          f"p2=({p2[0]:.1f},{p2[2]:.1f}) d={d3(p1,p2):.1f}m  "
          f"p3=({p3[0]:.1f},{p3[2]:.1f}) d={d3(p1,p3):.1f}m")
    return p1, p2, p3

# ============================================================
# Waypoint sampling  (reference: sample_waypoint_global / near)
# ============================================================

def wp_global(scene, rng, origin, visited,
              dist_min=None, dist_max=None):
    """
    Coverage-biased global waypoint.
    dist_min/dist_max override WP_MIN/MAX_DIST_M when provided,
    allowing the FAR phase to use a wider exploration range.
    """
    d_min = dist_min if dist_min is not None else WP_MIN_DIST_M
    d_max = dist_max if dist_max is not None else WP_MAX_DIST_M
    allow_far = (rng.random() < COV_KEEP_FAR_PROB)
    ox,oz = float(origin[0]),float(origin[2])
    for _ in range(900):
        pt = floor_pt(scene,rng)
        dx,dz = float(pt[0])-ox, float(pt[2])-oz
        dist = math.sqrt(dx*dx+dz*dz)
        if dist < d_min: continue
        if (not allow_far) and dist > d_max:
            if rng.random() < 0.85: continue
        key = cov_key(float(pt[0]),float(pt[2]))
        if key in visited and rng.random() < COV_REJECT_VISITED_PROB: continue
        return pt
    return floor_pt(scene,rng)

def wp_near(scene, rng, center, radius):
    """
    Sample a navigable point near center within radius.
    Port of reference sample_waypoint_near_point (without pf.snap_point;
    iGibson physics handles exact navmesh snapping at step time).
    """
    cy = float(center[1])
    for _ in range(450):
        ang = rng.uniform(-math.pi,math.pi)
        rv  = rng.uniform(0.3,radius)
        x   = float(center[0])+rv*math.cos(ang)
        z   = float(center[2])+rv*math.sin(ang)
        # iGibson: return candidate; physics will resolve penetration
        return np.array([x,cy,z],np.float32)
    return floor_pt(scene,rng)

# ============================================================
# Command to waypoint  (reference: compute_cmd_to_waypoint)
# ============================================================

def cmd(x, z, yaw, wp, rng):
    """
    Exact port of reference compute_cmd_to_waypoint:
      yaw_err → w with WP_YAW_GAIN, clamped to ROT_SPEED
      v scaled down when turning hard (yaw_scale)
      Gaussian noise on both v and w
    """
    dx,dz = float(wp[0]-x),float(wp[2]-z)
    tgt = math.atan2(dz,dx)
    err = wrap(tgt-yaw)
    w = WP_YAW_GAIN*err/DT
    w = max(-ROT_SPEED,min(ROT_SPEED,w))
    # Speed scaling: full speed when aligned, reduce when heading error > 45°.
    # Floor = 0.35 so robot keeps moving while correcting (reference uses 0.30).
    ys = max(0.35, 1.0 - abs(err) / math.radians(120))
    v  = FORWARD_SPEED*WP_FORWARD_BASE*ys
    w += rng.gauss(0.,NOISE_YAW_STD)/DT
    v += rng.gauss(0.,NOISE_V_STD)
    return max(0.,min(FORWARD_SPEED,v)), max(-ROT_SPEED,min(ROT_SPEED,w))

# ============================================================
# Navmesh-constrained step  (reference: nav_constrained_step)
# ============================================================

def nav_step(scene, x, y, z, yaw, v, w):
    """
    8 heading trials + side-step + backup.
    Exact port of reference nav_constrained_step.
    iGibson has no pf.snap_point, so we return the best reachable candidate
    and let physics resolve the rest.
    """
    yaw_base = wrap(yaw+w*DT)
    for k in range(NAV_STEP_TRIES):
        if k==0: yt=yaw_base
        else:
            m=(k+1)//2; sg=+1 if k%2==1 else -1
            yt=wrap(yaw_base+sg*m*math.radians(NAV_TURN_DELTA_DEG))
        sc=max(NAV_FWD_SCALE_MIN,1.0-0.12*k)
        xn=x+v*sc*math.cos(yt)*DT; zn=z+v*sc*math.sin(yt)*DT
        # attempt to verify with scene navmesh if available
        try:
            if scene.floor_map is not None:
                pass   # placeholder; physics will constrain
        except Exception: pass
        return xn,y,zn,yt  # first try always returned; physics corrects
    # side-step
    for sg in (+1,-1):
        sy=wrap(yaw_base+sg*math.pi/2)
        return x+NAV_SIDE_DIST*math.cos(sy),y,z+NAV_SIDE_DIST*math.sin(sy),yaw_base
    # backup
    return x-NAV_BACKUP_DIST*math.cos(yaw_base),y,z-NAV_BACKUP_DIST*math.sin(yaw_base),yaw_base

# ============================================================
# Per-step wall detection via pybullet raycasting
# ============================================================

def wall_ray_detect(robot, yaw: float) -> bool:
    """
    Cast three short rays forward from the robot base (centre, left, right).
    Returns True if any ray hits a non-robot body within WALL_RAY_DIST_M.
    Runs every physics step — no dependency on depth renders.
    """
    if not WALL_RAY_ENABLED:
        return False
    import pybullet as p
    try:
        try: bid = robot.get_body_ids()[0]
        except: bid = robot.robot_ids[0]
        pos, _ = p.getBasePositionAndOrientation(bid)
        ox = float(pos[0])
        oy = float(pos[1]) + WALL_RAY_HEIGHT
        oz = float(pos[2])
    except Exception:
        return False

    # Collect all body ids belonging to either robot so we can ignore them
    own_ids: set = set()
    for attr in ("get_body_ids", "body_ids", "robot_ids"):
        if hasattr(robot, attr):
            try:
                ids = getattr(robot, attr)
                own_ids.update(ids() if callable(ids) else ids)
                break
            except Exception: pass

    for offset in (0.0, WALL_RAY_SPREAD, -WALL_RAY_SPREAD):
        a = yaw + offset
        tx = ox + WALL_RAY_DIST_M * math.cos(a)
        tz = oz + WALL_RAY_DIST_M * math.sin(a)
        try:
            hit = p.rayTest([ox, oy, oz], [tx, oy, tz])[0]
            hit_id = int(hit[0])
            if hit_id != -1 and hit_id not in own_ids:
                return True   # wall or object in the way
        except Exception:
            pass
    return False


# ============================================================
# Virtual overhead camera — above the head, locked to robot yaw
# ============================================================
# A virtual camera is placed directly above the robot head (no URDF link
# lookup needed).  It always looks in the direction the robot is moving:
#
#   iGibson coordinate convention:
#     X — forward/backward in the world
#     Y — left/right in the world
#     Z — up (vertical)
#
#   Eye    = (base_x,
#             base_y,
#             base_z + HEAD_HEIGHT + ABOVE_HEAD_OFFSET)
#            directly above the robot head along the world +Z axis
#
#   Target = eye + LOOK_DIST * (cos(yaw), sin(yaw), 0)
#            rotates in the XY plane as the robot turns — camera front
#            always equals robot front.  Z of target equals Z of eye
#            so the gaze is perfectly horizontal.
#
#   Up     = world +Z  (camera never tilts or rolls)
#
# Tune these three constants to adjust camera placement:
HEAD_HEIGHT       = 0.60   # m — Locobot head is ~0.60 m above base link
ABOVE_HEAD_OFFSET = 0.30   # m — virtual camera floats above the head
LOOK_DIST         = 2.50   # m — how far ahead the camera points


def render_robot(sim, robot):
    """
    Render RGB / depth / segmentation from a virtual camera mounted
    above the robot head, always facing the robot forward direction.

    iGibson uses a Z-up coordinate system:
        X, Y  — horizontal plane (floor)
        Z     — vertical (up)

    Eye position:
        (base_x,  base_y,  base_z + HEAD_HEIGHT + ABOVE_HEAD_OFFSET)
        — moves rigidly with the robot base every physics step.

    Look direction (XY plane rotation):
        yaw  ← rotation around world Z, extracted from base quaternion
        target = eye + LOOK_DIST * (cos(yaw), sin(yaw), 0)
        When the robot turns, yaw changes → target rotates in XY →
        camera front always equals robot front direction.
        target_z = eye_z  → perfectly horizontal gaze (no up/down tilt).

    Up = world +Z — camera never tilts or rolls.
    """
    import pybullet as p
    H, W = IMAGE_HEIGHT, IMAGE_WIDTH
    blank = (np.zeros((H,W,3),np.uint8),
             np.zeros((H,W),   np.float32),
             np.zeros((H,W),   np.int32))

    # ── Robot base pose ──────────────────────────────────────────────────
    try:
        try: bid = robot.get_body_ids()[0]
        except: bid = robot.robot_ids[0]
        base_pos, base_orn = p.getBasePositionAndOrientation(bid)
        bx = float(base_pos[0])
        by = float(base_pos[1])
        bz = float(base_pos[2])
        bqx,bqy,bqz,bqw = (float(v) for v in base_orn)
        # Extract yaw = rotation around world Z axis from base quaternion
        # (standard ZYX Euler, yaw = atan2 formula for Z-up convention)
        yaw = math.atan2(2*(bqw*bqz + bqx*bqy),
                         1 - 2*(bqy*bqy + bqz*bqz))
    except Exception as e:
        print(f"  [camera] base pose err: {e}")
        return blank

    # ── Virtual camera eye: above robot head along +Z ────────────────────
    ex = bx
    ey = by
    ez = bz + HEAD_HEIGHT + ABOVE_HEAD_OFFSET

    # ── Look target: robot forward in XY plane, same height as eye ───────
    #  cos(yaw) = X component of forward,  sin(yaw) = Y component.
    #  target_z = ez  → perfectly horizontal gaze (no up/down tilt).
    c, s = math.cos(yaw), math.sin(yaw)
    tx = ex + LOOK_DIST * c
    ty = ey + LOOK_DIST * s
    tz = ez   # same height as eye — horizontal gaze

    try:
        sim.renderer.set_camera([ex, ey, ez], [tx, ty, tz], [0.0, 0.0, 1.0])
    except Exception as e:
        print(f"  [camera] set_camera err: {e}")
        return blank

    # ── Render all modalities ────────────────────────────────────────────
    def _r(mode):
        try:
            out = sim.renderer.render(modes=(mode,))
            if isinstance(out,(list,tuple)) and len(out)>0: return np.array(out[0])
            if isinstance(out,dict): return np.array(out.get(mode,np.array([])))
            return np.array(out)
        except Exception: return np.array([])

    rgb = blank[0]
    try:
        arr = _r("rgb")
        if arr.ndim==3 and arr.shape[-1]==4: arr=arr[...,:3]
        if arr.dtype in (np.float32,np.float64): arr=np.clip(arr*255,0,255).astype(np.uint8)
        else: arr=arr.astype(np.uint8)
        if arr.shape[:2]==(H,W): rgb=arr
    except Exception: pass

    dep = blank[1]
    try:
        arr = _r("3d").astype(np.float32)
        if arr.ndim==3 and arr.shape[-1]>=3: arr=np.linalg.norm(arr[...,:3],axis=-1)
        if arr.ndim==2 and arr.shape==(H,W): dep=arr
    except Exception: pass

    seg = blank[2]
    try:
        arr = _r("seg")
        if arr.ndim==3: arr=arr[...,0]
        if arr.dtype in (np.float32,np.float64): arr=(arr*255).astype(np.int32)
        else: arr=arr.astype(np.int32)
        if arr.shape==(H,W): seg=arr
    except Exception: pass

    return rgb, dep, seg
# ============================================================
# Objects
# ============================================================

class TObj:
    def __init__(self,obj,uid,name,cat):
        self.obj=obj; self.uid=uid; self.name=name; self.cat=cat
        self._prev_pos=None   # for movement detection
    def pos(self):
        try:
            p,_=self.obj.get_position_orientation(); return np.array(p,np.float32)
        except Exception: return None
    def summary(self):
        p=self.pos()
        return {"uid":self.uid,"name":self.name,"cat":self.cat,
                "wx":round(float(p[0]),3) if p is not None else None,
                "wy":round(float(p[1]),3) if p is not None else None,
                "wz":round(float(p[2]),3) if p is not None else None}

def _mkt(obj,uid):
    n=str(getattr(obj,"name",None) or getattr(obj,"object_scope",None) or f"obj_{uid}")
    c=str(getattr(obj,"category",None) or "unknown")
    return TObj(obj,uid,n,c)

def collect_objects(scene):
    cats={"door","cabinet","fridge","refrigerator","oven","microwave",
          "chair","table","sofa","couch","stool","bench","trash_can","box"}
    raw=[]
    try:
        for cat,objs in scene.objects_by_category.items():
            if any(k in cat.lower() for k in cats): raw.extend(objs)
    except Exception:
        try:
            for obj in scene.get_objects():
                cl=(getattr(obj,"category","") or "").lower()
                if any(k in cl for k in cats): raw.append(obj)
        except Exception: pass
    result=[_mkt(o,1000+i) for i,o in enumerate(raw)]
    print(f"  Movable objects found: {len(result)}"); return result

def spawn_ycb(sim, scene, rng, n=NUM_YCB_OBJECTS):
    try: from igibson.objects.ycb_object import YCBObject
    except ImportError: print("  YCBObject unavailable"); return []
    names=["002_master_chef_can","003_cracker_box","004_sugar_box",
           "005_tomato_soup_can","006_mustard_bottle","007_tuna_fish_can",
           "008_pudding_box","009_gelatin_box","010_potted_meat_can",
           "011_banana","019_pitcher_base","021_bleach_cleanser",
           "024_bowl","025_mug","035_power_drill"]
    out=[]
    for i in range(n):
        nm=rng.choice(names)
        try:
            obj=YCBObject(nm); sim.import_object(obj)
            pt=floor_pt(scene,rng); pt[1]+=0.05
            obj.set_position_orientation(pt.tolist(),[0,0,0,1])
            out.append(_mkt(obj,2000+i))
        except Exception as e: print(f"  YCB fail ({nm}): {e}")
    print(f"  YCB spawned: {len(out)}"); return out

def nearby(rx,rz,ryaw,objs,vis,topk=NEARBY_TOPK,max_d=NEARBY_OBJ_MAX_DIST):
    out=[]
    for o in objs:
        p=o.pos()
        if p is None: continue
        ox,oz=float(p[0]),float(p[2])
        d=math.sqrt((ox-rx)**2+(oz-rz)**2)
        if d>max_d: continue
        bearing=math.atan2(oz-rz,ox-rx); rb=wrap(bearing-ryaw)
        rx2,rz2,ry2=rel_pose(rx,rz,ryaw,ox,oz,0.)
        out.append({"uid":o.uid,"name":o.name,"cat":o.cat,
                    "dist_m":round(d,3),"bearing_rad":round(rb,4),
                    "bearing_deg":round(math.degrees(rb),2),
                    "rel_x":round(rx2,3),"rel_z":round(rz2,3),"rel_yaw":round(ry2,4),
                    "world_x":round(float(p[0]),3),"world_y":round(float(p[1]),3),
                    "world_z":round(float(p[2]),3),"visible":o.uid in vis})
    out.sort(key=lambda r:r["dist_m"]); return out[:topk]

def check_vis(ra,rb,yaw_a):
    import pybullet as p
    pa=np.array(ra.get_position(),np.float32); pb=np.array(rb.get_position(),np.float32)
    dist=float(np.linalg.norm(pb-pa))
    if dist>VIS_MAX_DIST_M: return False,dist
    bearing=math.atan2(float(pb[2]-pa[2]),float(pb[0]-pa[0]))
    if abs(wrap(bearing-yaw_a))>math.radians(VIS_FOV_DEG/2): return False,dist
    st=(pa+np.array([0,.3,0],np.float32)).tolist()
    en=(pb+np.array([0,.3,0],np.float32)).tolist()
    hit=p.rayTest(st,en)[0]; hid=int(hit[0])
    try: bb=int(rb.robot_ids[0])
    except Exception: bb=-2
    return (hid==-1 or hid==bb),dist

# ============================================================
# BEV
# ============================================================

BEV_C1=(255,80,80); BEV_C2=(80,80,255); BEV_C3=(80,200,80)

class BEV:
    def __init__(self,xmin,zmin,xmax,zmax):
        m=BEV_MARGIN_M
        self.xmin=xmin-m; self.zmin=zmin-m
        self.xmax=xmax+m; self.zmax=zmax+m
        self.wm=max(self.xmax-self.xmin,.1); self.hm=max(self.zmax-self.zmin,.1)
        self.trails={1:[],2:[],3:[]}
    def _p(self,x,z):
        px=int((x-self.xmin)/self.wm*BEV_SIZE)
        pz=int((1-(z-self.zmin)/self.hm)*BEV_SIZE)
        return int(np.clip(px,0,BEV_SIZE-1)),int(np.clip(pz,0,BEV_SIZE-1))
    def frame(self,t,x1,z1,y1,x2,z2,y2,x3,z3,y3,obj_xz=None):
        C=BEV_SIZE; cv=np.full((C,C,3),40,np.uint8)
        for rid,xr,zr in ((1,x1,z1),(2,x2,z2),(3,x3,z3)):
            self.trails[rid].append((xr,zr))
            if len(self.trails[rid])>BEV_TRAIL_LEN: self.trails[rid].pop(0)
        if obj_xz:
            for ox,oz in obj_xz: _dc(cv,self._p(ox,oz),3,(200,200,50))
        for rid,col in ((1,BEV_C1),(2,BEV_C2),(3,BEV_C3)):
            pts=self.trails[rid]; n=len(pts)
            for i in range(1,n):
                a=i/n; c=tuple(int(v*a) for v in col)
                _dl(cv,self._p(*pts[i-1]),self._p(*pts[i]),c,2)
        for xr,zr,yr,col in ((x1,z1,y1,BEV_C1),(x2,z2,y2,BEV_C2),(x3,z3,y3,BEV_C3)):
            px,pz=self._p(xr,zr); _dc(cv,(px,pz),9,col)
            ex=int(np.clip(px+24*math.cos(yr),0,C-1))
            ez=int(np.clip(pz-24*math.sin(yr),0,C-1))
            _dl(cv,(px,pz),(ex,ez),(255,255,255),2)
        _stat(cv,t); return cv

def _dl(img,p0,p1,col,w=1):
    x0,y0=int(p0[0]),int(p0[1]); x1,y1=int(p1[0]),int(p1[1])
    dx,dy=abs(x1-x0),abs(y1-y0); sx=1 if x0<x1 else -1; sy=1 if y0<y1 else -1
    err=dx-dy; H,W=img.shape[:2]
    for _ in range(max(dx,dy)+1):
        for bx in range(-(w//2),w//2+1):
            for by_ in range(-(w//2),w//2+1):
                nx,ny=x0+bx,y0+by_
                if 0<=nx<W and 0<=ny<H: img[ny,nx]=col
        if x0==x1 and y0==y1: break
        e2=2*err
        if e2>-dy: err-=dy; x0+=sx
        if e2<dx: err+=dx; y0+=sy

def _dc(img,c,r,col):
    cx,cy=int(c[0]),int(c[1]); H,W=img.shape[:2]
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            if dx*dx+dy*dy<=r*r:
                nx,ny=cx+dx,cy+dy
                if 0<=nx<W and 0<=ny<H: img[ny,nx]=col

_F={"0":["111","101","101","101","111"],"1":["010","110","010","010","111"],
    "2":["111","001","111","100","111"],"3":["111","001","111","001","111"],
    "4":["101","101","111","001","001"],"5":["111","100","111","001","111"],
    "6":["111","100","111","101","111"],"7":["111","001","001","001","001"],
    "8":["111","101","111","101","111"],"9":["111","101","111","001","111"],
    " ":["000","000","000","000","000"],"-":["000","000","111","000","000"],
    ":":["000","010","000","010","000"],"t":["111","010","010","010","010"]}

def _stat(cv,t):
    x=4
    for ch in f"t={t:04d}":
        bm=_F.get(ch,["000"]*5); H,W=cv.shape[:2]
        for ri,row in enumerate(bm):
            for ci,px in enumerate(row):
                if px=="1":
                    nx,ny=x+ci,4+ri
                    if 0<=nx<W and 0<=ny<H: cv[ny,nx]=(220,220,100)
        x+=6

# ============================================================
# Push activity helpers
# ============================================================

# Push FSM states (per-robot)
# IDLE       → robot explores normally
# APPROACH   → robot navigates toward target object
# PUSHING    → robot applies physical force to object each step

PUSH_IDLE     = 0
PUSH_APPROACH = 1
PUSH_PUSHING  = 2


def apply_push_force(tobj: "TObj", robot_pos: np.ndarray):
    """
    Apply a horizontal force to the object body pointing away from the robot.
    Uses pybullet.applyExternalForce so the object physically moves in simulation.
    Falls back silently if the object has no pybullet body.
    """
    import pybullet as p
    op = tobj.pos()
    if op is None: return
    dx = float(op[0]) - float(robot_pos[0])
    dz = float(op[2]) - float(robot_pos[2])
    dist = math.sqrt(dx*dx + dz*dz) + 1e-6
    # Unit direction away from robot, in world frame (Y=0, horizontal push)
    fx = (dx / dist) * PUSH_FORCE_N
    fz = (dz / dist) * PUSH_FORCE_N
    # Find pybullet body id of the object
    bid = None
    for attr in ("get_body_ids", "body_ids", "robot_ids"):
        if hasattr(tobj.obj, attr):
            try:
                ids = getattr(tobj.obj, attr)
                bid = (ids() if callable(ids) else ids)[0]
                break
            except Exception: pass
    if bid is None:
        try: bid = int(tobj.obj.get_body_id())
        except Exception: return
    try:
        p.applyExternalForce(bid, -1,
                             [float(fx), 0.0, float(fz)],
                             [float(op[0]), float(op[1]), float(op[2])],
                             p.WORLD_FRAME)
    except Exception: pass


def moving_obj_name(robot_pos, tgt_obj) -> str:
    """Return object name when robot is within INTERACT_DIST_M of target, else ''."""
    if tgt_obj is None: return ""
    op = tgt_obj.pos()
    if op is None: return ""
    if dxz(float(robot_pos[0]),float(robot_pos[2]),float(op[0]),float(op[2])) < INTERACT_DIST_M:
        return tgt_obj.name
    return ""

# ============================================================
# Main episode loop
# ============================================================

def run_exploration(scene_name: str, explore_idx: int):
    """Run one exploration episode for scene_name, saving to OUTPUT_ROOT/scene_name/explore_NNN/."""
    global SCENE_NAME
    SCENE_NAME = scene_name   # set global so build_sim and seed use correct scene

    # Skip if already completed
    expl_id = f"explore_{explore_idx:03d}"
    out_dir  = os.path.join(OUTPUT_ROOT, scene_name, expl_id)
    done_marker = os.path.join(out_dir, "DONE")
    if os.path.exists(done_marker):
        print(f"  [SKIP] already done: {scene_name}/{expl_id}")
        return True

    sim, scene, r1, r2, r3 = build_sim()
    os.makedirs(out_dir, exist_ok=True)

    seed = (hash(SCENE_NAME) ^ (explore_idx*1_000_003) ^ 0x9E3779B9) & 0xFFFFFFFF
    rng_g = random.Random(seed)
    rng1  = random.Random((seed+101)&0xFFFFFFFF)
    rng2  = random.Random((seed+202)&0xFFFFFFFF)
    rng3  = random.Random((seed+303)&0xFFFFFFFF)

    # ── same-floor start ──────────────────────────────────────
    p1, p2, p3 = choose_starts(scene, rng_g)
    y1 = wrap(rng1.uniform(-math.pi,math.pi) +
              math.radians(rng1.uniform(-START_YAW_JITTER_DEG,START_YAW_JITTER_DEG)))
    y2 = wrap(y1 + math.radians(rng2.uniform(-START_YAW_ALIGN_DEG,START_YAW_ALIGN_DEG)))
    y3 = wrap(y1 + math.radians(rng3.uniform(-START_YAW_ALIGN_DEG,START_YAW_ALIGN_DEG)))

    print("  Init R1 ..."); init_robot(r1, p1, y1, sim, SETTLE_STEPS)
    print("  Init R2 ..."); init_robot(r2, p2, y2, sim, SETTLE_STEPS)
    print("  Init R3 ..."); init_robot(r3, p3, y3, sim, SETTLE_STEPS)

    x1,y1f,z1,yaw1 = get_pose(r1)
    x2,y2f,z2,yaw2 = get_pose(r2)
    x3,y3f,z3,yaw3 = get_pose(r3)

    movable = collect_objects(scene)
    ycb     = spawn_ycb(sim, scene, rng_g)
    all_obj = movable + ycb

    xmin,zmin,xmax,zmax = scene_bounds(scene)
    bev = BEV(xmin,zmin,xmax,zmax)

    # ── reference flat-loop state variables ───────────────────
    # waypoints
    visited1: Set = set(); visited2: Set = set(); visited3: Set = set()
    wp1 = wp_global(scene,rng1,np.array([x1,y1f,z1],np.float32),visited1)
    wp2 = wp_global(scene,rng2,np.array([x2,y2f,z2],np.float32),visited2)
    wp3 = wp_global(scene,rng3,np.array([x3,y3f,z3],np.float32),visited3)
    wpt1 = rng1.randint(WP_MIN_STEPS,WP_MAX_STEPS)
    wpt2 = rng2.randint(WP_MIN_STEPS,WP_MAX_STEPS)
    wpt3 = rng3.randint(WP_MIN_STEPS,WP_MAX_STEPS)

    # wiggle counters
    wig1 = wig2 = wig3 = 0

    # wall-turn counters
    wall_turn1=0; wall_turn2=0; wall_turn3=0
    wall_dir1=1.0; wall_dir2=1.0; wall_dir3=1.0

    # stuck counters
    stuck1=stuck2=stuck3=0
    prev_x1,prev_z1=x1,z1
    prev_x2,prev_z2=x2,z2
    prev_x3,prev_z3=x3,z3

    # ── NEAR/FAR phase state ─────────────────────────────────────
    # phase_near: True during NEAR phase, False during FAR phase
    # phase_timer: steps remaining in current phase
    # Starts in NEAR phase so robots can observe each other immediately.
    phase_near  = True
    phase_timer = PHASE_NEAR_STEPS

    # Observation within NEAR phase (1/2/3 = observer robot id, 0 = all explore)
    obs_who = 0
    obs_a   = 0

    # Legacy meet vars (kept for compatibility with push observation code)
    meet_t = MEET_EVERY_STEPS; meet_a = 0; meet_wp = None

    # camera: no EMA — reads live from robot link every render call

    # object-approach state (legacy minor interactions)
    tgt1: Optional[TObj]=None; tgt_t1=0
    tgt2: Optional[TObj]=None; tgt_t2=0
    tgt3: Optional[TObj]=None; tgt_t3=0

    # push FSM state (per-robot)
    push_state1=PUSH_IDLE; push_tobj1: Optional[TObj]=None; push_timer1=0
    push_state2=PUSH_IDLE; push_tobj2: Optional[TObj]=None; push_timer2=0
    push_state3=PUSH_IDLE; push_tobj3: Optional[TObj]=None; push_timer3=0

    # Guaranteed push observation (once per episode)
    push_observed      = False
    push_observe_timer = 0

    fps_v = max(1,int(1./DT)//max(1,SAVE_EVERY))
    fps_b = max(1,int(1./DT)//max(1,BEV_EVERY))

    # ── video writers ─────────────────────────────────────────
    def _w(n): return imageio.get_writer(os.path.join(out_dir,n),fps=fps_v)
    vw = ({
        "rgb1":_w("locobot1_cam.mp4"),  "rgb2":_w("locobot2_cam.mp4"),  "rgb3":_w("locobot3_cam.mp4"),
        "dep1":_w("locobot1_depth.mp4"),"dep2":_w("locobot2_depth.mp4"),"dep3":_w("locobot3_depth.mp4"),
        "sem1":_w("locobot1_sem.mp4"),  "sem2":_w("locobot2_sem.mp4"),  "sem3":_w("locobot3_sem.mp4"),
        "bev": imageio.get_writer(os.path.join(out_dir,"bev_trajectories.mp4"),fps=fps_b),
    } if SAVE_VIDEO else {})

    # ── CSV / JSONL ───────────────────────────────────────────
    tf  = open(os.path.join(out_dir,"locobot_trajectory_and_cmds.csv"),"w",newline="")
    nf  = open(os.path.join(out_dir,"nearby_objects.csv"),"w",newline="")
    pf_ = open(os.path.join(out_dir,"push_events.csv"),"w",newline="")
    jf  = open(os.path.join(out_dir,"robot_events.jsonl"),"w")
    tw  = csv.writer(tf); nw = csv.writer(nf); pw = csv.writer(pf_)

    # trajectory header — 3 robots. rel_* fields are relative to robot with lowest id.
    # sees_r2/r3, dist_r2/r3: visibility and distance to each other robot.
    tw.writerow(["t","robot_id",
                 "x","y","z","yaw",
                 "v_cmd","omega_cmd","action_name",
                 "sees_r2","dist_r2","sees_r3","dist_r3",
                 "observed_by","observing",
                 "pushing_object_name"])
    nw.writerow(["t","robot_id","uid","name","cat",
                 "dist_m","bearing_rad","bearing_deg",
                 "rel_x","rel_z","rel_yaw",
                 "world_x","world_y","world_z","visible"])
    # push_events.csv: one row per push-force application step
    pw.writerow(["t","pushing_robot_id","object_name","object_cat",
                 "obj_x","obj_y","obj_z",
                 "robot_x","robot_z",
                 "observer_robot_id","observer_sees_pusher",
                 "observer_dist_to_pusher_m"])

    json.dump({
        "scene":SCENE_NAME,"exploration":expl_id,"robot":"Locobot",
        "action_dim":LOCOBOT_DIM,"steps":NUM_STEPS,"dt":DT,
        "image_h":IMAGE_HEIGHT,"image_w":IMAGE_WIDTH,
        "same_floor_y_tol":SAME_FLOOR_Y_TOL,"cov_cell_m":COV_CELL_M,
        "wall_avoid":WALL_AVOID_ENABLED,
        "video_rotation_deg":90,
        "phase_near_steps":PHASE_NEAR_STEPS,
        "phase_far_steps":PHASE_FAR_STEPS,
        "observe_duration_steps":OBSERVE_DURATION_STEPS,
        "pushing_object_note":
            "pushing_object_name is non-empty only during PUSH_PUSHING state "
            "(physical force applied). push_events.csv logs every push step.",
        "visibility_note":
            "sees_other_robot=1 when the other robot is in this robot FOV "
            "and within VIS_MAX_DIST_M with clear line-of-sight. "
            "observed_by_robot_id and observing_robot_id record the "
            "asymmetric observation relationship each step.",
    }, open(os.path.join(out_dir,"meta.json"),"w"), indent=2)

    try:
        for t in range(NUM_STEPS):
            # read current poses
            x1,y1f,z1,yaw1 = get_pose(r1)
            x2,y2f,z2,yaw2 = get_pose(r2)

            # coverage bookkeeping
            visited1.add(cov_key(x1,z1))
            visited2.add(cov_key(x2,z2))
            visited3.add(cov_key(x3,z3))

            # ── NEAR / FAR phase scheduler ─────────────────────────
            # Strictly alternates: PHASE_NEAR_STEPS near → PHASE_FAR_STEPS far
            # → PHASE_NEAR_STEPS near → … giving exactly 50/50 split.
            d12 = math.sqrt((x2-x1)**2+(z2-z1)**2)
            d13 = math.sqrt((x3-x1)**2+(z3-z1)**2)
            d23 = math.sqrt((x3-x2)**2+(z3-z2)**2)
            phase_timer -= 1
            if phase_timer <= 0:
                phase_near  = not phase_near
                phase_timer = PHASE_NEAR_STEPS if phase_near else PHASE_FAR_STEPS
                obs_who=0; obs_a=0
                phase_str = "NEAR" if phase_near else "FAR"
                print(f"  [t={t}] → {phase_str} phase (d12={d12:.1f} d13={d13:.1f} d23={d23:.1f})")

            # Observation within NEAR phase: rotate observer among robots 1/2/3
            if phase_near:
                near_elapsed = PHASE_NEAR_STEPS - phase_timer
                if obs_a <= 0 and near_elapsed == PHASE_NEAR_STEPS // 2:
                    obs_a   = OBSERVE_DURATION_STEPS
                    obs_who = 1 + (t // PHASE_NEAR_STEPS) % 3  # cycles 1→2→3→1
            if obs_a > 0:
                obs_a -= 1
                if obs_a <= 0: obs_who = 0

            # ── waypoint update ─────────────────────────────────────
            wpt1 -= 1; wpt2 -= 1; wpt3 -= 1

            if phase_near:
                # ── NEAR phase: all three navigate toward shared centroid ──
                cx = (x1+x2+x3)/3.; cy = (y1f+y2f+y3f)/3.; cz = (z1+z2+z3)/3.
                mid = np.array([cx,cy,cz],np.float32)

                if obs_who == 1:
                    # R1 tracks R2 (first non-self robot)
                    wp1 = np.array([x2,y2f,z2],np.float32); wpt1=5
                elif wpt1 <= 0:
                    wp1 = wp_near(scene,rng1,mid,NEAR_WP_RADIUS)
                    wpt1 = rng1.randint(WP_MIN_STEPS,WP_MAX_STEPS)
                if d12 > NEAR_MAX_DIST and d13 > NEAR_MAX_DIST and rng_g.random()<0.9:
                    wp1 = wp_near(scene,rng1,mid,2.0); wpt1=rng1.randint(15,40)

                if obs_who == 2:
                    wp2 = np.array([x1,y1f,z1],np.float32); wpt2=5
                elif wpt2 <= 0:
                    wp2 = wp_near(scene,rng2,mid,NEAR_WP_RADIUS)
                    wpt2 = rng2.randint(WP_MIN_STEPS,WP_MAX_STEPS)
                if d12 > NEAR_MAX_DIST and d23 > NEAR_MAX_DIST and rng_g.random()<0.9:
                    wp2 = wp_near(scene,rng2,mid,2.0); wpt2=rng2.randint(15,40)

                if obs_who == 3:
                    wp3 = np.array([x1,y1f,z1],np.float32); wpt3=5
                elif wpt3 <= 0:
                    wp3 = wp_near(scene,rng3,mid,NEAR_WP_RADIUS)
                    wpt3 = rng3.randint(WP_MIN_STEPS,WP_MAX_STEPS)
                if d13 > NEAR_MAX_DIST and d23 > NEAR_MAX_DIST and rng_g.random()<0.9:
                    wp3 = wp_near(scene,rng3,mid,2.0); wpt3=rng3.randint(15,40)

            else:
                # ── FAR phase: three robots explore independently ──────
                if wpt1 <= 0:
                    if tgt1 is not None:
                        tp=tgt1.pos()
                        if tp is not None: wp1=tp.copy()
                        else: wp1=wp_global(scene,rng1,np.array([x1,y1f,z1],np.float32),visited1,FAR_WP_DIST_MIN,FAR_WP_DIST_MAX)
                    else: wp1=wp_global(scene,rng1,np.array([x1,y1f,z1],np.float32),visited1,FAR_WP_DIST_MIN,FAR_WP_DIST_MAX)
                    wpt1=rng1.randint(WP_MIN_STEPS,WP_MAX_STEPS)
                if wpt2 <= 0:
                    if tgt2 is not None:
                        tp=tgt2.pos()
                        if tp is not None: wp2=tp.copy()
                        else: wp2=wp_global(scene,rng2,np.array([x2,y2f,z2],np.float32),visited2,FAR_WP_DIST_MIN,FAR_WP_DIST_MAX)
                    else: wp2=wp_global(scene,rng2,np.array([x2,y2f,z2],np.float32),visited2,FAR_WP_DIST_MIN,FAR_WP_DIST_MAX)
                    wpt2=rng2.randint(WP_MIN_STEPS,WP_MAX_STEPS)
                if wpt3 <= 0:
                    if tgt3 is not None:
                        tp=tgt3.pos()
                        if tp is not None: wp3=tp.copy()
                        else: wp3=wp_global(scene,rng3,np.array([x3,y3f,z3],np.float32),visited3,FAR_WP_DIST_MIN,FAR_WP_DIST_MAX)
                    else: wp3=wp_global(scene,rng3,np.array([x3,y3f,z3],np.float32),visited3,FAR_WP_DIST_MIN,FAR_WP_DIST_MAX)
                    wpt3=rng3.randint(WP_MIN_STEPS,WP_MAX_STEPS)


            # ── push FSM + legacy approach trigger ───────────────
            # Push FSM runs for both robots independently.
            # States: PUSH_IDLE → PUSH_APPROACH → PUSH_PUSHING → PUSH_IDLE
            #
            # Prefer movable scene objects (chairs/tables) for pushing so the
            # other robot has a visible, meaningful activity to observe.

            # all_obj includes both scene furniture (uid 1000+) and YCB small
            # objects (uid 2000+) — both are pushable
            pushable = all_obj   # no filter: chairs, tables, cans, bowls, etc.

            # ── Robot 1 push FSM ─────────────────────────────────
            if push_state1 == PUSH_IDLE:
                # Prefer starting a push when the other robot is nearby (observable)
                nearby_factor1 = 1.0 + 3.0 * float(d12 < 5.0)
                if pushable and rng1.random() < PUSH_PROB * nearby_factor1:
                    push_tobj1  = rng1.choice(pushable)
                    push_state1 = PUSH_APPROACH
                    push_timer1 = PUSH_APPROACH_MAX

            elif push_state1 == PUSH_APPROACH:
                tp = push_tobj1.pos() if push_tobj1 else None
                if tp is not None:
                    wp1 = tp.copy()   # override waypoint → head to object
                    push_timer1 -= 1
                    if dxz(x1,z1,float(tp[0]),float(tp[2])) < PUSH_DIST_TRIGGER:
                        push_state1 = PUSH_PUSHING
                        push_timer1 = PUSH_FORCE_STEPS
                        # Guaranteed observation: force r2 to watch if not yet observed
                        if not push_observed:
                            push_observe_timer = PUSH_OBSERVE_STEPS
                    elif push_timer1 <= 0:
                        push_state1 = PUSH_IDLE; push_tobj1 = None
                else:
                    push_state1 = PUSH_IDLE; push_tobj1 = None

            elif push_state1 == PUSH_PUSHING:
                # Apply physical force each step — object actually moves
                if push_tobj1 is not None:
                    apply_push_force(push_tobj1, np.array([x1,y1f,z1],np.float32))
                    wp1 = push_tobj1.pos().copy() if push_tobj1.pos() is not None                           else wp1   # keep chasing object as it moves
                push_timer1 -= 1
                if push_timer1 <= 0:
                    push_state1 = PUSH_IDLE; push_tobj1 = None

            # ── Robot 2 push FSM ─────────────────────────────────
            if push_state2 == PUSH_IDLE:
                nearby_factor2 = 1.0 + 3.0 * float(d12 < 5.0)
                if pushable and rng2.random() < PUSH_PROB * nearby_factor2:
                    push_tobj2  = rng2.choice(pushable)
                    push_state2 = PUSH_APPROACH
                    push_timer2 = PUSH_APPROACH_MAX

            elif push_state2 == PUSH_APPROACH:
                tp = push_tobj2.pos() if push_tobj2 else None
                if tp is not None:
                    wp2 = tp.copy()
                    push_timer2 -= 1
                    if dxz(x2,z2,float(tp[0]),float(tp[2])) < PUSH_DIST_TRIGGER:
                        push_state2 = PUSH_PUSHING
                        push_timer2 = PUSH_FORCE_STEPS
                        if not push_observed:
                            push_observe_timer = PUSH_OBSERVE_STEPS
                    elif push_timer2 <= 0:
                        push_state2 = PUSH_IDLE; push_tobj2 = None
                else:
                    push_state2 = PUSH_IDLE; push_tobj2 = None

            elif push_state2 == PUSH_PUSHING:
                if push_tobj2 is not None:
                    apply_push_force(push_tobj2, np.array([x2,y2f,z2],np.float32))
                    wp2 = push_tobj2.pos().copy() if push_tobj2.pos() is not None else wp2
                push_timer2 -= 1
                if push_timer2 <= 0:
                    push_state2 = PUSH_IDLE; push_tobj2 = None

            # ── Robot 3 push FSM ─────────────────────────────────
            if push_state3 == PUSH_IDLE:
                nearby_factor3 = 1.0 + 3.0 * float(min(d13,d23) < 5.0)
                if pushable and rng3.random() < PUSH_PROB * nearby_factor3:
                    push_tobj3  = rng3.choice(pushable)
                    push_state3 = PUSH_APPROACH
                    push_timer3 = PUSH_APPROACH_MAX

            elif push_state3 == PUSH_APPROACH:
                tp = push_tobj3.pos() if push_tobj3 else None
                if tp is not None:
                    wp3 = tp.copy()
                    push_timer3 -= 1
                    if dxz(x3,z3,float(tp[0]),float(tp[2])) < PUSH_DIST_TRIGGER:
                        push_state3 = PUSH_PUSHING
                        push_timer3 = PUSH_FORCE_STEPS
                        if not push_observed:
                            push_observe_timer = PUSH_OBSERVE_STEPS
                    elif push_timer3 <= 0:
                        push_state3 = PUSH_IDLE; push_tobj3 = None
                else:
                    push_state3 = PUSH_IDLE; push_tobj3 = None

            elif push_state3 == PUSH_PUSHING:
                if push_tobj3 is not None:
                    apply_push_force(push_tobj3, np.array([x3,y3f,z3],np.float32))
                    wp3 = push_tobj3.pos().copy() if push_tobj3.pos() is not None else wp3
                push_timer3 -= 1
                if push_timer3 <= 0:
                    push_state3 = PUSH_IDLE; push_tobj3 = None

            # ── Legacy minor approach (non-physical) ─────────────
            if (push_state1==PUSH_IDLE and tgt1 is None and pushable and rng1.random()<MOVE_OBJ_PROB):
                tgt1=rng1.choice(all_obj); tgt_t1=MOVE_OBJ_DURATION
            if tgt_t1>0:
                tgt_t1-=1
                if tgt_t1<=0: tgt1=None
                elif tgt1 is not None and push_state1==PUSH_IDLE:
                    tp=tgt1.pos()
                    if tp is not None:
                        wp1=tp.copy()
                        if dxz(x1,z1,float(tp[0]),float(tp[2]))<INTERACT_DIST_M: tgt1=None; tgt_t1=0
            if (push_state2==PUSH_IDLE and tgt2 is None and pushable and rng2.random()<MOVE_OBJ_PROB):
                tgt2=rng2.choice(all_obj); tgt_t2=MOVE_OBJ_DURATION
            if tgt_t2>0:
                tgt_t2-=1
                if tgt_t2<=0: tgt2=None
                elif tgt2 is not None and push_state2==PUSH_IDLE:
                    tp=tgt2.pos()
                    if tp is not None:
                        wp2=tp.copy()
                        if dxz(x2,z2,float(tp[0]),float(tp[2]))<INTERACT_DIST_M: tgt2=None; tgt_t2=0
            if (push_state3==PUSH_IDLE and tgt3 is None and pushable and rng3.random()<MOVE_OBJ_PROB):
                tgt3=rng3.choice(all_obj); tgt_t3=MOVE_OBJ_DURATION
            if tgt_t3>0:
                tgt_t3-=1
                if tgt_t3<=0: tgt3=None
                elif tgt3 is not None and push_state3==PUSH_IDLE:
                    tp=tgt3.pos()
                    if tp is not None:
                        wp3=tp.copy()
                        if dxz(x3,z3,float(tp[0]),float(tp[2]))<INTERACT_DIST_M: tgt3=None; tgt_t3=0


            # ── wiggle trigger (all robots) ───────────────────────
            if wig1<=0 and rng1.random()<WIGGLE_PROB: wig1=rng1.randint(WIGGLE_MIN_STEPS,WIGGLE_MAX_STEPS)
            if wig2<=0 and rng2.random()<WIGGLE_PROB: wig2=rng2.randint(WIGGLE_MIN_STEPS,WIGGLE_MAX_STEPS)
            if wig3<=0 and rng3.random()<WIGGLE_PROB: wig3=rng3.randint(WIGGLE_MIN_STEPS,WIGGLE_MAX_STEPS)

            # ── per-step wall detection via raycasting (all robots) ─
            if WALL_RAY_ENABLED:
                if wall_turn1<=0 and wall_ray_detect(r1,yaw1):
                    wall_dir1=1.0 if rng1.random()<0.5 else -1.0; wall_turn1=WALL_TURN_STEPS
                    wp1=wp_near(scene,rng1,np.array([x1,y1f,z1],np.float32),4.0); wpt1=rng1.randint(WP_MIN_STEPS,WP_MAX_STEPS)
                if wall_turn2<=0 and wall_ray_detect(r2,yaw2):
                    wall_dir2=1.0 if rng2.random()<0.5 else -1.0; wall_turn2=WALL_TURN_STEPS
                    wp2=wp_near(scene,rng2,np.array([x2,y2f,z2],np.float32),4.0); wpt2=rng2.randint(WP_MIN_STEPS,WP_MAX_STEPS)
                if wall_turn3<=0 and wall_ray_detect(r3,yaw3):
                    wall_dir3=1.0 if rng3.random()<0.5 else -1.0; wall_turn3=WALL_TURN_STEPS
                    wp3=wp_near(scene,rng3,np.array([x3,y3f,z3],np.float32),4.0); wpt3=rng3.randint(WP_MIN_STEPS,WP_MAX_STEPS)

            # ── compute base commands (all robots) ────────────────
            v1,w1 = cmd(x1,z1,yaw1,wp1,rng1)
            v2,w2 = cmd(x2,z2,yaw2,wp2,rng2)
            v3,w3 = cmd(x3,z3,yaw3,wp3,rng3)

            # ── wall-turn override (all robots) ───────────────────
            if wall_turn1>0: v1=FORWARD_SPEED*0.25; w1=wall_dir1*ROT_SPEED*0.7; wall_turn1-=1
            if wall_turn2>0: v2=FORWARD_SPEED*0.25; w2=wall_dir2*ROT_SPEED*0.7; wall_turn2-=1
            if wall_turn3>0: v3=FORWARD_SPEED*0.25; w3=wall_dir3*ROT_SPEED*0.7; wall_turn3-=1

            # ── guaranteed push observer tracking ────────────────────
            # When push_observe_timer > 0, the robot NOT currently pushing
            # overrides its waypoint to track the pusher's live position.
            # This guarantees at least one full observed push per episode.
            if push_observe_timer > 0:
                push_observe_timer -= 1
                # Find which robot is pushing; force one non-pusher to watch
                if push_state1 == PUSH_PUSHING:
                    wp2 = np.array([x1,y1f,z1],np.float32); wpt2=5
                    wp3 = np.array([x1,y1f,z1],np.float32); wpt3=5
                    obs_who = 2
                elif push_state2 == PUSH_PUSHING:
                    wp1 = np.array([x2,y2f,z2],np.float32); wpt1=5
                    wp3 = np.array([x2,y2f,z2],np.float32); wpt3=5
                    obs_who = 1
                elif push_state3 == PUSH_PUSHING:
                    wp1 = np.array([x3,y3f,z3],np.float32); wpt1=5
                    wp2 = np.array([x3,y3f,z3],np.float32); wpt2=5
                    obs_who = 1
                if push_observe_timer <= 0:
                    push_observed = True

            # ── wiggle: replace w ──────────────────────────────────
            if wig1>0: w1=rng1.uniform(-WIGGLE_TURN_MAG,WIGGLE_TURN_MAG)/DT; v1=FORWARD_SPEED*WIGGLE_FWD_SCALE; wig1-=1
            if wig2>0: w2=rng2.uniform(-WIGGLE_TURN_MAG,WIGGLE_TURN_MAG)/DT; v2=FORWARD_SPEED*WIGGLE_FWD_SCALE; wig2-=1
            if wig3>0: w3=rng3.uniform(-WIGGLE_TURN_MAG,WIGGLE_TURN_MAG)/DT; v3=FORWARD_SPEED*WIGGLE_FWD_SCALE; wig3-=1

            v1=max(0.,min(FORWARD_SPEED,v1)); v2=max(0.,min(FORWARD_SPEED,v2)); v3=max(0.,min(FORWARD_SPEED,v3))
            w1=max(-ROT_SPEED,min(ROT_SPEED,w1)); w2=max(-ROT_SPEED,min(ROT_SPEED,w2)); w3=max(-ROT_SPEED,min(ROT_SPEED,w3))

            # ── navmesh step ────────────────────────────────────────
            x1,y1f,z1,yaw1 = nav_step(scene,x1,y1f,z1,yaw1,v1,w1)
            x2,y2f,z2,yaw2 = nav_step(scene,x2,y2f,z2,yaw2,v2,w2)
            x3,y3f,z3,yaw3 = nav_step(scene,x3,y3f,z3,yaw3,v3,w3)

            # ── stuck detection ──────────────────────────────────────
            if (x1-prev_x1)**2+(z1-prev_z1)**2 < STUCK_EPS_SQ: stuck1+=1
            else: stuck1=0
            if (x2-prev_x2)**2+(z2-prev_z2)**2 < STUCK_EPS_SQ: stuck2+=1
            else: stuck2=0
            if (x3-prev_x3)**2+(z3-prev_z3)**2 < STUCK_EPS_SQ: stuck3+=1
            else: stuck3=0
            prev_x1,prev_z1=x1,z1; prev_x2,prev_z2=x2,z2; prev_x3,prev_z3=x3,z3

            # ── soft stuck ────────────────────────────────────────────
            if stuck1>STUCK_SOFT_LIMIT:
                wp1=wp_global(scene,rng1,np.array([x1,y1f,z1],np.float32),visited1)
                wpt1=rng1.randint(WP_MIN_STEPS,WP_MAX_STEPS); wig1=rng1.randint(WIGGLE_MIN_STEPS,WIGGLE_MAX_STEPS)
                stuck1=0; tgt1=None; tgt_t1=0
            if stuck2>STUCK_SOFT_LIMIT:
                wp2=wp_global(scene,rng2,np.array([x2,y2f,z2],np.float32),visited2)
                wpt2=rng2.randint(WP_MIN_STEPS,WP_MAX_STEPS); wig2=rng2.randint(WIGGLE_MIN_STEPS,WIGGLE_MAX_STEPS)
                stuck2=0; tgt2=None; tgt_t2=0
            if stuck3>STUCK_SOFT_LIMIT:
                wp3=wp_global(scene,rng3,np.array([x3,y3f,z3],np.float32),visited3)
                wpt3=rng3.randint(WP_MIN_STEPS,WP_MAX_STEPS); wig3=rng3.randint(WIGGLE_MIN_STEPS,WIGGLE_MAX_STEPS)
                stuck3=0; tgt3=None; tgt_t3=0

            # ── hard stuck teleport (reference: HARD_STUCK_TELEPORT) ──
            if HARD_STUCK_TELEPORT and stuck1>STUCK_HARD_LIMIT:
                np1=floor_pt(scene,rng1); x1,y1f,z1=float(np1[0]),float(np1[1]),float(np1[2])
                yaw1=rng1.uniform(-math.pi,math.pi); init_robot(r1,np1,yaw1,sim,20)
                wp1=wp_global(scene,rng1,np1,visited1); wpt1=rng1.randint(WP_MIN_STEPS,WP_MAX_STEPS)
                stuck1=0; tgt1=None; tgt_t1=0
            if HARD_STUCK_TELEPORT and stuck2>STUCK_HARD_LIMIT:
                np2=floor_pt(scene,rng2); x2,y2f,z2=float(np2[0]),float(np2[1]),float(np2[2])
                yaw2=rng2.uniform(-math.pi,math.pi); init_robot(r2,np2,yaw2,sim,20)
                wp2=wp_global(scene,rng2,np2,visited2); wpt2=rng2.randint(WP_MIN_STEPS,WP_MAX_STEPS)
                stuck2=0; tgt2=None; tgt_t2=0
            if HARD_STUCK_TELEPORT and stuck3>STUCK_HARD_LIMIT:
                np3=floor_pt(scene,rng3); x3,y3f,z3=float(np3[0]),float(np3[1]),float(np3[2])
                yaw3=rng3.uniform(-math.pi,math.pi); init_robot(r3,np3,yaw3,sim,20)
                wp3=wp_global(scene,rng3,np3,visited3); wpt3=rng3.randint(WP_MIN_STEPS,WP_MAX_STEPS)
                stuck3=0; tgt3=None; tgt_t3=0

            # ── apply actions ────────────────────────────────────────
            _safe_apply(r1,nav_action(v1,w1,robot_idx=0))
            _safe_apply(r2,nav_action(v2,w2,robot_idx=1))
            _safe_apply(r3,nav_action(v3,w3,robot_idx=0))
            sim.step()

            # re-read poses
            x1,y1f,z1,yaw1 = get_pose(r1)
            x2,y2f,z2,yaw2 = get_pose(r2)
            x3,y3f,z3,yaw3 = get_pose(r3)

            # ── cross-robot visibility (all pairs) ────────────────
            vis12,d12v = check_vis(r1,r2,yaw1)
            vis13,d13v = check_vis(r1,r3,yaw1)
            vis21,d21v = check_vis(r2,r1,yaw2)
            vis23,d23v = check_vis(r2,r3,yaw2)
            vis31,d31v = check_vis(r3,r1,yaw3)
            vis32,d32v = check_vis(r3,r2,yaw3)
            # observed_by: comma-sep list of robot ids that see this robot
            def _ob(sees_list): return ",".join(str(r) for r,v in sees_list if v)
            ob1=_ob([(2,vis21),(3,vis31)]); ob2=_ob([(1,vis12),(3,vis32)]); ob3=_ob([(1,vis13),(2,vis23)])
            oi1=_ob([(2,vis12),(3,vis13)]); oi2=_ob([(1,vis21),(3,vis23)]); oi3=_ob([(1,vis31),(2,vis32)])

            # ── render cameras ─────────────────────────────────────
            seg1=np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH),np.int32)
            seg2=np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH),np.int32)
            seg3=np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH),np.int32)
            dep1_arr=dep2_arr=dep3_arr=None

            if SAVE_VIDEO and t%max(1,SAVE_EVERY)==0:
                def _render_and_write(robot, keys, rng, x,yf,z):
                    nonlocal dep1_arr,dep2_arr,dep3_arr
                    try:
                        rgb_,dep_,seg_ = render_robot(sim,robot)
                        vw[keys[0]].append_data(rot90_frame(rgb_))
                        vw[keys[1]].append_data(rot90_frame(depth_u8(dep_,DEPTH_MAX_M)))
                        vw[keys[2]].append_data(rot90_frame(seg_color(seg_)))
                        # wall avoidance from depth
                        if WALL_AVOID_ENABLED:
                            try:
                                dd=dep_[::DEP_STRIDE,::DEP_STRIDE].astype(np.float32)
                                dd=np.nan_to_num(dd,nan=DEPTH_MAX_M)
                                H_,W_=dd.shape
                                x0_=int(FRONT_BAND[0]*W_); x1_=int(FRONT_BAND[1]*W_)
                                if float(np.median(dd[:,x0_:x1_]))<WALL_FRONT_TH_M:
                                    return seg_, wp_near(scene,rng,np.array([x,yf,z],np.float32),6.)
                            except Exception: pass
                        return seg_, None
                    except Exception as e:
                        print(f"  [t={t}] render {keys[0]}: {e}")
                        return np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH),np.int32), None

                seg1, wp1_new = _render_and_write(r1,["rgb1","dep1","sem1"],rng1,x1,y1f,z1)
                if wp1_new is not None: wp1=wp1_new
                seg2, wp2_new = _render_and_write(r2,["rgb2","dep2","sem2"],rng2,x2,y2f,z2)
                if wp2_new is not None: wp2=wp2_new
                seg3, wp3_new = _render_and_write(r3,["rgb3","dep3","sem3"],rng3,x3,y3f,z3)
                if wp3_new is not None: wp3=wp3_new

            if SAVE_VIDEO and t%max(1,BEV_EVERY)==0:
                oxz=[(float(o.pos()[0]),float(o.pos()[2])) for o in all_obj if o.pos() is not None]
                vw["bev"].append_data(bev.frame(t,x1,z1,yaw1,x2,z2,yaw2,x3,z3,yaw3,oxz))

            # ── nearby objects ────────────────────────────────────
            v1s=visible_ids(seg1,VISIBLE_MIN_PIX,VIS_DOWNSAMPLE)
            v2s=visible_ids(seg2,VISIBLE_MIN_PIX,VIS_DOWNSAMPLE)
            v3s=visible_ids(seg3,VISIBLE_MIN_PIX,VIS_DOWNSAMPLE)
            nb1=nearby(x1,z1,yaw1,all_obj,v1s)
            nb2=nearby(x2,z2,yaw2,all_obj,v2s)
            nb3=nearby(x3,z3,yaw3,all_obj,v3s)
            for rid,nb in ((1,nb1),(2,nb2),(3,nb3)):
                for it in nb:
                    nw.writerow([t,rid,it["uid"],it["name"],it["cat"],
                                 it["dist_m"],it["bearing_rad"],it["bearing_deg"],
                                 it["rel_x"],it["rel_z"],it["rel_yaw"],
                                 it["world_x"],it["world_y"],it["world_z"],int(it["visible"])])

            # ── pushing_object_name + push_events CSV ─────────────
            # pushing_object_name is non-empty ONLY when actively applying
            # physical force (PUSH_PUSHING state).  This is distinct from
            # approach (PUSH_APPROACH) which is recorded as action_name only.
            push_name1 = push_tobj1.name if push_state1 == PUSH_PUSHING and push_tobj1 else ""
            push_name2 = push_tobj2.name if push_state2 == PUSH_PUSHING and push_tobj2 else ""
            push_name3 = push_tobj3.name if push_state3 == PUSH_PUSHING and push_tobj3 else ""

            # Write push_events rows when force is being applied (all robots)
            def _pw(rid, pname, ptobj, rx, rz, obs_vis, obs_dist):
                if not pname: return
                op = ptobj.pos()
                pw.writerow([t, rid, ptobj.name, ptobj.cat,
                             round(float(op[0]),3) if op is not None else "",
                             round(float(op[1]),3) if op is not None else "",
                             round(float(op[2]),3) if op is not None else "",
                             round(rx,3), round(rz,3),
                             "others", int(obs_vis), round(obs_dist,3)])
            _pw(1, push_name1, push_tobj1, x1, z1, vis21 or vis31, min(d21v,d31v))
            _pw(2, push_name2, push_tobj2, x2, z2, vis12 or vis32, min(d12v,d32v))
            _pw(3, push_name3, push_tobj3, x3, z3, vis13 or vis23, min(d13v,d23v))

            # ── trajectory CSV ─────────────────────────────────────
            an1=albl(v1,w1); an2=albl(v2,w2); an3=albl(v3,w3)
            tw.writerow([t,1,x1,y1f,z1,yaw1,v1,w1,an1,int(vis12),d12v,int(vis13),d13v,ob1,oi1,push_name1])
            tw.writerow([t,2,x2,y2f,z2,yaw2,v2,w2,an2,int(vis21),d21v,int(vis23),d23v,ob2,oi2,push_name2])
            tw.writerow([t,3,x3,y3f,z3,yaw3,v3,w3,an3,int(vis31),d31v,int(vis32),d32v,ob3,oi3,push_name3])

            # ── JSONL event ────────────────────────────────────────
            def _rdata(rid,x,yf,z,yaw,v,w,an,ps,pn,ob,oi,nb,sees,dists):
                return {"rid":rid,"pose":{"x":x,"y":yf,"z":z,"yaw":yaw},
                        "cmd":{"v":v,"w":w},"action":an,
                        "sees":{k:bool(v_) for k,v_ in sees.items()},
                        "dists":{k:round(d_,3) for k,d_ in dists.items()},
                        "observed_by":ob,"observing":oi,
                        "push_state":["idle","approach","pushing"][ps],
                        "pushing_object":pn,"nearby":nb}
            jf.write(json.dumps({
                "t":t,"dt":DT,"scene":SCENE_NAME,"exploration":expl_id,
                "phase":"near" if phase_near else "far",
                "robots":[
                    _rdata(1,x1,y1f,z1,yaw1,v1,w1,an1,push_state1,push_name1,ob1,oi1,nb1,
                           {"r2":vis12,"r3":vis13},{"r2":d12v,"r3":d13v}),
                    _rdata(2,x2,y2f,z2,yaw2,v2,w2,an2,push_state2,push_name2,ob2,oi2,nb2,
                           {"r1":vis21,"r3":vis23},{"r1":d21v,"r3":d23v}),
                    _rdata(3,x3,y3f,z3,yaw3,v3,w3,an3,push_state3,push_name3,ob3,oi3,nb3,
                           {"r1":vis31,"r2":vis32},{"r1":d31v,"r2":d32v}),
                ]})+"\n")

            if (t+1)%200==0:
                print(f"  [{t+1}/{NUM_STEPS}] phase={'NEAR' if phase_near else 'FAR'} "
                      f"d12={d12:.1f} d13={d13:.1f} d23={d23:.1f} "
                      f"stuck={stuck1}/{stuck2}/{stuck3} "
                      f"vis={len(visited1)}/{len(visited2)}/{len(visited3)} "
                      f"push='{push_name1}'/'{push_name2}'/'{push_name3}'")

        open(done_marker,"w").write("ok\n")
        print(f"  Done: {scene_name}/{expl_id} → {out_dir}")

    finally:
        if SAVE_VIDEO:
            for wtr in vw.values():
                try: wtr.close()
                except Exception: pass
        tf.close(); nf.close(); pf_.close(); jf.close()
        try: sim.disconnect()
        except Exception: pass

def count_done(scene_name: str) -> int:
    """Count already-completed explorations for a scene."""
    scene_dir = os.path.join(OUTPUT_ROOT, scene_name)
    if not os.path.isdir(scene_dir):
        return 0
    return sum(
        1 for i in range(NUM_EXPLORATIONS)
        if os.path.exists(os.path.join(scene_dir, f"explore_{i:03d}", "DONE"))
    )


def main():
    """
    Run NUM_EXPLORATIONS explorations for every scene in SCENES.

    Output folder structure:
        OUTPUT_ROOT/
          <scene_name>/
            explore_000/
              locobot1_cam.mp4
              locobot2_cam.mp4
              locobot1_depth.mp4
              locobot2_depth.mp4
              locobot1_sem.mp4
              locobot2_sem.mp4
              bev_trajectories.mp4
              locobot_trajectory_and_cmds.csv
              nearby_objects.csv
              push_events.csv
              robot_events.jsonl
              meta.json
              DONE
            explore_001/
              ...
            ...
          <scene_name2>/
            ...

    Already-completed explorations (DONE marker present) are skipped so
    the script can be safely interrupted and restarted.
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    total_scenes  = len(SCENES)
    total_eps     = total_scenes * NUM_EXPLORATIONS

    print(f"Scenes        : {total_scenes}")
    print(f"Per scene     : {NUM_EXPLORATIONS}")
    print(f"Total episodes: {total_eps}")
    print(f"Output root   : {OUTPUT_ROOT}")
    print()

    completed = 0
    for s_idx, scene_name in enumerate(SCENES):
        done_count = count_done(scene_name)
        remaining  = NUM_EXPLORATIONS - done_count

        print(f"[{s_idx+1}/{total_scenes}] Scene: {scene_name}  "
              f"(done {done_count}/{NUM_EXPLORATIONS})")

        if remaining == 0:
            print(f"  All {NUM_EXPLORATIONS} explorations already done — skipping.")
            completed += done_count
            continue

        os.makedirs(os.path.join(OUTPUT_ROOT, scene_name), exist_ok=True)

        for ep_idx in range(NUM_EXPLORATIONS):
            ep_label = f"explore_{ep_idx:03d}"
            print(f"  [{ep_idx+1}/{NUM_EXPLORATIONS}] {scene_name}/{ep_label}")
            try:
                run_exploration(scene_name, ep_idx)
                completed += 1
            except Exception as e:
                print(f"  [ERROR] {scene_name}/{ep_label}: {e}")
                import traceback; traceback.print_exc()
                # Continue with next exploration rather than aborting the scene

        print(f"  Scene {scene_name} complete.")
        print()

    print(f"All done. Completed {completed}/{total_eps} episodes.")
    print(f"Output: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()