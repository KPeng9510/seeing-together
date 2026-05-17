from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))
from finetune_coopsr import (  # noqa: E402
    DATASET_CONFIGS,
    build_split_items,
    find_robot_videos,
    extract_frames,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sp_cor")

MC4_ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
MC4_IDX_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

# ── architecture dims ────────────────────────────────────────────────────────
D_VIS = 512
D_POSE = 64
D_EVENT = 32
N_FUSION_LAYERS = 3
N_HEADS_FUSION = 4
MAX_ROBOTS = 4
N_DISTILL_PROMPTS = 8
D_PROMPT = D_VIS
PROMPT_DISTILL_WEIGHT = 0.1#0.3

# ── CLIP → FFT frequency-energy sampler hyperparameters ─────────────────────
DENSE_FRAMES_PER_VIDEO = 48
SAMPLED_FRAMES_PER_VIDEO = 8

CLIP_DIM = 512                  # frozen CLIP ViT-B/32 output dim

# FFT is applied over temporal windows of CLIP visual features, not raw RGB.
FOURIER_WINDOW = 8              # local temporal window length
FOURIER_BINS = 16               # max non-DC temporal frequency components to keep

# Stage 1: CLIP semantic pre-selection
CLIP_CANDIDATE_MULTIPLIER = 4   # top-M = max(K * multiplier, min_candidates)
CLIP_MIN_CANDIDATES = 24        # lower bound on M before FFT refinement

# Stage 2: FFT energy selection
FREQ_ENERGY_WEIGHT = 1.0        # final score weight for normalized FFT energy
SEMANTIC_REFINE_WEIGHT = 0.0    # 0.0 = pure FFT top-K after CLIP top-M
ENERGY_Z_CLIP = 2.0             # clip normalized FFT energy to [-2, 2]

# Optional diversity. Keep 0.0 for exact top-K by frequency energy.
REDUNDANCY_WEIGHT = 0.0         # CLIP-feature redundancy penalty
TEMPORAL_REDUNDANCY_WEIGHT = 0.0
MIN_TEMPORAL_GAP = 2

# Optional cross-robot diversity. Keep 0.0 for exact per-robot CLIP→FFT sampling.
CROSS_ROB_LAMBDA = 0.0

# Kept only for CLI/backward compatibility with your current constructor.
QVRS_TAU = 1.0
QVRS_GAMMA = 0.5
# ── Fourier fusion dims ──────────────────────────────────────────────────────
D_FOURIER_FUSION = 128          # dim of per-robot Fourier embedding fed into EST

# ── Precomputation defaults ──────────────────────────────────────────────────
CLIP_BATCH_SIZE = 256           # images per CLIP forward pass (A100 40 GB)
QWEN_BATCH_SIZE = 128           # images per Qwen ViT forward pass (A100 40 GB)
CLIP_TEXT_BATCH_SIZE = 1024     # texts per CLIP text-encoder forward pass (A100)
PRECOMPUTE_NUM_WORKERS = 4      # DataLoader workers for precomputation


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – Frozen CLIP helpers (training-free)
# ════════════════════════════════════════════════════════════════════════════

class FrozenCLIP:
    """
    Wraps CLIP ViT-B/32 (via HuggingFace transformers) as a pure inference utility.
    No parameters are registered; nothing is ever trained.
    """
    _MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, device: str = "cpu"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(self._MODEL_ID)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(self._MODEL_ID)

    @staticmethod
    def _extract_tensor(out):
        """Handle both tensor and BaseModelOutputWithPooling returns."""
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            v = getattr(out, attr, None)
            if v is not None:
                return v if v.dim() <= 2 else v[:, 0]
        raise ValueError(f"Cannot extract tensor from {type(out)}")


    @torch.no_grad()
    def encode_images(self, pil_images: list[Image.Image],
                      batch_size: int = CLIP_BATCH_SIZE) -> torch.Tensor:
        """Returns [N, CLIP_DIM] float32 on CPU. Batched to avoid OOM."""
        all_feats = []
        for i in range(0, len(pil_images), batch_size):
            chunk = pil_images[i:i + batch_size]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**inputs)
            feats = (out.image_embeds if hasattr(out, "image_embeds") else
                     out.pooler_output if hasattr(out, "pooler_output") else out).float().cpu()
            all_feats.append(F.normalize(feats, dim=-1))
        return torch.cat(all_feats, dim=0)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Returns [N, CLIP_DIM] float32 on CPU."""
        inputs = self.processor(text=texts, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        feats = self._extract_tensor(self.model.get_text_features(**inputs)).float().cpu()
        return F.normalize(feats, dim=-1)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1b – Multi-GPU parallel CLIP precomputation
# ════════════════════════════════════════════════════════════════════════════

def _clip_worker_from_paths(gpu_id: int, work_items: list[tuple],
                            result_dict: dict, clip_batch_size: int,
                            dense_frames_per_video: int, max_robots: int):
    """
    Worker for multi-GPU CLIP extraction.  Decodes video frames on-the-fly
    so that only one group's frames are in RAM at a time — avoids pickling
    the entire image cache into every subprocess.

    work_items: list of (key, video_root, scene, exploration, n_robots)
    result_dict: shared dict; result_dict[(key, r)] = Tensor[T, CLIP_DIM]
    """
    device = f"cuda:{gpu_id}"
    clip = FrozenCLIP(device=device)
    T = dense_frames_per_video

    for key, video_root, scene, exploration, n_r in tqdm(
            work_items, desc=f"GPU-{gpu_id} CLIP", unit="group", position=gpu_id):
        # Decode this group's frames fresh, then discard after encoding
        robot_images: list[list] = []
        if video_root:
            vid_map = find_robot_videos(Path(video_root), scene, exploration)
            for _rid, vpath in sorted(vid_map.items()):
                frames = extract_frames(vpath, T)
                robot_images.append(frames if frames else [])
        while len(robot_images) < max_robots:
            robot_images.append([])
        for r in range(max_robots):
            frames = robot_images[r]
            if not frames:
                robot_images[r] = [Image.new("RGB", (128, 128))] * T
            elif len(frames) < T:
                robot_images[r] = frames + [frames[-1]] * (T - len(frames))
            else:
                robot_images[r] = frames[:T]

        for r in range(min(n_r, max_robots)):
            images = robot_images[r]
            if not images or all(img is None for img in images):
                result_dict[(key, r)] = torch.zeros(1, CLIP_DIM)
            else:
                feats = clip.encode_images(images, batch_size=clip_batch_size)
                result_dict[(key, r)] = feats


def precompute_clip_embeddings_parallel(
    all_items: list[dict],
    max_robots: int = MAX_ROBOTS,
    num_gpus: int = 4,
    clip_batch_size: int = CLIP_BATCH_SIZE,
    dense_frames_per_video: int = DENSE_FRAMES_PER_VIDEO,
) -> dict:
    """
    Precompute CLIP embeddings for all groups, decoding video frames
    inside each worker instead of passing a pre-built image cache.
    This keeps peak RAM proportional to one group's frames per worker,
    not the entire dataset.

    Args:
        all_items: list of dataset items (each has video_root, scene,
                   exploration, n_robots, config fields)
        max_robots: max robot slots
        num_gpus: number of GPUs to use
        dense_frames_per_video: frames to extract per video

    Returns:
        clip_cache: dict mapping group_key -> list[Tensor[T, CLIP_DIM]]
    """
    import torch.multiprocessing as mp

    num_gpus = min(num_gpus, torch.cuda.device_count())
    if num_gpus <= 0:
        log.warning("No GPUs available, falling back to CPU precomputation")
        num_gpus = 1

    # Collect unique groups; work_item = (key, video_root, scene, exploration, n_robots)
    seen: set = set()
    all_work: list[tuple] = []
    for item in all_items:
        key = (item.get("config", ""), item.get("scene", ""), item.get("exploration", ""))
        if key not in seen:
            seen.add(key)
            all_work.append((key,
                             item.get("video_root", ""),
                             item.get("scene", ""),
                             item.get("exploration", ""),
                             item.get("n_robots", 1)))

    log.info("Precomputing CLIP embeddings: %d robot-streams across %d GPUs",
             len(all_work) * max_robots, num_gpus)

    if num_gpus == 1:
        result: dict = {}
        _clip_worker_from_paths(0, all_work, result, clip_batch_size,
                                dense_frames_per_video, max_robots)
    else:
        gpu_work = [[] for _ in range(num_gpus)]
        for i, item in enumerate(all_work):
            gpu_work[i % num_gpus].append(item)

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        manager = mp.Manager()
        result = manager.dict()

        processes = []
        for gpu_id in range(num_gpus):
            if not gpu_work[gpu_id]:
                continue
            p = mp.Process(
                target=_clip_worker_from_paths,
                args=(gpu_id, gpu_work[gpu_id], result, clip_batch_size,
                      dense_frames_per_video, max_robots),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        result = dict(result)

    # Reshape into clip_cache[key] = [Tensor, ...] per robot
    clip_cache: dict[tuple, list[torch.Tensor]] = {}
    for key, *_ in all_work:
        robot_feats = []
        for r in range(max_robots):
            feat = result.get((key, r), torch.zeros(1, CLIP_DIM))
            robot_feats.append(feat)
        clip_cache[key] = robot_feats

    log.info("CLIP precomputation complete: %d groups cached", len(clip_cache))
    return clip_cache


def _clip_text_worker(gpu_id: int, texts: list[str], result_dict: dict,
                      batch_size: int):
    """
    Encode a shard of texts on one GPU.
    result_dict[text] = Tensor[CLIP_DIM]
    """
    device = f"cuda:{gpu_id}"
    clip = FrozenCLIP(device=device)
    for i in tqdm(range(0, len(texts), batch_size),
                  desc=f"GPU-{gpu_id} text", unit="batch", position=gpu_id):
        chunk = texts[i:i + batch_size]
        feats = clip.encode_text(chunk)   # [chunk_size, CLIP_DIM] on CPU
        for j, text in enumerate(chunk):
            result_dict[text] = feats[j]


def precompute_clip_text_embeddings(
    items: list[dict],
    num_gpus: int = 4,
    batch_size: int = CLIP_TEXT_BATCH_SIZE,
) -> dict[str, torch.Tensor]:
    """
    Precompute CLIP text embeddings for all unique prompt texts, split across
    all available GPUs.  Each GPU gets a FrozenCLIP instance and processes its
    shard with large batches, fully utilising A100 VRAM.

    Returns: dict mapping prompt_text -> Tensor[CLIP_DIM]
    """
    import torch.multiprocessing as mp

    # Collect unique prompt texts
    unique_texts: set[str] = set()
    for item in items:
        raw_text = item.get("text", "")
        fmt = item.get("answer_format", "MC4")
        options = item.get("options", None)
        if fmt == "MC4" and options and len(options) == 4:
            opt_text = " ".join(f"{MC4_IDX_MAP[i]}) {options[i]}" for i in range(4))
            prompt_text = (f"Question: {raw_text}\nOptions: {opt_text}\n"
                           f"Answer with a single letter only.\nAnswer:")
        else:
            prompt_text = f"Question: {raw_text}\nAnswer:"
        unique_texts.add(prompt_text)

    unique_list = list(unique_texts)
    n = len(unique_list)
    log.info("Precomputing CLIP text embeddings for %d unique prompts", n)

    num_gpus = min(num_gpus, torch.cuda.device_count())
    if num_gpus <= 0:
        log.warning("No GPUs found for text embeddings, falling back to CPU")
        # CPU fallback — load once, run with the tuned batch size
        clip = FrozenCLIP(device="cpu")
        text_cache: dict[str, torch.Tensor] = {}
        for i in range(0, n, batch_size):
            chunk = unique_list[i:i + batch_size]
            feats = clip.encode_text(chunk)
            for j, text in enumerate(chunk):
                text_cache[text] = feats[j]
        log.info("CLIP text precomputation complete: %d texts (CPU)", n)
        return text_cache

    if num_gpus == 1 or n <= batch_size:
        # Single GPU — no subprocess overhead
        text_cache = {}
        _clip_text_worker(0, unique_list, text_cache, batch_size)
        log.info("CLIP text precomputation complete: %d texts (1 GPU)", n)
        return text_cache

    # Split texts round-robin across GPUs for even load
    shards: list[list[str]] = [[] for _ in range(num_gpus)]
    for i, text in enumerate(unique_list):
        shards[i % num_gpus].append(text)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    manager = mp.Manager()
    result = manager.dict()

    processes = []
    for gpu_id in range(num_gpus):
        if not shards[gpu_id]:
            continue
        p = mp.Process(
            target=_clip_text_worker,
            args=(gpu_id, shards[gpu_id], result, batch_size),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    text_cache = dict(result)
    log.info("CLIP text precomputation complete: %d texts across %d GPUs",
             len(text_cache), num_gpus)
    return text_cache


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1c – Multi-GPU parallel Qwen backbone precomputation
# ════════════════════════════════════════════════════════════════════════════

def _decode_group(video_root: str, scene: str, exploration: str,
                  n_r: int, T: int, max_robots: int) -> list[list]:
    """
    Decode all robot video streams for one group into padded PIL image lists.
    Kept as a standalone function so it can be called from a background thread.
    """
    robot_images: list[list] = []
    if video_root:
        vid_map = find_robot_videos(Path(video_root), scene, exploration)
        for _rid, vpath in sorted(vid_map.items()):
            frames = extract_frames(vpath, T)
            robot_images.append(frames if frames else [])
    while len(robot_images) < max_robots:
        robot_images.append([])
    for r in range(max_robots):
        frames = robot_images[r]
        if not frames:
            robot_images[r] = [Image.new("RGB", (128, 128))] * T
        elif len(frames) < T:
            robot_images[r] = frames + [frames[-1]] * (T - len(frames))
        else:
            robot_images[r] = frames[:T]
    return robot_images


def _get_vit_hidden_size(vit_module, full_model_config) -> int:
    """Return the vision encoder output dimension."""
    for attr in ("hidden_size", "embed_dim", "d_model"):
        v = getattr(getattr(vit_module, "config", None), attr, None)
        if v is not None:
            return v
    vcfg = getattr(full_model_config, "vision_config", None)
    if vcfg is not None:
        for attr in ("hidden_size", "embed_dim", "d_model"):
            v = getattr(vcfg, attr, None)
            if v is not None:
                return v
    tcfg = getattr(full_model_config, "text_config", full_model_config)
    return getattr(tcfg, "hidden_size", getattr(full_model_config, "hidden_size", 3584))


def _get_vit_output_dim(vit_module, full_model_config) -> int:
    """Return the actual forward-pass output dim of vit_module.

    For models with a patch-merger MLP (e.g. Qwen2.5-VL) the ViT config
    stores the *encoder* hidden size (e.g. 1280) while vit(x) returns the
    post-merger dim (e.g. 3584 for the 7B variant).  This function probes
    the merger MLP first so the returned value matches what encode_frames
    actually produces.
    """
    merger = getattr(vit_module, "merger", None)
    mlp = getattr(merger, "mlp", None)
    if mlp is not None:
        for layer in reversed(list(mlp)):
            out_features = getattr(layer, "out_features", None)
            if out_features is not None:
                return int(out_features)
    return _get_vit_hidden_size(vit_module, full_model_config)


def _qwen_visual_split_sizes(thw: torch.Tensor, hidden_len: int,
                             merge_size: int) -> list[int]:
    """Infer per-image token counts from Qwen-VL visual output.

    Some Qwen2.5-VL transformer versions return raw patch tokens
    (t*h*w), while others return spatially merged tokens
    (t*(h/merge)*(w/merge)).  Pick the one that matches the actual visual
    tower output so batched precompute does not crash on split().
    """
    thw_cpu = thw.detach().to("cpu").long()
    raw = (thw_cpu[:, 0] * thw_cpu[:, 1] * thw_cpu[:, 2]).tolist()
    m = max(1, int(merge_size or 1))
    merged = (
        thw_cpu[:, 0]
        * torch.div(thw_cpu[:, 1], m, rounding_mode="floor")
        * torch.div(thw_cpu[:, 2], m, rounding_mode="floor")
    ).tolist()

    raw_sum = int(sum(raw))
    merged_sum = int(sum(merged))
    if raw_sum == hidden_len:
        return [int(x) for x in raw]
    if merged_sum == hidden_len:
        return [int(x) for x in merged]
    if len(raw) > 0 and hidden_len % len(raw) == 0:
        return [hidden_len // len(raw)] * len(raw)
    raise RuntimeError(
        "Cannot split Qwen visual tokens: "
        f"hidden_len={hidden_len}, raw_sum={raw_sum}, "
        f"merged_sum={merged_sum}, n_images={len(raw)}, merge_size={m}"
    )


def _encode_images_qwen(images: list, vit, img_proc, device: str,
                        qwen_batch_size: int, d_backbone: int, _M: int
                        ) -> list[torch.Tensor]:
    """
    Run the Qwen ViT on a flat list of PIL images in large batches.
    Returns a list of [D] feature tensors (one per input image).
    """
    all_feats: list[torch.Tensor] = []
    for i in range(0, len(images), qwen_batch_size):
        sub = [img.convert("RGB") for img in images[i:i + qwen_batch_size]]
        with torch.no_grad():
            enc = img_proc(images=sub, return_tensors="pt")
            pv  = enc["pixel_values"].to(device, dtype=torch.bfloat16)
            thw = enc["image_grid_thw"].to(device)
            out = vit(pv, thw)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            split_sizes = _qwen_visual_split_sizes(thw, hidden.shape[0], _M)
            splits = hidden.split(split_sizes, dim=0)
            for seq_tokens, seq_thw in zip(splits, thw):
                temporal = int(seq_thw[0].item())
                feat_vec = seq_tokens.mean(dim=0).float().cpu()
                for _ in range(max(1, temporal)):
                    all_feats.append(feat_vec)
    return all_feats


def _qwen_worker_from_paths(gpu_id: int, work_items: list[tuple],
                            result_dict: dict, model_name: str,
                            qwen_batch_size: int, dense_frames_per_video: int,
                            max_robots: int):
    """
    Worker for multi-GPU Qwen ViT feature extraction.

    Two key optimisations vs the naive approach:
    1. A background thread prefetches and decodes the *next* group's frames
       while the GPU is encoding the *current* group — hides I/O latency.
    2. All robots in a group are concatenated into one flat image list and
       encoded together in large batches, maximising GPU utilisation.

    work_items: list of (key, video_root, scene, exploration, n_robots)
    result_dict[(key, r)] = Tensor[T, d_backbone]
    """
    import queue, threading
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    device = f"cuda:{gpu_id}"
    log.info("GPU-%d: Loading Qwen ViT from %s", gpu_id, model_name)
    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True)
    vit = getattr(getattr(full_model, "model", full_model), "visual").eval()
    d_backbone = _get_vit_output_dim(vit, full_model.config)
    del full_model
    torch.cuda.empty_cache()

    img_proc = proc.image_processor
    _M = 2
    try:
        _M = img_proc.merge_size
    except AttributeError:
        pass

    T = dense_frames_per_video

    # ── Prefetch queue: background thread decodes frames while GPU encodes ──
    # Buffer up to 4 decoded groups ahead so the GPU is never starved.
    prefetch_q: queue.Queue = queue.Queue(maxsize=4)

    def _decode_producer():
        for item in work_items:
            key, video_root, scene, exploration, n_r = item
            robot_images = _decode_group(video_root, scene, exploration,
                                         n_r, T, max_robots)
            prefetch_q.put((key, n_r, robot_images))
        prefetch_q.put(None)  # sentinel

    producer = threading.Thread(target=_decode_producer, daemon=True)
    producer.start()

    pbar = tqdm(total=len(work_items), desc=f"GPU-{gpu_id} Qwen",
                unit="group", position=gpu_id)

    while True:
        item = prefetch_q.get()
        if item is None:
            break

        key, n_r, robot_images = item
        n_active = min(n_r, max_robots)

        # Flatten all robots → one big image list; track slice boundaries
        flat_images: list = []
        slices: list[tuple[int, int]] = []  # (start, end) per robot
        for r in range(n_active):
            imgs = robot_images[r]
            start = len(flat_images)
            flat_images.extend(imgs)
            slices.append((start, len(flat_images)))

        # Single large GPU forward pass covering all robots in the group
        all_feats = _encode_images_qwen(flat_images, vit, img_proc, device,
                                        qwen_batch_size, d_backbone, _M)

        # Split features back per robot and store
        for r, (s, e) in enumerate(slices):
            robot_feats = all_feats[s:e]
            n = e - s
            if len(robot_feats) > n:
                robot_feats = robot_feats[:n]
            elif len(robot_feats) < n and robot_feats:
                robot_feats += [robot_feats[-1]] * (n - len(robot_feats))
            elif not robot_feats:
                robot_feats = [torch.zeros(d_backbone)] * n
            result_dict[(key, r)] = torch.stack(robot_feats)

        pbar.update(1)

    pbar.close()
    producer.join()
    del vit
    torch.cuda.empty_cache()


def precompute_qwen_embeddings_parallel(
    all_items: list[dict],
    model_name: str,
    max_robots: int = MAX_ROBOTS,
    num_gpus: int = 4,
    qwen_batch_size: int = QWEN_BATCH_SIZE,
    dense_frames_per_video: int = DENSE_FRAMES_PER_VIDEO,
) -> tuple[dict, int]:
    """
    Precompute Qwen ViT embeddings for all groups, decoding video frames
    inside each worker to avoid holding the entire image dataset in RAM.

    Returns:
        qwen_cache: dict[key] -> list[Tensor[T, d_backbone]] per robot
        d_backbone: int
    """
    import torch.multiprocessing as mp

    num_gpus = min(num_gpus, torch.cuda.device_count())
    if num_gpus <= 0:
        num_gpus = 1

    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        vcfg = getattr(cfg, "vision_config", None)
        d_backbone = (
            next((getattr(vcfg, a, None) for a in ("hidden_size", "embed_dim", "d_model")
                  if vcfg and getattr(vcfg, a, None)), None)
            or getattr(cfg, "hidden_size", None)
            or getattr(getattr(cfg, "text_config", cfg), "hidden_size", 3584)
        )
    except Exception:
        d_backbone = 1280  # Qwen2.5-VL-7B vision encoder default

    # Collect unique groups
    seen: set = set()
    all_work: list[tuple] = []
    for item in all_items:
        key = (item.get("config", ""), item.get("scene", ""), item.get("exploration", ""))
        if key not in seen:
            seen.add(key)
            all_work.append((key,
                             item.get("video_root", ""),
                             item.get("scene", ""),
                             item.get("exploration", ""),
                             item.get("n_robots", 1)))

    log.info("Precomputing Qwen ViT embeddings: %d robot-streams across %d GPUs",
             len(all_work) * max_robots, num_gpus)

    if num_gpus == 1:
        result: dict = {}
        _qwen_worker_from_paths(0, all_work, result, model_name, qwen_batch_size,
                                dense_frames_per_video, max_robots)
    else:
        gpu_work = [[] for _ in range(num_gpus)]
        for i, item in enumerate(all_work):
            gpu_work[i % num_gpus].append(item)

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        manager = mp.Manager()
        result = manager.dict()

        processes = []
        for gpu_id in range(num_gpus):
            if not gpu_work[gpu_id]:
                continue
            p = mp.Process(
                target=_qwen_worker_from_paths,
                args=(gpu_id, gpu_work[gpu_id], result, model_name, qwen_batch_size,
                      dense_frames_per_video, max_robots),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        result = dict(result)

    qwen_cache: dict[tuple, list[torch.Tensor]] = {}
    for key, *_ in all_work:
        robot_feats = []
        for r in range(max_robots):
            feat = result.get((key, r), torch.zeros(1, d_backbone))
            robot_feats.append(feat)
        qwen_cache[key] = robot_feats

    log.info("Qwen precomputation complete: %d groups, d_backbone=%d",
             len(qwen_cache), d_backbone)
    return qwen_cache, d_backbone


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – CLIP-feature temporal FFT energy extraction
# ════════════════════════════════════════════════════════════════════════════

def _safe_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Numerically stable z-score. If the input has almost no variance, returns zeros.
    """
    x = x.float()
    if x.numel() <= 1:
        return torch.zeros_like(x)
    std = x.std(unbiased=False)
    if not torch.isfinite(std) or std < eps:
        return torch.zeros_like(x)
    return (x - x.mean()) / (std + eps)


def _replicate_pad_time(x: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
    """
    Replicate-pad a [T, D] sequence along time.
    This avoids zero-padding artifacts at the beginning/end of videos.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected [T, D] tensor, got shape {tuple(x.shape)}")

    if x.shape[0] == 0:
        return x

    pieces = []
    if pad_left > 0:
        pieces.append(x[:1].expand(pad_left, -1))
    pieces.append(x)
    if pad_right > 0:
        pieces.append(x[-1:].expand(pad_right, -1))
    return torch.cat(pieces, dim=0)


def compute_spectral_features(
    clip_feats: torch.Tensor,   # [T, D_clip]
    window: int = FOURIER_WINDOW,
    n_bins: int = FOURIER_BINS,
    exclude_dc: bool = True,
    log_power: bool = True,
    normalize_input: bool = True,
) -> torch.Tensor:
    """
    Computes a local temporal FFT power profile for every frame.

    Important:
        FFT is applied over the temporal sequence of CLIP visual features,
        not over raw RGB pixels.

    Returns:
        spectral_profile: [T, n_bins]

    The returned profile is mostly for diagnostics / optional use. The sampler
    below converts it into a scalar frequency-energy score.
    """
    if clip_feats.ndim != 2:
        raise ValueError(f"clip_feats must be [T, D], got {tuple(clip_feats.shape)}")

    T, D = clip_feats.shape
    device = clip_feats.device

    if T == 0:
        return torch.zeros(0, n_bins, device=device)

    if T == 1:
        return torch.zeros(1, n_bins, device=device)

    x = clip_feats.float()
    if normalize_input:
        x = F.normalize(x, dim=-1)

    # Need at least length 2 for a meaningful temporal FFT.
    window = max(2, int(window))

    half = window // 2
    pad_left = half
    pad_right = window - half - 1

    padded = _replicate_pad_time(x, pad_left, pad_right)

    # padded.unfold(0, window, 1) gives [T, D, window].
    # Convert to [T, window, D].
    win = padded.unfold(0, window, 1).permute(0, 2, 1).contiguous()

    # Remove local mean so the DC component does not dominate.
    win = win - win.mean(dim=1, keepdim=True)

    # FFT along temporal window dimension.
    fft = torch.fft.rfft(win, dim=1)
    power = fft.abs().pow(2).mean(dim=-1)  # [T, F], averaged over CLIP dims

    start = 1 if exclude_dc else 0
    if start >= power.shape[1]:
        return torch.zeros(T, n_bins, device=device)

    profile = power[:, start:]  # remove DC if requested

    if log_power:
        profile = torch.log1p(profile)

    n_freq = profile.shape[1]
    out = torch.zeros(T, n_bins, device=device, dtype=profile.dtype)

    if n_freq <= n_bins:
        out[:, :n_freq] = profile
    else:
        # Average neighboring frequencies into n_bins bins.
        edges = torch.linspace(0, n_freq, n_bins + 1, device=device)
        edges = edges.round().long().clamp(0, n_freq)
        for b in range(n_bins):
            s = int(edges[b].item())
            e = int(edges[b + 1].item())
            if e > s:
                out[:, b] = profile[:, s:e].mean(dim=1)

    return out


def compute_temporal_fft_energy(
    clip_feats: torch.Tensor,   # [T, D_clip]
    window: int = FOURIER_WINDOW,
    n_freq_keep: int = FOURIER_BINS,
) -> torch.Tensor:
    """
    Converts the local FFT profile into one scalar motion/frequency-energy
    score per frame.

    Returns:
        energy: [T]
    """
    profile = compute_spectral_features(
        clip_feats=clip_feats,
        window=window,
        n_bins=n_freq_keep,
        exclude_dc=True,
        log_power=True,
        normalize_input=True,
    )

    if profile.numel() == 0:
        return torch.zeros(clip_feats.shape[0], device=clip_feats.device)

    # Sum across retained non-DC frequency components.
    energy = profile.sum(dim=-1)
    return energy


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – Small selection helpers
# ════════════════════════════════════════════════════════════════════════════

def _pad_or_trim_indices(
    idx: torch.Tensor,
    k: int,
    T: int,
) -> torch.Tensor:
    """
    Ensures selected indices have exactly length k.
    """
    device = idx.device

    if T <= 0:
        return torch.zeros(k, dtype=torch.long, device=device)

    if idx.numel() == 0:
        idx = torch.zeros(1, dtype=torch.long, device=device)

    idx = idx.long().clamp(0, T - 1)

    if idx.numel() >= k:
        return idx[:k]

    pad = idx[-1:].repeat(k - idx.numel())
    return torch.cat([idx, pad], dim=0)


def _uniform_indices(
    k: int,
    T: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Uniform fallback indices over a sequence of length T.
    """
    device = device or torch.device("cpu")
    if T <= 1:
        return torch.zeros(k, dtype=torch.long, device=device)
    return torch.linspace(0, T - 1, k, device=device).round().long()


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – CLIP-guided FFT frequency-energy frame sampler
# ════════════════════════════════════════════════════════════════════════════

class SE_MR2FS:
    """
    Spectral Energy-aware Multi-Robot Relevant Frame Sampler (SE-MR2FS).

    Paper role:
        1. For each robot, compute CLIP image-text similarity.
        2. Select top-M semantically relevant candidate frames.
        3. Compute temporal FFT energy over CLIP visual features.
        4. Select final top-K frames from the candidate set according to
           normalized FFT energy.

    Python name uses SE_MR2FS because hyphens and superscripts are not valid
    Python identifiers.
    """

    def __init__(
        self,
        total_budget: int = SAMPLED_FRAMES_PER_VIDEO,
        fourier_window: int = FOURIER_WINDOW,
        n_bins: int = FOURIER_BINS,
        qvrs_tau: float = QVRS_TAU,          # kept for CLI compatibility
        qvrs_gamma: float = QVRS_GAMMA,      # kept for CLI compatibility
        cross_robot_lambda: float = CROSS_ROB_LAMBDA,
        min_frames_per_robot: int = 1,
        candidate_multiplier: int = CLIP_CANDIDATE_MULTIPLIER,
        min_candidates: int = CLIP_MIN_CANDIDATES,
        freq_energy_weight: float = FREQ_ENERGY_WEIGHT,
        semantic_refine_weight: float = SEMANTIC_REFINE_WEIGHT,
        redundancy_weight: float = REDUNDANCY_WEIGHT,
        temporal_redundancy_weight: float = TEMPORAL_REDUNDANCY_WEIGHT,
        min_temporal_gap: int = MIN_TEMPORAL_GAP,
        energy_z_clip: float = ENERGY_Z_CLIP,
    ):
        # In the new sampler, total_budget means K frames PER ROBOT.
        self.frames_per_robot = int(total_budget)
        self.fourier_window = int(fourier_window)
        self.n_freq_keep = int(n_bins)

        # Kept only so your current argument plumbing still works.
        self.qvrs_tau = qvrs_tau
        self.qvrs_gamma = qvrs_gamma

        self.cross_lambda = float(cross_robot_lambda)
        self.min_frames = int(min_frames_per_robot)

        self.candidate_multiplier = int(candidate_multiplier)
        self.min_candidates = int(min_candidates)

        self.freq_energy_weight = float(freq_energy_weight)
        self.semantic_refine_weight = float(semantic_refine_weight)

        self.redundancy_weight = float(redundancy_weight)
        self.temporal_redundancy_weight = float(temporal_redundancy_weight)
        self.min_temporal_gap = int(min_temporal_gap)

        self.energy_z_clip = float(energy_z_clip)

    def _topm_clip_candidates(
        self,
        sim: torch.Tensor,   # [T]
        k: int,
    ) -> torch.Tensor:
        """
        Stage 1: choose top-M frames by CLIP image-text similarity.
        """
        T = sim.shape[0]
        if T == 0:
            return torch.zeros(0, dtype=torch.long, device=sim.device)

        M = max(k, self.min_candidates, k * self.candidate_multiplier)
        M = min(T, M)

        if M >= T:
            return torch.arange(T, device=sim.device, dtype=torch.long)

        _, idx = torch.topk(sim.float(), k=M, largest=True, sorted=False)

        # Sort by time only for stable downstream behavior.
        return idx.sort().values.long()

    def _cross_robot_diversity_bonus(
        self,
        clip_feats_per_robot: list[torch.Tensor],
        r: int,
        cand_idx: torch.Tensor,
        n_robots: int,
    ) -> torch.Tensor:
        """
        Optional cross-robot diversity bonus.

        If cross_robot_lambda == 0, this returns zeros and the sampler becomes
        pure per-robot CLIP→FFT selection.
        """
        device = cand_idx.device

        if self.cross_lambda <= 0.0 or n_robots <= 1 or cand_idx.numel() == 0:
            return torch.zeros(cand_idx.numel(), device=device)

        other_means = []
        for j in range(n_robots):
            if j == r:
                continue
            fj = clip_feats_per_robot[j].float().to(device)
            if fj.numel() == 0:
                continue
            fj = F.normalize(fj, dim=-1)
            other_means.append(fj.mean(dim=0))

        if not other_means:
            return torch.zeros(cand_idx.numel(), device=device)

        other_mean = torch.stack(other_means, dim=0).mean(dim=0)
        other_mean = F.normalize(other_mean, dim=0)

        cur = clip_feats_per_robot[r].float().to(device)
        cur = F.normalize(cur, dim=-1)
        cand_feats = cur[cand_idx]

        # Higher means less redundant with other robots' average view.
        bonus = 1.0 - (cand_feats @ other_mean).clamp(-1.0, 1.0)
        return _safe_zscore(bonus)

    def _select_topk_with_optional_redundancy(
        self,
        cand_idx: torch.Tensor,       # [M]
        base_score: torch.Tensor,     # [M]
        cand_clip_feats: torch.Tensor,  # [M, D], normalized
        k: int,
        T: int,
    ) -> torch.Tensor:
        """
        Stage 2: select final K frames.

        If redundancy weights are 0, this is exactly top-K by base_score.
        If redundancy weights are positive, this becomes a simple MMR-style
        diversity-aware top-K.
        """
        device = cand_idx.device
        M = cand_idx.numel()

        if M == 0:
            return _uniform_indices(k, T, device=device)

        k_eff = min(k, M)

        no_diversity = (
            self.redundancy_weight <= 0.0
            and self.temporal_redundancy_weight <= 0.0
        )

        if no_diversity:
            _, pos = torch.topk(base_score.float(), k=k_eff, largest=True, sorted=False)
            selected = cand_idx[pos].sort().values
            return _pad_or_trim_indices(selected, k, T)

        selected_positions: list[int] = []
        available = torch.ones(M, dtype=torch.bool, device=device)
        running_score = base_score.float().clone()

        for _ in range(k_eff):
            masked_score = running_score.masked_fill(~available, -1e9)
            pos = int(masked_score.argmax().item())

            selected_positions.append(pos)
            available[pos] = False

            # CLIP-feature redundancy penalty.
            if self.redundancy_weight > 0.0:
                sim_penalty = (cand_clip_feats @ cand_clip_feats[pos]).clamp(min=0.0)
                running_score = running_score - self.redundancy_weight * sim_penalty

            # Temporal near-duplicate penalty.
            if self.temporal_redundancy_weight > 0.0 and self.min_temporal_gap > 0:
                dt = (cand_idx - cand_idx[pos]).abs().float()
                close_penalty = (1.0 - dt / float(max(self.min_temporal_gap, 1))).clamp(min=0.0)
                running_score = running_score - self.temporal_redundancy_weight * close_penalty

        selected_pos_t = torch.tensor(selected_positions, dtype=torch.long, device=device)
        selected = cand_idx[selected_pos_t].sort().values
        return _pad_or_trim_indices(selected, k, T)

    @torch.no_grad()
    def select(
        self,
        clip_feats_per_robot: list[torch.Tensor],
        question_vec: torch.Tensor,
        n_robots: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[list[torch.Tensor], dict]:
        """
        Args:
            clip_feats_per_robot:
                list of length n_robots. Each tensor is [T, CLIP_DIM].
            question_vec:
                frozen CLIP text feature, [CLIP_DIM].
            n_robots:
                active robot count.

        Returns:
            selected_indices:
                list of length n_robots. Each tensor has shape [K].
            info:
                logging/debug dictionary.
        """
        if n_robots is None:
            n_robots = len(clip_feats_per_robot)

        n_robots = min(n_robots, len(clip_feats_per_robot))
        k = max(self.frames_per_robot, self.min_frames)

        if device is None:
            if clip_feats_per_robot:
                device = clip_feats_per_robot[0].device
            else:
                device = torch.device("cpu")

        if n_robots <= 0:
            return [], {
                "sampler": "SE-MR2FS",
                "budgets": [],
                "candidate_counts": [],
            }

        q = question_vec.float().to(device)
        q = F.normalize(q, dim=-1)

        # Move/normalize CLIP features.
        robot_feats: list[torch.Tensor] = []
        for r in range(n_robots):
            f = clip_feats_per_robot[r].float().to(device)
            if f.ndim != 2 or f.shape[0] == 0:
                f = torch.zeros(1, CLIP_DIM, device=device)
            f = F.normalize(f, dim=-1)
            robot_feats.append(f)

        selected_indices: list[torch.Tensor] = []

        candidate_counts: list[int] = []
        budgets: list[int] = []
        mean_clip_sim_per_robot: list[float] = []
        mean_freq_energy_per_robot: list[float] = []
        selected_clip_sim_mean: list[float] = []
        selected_freq_energy_mean: list[float] = []

        for r in range(n_robots):
            cf = robot_feats[r]
            T = cf.shape[0]

            if T <= 0:
                idx = torch.zeros(k, dtype=torch.long, device=device)
                selected_indices.append(idx)
                candidate_counts.append(0)
                budgets.append(k)
                mean_clip_sim_per_robot.append(0.0)
                mean_freq_energy_per_robot.append(0.0)
                selected_clip_sim_mean.append(0.0)
                selected_freq_energy_mean.append(0.0)
                continue

            # -----------------------------------------------------------------
            # Stage 1: CLIP semantic candidate set
            # -----------------------------------------------------------------
            sim = (cf @ q).clamp(-1.0, 1.0)  # [T]
            cand_idx = self._topm_clip_candidates(sim, k=k)  # [M]

            # -----------------------------------------------------------------
            # Stage 2: temporal FFT energy over CLIP visual features
            # -----------------------------------------------------------------
            energy = compute_temporal_fft_energy(
                clip_feats=cf,
                window=self.fourier_window,
                n_freq_keep=self.n_freq_keep,
            )  # [T]

            cand_sim = sim[cand_idx]
            cand_energy = energy[cand_idx]

            # Normalize only inside the candidate set.
            cand_sim_z = _safe_zscore(cand_sim)

            cand_energy_z = _safe_zscore(cand_energy)
            if self.energy_z_clip > 0:
                cand_energy_z = cand_energy_z.clamp(
                    -self.energy_z_clip,
                    self.energy_z_clip,
                )

            # Default final score is pure FFT energy after CLIP top-M because
            # SEMANTIC_REFINE_WEIGHT defaults to 0.0.
            final_score = (
                self.freq_energy_weight * cand_energy_z
                + self.semantic_refine_weight * cand_sim_z
            )

            # Optional cross-robot diversity.
            if self.cross_lambda > 0.0:
                cross_bonus = self._cross_robot_diversity_bonus(
                    clip_feats_per_robot=robot_feats,
                    r=r,
                    cand_idx=cand_idx,
                    n_robots=n_robots,
                )
                final_score = final_score + self.cross_lambda * cross_bonus

            cand_feats = cf[cand_idx]
            cand_feats = F.normalize(cand_feats, dim=-1)

            idx = self._select_topk_with_optional_redundancy(
                cand_idx=cand_idx,
                base_score=final_score,
                cand_clip_feats=cand_feats,
                k=k,
                T=T,
            )

            selected_indices.append(idx)

            candidate_counts.append(int(cand_idx.numel()))
            budgets.append(int(k))

            mean_clip_sim_per_robot.append(float(cand_sim.mean().item()))
            mean_freq_energy_per_robot.append(float(cand_energy.mean().item()))

            selected_clip_sim_mean.append(float(sim[idx].mean().item()))
            selected_freq_energy_mean.append(float(energy[idx].mean().item()))

        # Backward-compatible keys: evaluation currently reads qvrs_per_robot/budgets.
        # Here qvrs_per_robot is replaced by mean CLIP relevance, only for logging.
        info = {
            "sampler": "SE-MR2FS",
            "budgets": budgets,
            "candidate_counts": candidate_counts,
            "mean_clip_sim_per_robot": mean_clip_sim_per_robot,
            "mean_freq_energy_per_robot": mean_freq_energy_per_robot,
            "selected_clip_sim_mean": selected_clip_sim_mean,
            "selected_freq_energy_mean": selected_freq_energy_mean,

            # Legacy logging compatibility.
            "qvrs_per_robot": mean_clip_sim_per_robot,
            "qvrs_fused": float(max(mean_clip_sim_per_robot)) if mean_clip_sim_per_robot else 0.0,
        }

        return selected_indices, info


# Backward compatibility with previous code/checkpoints.
MS3ASCS = SE_MR2FS

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – Fourier embedding for EST
# ════════════════════════════════════════════════════════════════════════════

class SpectralTokenExtractor(nn.Module):
    """
    Spectral token extraction module used by SPI-MRF.

    Paper role:
        Computes compact robot-level Fourier/spectral tokens from sampled
        visual embeddings.
    """
    def __init__(self, d_clip: int, d_out: int = D_FOURIER_FUSION, n_freq_keep: int = 8):
        super().__init__()
        self.n_freq_keep = n_freq_keep
        d_fft_in = n_freq_keep * d_clip
        self.proj = nn.Sequential(
            nn.Linear(d_fft_in, d_out * 2), nn.GELU(),
            nn.Linear(d_out * 2, d_out),
        )
        self.norm = nn.LayerNorm(d_out)

    def forward(self, frame_embeds: torch.Tensor) -> torch.Tensor:
        B, N, K, D = frame_embeds.shape
        fft_out = torch.fft.rfft(frame_embeds.float(), dim=2)
        fft_mag = fft_out.abs()
        n_keep = min(self.n_freq_keep, fft_mag.shape[2])
        fft_mag = fft_mag[:, :, :n_keep, :]
        flat = fft_mag.reshape(B, N, n_keep * D)
        return self.norm(self.proj(flat.to(frame_embeds.dtype)))



# Backward compatibility with previous code/checkpoints.
RobotFourierEmbedding = SpectralTokenExtractor

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – Architectural building blocks (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class CuDNNFreeGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell_l0 = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.cell_l1 = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, T, _ = x.shape
        dev = x.device
        h0 = torch.zeros(batch, self.hidden_size, device=dev, dtype=x.dtype)
        h1 = torch.zeros(batch, self.hidden_size, device=dev, dtype=x.dtype)
        outs_l1 = []
        for t in range(T):
            h0 = self.cell_l0(x[:, t, :], h0)
            h0_d = self.drop(h0)
            h1 = self.cell_l1(h0_d, h1)
            outs_l1.append(h1)
        out = torch.stack(outs_l1, dim=1)
        h_n = torch.stack([h0, h1], dim=0)
        return out, h_n


class PhysicsConstraintModule(nn.Module):
    def __init__(self, d_pose: int = D_POSE, d_event: int = D_EVENT):
        super().__init__()
        self.d_pose = d_pose
        self.pose_gru = CuDNNFreeGRU(input_size=3, hidden_size=d_pose, dropout=0.1)

    def _integrate_poses(self, cmds, dt):
        B, N, T, _ = cmds.shape
        poses = torch.zeros(B, N, T + 1, 3, device=cmds.device)
        for t in range(T):
            theta = poses[:, :, t, 2]
            vx, vy, omega = cmds[:, :, t, 0], cmds[:, :, t, 1], cmds[:, :, t, 2]
            poses[:, :, t+1, 0] = poses[:, :, t, 0] + dt*vx*torch.cos(theta) - dt*vy*torch.sin(theta)
            poses[:, :, t+1, 1] = poses[:, :, t, 1] + dt*vx*torch.sin(theta) + dt*vy*torch.cos(theta)
            poses[:, :, t+1, 2] = theta + dt*omega
        return poses[:, :, 1:, :]

    def forward(self, cmds, dt=0.1, gt_poses=None):
        B, N, T, _ = cmds.shape
        cmds_flat = cmds.view(B * N, T, 3)
        _, h_n = self.pose_gru(cmds_flat)
        pose_emb = h_n[-1].view(B, N, self.d_pose)
        return pose_emb


class ProbabilisticVisualHead(nn.Module):
    def __init__(self, d_in: int, d_out: int = D_VIS):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.proj = nn.Linear(d_in, d_out)
        self.unc_head = nn.Sequential(
            nn.Linear(d_out, d_out // 2), nn.GELU(),
            nn.Linear(d_out // 2, d_out), nn.Softplus(),
        )
        self.temp_query = nn.Parameter(torch.randn(d_out))

    def forward(self, frame_embeds, vel_magnitudes=None):
        B, N, T, _ = frame_embeds.shape
        h = self.proj(frame_embeds)
        q = self.temp_query.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        att = (h * q).sum(-1) / math.sqrt(self.d_out)
        if vel_magnitudes is not None:
            att = att - 2.0 * vel_magnitudes
        att = F.softmax(att, dim=-1).unsqueeze(-1)
        mu = (h * att).sum(dim=2)
        frame_var = ((h - mu.unsqueeze(2)) ** 2 * att).sum(dim=2)
        sigma = self.unc_head(mu) + frame_var.clamp(min=0.0).sqrt().detach()
        return mu, sigma


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 7 – Simple cross-attention fusion ablation
# ════════════════════════════════════════════════════════════════════════════

class SPI_MRF(nn.Module):
    """
    Spectral and Physics-Informed Multi-Robot Fusion (SPI-MRF).

    Paper role:
        Visual robot states attend to auxiliary pose, spectral, and robot-role
        tokens through cross-attention, followed by uncertainty-aware pooling.
    """

    def __init__(self, d_vis=D_VIS, d_pose=D_POSE, d_event=D_EVENT,
                 d_fourier=D_FOURIER_FUSION, n_layers=N_FUSION_LAYERS,
                 n_heads=N_HEADS_FUSION, max_robots=MAX_ROBOTS):
        super().__init__()
        self.vis_norm = nn.LayerNorm(d_vis)
        self.pose_proj = nn.Linear(d_pose, d_vis)
        self.fourier_proj = nn.Linear(d_fourier, d_vis)
        self.role_emb = nn.Embedding(max_robots, d_vis)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_vis,
            num_heads=n_heads,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(d_vis)
        self.ff = nn.Sequential(
            nn.Linear(d_vis, d_vis * 2), nn.GELU(),
            nn.Linear(d_vis * 2, d_vis),
        )
        self.sigma_proj = nn.Linear(d_vis, d_vis)
        self.belief_proj = nn.Linear(d_vis, d_vis)

    def forward(self, mu_vis, sigma_vis, fourier_emb=None, pose_emb=None,
                n_robots=None):
        B, N, _ = mu_vis.shape
        dev = mu_vis.device
        fusion_dtype = next(self.pose_proj.parameters()).dtype
        if pose_emb is None:
            pose_emb = torch.zeros(B, N, D_POSE, device=dev, dtype=fusion_dtype)

        role_idx = torch.arange(N, device=dev).unsqueeze(0).expand(B, -1)
        q = self.vis_norm(mu_vis.to(dtype=fusion_dtype))
        aux = (
            self.pose_proj(pose_emb.to(device=dev, dtype=fusion_dtype))
            + self.role_emb(role_idx)
        )

        if fourier_emb is not None:
            aux = aux + self.fourier_proj(fourier_emb.to(device=dev, dtype=fusion_dtype))

        if n_robots is not None:
            mask = torch.arange(N, device=dev).unsqueeze(0) >= n_robots.unsqueeze(1)
        else:
            mask = None

        attn_out, _ = self.cross_attn(
            query=q,
            key=aux,
            value=aux,
            key_padding_mask=mask,
            need_weights=False,
        )
        mu = self.out_norm(q + attn_out)
        mu = self.out_norm(mu + self.ff(mu))

        sigma = F.softplus(
            self.sigma_proj(sigma_vis.to(dtype=fusion_dtype))
        ).clamp(1e-4, 50.0)
        if fourier_emb is not None:
            fourier_energy = fourier_emb.norm(dim=-1, keepdim=True).to(
                device=dev, dtype=fusion_dtype)
            fourier_energy = fourier_energy / fourier_energy.mean().clamp(min=1e-8)
            fourier_energy = torch.nan_to_num(
                fourier_energy, nan=0.0, posinf=10.0, neginf=0.0)
            sigma = sigma + 0.1 * fourier_energy.expand_as(sigma)

        if mask is not None:
            mu = mu.masked_fill(mask.unsqueeze(-1), 0.0)
            sigma = sigma.masked_fill(mask.unsqueeze(-1), 1e4)

        weights = torch.exp(-sigma.norm(dim=-1, keepdim=True).clamp(max=20.0))
        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(-1), 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        global_belief = self.belief_proj((mu * weights).sum(dim=1))
        return global_belief, mu, sigma



# Backward compatibility with previous code/checkpoints.
EvidentialSetTransformer = SPI_MRF

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 8 – Pose-only KinematicsPromptDistiller
# ════════════════════════════════════════════════════════════════════════════

class PAPSD(nn.Module):
    """
    Physics-Aligned Prompt-Space Distillation (PAPSD).

    Paper role:
        Distills privileged pose-aware teacher prompts into student prompts
        that are available at inference without pose input.
    """
    def __init__(self, d_vis=D_VIS, d_pose=D_POSE, d_event=D_EVENT,
                 n_prompts=N_DISTILL_PROMPTS, d_prompt=D_PROMPT):
        super().__init__()
        self.n_prompts = n_prompts
        self.d_prompt = d_prompt
        teacher_in = d_pose
        self.teacher_mlp = nn.Sequential(
            nn.Linear(teacher_in, d_vis), nn.GELU(),
            nn.Linear(d_vis, n_prompts * d_prompt),
        )
        student_in = d_vis * 3 + 1
        self.student_mlp = nn.Sequential(
            nn.Linear(student_in, d_vis * 2), nn.GELU(),
            nn.Linear(d_vis * 2, n_prompts * d_prompt),
        )
        self.teacher_ln = nn.LayerNorm(d_prompt)
        self.student_ln = nn.LayerNorm(d_prompt)
        self.teacher_pos = nn.Parameter(torch.randn(n_prompts, d_prompt) * 0.02)
        self.student_pos = nn.Parameter(torch.randn(n_prompts, d_prompt) * 0.02)

    def build_teacher_prompts(self, pose_emb):
        B = pose_emb.shape[0]
        pose_mean = pose_emb.mean(dim=1)
        teacher_feat = pose_mean
        prompts = self.teacher_mlp(teacher_feat).view(B, self.n_prompts, self.d_prompt)
        return self.teacher_ln(prompts + self.teacher_pos.unsqueeze(0))

    def build_student_prompts(self, global_belief, mu_vis, sigma_vis):
        B = global_belief.shape[0]
        mu_mean = mu_vis.mean(dim=1)
        sigma_mean = sigma_vis.mean(dim=1)
        sigma_norm = sigma_mean.norm(dim=-1, keepdim=True) / math.sqrt(sigma_mean.shape[-1])
        student_feat = torch.cat([global_belief, mu_mean, sigma_mean, sigma_norm], dim=-1)
        prompts = self.student_mlp(student_feat).view(B, self.n_prompts, self.d_prompt)
        return self.student_ln(prompts + self.student_pos.unsqueeze(0))

    def distill_loss(self, student_prompts, teacher_prompts):
        mse = F.mse_loss(student_prompts, teacher_prompts.detach())
        s = F.normalize(student_prompts, dim=-1)
        t = F.normalize(teacher_prompts.detach(), dim=-1)
        cos = 1.0 - (s * t).sum(dim=-1).mean()
        return 0.7 * mse + 0.3 * cos



# Backward compatibility with previous code/checkpoints.
KinematicsPromptDistiller = PAPSD

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 9 – Backbone / LLM adapters (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class BackboneAdapter(nn.Module):
    d_backbone: int = 0
    def load(self, model_name, device="auto"): raise NotImplementedError
    def encode_frames(self, images, device=None) -> torch.Tensor: raise NotImplementedError


class QwenVLBackboneAdapter(BackboneAdapter):
    d_backbone = 3584

    def load(self, model_name, device="auto"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        log.info("BackboneAdapter: loading Qwen2.5-VL from %s", model_name)
        self.proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        self.vit = getattr(getattr(full_model, "model", full_model), "visual").eval()
        self.d_backbone = _get_vit_output_dim(self.vit, full_model.config)
        if self.proc.tokenizer.pad_token_id is None:
            self.proc.tokenizer.pad_token_id = self.proc.tokenizer.eos_token_id
        del full_model
        torch.cuda.empty_cache()
        self._spatial_merge_size = 2
        for attr in ("proc.image_processor.merge_size", "vit.config.spatial_merge_size"):
            try:
                obj = self
                for part in attr.split("."):
                    obj = getattr(obj, part)
                self._spatial_merge_size = obj
                break
            except AttributeError:
                pass

    @torch.no_grad()
    def encode_frames(self, images, device=None):
        img_proc = self.proc.image_processor
        dev = device if device is not None else next(self.vit.parameters()).device
        _M = self._spatial_merge_size
        all_feats = []
        for i in range(0, len(images), 512):
            sub = [img.convert("RGB") for img in images[i:i+512]]
            enc = img_proc(images=sub, return_tensors="pt")
            pv = enc["pixel_values"].to(dev, dtype=torch.bfloat16)
            thw = enc["image_grid_thw"].to(dev)
            out = self.vit(pv, thw)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            split_sizes = _qwen_visual_split_sizes(thw, hidden.shape[0], _M)
            splits = hidden.split(split_sizes, dim=0)
            for seq_tokens, seq_thw in zip(splits, thw):
                temporal = int(seq_thw[0].item())
                feat_vec = seq_tokens.mean(dim=0).float().cpu()
                for _ in range(max(1, temporal)):
                    all_feats.append(feat_vec)
        n = len(images)
        if len(all_feats) > n:
            all_feats = all_feats[:n]
        elif len(all_feats) < n and all_feats:
            all_feats += [all_feats[-1]] * (n - len(all_feats))
        elif not all_feats:
            all_feats = [torch.zeros(self.d_backbone)] * n
        return torch.stack(all_feats)


BACKBONE_REGISTRY: dict[str, type] = {"qwen_vl": QwenVLBackboneAdapter}
LLM_REGISTRY: dict[str, type] = {}


class LLMDecoderAdapter(nn.Module):
    def load(self, model_name, device="auto"): raise NotImplementedError
    def get_hidden_size(self) -> int: raise NotImplementedError
    def get_tokenizer(self): raise NotImplementedError
    def forward_with_context(self, input_ids, attention_mask, global_ctx,
                             labels=None, output_hidden_states=True):
        raise NotImplementedError


class QwenVLLLMAdapter(LLMDecoderAdapter):
    def load(self, model_name, device="cuda:0"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        log.info("LLMAdapter: loading Qwen2.5-VL from %s on %s", model_name, device)
        self.full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map={"": device},   # single-device — required for DDP
            trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        _cfg = self.full_model.config
        self._hidden = getattr(_cfg, "hidden_size", None) or getattr(getattr(_cfg, "text_config", _cfg), "hidden_size")
        self._lora_active = False

    def apply_lora(self, r: int = 16, lora_alpha: int = 32,
                   lora_dropout: float = 0.05,
                   target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj")):
        """Wrap the LLM with a LoRA adapter (PEFT).  Call once before training."""
        from peft import LoraConfig, get_peft_model, TaskType
        if self._lora_active:
            log.warning("LoRA already applied — skipping duplicate apply_lora()")
            return
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
        )
        self.full_model = get_peft_model(self.full_model, lora_cfg)
        self.full_model.print_trainable_parameters()
        self._lora_active = True
        self._lora_path = None
        log.info("LoRA applied: r=%d alpha=%d dropout=%.2f targets=%s",
                 r, lora_alpha, lora_dropout, list(target_modules))

    def lora_parameters(self):
        """Return only the trainable LoRA parameters."""
        if not self._lora_active:
            return []
        return [p for p in self.full_model.parameters() if p.requires_grad]

    def save_lora(self, path):
        """Save LoRA adapter weights to *path* (a directory)."""
        if not self._lora_active:
            return
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.full_model.save_pretrained(str(path))
        self._lora_path = path.resolve()
        log.info("LoRA adapter saved to %s", path)

    def load_lora(self, path):
        """Load LoRA adapter weights from *path*."""
        from peft import PeftModel
        path = Path(path)
        if not path.exists():
            log.warning("LoRA path %s not found — skipping load", path)
            return
        resolved_path = path.resolve()
        if isinstance(self.full_model, PeftModel):
            loaded_path = getattr(self, "_lora_path", None)
            if loaded_path is not None and Path(loaded_path).resolve() == resolved_path:
                self._lora_active = True
                self._lora_path = resolved_path
                log.info("LoRA adapter already active for %s — skipping duplicate wrap",
                         resolved_path)
                return
            from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
            adapter_name = getattr(self.full_model, "active_adapter", "default")
            if callable(adapter_name):
                adapter_name = adapter_name()
            weights = load_peft_weights(str(path), device=str(next(self.full_model.parameters()).device))
            set_peft_model_state_dict(self.full_model, weights, adapter_name=adapter_name)
            self._lora_active = True
            self._lora_path = resolved_path
            log.info("LoRA adapter weights refreshed from %s", path)
            return
        self.full_model = PeftModel.from_pretrained(self.full_model, str(path))
        self._lora_active = True
        self._lora_path = resolved_path
        log.info("LoRA adapter loaded from %s", path)

    def get_hidden_size(self): return self._hidden
    def get_tokenizer(self): return self.tokenizer

    def forward_with_context(self, input_ids, attention_mask, global_ctx,
                             labels=None, output_hidden_states=True):
        dev = next(self.full_model.parameters()).device
        llm_dtype = next(self.full_model.parameters()).dtype
        embed_layer = self.full_model.get_input_embeddings()
        text_embs = embed_layer(input_ids.to(dev)).to(llm_dtype)
        if global_ctx.dim() == 2:
            prefix_embs = global_ctx.unsqueeze(1).to(device=dev, dtype=llm_dtype)
        elif global_ctx.dim() == 3:
            prefix_embs = global_ctx.to(device=dev, dtype=llm_dtype)
        else:
            raise ValueError("global_ctx must be [B,D] or [B,P,D]")
        embs = torch.cat([prefix_embs, text_embs], dim=1)
        prefix_len = prefix_embs.shape[1]
        ctx_mask = torch.ones(attention_mask.shape[0], prefix_len,
                              device=dev, dtype=attention_mask.dtype)
        combined_mask = torch.cat([ctx_mask, attention_mask.to(dev)], dim=1)
        combined_labels = None
        if labels is not None:
            prefix_labels = torch.full((labels.shape[0], prefix_len), -100,
                                       device=dev, dtype=labels.dtype)
            combined_labels = torch.cat([prefix_labels, labels.to(dev)], dim=1)
        out = self.full_model(inputs_embeds=embs, attention_mask=combined_mask,
                              labels=combined_labels,
                              output_hidden_states=output_hidden_states,
                              return_dict=True)
        return out, combined_mask


LLM_REGISTRY["qwen_vl"] = QwenVLLLMAdapter


class DirichletMC4Head(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        d_mid = d_in // 2
        self.fc1 = nn.Linear(d_in, d_mid)
        self.fc2 = nn.Linear(d_mid, d_mid)
        self.fc3 = nn.Linear(d_mid, 4)
        self.drop = nn.Dropout(0.1)
        self.act = nn.GELU()
        self.res_proj = nn.Linear(d_in, d_mid) if d_in != d_mid else nn.Identity()

    def forward(self, h):
        h1 = self.drop(self.act(self.fc1(h)))
        h2 = self.drop(self.act(self.fc2(h1) + self.res_proj(h)))
        alpha = F.softplus(self.fc3(h2)) + 1.0
        s = alpha.sum(dim=-1, keepdim=True)
        return alpha, alpha / s, s.squeeze(-1)

    def loss(self, alpha, targets, label_smoothing=0.05):
        valid = (targets >= 0) & (targets < 4)
        if not valid.any():
            return torch.tensor(0.0, device=alpha.device, requires_grad=True)
        alpha_v, targets_v = alpha[valid], targets[valid]
        s = alpha_v.sum(dim=-1, keepdim=True)
        nll = -torch.log(alpha_v[torch.arange(len(targets_v), device=alpha_v.device),
                                  targets_v] / s.squeeze(-1) + 1e-8)
        log_probs = torch.log(alpha_v / s + 1e-8)
        ce = (1 - label_smoothing) * F.nll_loss(log_probs, targets_v) \
             - label_smoothing * log_probs.mean(dim=-1).mean()
        return 0.7 * ce + 0.3 * nll.mean()


class GaussianOpenHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.mean_head = nn.Linear(d_in, 1)
        self.log_var_head = nn.Linear(d_in, 1)

    def forward(self, h):
        return self.mean_head(h).squeeze(-1), \
               self.log_var_head(h).squeeze(-1).clamp(-10.0, 10.0)

    def loss(self, mean, log_var, targets):
        loss = 0.5 * (((targets - mean)**2).clamp(max=1e6) * (-log_var).exp() + log_var)
        return loss.mean() if torch.isfinite(loss).all() \
            else torch.tensor(0.0, device=mean.device, requires_grad=True)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 10 – UACoPT (main model — uses precomputed caches)
# ════════════════════════════════════════════════════════════════════════════

def gather_dense_by_indices(dense_imgs: list[list[Image.Image]],
                            idx_tensor: torch.Tensor) -> list[Image.Image]:
    out = []
    for r in range(min(len(dense_imgs), idx_tensor.shape[0])):
        T = len(dense_imgs[r])
        if T == 0: continue
        for k in range(idx_tensor.shape[1]):
            t = max(0, min(T-1, int(idx_tensor[r, k].item())))
            out.append(dense_imgs[r][t])
    return out


def infer_qwen_cache_dim(qwen_cache: Optional[dict]) -> Optional[int]:
    if not qwen_cache:
        return None
    for per_robot in qwen_cache.values():
        if not per_robot:
            continue
        for feats in per_robot:
            if isinstance(feats, torch.Tensor) and feats.ndim >= 2 and feats.shape[-1] > 0:
                return int(feats.shape[-1])
    return None


class SPCoR(nn.Module):
    """
    SP-CoR: Spectral and Physics-Informed Cooperative Reasoner.

    Paper modules:
        - SE-MR2FS: spectral energy-aware multi-robot frame sampler
        - SpectralTokenExtractor: Fourier/spectral token extraction
        - SPI-MRF: spectral and physics-informed multi-robot fusion
        - PAPSD: physics-aligned prompt-space distillation
    """
    def __init__(self, backbone: str, d_vis=D_VIS, d_pose=D_POSE,
                 d_event=D_EVENT, d_fourier=D_FOURIER_FUSION,
                 n_fusion_layers=N_FUSION_LAYERS, n_heads=N_HEADS_FUSION,
                 use_aux_heads=True, n_distill_prompts=N_DISTILL_PROMPTS,
                 sampled_frames_per_video=SAMPLED_FRAMES_PER_VIDEO,
                 fourier_window=FOURIER_WINDOW, fourier_bins=FOURIER_BINS,
                 qvrs_tau=QVRS_TAU, qvrs_gamma=QVRS_GAMMA,
                 cross_robot_lambda=CROSS_ROB_LAMBDA,
                 clip_device="cpu"):
        super().__init__()
        self.backbone_name = backbone
        self.use_aux_heads = use_aux_heads
        self.n_distill_prompts = n_distill_prompts
        self.sampled_frames_per_video = sampled_frames_per_video
        self.d_fourier = d_fourier
        self._fourier_window = fourier_window
        self._fourier_bins = fourier_bins

        # Training-free sampler (no nn.Module, no parameters)
        self.se_mr2fs = SE_MR2FS(
            total_budget=sampled_frames_per_video,
            fourier_window=fourier_window, n_bins=fourier_bins,
            qvrs_tau=qvrs_tau, qvrs_gamma=qvrs_gamma,
            cross_robot_lambda=cross_robot_lambda,
        )

        # Heavy backbone adapters
        self.vis_adapter = BACKBONE_REGISTRY[backbone]()
        self.llm_adapter = LLM_REGISTRY[backbone]()

        # Fourier embedding (learned)
        self.spectral_token_extractor = SpectralTokenExtractor(
            d_clip=self.vis_adapter.d_backbone, d_out=d_fourier,
            n_freq_keep=sampled_frames_per_video // 2,
        )

        self.vis_head = ProbabilisticVisualHead(
            d_in=self.vis_adapter.d_backbone, d_out=d_vis)
        self.physics = PhysicsConstraintModule(d_pose=d_pose, d_event=d_event)
        self.spi_mrf = SPI_MRF(
            d_vis=d_vis, d_pose=d_pose, d_event=d_event, d_fourier=d_fourier,
            n_layers=n_fusion_layers, n_heads=n_heads)
        self.papsd = PAPSD(
            d_vis=d_vis, d_pose=d_pose, d_event=d_event,
            n_prompts=n_distill_prompts, d_prompt=d_vis)

        self._d_llm: Optional[int] = None
        self._ctx_proj: Optional[nn.Linear] = None
        self._prompt_proj: Optional[nn.Linear] = None
        self._context_gate: Optional[nn.Sequential] = None
        self._mc4_head: Optional[DirichletMC4Head] = None
        self._open_head: Optional[GaussianOpenHead] = None
        self._loaded = False


    @property
    def ms3_sampler(self):
        return self.se_mr2fs

    @property
    def fourier_emb_module(self):
        return self.spectral_token_extractor

    @fourier_emb_module.setter
    def fourier_emb_module(self, value):
        self.spectral_token_extractor = value

    @property
    def fusion(self):
        return self.spi_mrf

    @fusion.setter
    def fusion(self, value):
        self.spi_mrf = value

    @property
    def prompt_distiller(self):
        return self.papsd

    @prompt_distiller.setter
    def prompt_distiller(self, value):
        self.papsd = value

    def reconfigure_visual_input_dim(self, d_in: int) -> None:
        d_in = int(d_in)
        if d_in <= 0 or d_in == self.vis_head.d_in:
            return

        old_d = self.vis_head.d_in
        module_device = next(self.vis_head.parameters()).device
        module_dtype = next(self.vis_head.parameters()).dtype

        self.vis_head = ProbabilisticVisualHead(d_in=d_in, d_out=self.vis_head.d_out)
        self.spectral_token_extractor = SpectralTokenExtractor(
            d_clip=d_in,
            d_out=self.d_fourier,
            n_freq_keep=self.sampled_frames_per_video // 2,
        )
        self.vis_head.to(device=module_device, dtype=module_dtype)
        self.fourier_emb_module.to(device=module_device, dtype=module_dtype)
        self.vis_adapter.d_backbone = d_in
        log.warning(
            "Reconfigured visual modules to accept %d-dim frame embeddings "
            "(previously %d).",
            d_in,
            old_d,
        )

    def load_backbone(self, model_name, device="auto"):
        self.vis_adapter.load(model_name, device=device)
        self.llm_adapter.load(model_name, device=device)
        self._d_llm = self.llm_adapter.get_hidden_size()

        actual_d = self.vis_adapter.d_backbone
        if actual_d != self.vis_head.d_in:
            self.reconfigure_visual_input_dim(actual_d)

        d_vis = self.vis_head.d_out
        self._ctx_proj = nn.Linear(d_vis, self._d_llm)
        self._prompt_proj = nn.Linear(d_vis, self._d_llm)
        self._context_gate = nn.Sequential(
            nn.Linear(self._d_llm * 2, self._d_llm), nn.Sigmoid())
        # MC4 and open heads intentionally not instantiated — LM loss only
        self._loaded = True
        log.info("SP-CoR loaded: backbone=%s d_backbone=%d d_llm=%d",
                 self.backbone_name, actual_d, self._d_llm)

    def encode_scene(self, frame_embeds, vel_cmds=None, n_robots=None,
                     gt_poses=None, dt=0.1):
        vel_mag = vel_cmds.norm(dim=-1) if (self.training and vel_cmds is not None) else None
        mu_vis, sigma_vis = self.vis_head(frame_embeds, vel_mag)
        fourier_emb = self.spectral_token_extractor(frame_embeds)
        if vel_cmds is not None:
            pose_emb = self.physics(vel_cmds, dt, gt_poses)
        else:
            pose_emb = None
        global_belief, mu_out, sigma_out = self.spi_mrf(
            mu_vis, sigma_vis, fourier_emb=fourier_emb,
            pose_emb=pose_emb, n_robots=n_robots,
        )
        return {
            "mu_vis": mu_vis, "sigma_vis": sigma_vis, "fourier_emb": fourier_emb,
            "pose_emb": pose_emb,
            "global_belief": global_belief,
            "mu_fused": mu_out, "sigma_fused": sigma_out,
        }

    def forward(self, dense_images, dense_vel_cmds, dense_gt_poses, n_robots,
                prompt_texts, input_ids, attention_mask, answer_format,
                lm_labels=None, mc4_labels=None, numeric_labels=None,
                dt=0.1, skip_llm=False,
                lm_loss_weight=1.0, aux_loss_weight=0.3,
                prompt_distill_weight=PROMPT_DISTILL_WEIGHT,
                # ── NEW: precomputed cache inputs ────────────────────────
                precomputed_qwen_feats=None,    # [B, MAX_ROBOTS, K, D_backbone]
                precomputed_selected_idx=None,  # [B, MAX_ROBOTS, K]
                precomputed_sampler_info=None,   # list[dict]
                ):
        assert self._loaded, "Call load_backbone() before forward()"
        dev = next(self._ctx_proj.parameters()).device
        B = len(prompt_texts) if precomputed_qwen_feats is not None else len(dense_images)

        # ── Step 1+2: Use precomputed features or fall back ──────────────
        if precomputed_qwen_feats is not None and precomputed_selected_idx is not None:
            # FAST PATH: everything precomputed
            frame_embeds = precomputed_qwen_feats.to(dev)
            selected_idx = precomputed_selected_idx
            sampler_info = precomputed_sampler_info or [{}] * B
        else:
            # SLOW PATH: original online computation (for eval / compatibility)
            clip = FrozenCLIP(device="cpu")
            q_vecs = clip.encode_text(prompt_texts)
            all_idx = torch.zeros(B, MAX_ROBOTS, self.sampled_frames_per_video, dtype=torch.long)
            sampler_info = []
            for b in range(B):
                n_r = int(n_robots[b].item())
                robot_clip = []
                for r in range(n_r):
                    imgs = dense_images[b][r]
                    if imgs:
                        robot_clip.append(clip.encode_images(imgs))
                    else:
                        robot_clip.append(torch.zeros(1, CLIP_DIM))
                indices, info = self.ms3_sampler.select(
                    robot_clip, q_vecs[b], n_r, torch.device("cpu"))
                sampler_info.append(info)
                K = self.sampled_frames_per_video
                for r in range(n_r):
                    idx_r = indices[r]
                    k_r = len(idx_r)
                    if k_r >= K:
                        all_idx[b, r, :K] = idx_r[:K]
                    else:
                        pad = idx_r[-1:].repeat(K - k_r)
                        all_idx[b, r, :] = torch.cat([idx_r, pad])
            selected_idx = all_idx

            # Gather Qwen features for selected frames
            K = self.sampled_frames_per_video
            D = self.vis_adapter.d_backbone
            batch_feats = []
            for b in range(B):
                cur_n = int(n_robots[b].item())
                cur_imgs = gather_dense_by_indices(
                    dense_images[b][:cur_n], selected_idx[b, :cur_n].cpu())
                if not cur_imgs:
                    feats = torch.zeros(MAX_ROBOTS, K, D)
                else:
                    enc = self.vis_adapter.encode_frames(cur_imgs)
                    per_robot = []
                    offset = 0
                    for r in range(cur_n):
                        cur = enc[offset:offset + K]
                        if cur.shape[0] < K:
                            cur = torch.cat([cur, cur[-1:].repeat(K - cur.shape[0], 1)], dim=0)
                        per_robot.append(cur)
                        offset += K
                    while len(per_robot) < MAX_ROBOTS:
                        per_robot.append(torch.zeros(K, D))
                    feats = torch.stack(per_robot, dim=0)
                batch_feats.append(feats)
            frame_embeds = torch.stack(batch_feats, dim=0).to(dev)

        # ── Step 3: gather aligned cmd/pose at selected timesteps ────────
        gathered_cmds, gathered_poses = [], []
        if dense_vel_cmds is not None:
            for b in range(B):
                cmd_b, pose_b = [], []
                cur_n = int(n_robots[b].item())
                for r in range(cur_n):
                    idx = selected_idx[b, r].clamp(
                        min=0, max=dense_vel_cmds.shape[2]-1).cpu()
                    cmd_b.append(dense_vel_cmds[b, r, idx])
                    pose_b.append(dense_gt_poses[b, r, idx])
                pad_dev = dense_vel_cmds.device
                while len(cmd_b) < MAX_ROBOTS:
                    cmd_b.append(torch.zeros(self.sampled_frames_per_video, 3, device=pad_dev))
                    pose_b.append(torch.zeros(self.sampled_frames_per_video, 3, device=pad_dev))
                gathered_cmds.append(torch.stack(cmd_b))
                gathered_poses.append(torch.stack(pose_b))
            vel_cmds = torch.stack(gathered_cmds).to(dev)
            gt_poses = torch.stack(gathered_poses).to(dev)
        else:
            vel_cmds = gt_poses = None

        # ── Step 4: encode scene ─────────────────────────────────────────
        if frame_embeds.shape[-1] != self.vis_head.d_in:
            raise RuntimeError(
                "Visual feature width mismatch: frame_embeds has last dim "
                f"{frame_embeds.shape[-1]}, but vis_head expects {self.vis_head.d_in}. "
                "This usually means the cached Qwen features were produced with a "
                "different visual output width than the currently configured model. "
                "Rebuild the Qwen cache or realign the model before training."
            )
        enc = self.encode_scene(frame_embeds, vel_cmds, n_robots.to(dev),
                                gt_poses, dt)

        # ── Step 5: build LLM prefix context ─────────────────────────────
        _llm_dtype = next(self.llm_adapter.full_model.parameters()).dtype
        _proj_dtype = next(self._ctx_proj.parameters()).dtype

        global_belief = enc["global_belief"].to(_proj_dtype)
        base_ctx = self._ctx_proj(global_belief).to(_llm_dtype)

        student_prompts = self.papsd.build_student_prompts(
            global_belief=enc["global_belief"], mu_vis=enc["mu_vis"],
            sigma_vis=enc["sigma_vis"],
        )
        prefix_ctx = torch.cat([
            base_ctx.unsqueeze(1),
            self._prompt_proj(student_prompts.to(_proj_dtype)).to(_llm_dtype),
        ], dim=1)

        output = {
            "selected_idx": selected_idx,
            "sampler_info": sampler_info,
            "mu_vis": enc["mu_vis"], "fourier_emb": enc["fourier_emb"],
            "global_belief": enc["global_belief"],
            "sigma_vis": enc["sigma_vis"], "sigma_fused": enc["sigma_fused"],
            "student_prompts": student_prompts,
        }

        prompt_distill_loss = torch.tensor(0.0, device=dev)
        if vel_cmds is not None and enc["pose_emb"] is not None:
            teacher_prompts = self.papsd.build_teacher_prompts(
                pose_emb=enc["pose_emb"])
            output["teacher_prompts"] = teacher_prompts
            prompt_distill_loss = self.papsd.distill_loss(
                student_prompts, teacher_prompts)
        output["prompt_distill_loss"] = prompt_distill_loss

        lm_out, combined_mask = self.llm_adapter.forward_with_context(
            input_ids=input_ids, attention_mask=attention_mask,
            global_ctx=prefix_ctx, labels=lm_labels, output_hidden_states=True)
        hidden = lm_out.hidden_states[-1]
        output["lm_loss"] = (lm_out.loss if lm_out.loss is not None
                             else torch.tensor(0.0, device=dev))
        output["logits"] = lm_out.logits

        prefix_len = prefix_ctx.shape[1]
        text_hidden = hidden[:, prefix_len:, :]
        text_mask_f = combined_mask[:, prefix_len:].unsqueeze(-1).float()
        text_pooled = ((text_hidden.float() * text_mask_f).sum(dim=1)
                       / text_mask_f.sum(dim=1).clamp(min=1.0))
        if not torch.isfinite(text_pooled).all():
            text_pooled = torch.nan_to_num(text_pooled, nan=0.0)

        vis_ctx = base_ctx.float().to(text_pooled.device)
        gate = self._context_gate(
            torch.cat([text_pooled.float(), vis_ctx], dim=-1).to(
                next(self._context_gate.parameters()).device)
        ).to(text_pooled.device)
        cls_hidden = text_pooled.float() + gate * vis_ctx

        total_loss = (lm_loss_weight * output["lm_loss"]
                      + prompt_distill_weight * prompt_distill_loss)

        if self.use_aux_heads and (self._mc4_head is not None or self._open_head is not None):
            mc4_mask = torch.tensor([f == "MC4" for f in answer_format], device=dev)
            open_mask = torch.tensor([f == "OPEN" for f in answer_format], device=dev)
            aux_loss = torch.tensor(0.0, device=dev)
            if self._mc4_head is not None and mc4_mask.any() and mc4_labels is not None:
                alpha, probs, conf = self._mc4_head(cls_hidden[mc4_mask])
                output.update(mc4_alpha=alpha, mc4_probs=probs, mc4_conf=conf)
                aux_loss = aux_loss + self._mc4_head.loss(
                    alpha, mc4_labels[mc4_mask].to(dev))
            if self._open_head is not None and open_mask.any() and numeric_labels is not None:
                mean, log_var = self._open_head(cls_hidden[open_mask])
                output.update(open_mean=mean, open_log_var=log_var, open_var=log_var.exp())
                aux_loss = aux_loss + self._open_head.loss(
                    mean, log_var, numeric_labels[open_mask].to(dev))
            output["aux_loss"] = aux_loss
            total_loss = total_loss + aux_loss_weight * aux_loss

        output["loss"] = total_loss
        return output



# Backward compatibility with previous script names.
UACoPT = SPCoR

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 11 – Dataset (updated: serves precomputed embeddings)
# ════════════════════════════════════════════════════════════════════════════

class UACoPTDataset(Dataset):
    CSV_FILENAME = "locobot_trajectory_and_cmds.csv"

    def __init__(self, items, dense_frames_per_video=DENSE_FRAMES_PER_VIDEO,
                 max_robots=MAX_ROBOTS, numeric_decimals=2,
                 # ── NEW: precomputed caches ─────────────────────────────
                 qwen_cache=None, sampler_cache=None):
        self.items = items
        self.dense_frames_per_video = dense_frames_per_video
        self.max_robots = max_robots
        self.numeric_decimals = numeric_decimals
        self.qwen_cache = qwen_cache          # dict[key] -> list[Tensor[T, D]]
        self.sampler_cache = sampler_cache    # dict[(key, prompt)] -> {idx, info}
        self._backfill_provenance(items)
        self._vel_cache: dict = {}
        self._pose_cache: dict = {}
        self._image_cache: dict = {}  # only needed if qwen_cache is None
        self._build_caches()

    @staticmethod
    def _backfill_provenance(items):
        for item in items:
            item.setdefault("scene", item.get("_scene", ""))
            item.setdefault("exploration", item.get("_exploration", ""))
            item.setdefault("video_root",
                            DATASET_CONFIGS.get(item.get("config", ""), {}).get("video_root", ""))

    def _load_csv_trajectory(self, video_root, scene, exploration, robot_ids):
        if not video_root:
            return None, None
        csv_path = Path(video_root) / scene / exploration / self.CSV_FILENAME
        if not csv_path.exists():
            return None, None
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
        except Exception as e:
            log.warning("Could not read %s: %s", csv_path, e)
            return None, None
        T_max = self.dense_frames_per_video
        all_cmds, all_poses = [], []
        for rid in robot_ids:
            rob = df[df["robot_id"] == rid].sort_values("t")
            if rob.empty:
                all_cmds.append(np.zeros((T_max, 3), dtype=np.float32))
                all_poses.append(np.zeros((T_max, 3), dtype=np.float32))
                continue
            xs = np.linspace(0, 1, len(rob))
            xq = np.linspace(0, 1, T_max)
            raw_v = np.interp(xq, xs, rob["v_cmd"].values.astype(np.float32))
            raw_omega = np.interp(xq, xs, rob["omega_cmd"].values.astype(np.float32))
            raw_x = np.interp(xq, xs, rob["x"].values.astype(np.float32))
            raw_z = np.interp(xq, xs, rob["z"].values.astype(np.float32))
            raw_yaw = np.interp(xq, xs, rob["yaw"].values.astype(np.float32))
            vy = np.zeros(T_max, dtype=np.float32)
            all_cmds.append(np.stack([raw_v, vy, raw_omega], axis=1))
            all_poses.append(np.stack([raw_x, raw_z, raw_yaw], axis=1))
        zero = np.zeros((T_max, 3), dtype=np.float32)
        while len(all_cmds) < self.max_robots:
            all_cmds.append(zero); all_poses.append(zero)
        return (torch.tensor(np.stack(all_cmds), dtype=torch.float32),
                torch.tensor(np.stack(all_poses), dtype=torch.float32))

    def _build_caches(self):
        from collections import OrderedDict as OD
        groups: dict = OD()
        for item in self.items:
            key = (item.get("config",""), item.get("scene",""), item.get("exploration",""))
            if key not in groups:
                groups[key] = item

        log.info("_build_caches: %d unique groups from %d items", len(groups), len(self.items))

        for key in tqdm(groups, desc="Dense video decode", unit="exp"):
            item = groups[key]
            T = self.dense_frames_per_video
            video_root = item.get("video_root", "")
            n_r = item.get("n_robots", 1)

            # Only decode images if we don't have precomputed Qwen features
            if self.qwen_cache is None or key not in self.qwen_cache:
                dense_images: list[list[Image.Image]] = []
                if video_root:
                    vid_map = find_robot_videos(Path(video_root),
                                                item.get("scene",""), item.get("exploration",""))
                    for rid, vpath in sorted(vid_map.items()):
                        frames = extract_frames(vpath, T)
                        dense_images.append(frames if frames else [])
                    n_r = max(n_r, len(dense_images))
                while len(dense_images) < self.max_robots:
                    dense_images.append([])
                for r in range(self.max_robots):
                    frames = dense_images[r]
                    if not frames:
                        dense_images[r] = [Image.new("RGB", (128, 128))] * T
                    elif len(frames) < T:
                        dense_images[r] = frames + [frames[-1]] * (T - len(frames))
                    else:
                        dense_images[r] = frames[:T]
                self._image_cache[key] = dense_images

            vel_cmds, gt_poses = self._load_csv_trajectory(
                video_root, item.get("scene",""), item.get("exploration",""),
                list(range(1, n_r + 1)))
            if vel_cmds is None:
                vel_cmds = torch.zeros(self.max_robots, T, 3)
            if gt_poses is None:
                gt_poses = torch.zeros(self.max_robots, T, 3)
            self._vel_cache[key] = vel_cmds
            self._pose_cache[key] = gt_poses

    def _format_numeric_target(self, answer) -> str:
        try:
            return f"{float(answer):.{self.numeric_decimals}f}"
        except Exception:
            return str(answer).strip()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        key = (item.get("config",""), item.get("scene",""), item.get("exploration",""))
        T = self.dense_frames_per_video
        K = SAMPLED_FRAMES_PER_VIDEO

        vel_cmds = self._vel_cache.get(key, torch.zeros(self.max_robots, T, 3))
        gt_poses = self._pose_cache.get(key, torch.zeros(self.max_robots, T, 3))
        n_robots = min(item.get("n_robots", 1), self.max_robots)

        fmt = item.get("answer_format", "MC4")
        answer = str(item.get("answer", "")).strip()
        options = item.get("options", None)
        mc4_label = -1
        num_label = float("nan")
        if fmt == "MC4":
            mc4_label = MC4_ANSWER_MAP.get(answer[:1].upper(), -1)
        elif fmt == "OPEN":
            try: num_label = float(item.get("answer"))
            except Exception: pass

        raw_text = item.get("text", "")
        if fmt == "MC4" and options and len(options) == 4:
            opt_text = " ".join(f"{MC4_IDX_MAP[i]}) {options[i]}" for i in range(4))
            prompt_text = (f"Question: {raw_text}\nOptions: {opt_text}\n"
                           f"Answer with a single letter only.\nAnswer:")
            target_text = answer[:1].upper()
        else:
            prompt_text = f"Question: {raw_text}\nAnswer:"
            target_text = self._format_numeric_target(answer) if fmt == "OPEN" else answer

        result = {
            "dense_vel_cmds": vel_cmds,
            "dense_gt_poses": gt_poses,
            "n_robots": torch.tensor(n_robots, dtype=torch.long),
            "prompt_text": prompt_text,
            "target_text": target_text,
            "answer_format": fmt,
            "mc4_label": torch.tensor(mc4_label, dtype=torch.long),
            "num_label": torch.tensor(num_label, dtype=torch.float32),
            "qa_type": item.get("qa_type", ""),
            "id": item.get("id", ""),
        }

        # ── Serve precomputed Qwen features + sampler indices ────────────
        if self.qwen_cache is not None and key in self.qwen_cache:
            # Lookup precomputed sampler indices
            sampler_entry = None
            if self.sampler_cache is not None:
                sampler_entry = self.sampler_cache.get((key, prompt_text))

            if sampler_entry is not None:
                selected_idx = sampler_entry["selected_idx"]  # [MAX_ROBOTS, K_cached]
                K_cached = selected_idx.shape[1]
                if K_cached != K:
                    # Cache was built with a different SAMPLED_FRAMES_PER_VIDEO; resize.
                    if K_cached < K:
                        pad = selected_idx[:, -1:].repeat(1, K - K_cached)
                        selected_idx = torch.cat([selected_idx, pad], dim=1)
                    else:
                        selected_idx = selected_idx[:, :K]
                sampler_info_item = sampler_entry["info"]
            else:
                # Fallback: uniform sampling instead of repeated frame 0.
                base = _uniform_indices(K, T, device=torch.device("cpu"))
                selected_idx = base.unsqueeze(0).repeat(self.max_robots, 1)
                sampler_info_item = {
                    "sampler": "uniform_fallback",
                    "budgets": [K] * self.max_robots,
                }

            # Gather Qwen features at selected indices
            qwen_per_robot = self.qwen_cache[key]  # list of [T, D_backbone]
            D = qwen_per_robot[0].shape[-1]
            gathered = torch.zeros(self.max_robots, K, D)
            for r in range(min(n_robots, self.max_robots)):
                feats_r = qwen_per_robot[r]  # [T, D_backbone]
                T_r = feats_r.shape[0]
                idx_r = selected_idx[r].clamp(0, T_r - 1)
                gathered[r] = feats_r[idx_r]

            result["precomputed_qwen_feats"] = gathered        # [MAX_ROBOTS, K, D]
            result["precomputed_selected_idx"] = selected_idx  # [MAX_ROBOTS, K]
            result["precomputed_sampler_info"] = sampler_info_item
            result["dense_images"] = None  # not needed
        else:
            # Fallback: serve images for online computation
            dense_images = self._image_cache.get(key,
                [[Image.new("RGB",(128,128))] * T for _ in range(self.max_robots)])
            result["dense_images"] = dense_images

        return result


def collate_ua_copt(batch: list[dict]) -> dict:
    has_precomputed = batch[0].get("precomputed_qwen_feats") is not None

    result = {
        "dense_vel_cmds": torch.stack([b["dense_vel_cmds"] for b in batch]),
        "dense_gt_poses": torch.stack([b["dense_gt_poses"] for b in batch]),
        "n_robots": torch.stack([b["n_robots"] for b in batch]),
        "prompt_text": [b["prompt_text"] for b in batch],
        "target_text": [b["target_text"] for b in batch],
        "answer_format": [b["answer_format"] for b in batch],
        "mc4_label": torch.stack([b["mc4_label"] for b in batch]),
        "num_label": torch.stack([b["num_label"] for b in batch]),
        "qa_type": [b["qa_type"] for b in batch],
        "id": [b["id"] for b in batch],
    }

    if has_precomputed:
        result["precomputed_qwen_feats"] = torch.stack(
            [b["precomputed_qwen_feats"] for b in batch])
        result["precomputed_selected_idx"] = torch.stack(
            [b["precomputed_selected_idx"] for b in batch])
        result["precomputed_sampler_info"] = [
            b["precomputed_sampler_info"] for b in batch]
        result["dense_images"] = None
    else:
        result["dense_images"] = [b["dense_images"] for b in batch]
        result["precomputed_qwen_feats"] = None
        result["precomputed_selected_idx"] = None
        result["precomputed_sampler_info"] = None

    return result


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 12 – Trainer (updated for precomputed caches)
# ════════════════════════════════════════════════════════════════════════════

class UACoPTTrainer:
    def __init__(self, model: UACoPT, output_dir: Path, device="cuda",
                 aux_loss_weight=0.3, lm_loss_weight=1.0,
                 prompt_distill_weight=PROMPT_DISTILL_WEIGHT,
                 dataloader_num_workers: int = 0,
                 pin_memory: bool = False,
                 min_lr_ratio: float = 0.05,
                 max_grad_norm: float = 1.0,
                 eval_every_n_epochs: int = 3,
                 checkpoint_every_n_epochs: int = 3):
        self.model = model
        self.output_dir = Path(output_dir)
        self.device = device
        self.aux_loss_weight = aux_loss_weight
        self.lm_loss_weight = lm_loss_weight
        self.prompt_distill_weight = prompt_distill_weight
        self.dataloader_num_workers = max(0, int(dataloader_num_workers))
        self.pin_memory = bool(pin_memory)
        self.min_lr_ratio = min(1.0, max(0.0, float(min_lr_ratio)))
        self.max_grad_norm = float(max_grad_norm)
        self.eval_every_n_epochs = max(1, int(eval_every_n_epochs))
        self.checkpoint_every_n_epochs = max(1, int(checkpoint_every_n_epochs))
        self._ddp_model = None   # set in train_stage when DDP is active
        import torch.distributed as dist
        self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self._is_main = (self._rank == 0)
        if self._is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _set_trainable(self, modules, trainable):
        for m in modules:
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(trainable)

    def _freeze_for_stage(self, stage):
        all_modules = [
            self.model.fourier_emb_module,
            self.model.vis_head, self.model.physics, self.model.fusion,
            self.model.prompt_distiller,
            self.model.llm_adapter,
            self.model._ctx_proj, self.model._prompt_proj,
            self.model._context_gate,
        ]
        if stage == "pretrain_encoder":
            self._set_trainable(all_modules, False)
            self._set_trainable([self.model.fourier_emb_module,
                                 self.model.vis_head, self.model.physics,
                                 self.model.prompt_distiller], True)
        elif stage == "pretrain_fusion":
            self._set_trainable(all_modules, False)
            self._set_trainable([self.model.fourier_emb_module,
                                 self.model.physics, self.model.fusion,
                                 self.model.prompt_distiller], True)
        elif stage == "align_llm":
            self._set_trainable(all_modules, False)
            self._set_trainable([
                self.model.fourier_emb_module, self.model.prompt_distiller,
                self.model._ctx_proj, self.model._prompt_proj,
                self.model._context_gate,
            ], True)
        elif stage == "finetune":
            self._set_trainable(all_modules, True)
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        log.info("Stage '%s': %d / %d params trainable (%.1f%%)",
                 stage, n_train, n_total, 100 * n_train / max(n_total, 1))

    def _calibration_loss(self, alpha, labels):
        s = alpha.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        probs = (alpha / s).clamp(0, 1)
        conf = probs.max(dim=-1).values
        correct = (probs.argmax(dim=-1) == labels).float()
        return ((conf - correct) ** 2).mean()

    def _build_lm_batch(self, prompts, targets):
        tok = self.model.llm_adapter.get_tokenizer()
        pad_id = tok.pad_token_id
        input_id_list, attn_list, label_list = [], [], []
        for prompt, target in zip(prompts, targets):
            p_ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            t_ids = tok(target, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            eos = torch.tensor([tok.eos_token_id], dtype=torch.long)
            full_ids = torch.cat([p_ids, t_ids, eos])
            labels = torch.cat([torch.full_like(p_ids, -100), t_ids, eos])
            input_id_list.append(full_ids)
            attn_list.append(torch.ones_like(full_ids))
            label_list.append(labels)
        max_len = max(x.size(0) for x in input_id_list)
        input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(prompts), max_len), dtype=torch.long)
        lab = torch.full((len(prompts), max_len), -100, dtype=torch.long)
        for i, (ids, a, l) in enumerate(zip(input_id_list, attn_list, label_list)):
            L = ids.size(0)
            input_ids[i, :L] = ids; attn[i, :L] = a; lab[i, :L] = l
        return input_ids, attn, lab

    def _step(self, batch, stage, lambda_vis=0.5, lambda_consist=0.3, lambda_cal=0.1):
        dev = self.device
        input_ids, attn_mask, lm_labels = self._build_lm_batch(
            batch["prompt_text"], batch["target_text"])
        skip_llm = False
        
        fwd = self._ddp_model if self._ddp_model is not None else self.model
        out = fwd(
            dense_images=batch["dense_images"],
            dense_vel_cmds=batch["dense_vel_cmds"].to(dev, dtype=torch.float32),
            dense_gt_poses=batch["dense_gt_poses"].to(dev, dtype=torch.float32),
            n_robots=batch["n_robots"].to(dev),
            prompt_texts=batch["prompt_text"],
            input_ids=input_ids.to(dev),
            attention_mask=attn_mask.to(dev),
            answer_format=batch["answer_format"],
            lm_labels=lm_labels.to(dev),
            mc4_labels=batch["mc4_label"].to(dev),
            numeric_labels=torch.nan_to_num(batch["num_label"].to(dev), nan=0.0),
            skip_llm=skip_llm,
            lm_loss_weight=self.lm_loss_weight,
            aux_loss_weight=self.aux_loss_weight,
            prompt_distill_weight=self.prompt_distill_weight,
            # ── Pass precomputed data ────────────────────────────────────
            precomputed_qwen_feats=batch.get("precomputed_qwen_feats"),
            precomputed_selected_idx=batch.get("precomputed_selected_idx"),
            precomputed_sampler_info=batch.get("precomputed_sampler_info"),
        )
        loss = out["loss"]

        if not torch.isfinite(loss):
            log.warning("_step: loss is non-finite; returning zero")
            return torch.tensor(0.0, device=dev, requires_grad=True)
        return loss

    def train_stage(self, stage, train_dataset, val_dataset=None,
                    test_dataset=None,
                    epochs=5, batch_size=4, lr=1e-4, vision_lr=1e-5, grad_accum=1,
                    finetune_llm=False,
                    use_lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler

        ddp_active = dist.is_available() and dist.is_initialized() and self._world_size > 1
        grad_accum = max(1, int(grad_accum))

        self._freeze_for_stage(stage)

        # ── Remove Qwen ViT from the module tree before DDP wrapping ────────
        # All features are precomputed; the ViT is never called during training.
        # Keeping it even on CPU causes DDP to reject the model ("parameters on
        # {'cpu', 'cuda'}").  We stash it aside and restore after training.
        _stashed_vit = None
        if hasattr(self.model.vis_adapter, "vit") and self.model.vis_adapter.vit is not None:
            _stashed_vit = self.model.vis_adapter.vit
            self.model.vis_adapter.vit = None   # remove from module tree
            torch.cuda.empty_cache()
            log.info("vis_adapter.vit removed from module tree for DDP (stashed)")

        for m in [self.model.fourier_emb_module,
                  self.model.vis_head, self.model.physics, self.model.fusion,
                  self.model.prompt_distiller,
                  self.model._ctx_proj, self.model._prompt_proj,
                  self.model._context_gate]:
            if m is not None:
                m.to(self.device)
        if hasattr(self.model.llm_adapter, "full_model"):
            try:
                self.model.llm_adapter.full_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                log.info("Gradient checkpointing enabled on LLM backbone")
            except Exception as e:
                log.warning("gradient_checkpointing_enable() failed (%s); "
                            "activation memory will be higher — reduce batch_size "
                            "or grad_accum if OOM", e)

        # ── Wrap in DDP for multi-GPU training ───────────────────────────
        if ddp_active:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            # find_unused_parameters=True handles frozen modules per stage
            self._ddp_model = DDP(self.model, device_ids=[local_rank],
                                  output_device=local_rank,
                                  find_unused_parameters=True)
        else:
            self._ddp_model = None

        fourier_params = list(self.model.fourier_emb_module.parameters())
        vis_params = (list(self.model.vis_head.parameters())
                      + list(self.model.physics.parameters()))
        fus_params = list(self.model.fusion.parameters())
        proj_params = []
        for m in [self.model.prompt_distiller, self.model._ctx_proj,
                  self.model._prompt_proj, self.model._context_gate]:
            if m is not None:
                proj_params.extend(p for p in m.parameters() if p.requires_grad)
        # ── LLM backbone: freeze base weights; optionally add LoRA or full FT ──
        lora_this_stage = (stage == "finetune" and use_lora)
        full_ft_this_stage = (stage == "finetune" and finetune_llm and not use_lora)

        if hasattr(self.model.llm_adapter, "full_model"):
            if lora_this_stage:
                # Freeze all base weights; LoRA adds its own requires_grad=True params
                for p in self.model.llm_adapter.full_model.parameters():
                    p.requires_grad_(False)
                self.model.llm_adapter.apply_lora(
                    r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            else:
                for p in self.model.llm_adapter.full_model.parameters():
                    p.requires_grad_(full_ft_this_stage)

        lora_params = (self.model.llm_adapter.lora_parameters()
                       if lora_this_stage else [])
        llm_params  = ([]  # LoRA handles its own params
                       if lora_this_stage
                       else ([p for p in self.model.llm_adapter.full_model.parameters()
                               if p.requires_grad]
                             if hasattr(self.model.llm_adapter, "full_model") else []))

        param_groups = [
            {"params": [p for p in fourier_params if p.requires_grad],
             "lr": lr, "name": "fourier_emb"},
            {"params": [p for p in vis_params if p.requires_grad],
             "lr": vision_lr, "name": "vision+physics"},
            {"params": [p for p in fus_params if p.requires_grad],
             "lr": lr * 0.8, "name": "fusion"},
            {"params": [p for p in proj_params if p.requires_grad],
             "lr": lr, "name": "prompts+proj+heads"},
        ]
        if lora_params:
            lora_lr = lr * 0.3
            param_groups.append({"params": lora_params,
                                  "lr": lora_lr,   # smaller than heads
                                  "name": "llm_lora", "weight_decay": 0.0})
            log.info("LoRA parameters added to optimizer (lr=%.2e)", lora_lr)
        if llm_params:
            # only reached when finetune_llm=True (full fine-tune, no LoRA)
            param_groups.append({"params": llm_params,
                                  "lr": lr * 0.01,   # 100× smaller — avoid forgetting
                                  "name": "llm_body", "weight_decay": 0.0})
            log.warning("LLM backbone is being fully fine-tuned. "
                        "Ensure each GPU has >80 GB VRAM for AdamW states.")
        param_groups = [g for g in param_groups if g["params"]]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # ── DataLoader: DistributedSampler when DDP, else shuffle ────────
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=self._world_size,
                                           rank=self._rank,
                                           shuffle=True) if ddp_active else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  collate_fn=collate_ua_copt,
                                  sampler=train_sampler,
                                  shuffle=(train_sampler is None),
                                  num_workers=self.dataloader_num_workers,
                                  pin_memory=self.pin_memory,
                                  persistent_workers=self.dataloader_num_workers > 0)

        optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / max(grad_accum, 1)))
        total_steps = max(1, optimizer_steps_per_epoch * epochs)
        warmup = min(max(total_steps // 10, 50), total_steps)

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            prog = (step - warmup) / max(total_steps - warmup, 1)
            cosine = 0.5 * (1 + math.cos(math.pi * prog))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        stage_dir = self.output_dir / f"stage_{stage}"
        if self._is_main:
            stage_dir.mkdir(exist_ok=True)
        best_val_loss = float("inf")
        history = []

        for epoch in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)   # ensures different shuffles per epoch
            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            loader_iter = tqdm(train_loader, desc=f"[{stage}] epoch {epoch}",
                               unit="batch", disable=not self._is_main)
            for step, batch in enumerate(loader_iter, start=1):
                loss = self._step(batch, stage) / grad_accum
                loss.backward()
                total_loss += loss.item() * grad_accum
                if step % grad_accum == 0:
                    if self.max_grad_norm and self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            if len(train_loader) % grad_accum != 0:
                if self.max_grad_norm and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            avg_loss = total_loss / max(len(train_loader), 1)
            # Average loss across ranks for consistent logging
            if ddp_active:
                loss_t = torch.tensor(avg_loss, device=self.device)
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                avg_loss = loss_t.item()

            record = {"epoch": epoch, "train_loss": avg_loss}
            if self._is_main:
                log.info("[%s] epoch %d train_loss=%.4f", stage, epoch, avg_loss)

            if val_dataset is not None and self._is_main:
                # Evaluation only on rank 0 to avoid DistributedSampler overhead
                val_loss = self._evaluate_loss(val_dataset, batch_size, stage)
                record["val_loss"] = val_loss
                log.info("[%s] epoch %d val_loss=%.4f", stage, epoch, val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save(stage_dir / "best")

            if (test_dataset is not None and self._is_main
                    and (epoch % self.eval_every_n_epochs == 0 or epoch == epochs)):
                log.info("[%s] epoch %d running test evaluation", stage, epoch)
                metrics = evaluate_ua_copt(
                    model=self.model,
                    dataset=test_dataset,
                    output_dir=stage_dir / f"test_epoch_{epoch}",
                    batch_size=batch_size,
                    device=self.device,
                    split_name=f"{stage}_epoch_{epoch}",
                    num_workers=self.dataloader_num_workers,
                    pin_memory=self.pin_memory,
                )
                record["test_metrics"] = metrics
                self.model.train()

            history.append(record)

            if self._is_main and epoch % self.checkpoint_every_n_epochs == 0:
                epoch_ckpt = stage_dir / "checkpoints" / f"epoch_{epoch:03d}"
                self._save_training_state(epoch_ckpt, optimizer, scheduler,
                                          epoch, history, best_val_loss)

        if self._is_main:
            self._save(stage_dir / "final")
            with open(stage_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        # Synchronise all ranks before moving to next stage
        if ddp_active:
            dist.barrier()
            self._ddp_model = None   # release DDP wrapper

        # Restore stashed ViT (needed for inference / subsequent stages)
        if _stashed_vit is not None:
            self.model.vis_adapter.vit = _stashed_vit
            log.info("vis_adapter.vit restored after training stage")

        best_path = stage_dir / "best"
        return best_path if (best_path / "ua_copt_modules.pt").exists() else stage_dir / "final"

    @torch.no_grad()
    def _evaluate_loss(self, dataset, batch_size, stage="finetune"):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=collate_ua_copt,
                            num_workers=self.dataloader_num_workers,
                            pin_memory=self.pin_memory,
                            persistent_workers=self.dataloader_num_workers > 0)
        total = sum(self._step(b, stage=stage).item() for b in loader)
        self.model.train()
        return total / max(len(loader), 1)

    def _save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "fourier_emb_module": self.model.fourier_emb_module.state_dict(),
            "spectral_token_extractor": self.model.spectral_token_extractor.state_dict(),
            "vis_head": self.model.vis_head.state_dict(),
            "physics": self.model.physics.state_dict(),
            "fusion": self.model.fusion.state_dict(),
            "spi_mrf": self.model.spi_mrf.state_dict(),
            "prompt_distiller": self.model.prompt_distiller.state_dict(),
            "papsd": self.model.papsd.state_dict(),
            "ctx_proj": self.model._ctx_proj.state_dict(),
            "prompt_proj": self.model._prompt_proj.state_dict(),
            "context_gate": self.model._context_gate.state_dict(),
        }
        torch.save(state, path / "ua_copt_modules.pt")
        # Save LoRA adapter alongside the other weights (no-op if LoRA not active)
        if hasattr(self.model.llm_adapter, "_lora_active") \
                and self.model.llm_adapter._lora_active:
            self.model.llm_adapter.save_lora(path / "lora_adapter")

    def _save_training_state(self, path: Path, optimizer, scheduler,
                             epoch: int, history: list, best_val_loss: float):
        self._save(path)
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_val_loss": best_val_loss,
        }, path / "training_state.pt")

    def load_checkpoint(self, path: Path):
        ckpt = torch.load(path / "ua_copt_modules.pt", map_location="cpu",
                          weights_only=True)
        for key, mod in [
            ("fourier_emb_module", self.model.fourier_emb_module),
            ("vis_head", self.model.vis_head),
            ("physics", self.model.physics),
            ("fusion", self.model.fusion),
            ("prompt_distiller", self.model.prompt_distiller),
            ("ctx_proj", self.model._ctx_proj),
            ("prompt_proj", self.model._prompt_proj),
            ("context_gate", self.model._context_gate),
        ]:
            if mod is not None:
                try:
                    mod.load_state_dict(ckpt[key])
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Checkpoint module '{key}' is incompatible with the current model "
                        "shape. This often means the checkpoint was trained with a different "
                        "Qwen visual feature width or cached backbone output size."
                    ) from exc
                mod.to(self.device)
        # Load LoRA adapter if present
        lora_path = path / "lora_adapter"
        if lora_path.exists() and hasattr(self.model.llm_adapter, "load_lora"):
            self.model.llm_adapter.load_lora(lora_path)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 13 – Evaluation (unchanged logic, supports precomputed)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_ua_copt(model, dataset, output_dir, batch_size=2,
                     device="cuda", split_name="", num_workers=0,
                     pin_memory=False):
    model.eval()
    for mod in [model.fourier_emb_module, model.vis_head, model.physics,
                model.fusion, model.prompt_distiller,
                model._ctx_proj, model._prompt_proj, model._context_gate]:
        if mod is not None:
            mod.to(device)
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=collate_ua_copt,
                        num_workers=max(0, int(num_workers)),
                        pin_memory=bool(pin_memory),
                        persistent_workers=max(0, int(num_workers)) > 0)
    tok = model.llm_adapter.get_tokenizer()
    mc4_letters = ["A", "B", "C", "D"]
    mc4_token_ids = []
    for letter in mc4_letters:
        token_ids = tok(letter, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(
                f"Expected {letter!r} to map to a single token, got token ids {token_ids}"
            )
        mc4_token_ids.append(int(token_ids[0]))
    mc4_token_ids_tensor = torch.tensor(mc4_token_ids, device=device, dtype=torch.long)
    records = []
    for batch in tqdm(loader, desc="Evaluating", unit="batch"):
        enc = tok(batch["prompt_text"], padding=True, return_tensors="pt",
                  add_special_tokens=False)
        out = model(
            dense_images=batch["dense_images"],
            dense_vel_cmds=None, dense_gt_poses=None,
            n_robots=batch["n_robots"].to(device),
            prompt_texts=batch["prompt_text"],
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            answer_format=batch["answer_format"],
            skip_llm=False,
            precomputed_qwen_feats=batch.get("precomputed_qwen_feats"),
            precomputed_selected_idx=batch.get("precomputed_selected_idx"),
            precomputed_sampler_info=batch.get("precomputed_sampler_info"),
        )
        prefix_len = 1 + model.n_distill_prompts
        lm_logits = out["logits"][:, prefix_len:, :]
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        for i in range(len(batch["id"])):
            L = int(prompt_lens[i])
            qvrs = (out["sampler_info"][i].get("qvrs_per_robot", [])
                    if isinstance(out["sampler_info"][i], dict) else [])
            budgets = (out["sampler_info"][i].get("budgets", [])
                       if isinstance(out["sampler_info"][i], dict) else [])
            pred_text = ""
            rec = {
                "id": batch["id"][i],
                "qa_type": batch["qa_type"][i],
                "answer_format": batch["answer_format"][i],
                "target_text": batch["target_text"][i],
                "predicted_text": pred_text,
                "n_robots": int(batch["n_robots"][i].item()),
                "selected_idx": out["selected_idx"][i].detach().cpu().tolist(),
                "qvrs_per_robot": qvrs,
                "budgets": budgets,
            }
            if batch["answer_format"][i] == "MC4":
                mc4_logits = lm_logits[i, L - 1].index_select(0, mc4_token_ids_tensor)
                choice_idx = int(mc4_logits.argmax().item())
                choice = mc4_letters[choice_idx]
                pred_text = choice
                rec["predicted_text"] = pred_text
                rec["label"] = int(batch["mc4_label"][i].item())
                rec["predicted"] = choice
                rec["is_correct"] = MC4_ANSWER_MAP.get(choice, -999) == rec["label"]
            else:
                pred_token_id = int(lm_logits[i, L - 1].argmax().item())
                pred_text = tok.decode([pred_token_id], skip_special_tokens=True).strip()
                rec["predicted_text"] = pred_text
                try: rec["predicted_value"] = float(pred_text)
                except Exception: rec["predicted_value"] = float("nan")
                rec["label_value"] = float(batch["num_label"][i].item())
            records.append(rec)

    mc4_recs = [r for r in records if r["answer_format"] == "MC4" and "label" in r]
    open_recs = [r for r in records if r["answer_format"] == "OPEN"]
    def _acc(recs): return sum(r["is_correct"] for r in recs) / max(len(recs), 1) * 100
    open_errs = [(r["predicted_value"] - r["label_value"])**2
                 for r in open_recs
                 if math.isfinite(r.get("predicted_value", float("nan")))
                 and math.isfinite(r.get("label_value", float("nan")))]
    open_abs = [abs(r["predicted_value"] - r["label_value"])
                for r in open_recs
                if math.isfinite(r.get("predicted_value", float("nan")))
                and math.isfinite(r.get("label_value", float("nan")))]
    mse = float(np.mean(open_errs)) if open_errs else 0.0
    metrics = {
        "mc4": {"total": len(mc4_recs), "accuracy": _acc(mc4_recs)},
        "open": {"total": len(open_recs), "mse": mse,
                 "rmse": float(math.sqrt(mse)) if mse > 0 else 0.0,
                 "mae": float(np.mean(open_abs)) if open_abs else 0.0},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sp_cor_eval_se_mr2fs.json"
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "records": records}, f, indent=2)
    log.info("Evaluation results -> %s", out_path)
    return metrics


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 14 – Dataset loading helpers
# ════════════════════════════════════════════════════════════════════════════

def _enrich_items(items, cfg_name, cfg, qa_json_path):
    if items and "scene" in items[0] and "video_root" in items[0]:
        return items
    video_root = str(cfg.get("video_root") or "")
    id_to_prov = {}
    try:
        with open(qa_json_path, encoding="utf-8") as fh:
            for raw in json.load(fh):
                iid = raw.get("id", "")
                if iid:
                    id_to_prov[iid] = (raw.get("scene",""), raw.get("exploration",""))
    except Exception as exc:
        log.warning("_enrich_items: %s", exc)
    return [{**item,
             "scene": id_to_prov.get(item.get("id",""), ("",""))[0],
             "exploration": id_to_prov.get(item.get("id",""), ("",""))[1],
             "video_root": video_root}
            for item in items]


def _build_image_cache_for_items(items, dense_frames_per_video, max_robots):
    """Build image cache from items (shared across train/val/test)."""
    from collections import OrderedDict as OD
    groups = OD()
    for item in items:
        key = (item.get("config",""), item.get("scene",""), item.get("exploration",""))
        if key not in groups:
            groups[key] = item

    image_cache = {}
    T = dense_frames_per_video
    for key in tqdm(groups, desc="Decoding dense frames", unit="exp"):
        item = groups[key]
        video_root = item.get("video_root", "")
        n_r = item.get("n_robots", 1)
        dense_images: list[list[Image.Image]] = []
        if video_root:
            vid_map = find_robot_videos(Path(video_root),
                                        item.get("scene",""), item.get("exploration",""))
            for rid, vpath in sorted(vid_map.items()):
                frames = extract_frames(vpath, T)
                dense_images.append(frames if frames else [])
            n_r = max(n_r, len(dense_images))
        while len(dense_images) < max_robots:
            dense_images.append([])
        for r in range(max_robots):
            frames = dense_images[r]
            if not frames:
                dense_images[r] = [Image.new("RGB", (128, 128))] * T
            elif len(frames) < T:
                dense_images[r] = frames + [frames[-1]] * (T - len(frames))
            else:
                dense_images[r] = frames[:T]
        image_cache[key] = dense_images
    return image_cache


def load_split_datasets(cfg_names, splits=None, qa_types_filter=None,
                        dense_frames_per_video=DENSE_FRAMES_PER_VIDEO,
                        # ── NEW: precomputed caches ──────────────────────
                        qwen_cache=None, sampler_cache=None):
    if splits is None:
        splits = ["train", "val", "test"]
    collectors = {s: [] for s in splits}
    for cfg_name in cfg_names:
        if cfg_name not in DATASET_CONFIGS:
            continue
        cfg = DATASET_CONFIGS[cfg_name]
        qa_json_path = Path(cfg["qa_json"])
        for split in splits:
            items = build_split_items(cfg_name, cfg, split, "none", 0, qa_types_filter)
            items = _enrich_items(items, cfg_name, cfg, qa_json_path)
            collectors[split].extend(items)

    def _make(items):
        return UACoPTDataset(items=items,
                             dense_frames_per_video=dense_frames_per_video,
                             qwen_cache=qwen_cache,
                             sampler_cache=sampler_cache)
    return (
        _make(collectors["train"]) if "train" in collectors else None,
        _make(collectors["val"]) if "val" in collectors else None,
        _make(collectors["test"]) if "test" in collectors else None,
    )


def precompute_sampler_indices(
    clip_cache: dict,
    text_cache: dict,
    items: list[dict],
    sampler: SE_MR2FS,
    max_robots: int = MAX_ROBOTS,
    sampled_frames: int = SAMPLED_FRAMES_PER_VIDEO,
) -> dict:
    """
    Precompute MS³-ASCS selected frame indices for every (group_key, prompt_text)
    pair. Since the sampler is training-free, indices are deterministic given
    CLIP embeddings and question text.

    Returns:
        sampler_cache: dict[(group_key, prompt_text)] ->
            {"selected_idx": Tensor[MAX_ROBOTS, K], "info": dict}
    """
    # Collect unique (key, prompt_text) pairs
    unique_pairs = set()
    for item in items:
        key = (item.get("config", ""), item.get("scene", ""), item.get("exploration", ""))
        raw_text = item.get("text", "")
        fmt = item.get("answer_format", "MC4")
        options = item.get("options", None)
        if fmt == "MC4" and options and len(options) == 4:
            opt_text = " ".join(f"{MC4_IDX_MAP[i]}) {options[i]}" for i in range(4))
            prompt_text = (f"Question: {raw_text}\nOptions: {opt_text}\n"
                           f"Answer with a single letter only.\nAnswer:")
        else:
            prompt_text = f"Question: {raw_text}\nAnswer:"
        n_robots = min(item.get("n_robots", 1), max_robots)
        unique_pairs.add((key, prompt_text, n_robots))

    log.info("Precomputing sampler indices for %d unique (scene, question) pairs",
             len(unique_pairs))

    sampler_cache = {}
    K = sampled_frames

    for key, prompt_text, n_robots in tqdm(unique_pairs, desc="Sampler indices"):
        clip_feats = clip_cache.get(key)
        q_vec = text_cache.get(prompt_text)
        if clip_feats is None or q_vec is None:
            # Fallback: uniform indices
            idx = torch.zeros(max_robots, K, dtype=torch.long)
            sampler_cache[(key, prompt_text)] = {"selected_idx": idx, "info": {}}
            continue

        robot_clip = clip_feats[:n_robots]
        indices, info = sampler.select(
            clip_feats_per_robot=robot_clip,
            question_vec=q_vec,
            n_robots=n_robots,
            device=torch.device("cpu"),
        )

        all_idx = torch.zeros(max_robots, K, dtype=torch.long)
        for r in range(n_robots):
            idx_r = indices[r]
            k_r = len(idx_r)
            if k_r >= K:
                all_idx[r, :K] = idx_r[:K]
            else:
                pad = idx_r[-1:].repeat(K - k_r)
                all_idx[r, :] = torch.cat([idx_r, pad])

        # Detach coherence tensor for storage
        info_clean = {k: v for k, v in info.items() if k != "cross_coherence"}
        sampler_cache[(key, prompt_text)] = {"selected_idx": all_idx, "info": info_clean}

    log.info("Sampler index precomputation complete")
    return sampler_cache

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 15 – main (updated: precomputation stage before training)
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SP-CoR with SE-MR2FS, SPI-MRF, and PAPSD")
    parser.add_argument("--stage",
        choices=["pretrain_encoder","pretrain_fusion","align_llm","finetune",
                 "evaluate","all"], default="finetune")
    parser.add_argument("--backbone", default="qwen_vl",
                        choices=list(BACKBONE_REGISTRY))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--configs", nargs="+", choices=list(DATASET_CONFIGS),
                        metavar="CFG")
    parser.add_argument("--all_configs", action="store_true")
    parser.add_argument("--qa_types", nargs="+", default=None)
    parser.add_argument("--output_dir", type=Path,
                        default=Path("runs_new/sp_cor_qwen25"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dense_frames_per_video", type=int,
                        default=DENSE_FRAMES_PER_VIDEO) #
    parser.add_argument("--sampled_frames_per_video", type=int,
                        default=SAMPLED_FRAMES_PER_VIDEO) #SAMPLED_FRAMES_PER_VIDEO
    parser.add_argument("--d_vis", type=int, default=D_VIS)
    parser.add_argument("--d_pose", type=int, default=D_POSE)
    parser.add_argument("--d_event", type=int, default=D_EVENT)
    parser.add_argument("--d_fourier", type=int, default=D_FOURIER_FUSION)
    parser.add_argument("--n_fusion_layers", type=int, default=N_FUSION_LAYERS)
    parser.add_argument("--n_heads", type=int, default=N_HEADS_FUSION)
    parser.add_argument("--n_distill_prompts", type=int, default=N_DISTILL_PROMPTS) #8 
    parser.add_argument("--fourier_window", type=int, default=FOURIER_WINDOW)
    parser.add_argument("--fourier_bins", type=int, default=FOURIER_BINS)
    parser.add_argument("--qvrs_tau", type=float, default=QVRS_TAU)
    parser.add_argument("--qvrs_gamma", type=float, default=QVRS_GAMMA)
    parser.add_argument("--cross_robot_lambda", type=float, default=CROSS_ROB_LAMBDA)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval_every_n_epochs", type=int, default=2,
                        help="Run full test-set evaluation every N training epochs.")
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=2,
                        help="Save a numbered training checkpoint every N epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-GPU batch size (effective = batch_size × num_gpus × grad_accum)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vision_lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (1 = disabled; "
                             "increase if GPU memory is tight)")
    parser.add_argument("--aux_loss_weight", type=float, default=0.2)
    parser.add_argument("--lm_loss_weight", type=float, default=1.0)
    parser.add_argument("--prompt_distill_weight", type=float, default=PROMPT_DISTILL_WEIGHT)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05,
                        help="Cosine schedule floor as a fraction of each parameter-group LR.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm. Use 0 to disable clipping.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="DataLoader workers. Default 0 avoids mmap/shared-memory pressure "
                             "with large cached tensors.")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Enable pinned host memory for DataLoaders.")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Retrain stages even if stage checkpoints already exist.")
    parser.add_argument("--disable_aux_heads", action="store_true")
    parser.add_argument("--finetune_llm", action="store_true",
                        help="Full fine-tune of the Qwen LLM backbone during finetune stage. "
                             "Requires ~80 GB VRAM per GPU for AdamW optimizer states. "
                             "Use --lora instead for memory-efficient fine-tuning.")
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument("--lora", dest="lora", action="store_true",
                            help="Apply LoRA to the LLM during the finetune stage.")
    lora_group.add_argument("--no_lora", dest="lora", action="store_false",
                            help="Disable LoRA and train only the non-LLM modules unless "
                                 "--finetune_llm is also set.")
    parser.set_defaults(lora=True)
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (default 32; higher = more capacity)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling factor (default 64)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability (default 0.05)")
    parser.add_argument("--clip_device", default="cpu")
    parser.add_argument("--device", default=None,
                        help="Training device. Defaults to cuda:LOCAL_RANK when using "
                             "torchrun, otherwise cuda:0.")
    parser.add_argument("--verbose", action="store_true")
    # ── NEW: precomputation args ─────────────────────────────────────────
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Number of GPUs for parallel CLIP/Qwen precomputation")
    parser.add_argument("--clip_batch_size", type=int, default=CLIP_BATCH_SIZE,
                        help="Batch size for CLIP encoding per GPU")
    parser.add_argument("--qwen_batch_size", type=int, default=QWEN_BATCH_SIZE,
                        help="Batch size for Qwen ViT encoding per GPU")
    parser.add_argument("--precompute_cache_dir", type=Path, default=None,
                        help="Directory to save/load precomputed caches (skip recomputation)")
    parser.add_argument("--skip_precompute", action="store_true",
                        help="Skip precomputation; run original online path")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # ── DDP initialisation (torchrun sets LOCAL_RANK / WORLD_SIZE) ────────
    import torch.distributed as dist
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    ddp_active  = world_size > 1

    if ddp_active:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        log.info("DDP initialised: rank %d / %d", local_rank, world_size)

    is_main = (local_rank == 0)
    device  = args.device or f"cuda:{local_rank}"

    cfg_names = (list(DATASET_CONFIGS.keys()) if args.all_configs
                 else (args.configs or list(DATASET_CONFIGS.keys())))

    model = SPCoR(
        backbone=args.backbone,
        d_vis=args.d_vis, d_pose=args.d_pose, d_event=args.d_event,
        d_fourier=args.d_fourier,
        n_fusion_layers=args.n_fusion_layers, n_heads=args.n_heads,
        use_aux_heads=not args.disable_aux_heads,
        n_distill_prompts=args.n_distill_prompts,
        sampled_frames_per_video=args.sampled_frames_per_video,
        fourier_window=args.fourier_window, fourier_bins=args.fourier_bins,
        qvrs_tau=args.qvrs_tau, qvrs_gamma=args.qvrs_gamma,
        cross_robot_lambda=args.cross_robot_lambda,
        clip_device=args.clip_device,
    )

    eval_only = args.stage == "evaluate"

    # ══════════════════════════════════════════════════════════════════════
    #  PRECOMPUTATION PHASE — rank 0 only; other ranks wait at barrier
    # ══════════════════════════════════════════════════════════════════════
    qwen_cache = None
    qwen_cache_d_backbone = None
    clip_cache = None
    sampler_cache = None
    text_cache = None

    # Resolve cache directory and paths unconditionally so all ranks share them
    cache_dir = Path(args.precompute_cache_dir or (args.output_dir / "precomputed"))
    clip_cache_path    = cache_dir / "clip_cache.pt"
    text_cache_path    = cache_dir / "text_cache.pt"
    qwen_cache_path    = cache_dir / "qwen_cache.pt"
    sampler_cache_path = cache_dir / "sampler_cache.pt"

    if is_main:
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Determine which caches still need to be computed
        missing = [
            not clip_cache_path.exists(),
            not text_cache_path.exists(),
            not sampler_cache_path.exists(),
            not qwen_cache_path.exists(),
        ]
        need_compute = any(missing) and not args.skip_precompute

        if need_compute:
            all_items = []
            needed_splits = ["test"] if eval_only else ["train", "val", "test"]
            for cfg_name in cfg_names:
                if cfg_name not in DATASET_CONFIGS:
                    continue
                cfg = DATASET_CONFIGS[cfg_name]
                qa_json_path = Path(cfg["qa_json"])
                for split in needed_splits:
                    items = build_split_items(cfg_name, cfg, split, "none", 0, args.qa_types)
                    items = _enrich_items(items, cfg_name, cfg, qa_json_path)
                    all_items.extend(items)

            log.info("=" * 70)
            log.info("PRECOMPUTATION PHASE: Extracting frozen embeddings in parallel")
            log.info("=" * 70)

            if clip_cache_path.exists():
                log.info("Loading cached CLIP embeddings from %s", clip_cache_path)
                clip_cache = torch.load(clip_cache_path, map_location="cpu", weights_only=False)
            else:
                clip_cache = precompute_clip_embeddings_parallel(
                    all_items, MAX_ROBOTS, args.num_gpus, args.clip_batch_size,
                    args.dense_frames_per_video)
                torch.save(clip_cache, clip_cache_path)
                log.info("Saved CLIP cache to %s", clip_cache_path)

            if text_cache_path.exists():
                log.info("Loading cached CLIP text embeddings from %s", text_cache_path)
                text_cache = torch.load(text_cache_path, map_location="cpu", weights_only=False)
            else:
                text_cache = precompute_clip_text_embeddings(
                    all_items, num_gpus=args.num_gpus)
                torch.save(text_cache, text_cache_path)
                log.info("Saved text cache to %s", text_cache_path)

            if sampler_cache_path.exists():
                log.info("Loading cached sampler indices from %s", sampler_cache_path)
                sampler_cache = torch.load(sampler_cache_path, map_location="cpu",
                                           weights_only=False)
            else:
                sampler_cache = precompute_sampler_indices(
                    clip_cache, text_cache, all_items, model.se_mr2fs,
                    MAX_ROBOTS, args.sampled_frames_per_video)
                torch.save(sampler_cache, sampler_cache_path)
                log.info("Saved sampler cache to %s", sampler_cache_path)

            if qwen_cache_path.exists():
                log.info("Loading cached Qwen embeddings from %s", qwen_cache_path)
                loaded = torch.load(qwen_cache_path, map_location="cpu", weights_only=False)
                qwen_cache = loaded["qwen_cache"]
                qwen_cache_d_backbone = loaded.get("d_backbone")
            else:
                qwen_cache, d_backbone = precompute_qwen_embeddings_parallel(
                    all_items, args.model, MAX_ROBOTS, args.num_gpus,
                    args.qwen_batch_size, args.dense_frames_per_video)
                qwen_cache_d_backbone = d_backbone
                torch.save({"qwen_cache": qwen_cache, "d_backbone": d_backbone},
                           qwen_cache_path)
                log.info("Saved Qwen cache to %s", qwen_cache_path)

            log.info("=" * 70)
            log.info("PRECOMPUTATION COMPLETE — starting training with cached features")
            log.info("=" * 70)

        # Load any caches that exist but weren't loaded yet (covers skip_precompute
        # or the case where caches were produced by a previous run)
        if clip_cache is None and clip_cache_path.exists():
            log.info("Auto-loading cached CLIP embeddings from %s", clip_cache_path)
            clip_cache = torch.load(clip_cache_path, map_location="cpu", weights_only=False)
        if text_cache is None and text_cache_path.exists():
            log.info("Auto-loading cached CLIP text embeddings from %s", text_cache_path)
            text_cache = torch.load(text_cache_path, map_location="cpu", weights_only=False)
        if sampler_cache is None and sampler_cache_path.exists():
            log.info("Auto-loading cached sampler indices from %s", sampler_cache_path)
            sampler_cache = torch.load(sampler_cache_path, map_location="cpu",
                                       weights_only=False)
        if qwen_cache is None and qwen_cache_path.exists():
            log.info("Auto-loading cached Qwen embeddings from %s", qwen_cache_path)
            loaded = torch.load(qwen_cache_path, map_location="cpu", weights_only=False)
            qwen_cache = loaded["qwen_cache"]
            qwen_cache_d_backbone = loaded.get("d_backbone")

    # Release any GPU memory held by precomputation workers before training loads
    # the backbone.  Workers' CUDA allocations stay in the per-process cache
    # until explicitly flushed — this prevents a brief double-load spike.
    torch.cuda.empty_cache()

    # All ranks wait until rank 0 has finished writing / loading caches
    if ddp_active:
        dist.barrier()

    # Non-main ranks: always load whatever caches exist on disk
    if not is_main:
        if qwen_cache_path.exists():
            loaded     = torch.load(qwen_cache_path, map_location="cpu", weights_only=False)
            qwen_cache = loaded["qwen_cache"]
            qwen_cache_d_backbone = loaded.get("d_backbone")
        if sampler_cache_path.exists():
            sampler_cache = torch.load(sampler_cache_path, map_location="cpu",
                                       weights_only=False)

    # ══════════════════════════════════════════════════════════════════════
    #  Load backbone & build datasets
    # ══════════════════════════════════════════════════════════════════════
    model.load_backbone(args.model, device=device)
    cache_d = qwen_cache_d_backbone or infer_qwen_cache_dim(qwen_cache)
    if cache_d is not None and cache_d != model.vis_head.d_in:
        log.warning(
            "Cached Qwen feature width (%d) does not match model visual width (%d). "
            "Realigning visual modules to the cached feature width.",
            cache_d,
            model.vis_head.d_in,
        )
        model.reconfigure_visual_input_dim(cache_d)

    trainer = UACoPTTrainer(
        model, args.output_dir, device=device,
        aux_loss_weight=args.aux_loss_weight,
        lm_loss_weight=args.lm_loss_weight,
        prompt_distill_weight=args.prompt_distill_weight,
        dataloader_num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        min_lr_ratio=args.min_lr_ratio,
        max_grad_norm=args.max_grad_norm,
        eval_every_n_epochs=args.eval_every_n_epochs,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
    )
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    if eval_only:
        _, _, test_ds = load_split_datasets(
            cfg_names=cfg_names, qa_types_filter=args.qa_types,
            dense_frames_per_video=args.dense_frames_per_video,
            splits=["test"],
            qwen_cache=qwen_cache, sampler_cache=sampler_cache)
        train_ds = val_ds = None
    else:
        train_ds, val_ds, test_ds = load_split_datasets(
            cfg_names=cfg_names, qa_types_filter=args.qa_types,
            dense_frames_per_video=args.dense_frames_per_video,
            qwen_cache=qwen_cache, sampler_cache=sampler_cache)
        if is_main:
            log.info("Dataset sizes: train=%d val=%d test=%d",
                     len(train_ds), len(val_ds), len(test_ds))

    stages_to_run = (["finetune", "evaluate"]
                     if args.stage == "all" else [args.stage])
    stage_epochs = {"pretrain_encoder": args.epochs,
                    "pretrain_fusion": args.epochs,
                    "align_llm": max(1, args.epochs // 2),
                    "finetune": args.epochs}

    ckpt_dir = args.checkpoint
    for stage in stages_to_run:
        if stage == "evaluate":
            if is_main:
                if ckpt_dir and (Path(ckpt_dir) / "ua_copt_modules.pt").exists():
                    log.info("Loading best available checkpoint for evaluation: %s", ckpt_dir)
                    trainer.load_checkpoint(Path(ckpt_dir))
                evaluate_ua_copt(model=model, dataset=test_ds,
                                 output_dir=args.output_dir / "eval",
                                 batch_size=args.batch_size, device=device,
                                 num_workers=args.dataloader_num_workers,
                                 pin_memory=args.pin_memory)
        else:
            stage_best  = args.output_dir / f"stage_{stage}" / "best"
            stage_final = args.output_dir / f"stage_{stage}" / "final"
            if not args.force_retrain and (stage_best / "ua_copt_modules.pt").exists():
                ckpt_dir = stage_best
                log.info("Stage '%s' already done, loading from %s", stage, ckpt_dir)
                trainer.load_checkpoint(ckpt_dir)
                continue
            elif not args.force_retrain and (stage_final / "ua_copt_modules.pt").exists():
                ckpt_dir = stage_final
                log.info("Stage '%s' already done (final), loading from %s", stage, ckpt_dir)
                trainer.load_checkpoint(ckpt_dir)
                continue
            if ckpt_dir:
                trainer.load_checkpoint(ckpt_dir)
            ckpt_dir = trainer.train_stage(
                stage=stage, train_dataset=train_ds, val_dataset=val_ds,
                test_dataset=test_ds,
                epochs=stage_epochs.get(stage, args.epochs),
                batch_size=args.batch_size, lr=args.lr,
                vision_lr=args.vision_lr, grad_accum=args.grad_accum,
                finetune_llm=args.finetune_llm,
                use_lora=args.lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )

    if ddp_active:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
