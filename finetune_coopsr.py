

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONFIGURATIONS  ← edit paths and scene lists here
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS: dict[str, dict] = {

    # ── Non-iGibson ──────────────────────────────────────────────────────────
    "two_robot": {
        "label":      "2-robot (non-iGibson)",
        "source":     "non_igibson",
        "n_robots":   2,
        "qa_json":    "your path",
        "video_root": "your path",
        "qa_types":   None,   
        "train_scenes": [
    "1S7LAXRdDqK", "aRKASs4e8j1", "DoSbsoo4EAg", "fRZhp6vWGw7", "gQgtJ9Stk5s",
    "HxmXPBbFCkH", "JNiWU5TZLtt", "nACV8wLu1u5", "oPj9qMxrDEa", "qZ4B7U6XE5Y",
    "u5atqC7vRCY", "w8GiikYuFRk", "XiJhRLvpKpX",
    "1UnKg1rAb8A", "6imZUJGRUq4", "HZ2iMMBsBQ9", "JptJPosx1Z6", "NEVASPhcrxR",
    "oStKKWkQ1id", "R9fYpvCUkV7", "u9rPN5cHWBg", "W9YAR9qcuvN", "XVSZJAtHKdi",
    "Z2DQddYp1fn",
    "1xGrZPxG1Hz", "6YtDG3FhNvx", "bB6nKqfsb1z", "dQrLTxHvLXU", "g8Xrdbe9fir",
    "GTV2Y73Sn5t", "RaYrxWt5pR1", "URjpCob8MGw", "WhNyDTnd9g5", "xWvSkKiWQpC",
    "zepmXAdrpjR",
    "226REUyJh2K", "741Fdj7NLF9", "bdp1XNEdvmW", "DsEJeNPcZtE", "g8Xrdbe9fir",
    "GTV2Y73Sn5t", "NGyoyh91xXJ", "PE6kVEtrxtj", "RTV2n6fXB2w", "UuwwmrTsfBN",
    "Wo6kuutE9i7", "XYyR54sxe6b", "ZNanfzgCdm3",
    "2Pc8W48bu21", "77mMEyxhs44", "bHKTDQFJxTw", "E1NrAhMoqvB", "GGBvSFddQgs",
    "h6nwVLpAKQz", "iKFn6fzyRqs", "PPTLa8SkUfo", "S7uMvxjBVZq",
    "v7DzfFFEpsD", "wPLokgvCnuk", "yHLr6bvWsVm", "zUG6FL9TYeR",
    "3CBBjsNkhqW", "8B43pG641ff", "ceJTwFNjqCt", "ENiCjXWB6aQ", "ggNAcMh8JPT",
    "H8rQCnvBgo6", "iLDo95ZbDJq", "L5QEsaVqwrY", "SgkmkWjjmDJ", "vDfkYo5VqEQ",
    "wsAYBFtQaL7", "YHmAkqgwe2p",
    "3XYAD64HpDr", "8wJuSPJ9FXG", "CQWES1bawee", "erXNfWVjqZ8", "gjhYih4upQ9",
    "HeSYRw7eMtG", "ixTj1aTMup2", "LcAd9dhvVwh", "NtnvZSMK3en", "vLpv2VX547B",
    "X6Pct1msZv5", "YJDUB7hWg9h",
    "4vwGX7U38Ux", "92vYG1q49FY", "CthA7sQNTPK", "fK2vEV32Lag", "gmuS7Wgsbrx",
    "HfMobPm86Xn", "j2EJhFEQGCL", "LVgQNuK8vtv", "oahi4u45xMf", "QN2dRqwd84J",
    "xAHnY3QzFUN", "YMNvYDhK8mB",
    "5biL7VEkByM", "9h5JJxM6E5S", "DBBESbk4Y3k", "FnDDfrBZPhh", "GPyDUnjwZQy",
    "HkseAnWCgqk", "j6fHrce9pHR", "mt9H8KcxRKD", "oEPjPNSPmzL", "QVAA6zecMHu",
    "TYDavTf8oyy", "YmWinf3mhb5",
    "5Kw4nGdqYtS", "ACZZiU6BXLz", "DNWbUAJYsPy", "FRQ75PjD278", "gQ3xxshDiCz",
    "hWDDQnSDMXb", "Jfyvj3xn2aJ", "MVVzj944atG", "ooq3SnvC79d", "qz3829g1Lzf",
    "U3oQjwTuMX8", "W16Bm4ysK8v",
        ],
        "val_scenes": [
    "5Kw4nGdqYtS", "ACZZiU6BXLz", "DNWbUAJYsPy", "FRQ75PjD278", "gQ3xxshDiCz",
    "hWDDQnSDMXb", "Jfyvj3xn2aJ", "MVVzj944atG", "ooq3SnvC79d", "qz3829g1Lzf",
    "U3oQjwTuMX8", "W16Bm4ysK8v",
        ],
        "test_scenes": [
    "fxbzYAGkrtm", "GsQBY83r3hb", "iePHCSf119p", "kJxT5qssH4H", "NPHxDe6VeCc",
    "qgZhhx1MpTi", "sX9xad6ULKc", "VoVGtfYrpuQ", "XfUxBGTFQQb", "yX5efd48dLf",
    "6HRFAUDqpTb", "b3WpMbPFB6q", "DqJKU7YU7dA", "g7hUFVNac26", "GtM3JtRvvvR",
    "iigzG1rtanx", "KjZrPggnHm8", "nS8T59Aw3sf", "qk9eeNeR4vw", "TSJmdttd2GV",
    "VSxVP19Cdyw", "xgLmjqzoAzF", "YY8rqV6L6rf",
        ],
    },

    "three_robot": {
        "label":      "3-robot (non-iGibson)",
        "source":     "non_igibson",
        "n_robots":   3,
        "qa_json":    "your path",
        "video_root": "your path",
        "qa_types":   None,
        "train_scenes": [
    "1S7LAXRdDqK", "aRKASs4e8j1", "DoSbsoo4EAg", "fRZhp6vWGw7", "gQgtJ9Stk5s",
    "HxmXPBbFCkH", "JNiWU5TZLtt", "nACV8wLu1u5", "oPj9qMxrDEa", "qZ4B7U6XE5Y",
    "u5atqC7vRCY", "w8GiikYuFRk", "XiJhRLvpKpX",
    "1UnKg1rAb8A", "6imZUJGRUq4", "HZ2iMMBsBQ9", "JptJPosx1Z6", "NEVASPhcrxR",
    "oStKKWkQ1id", "R9fYpvCUkV7", "u9rPN5cHWBg", "W9YAR9qcuvN", "XVSZJAtHKdi",
    "Z2DQddYp1fn",
    "1xGrZPxG1Hz", "6YtDG3FhNvx", "bB6nKqfsb1z", "dQrLTxHvLXU", "g8Xrdbe9fir",
    "GTV2Y73Sn5t", "RaYrxWt5pR1", "URjpCob8MGw", "WhNyDTnd9g5", "xWvSkKiWQpC",
    "zepmXAdrpjR",
    "226REUyJh2K", "741Fdj7NLF9", "bdp1XNEdvmW", "DsEJeNPcZtE", "g8Xrdbe9fir",
    "GTV2Y73Sn5t", "NGyoyh91xXJ", "PE6kVEtrxtj", "RTV2n6fXB2w", "UuwwmrTsfBN",
    "Wo6kuutE9i7", "XYyR54sxe6b", "ZNanfzgCdm3",
    "2Pc8W48bu21", "77mMEyxhs44", "bHKTDQFJxTw", "E1NrAhMoqvB", "GGBvSFddQgs",
    "h6nwVLpAKQz", "iKFn6fzyRqs", "PPTLa8SkUfo", "S7uMvxjBVZq",
    "v7DzfFFEpsD", "wPLokgvCnuk", "yHLr6bvWsVm", "zUG6FL9TYeR",
    "3CBBjsNkhqW", "8B43pG641ff", "ceJTwFNjqCt", "ENiCjXWB6aQ", "ggNAcMh8JPT",
    "H8rQCnvBgo6", "iLDo95ZbDJq", "L5QEsaVqwrY", "SgkmkWjjmDJ", "vDfkYo5VqEQ",
    "wsAYBFtQaL7", "YHmAkqgwe2p",
    "3XYAD64HpDr", "8wJuSPJ9FXG", "CQWES1bawee", "erXNfWVjqZ8", "gjhYih4upQ9",
    "HeSYRw7eMtG", "ixTj1aTMup2", "LcAd9dhvVwh", "NtnvZSMK3en", "vLpv2VX547B",
    "X6Pct1msZv5", "YJDUB7hWg9h",
    "4vwGX7U38Ux", "92vYG1q49FY", "CthA7sQNTPK", "fK2vEV32Lag", "gmuS7Wgsbrx",
    "HfMobPm86Xn", "j2EJhFEQGCL", "LVgQNuK8vtv", "oahi4u45xMf", "QN2dRqwd84J",
    "xAHnY3QzFUN", "YMNvYDhK8mB",
    "5biL7VEkByM", "9h5JJxM6E5S", "DBBESbk4Y3k", "FnDDfrBZPhh", "GPyDUnjwZQy",
    "HkseAnWCgqk", "j6fHrce9pHR", "mt9H8KcxRKD", "oEPjPNSPmzL", "QVAA6zecMHu",
    "TYDavTf8oyy", "YmWinf3mhb5",
    "5Kw4nGdqYtS", "ACZZiU6BXLz", "DNWbUAJYsPy", "FRQ75PjD278", "gQ3xxshDiCz",
    "hWDDQnSDMXb", "Jfyvj3xn2aJ", "MVVzj944atG", "ooq3SnvC79d", "qz3829g1Lzf",
    "U3oQjwTuMX8", "W16Bm4ysK8v",
        ],
        "val_scenes": [
    "5Kw4nGdqYtS", "ACZZiU6BXLz", "DNWbUAJYsPy", "FRQ75PjD278", "gQ3xxshDiCz",
    "hWDDQnSDMXb", "Jfyvj3xn2aJ", "MVVzj944atG", "ooq3SnvC79d", "qz3829g1Lzf",
    "U3oQjwTuMX8", "W16Bm4ysK8v",
        ],
        "test_scenes": [
    "fxbzYAGkrtm", "GsQBY83r3hb", "iePHCSf119p", "kJxT5qssH4H", "NPHxDe6VeCc",
    "qgZhhx1MpTi", "sX9xad6ULKc", "VoVGtfYrpuQ", "XfUxBGTFQQb", "yX5efd48dLf",
    "6HRFAUDqpTb", "b3WpMbPFB6q", "DqJKU7YU7dA", "g7hUFVNac26", "GtM3JtRvvvR",
    "iigzG1rtanx", "KjZrPggnHm8", "nS8T59Aw3sf", "qk9eeNeR4vw", "TSJmdttd2GV",
    "VSxVP19Cdyw", "xgLmjqzoAzF", "YY8rqV6L6rf",
        ],
    },

    "four_robot": {
        "label":      "4-robot (non-iGibson)",
        "source":     "non_igibson",
        "n_robots":   4,
        "qa_json":    "your path",
        "video_root": "your path",
        "qa_types":   None,
        "train_scenes": [
    "1S7LAXRdDqK", "aRKASs4e8j1", "DoSbsoo4EAg", "fRZhp6vWGw7", "gQgtJ9Stk5s",
    "HxmXPBbFCkH", "JNiWU5TZLtt", "nACV8wLu1u5", "oPj9qMxrDEa", "qZ4B7U6XE5Y",
    "u5atqC7vRCY", "w8GiikYuFRk", "XiJhRLvpKpX",
    "1UnKg1rAb8A", "6imZUJGRUq4", "HZ2iMMBsBQ9", "JptJPosx1Z6", "NEVASPhcrxR",
    "oStKKWkQ1id", "R9fYpvCUkV7", "u9rPN5cHWBg", "W9YAR9qcuvN", "XVSZJAtHKdi",
    "Z2DQddYp1fn",
    "1xGrZPxG1Hz", "6YtDG3FhNvx", "bB6nKqfsb1z", "dQrLTxHvLXU", "g8Xrdbe9fir",
    "GTV2Y73Sn5t", "RaYrxWt5pR1", "URjpCob8MGw", "WhNyDTnd9g5", "xWvSkKiWQpC",
    "zepmXAdrpjR",
    "226REUyJh2K", "741Fdj7NLF9", "bdp1XNEdvmW", "DsEJeNPcZtE", "g8Xrdbe9fir",
    "GTV2Y73Sn5t", "NGyoyh91xXJ", "PE6kVEtrxtj", "RTV2n6fXB2w", "UuwwmrTsfBN",
    "Wo6kuutE9i7", "XYyR54sxe6b", "ZNanfzgCdm3",
    "2Pc8W48bu21", "77mMEyxhs44", "bHKTDQFJxTw", "E1NrAhMoqvB", "GGBvSFddQgs",
    "h6nwVLpAKQz", "iKFn6fzyRqs", "PPTLa8SkUfo", "S7uMvxjBVZq",
    "v7DzfFFEpsD", "wPLokgvCnuk", "yHLr6bvWsVm", "zUG6FL9TYeR",
    "3CBBjsNkhqW", "8B43pG641ff", "ceJTwFNjqCt", "ENiCjXWB6aQ", "ggNAcMh8JPT",
    "H8rQCnvBgo6", "iLDo95ZbDJq", "L5QEsaVqwrY", "SgkmkWjjmDJ", "vDfkYo5VqEQ",
    "wsAYBFtQaL7", "YHmAkqgwe2p",
    "3XYAD64HpDr", "8wJuSPJ9FXG", "CQWES1bawee", "erXNfWVjqZ8", "gjhYih4upQ9",
    "HeSYRw7eMtG", "ixTj1aTMup2", "LcAd9dhvVwh", "NtnvZSMK3en", "vLpv2VX547B",
    "X6Pct1msZv5", "YJDUB7hWg9h",
    "4vwGX7U38Ux", "92vYG1q49FY", "CthA7sQNTPK", "fK2vEV32Lag", "gmuS7Wgsbrx",
    "HfMobPm86Xn", "j2EJhFEQGCL", "LVgQNuK8vtv", "oahi4u45xMf", "QN2dRqwd84J",
    "xAHnY3QzFUN", "YMNvYDhK8mB",
    "5biL7VEkByM", "9h5JJxM6E5S", "DBBESbk4Y3k", "FnDDfrBZPhh", "GPyDUnjwZQy",
    "HkseAnWCgqk", "j6fHrce9pHR", "mt9H8KcxRKD", "oEPjPNSPmzL", "QVAA6zecMHu",
    "TYDavTf8oyy", "YmWinf3mhb5",
    "5Kw4nGdqYtS", "ACZZiU6BXLz", "DNWbUAJYsPy", "FRQ75PjD278", "gQ3xxshDiCz",
    "hWDDQnSDMXb", "Jfyvj3xn2aJ", "MVVzj944atG", "ooq3SnvC79d", "qz3829g1Lzf",
    "U3oQjwTuMX8", "W16Bm4ysK8v",
        ],
        "val_scenes": [
    "5Kw4nGdqYtS", "ACZZiU6BXLz", "DNWbUAJYsPy", "FRQ75PjD278", "gQ3xxshDiCz",
    "hWDDQnSDMXb", "Jfyvj3xn2aJ", "MVVzj944atG", "ooq3SnvC79d", "qz3829g1Lzf",
    "U3oQjwTuMX8", "W16Bm4ysK8v",
        ],
        "test_scenes": [
    "fxbzYAGkrtm", "GsQBY83r3hb", "iePHCSf119p", "kJxT5qssH4H", "NPHxDe6VeCc",
    "qgZhhx1MpTi", "sX9xad6ULKc", "VoVGtfYrpuQ", "XfUxBGTFQQb", "yX5efd48dLf",
    "6HRFAUDqpTb", "b3WpMbPFB6q", "DqJKU7YU7dA", "g7hUFVNac26", "GtM3JtRvvvR",
    "iigzG1rtanx", "KjZrPggnHm8", "nS8T59Aw3sf", "qk9eeNeR4vw", "TSJmdttd2GV",
    "VSxVP19Cdyw", "xgLmjqzoAzF", "YY8rqV6L6rf",
        ],
    },

    # ── iGibson ───────────────────────────────────────────────────────────────
    "igbson_two": {
        "label":      "2-robot (iGibson)",
        "source":     "igibson",
        "n_robots":   2,
        "qa_json":    "your path",
        "video_root": "your path",
        "qa_types":   None,
        "train_scenes": [
            "Merom_0_int", "Merom_1_int",
            "Pomaria_0_int", "Pomaria_1_int", "Pomaria_2_int",
            "Rs_int",
            "Wainscott_0_int", "Wainscott_1_int",
        ],
        "val_scenes": [
            "Ihlen_0_int", "Ihlen_1_int",
        ],
        "test_scenes": [
            "Beechwood_0_int", "Beechwood_1_int",
            "Benevolence_0_int", "Benevolence_1_int",
        ],
    },

    "igbson_three": {
        "label":      "3-robot (iGibson)",
        "source":     "igibson",
        "n_robots":   3,
        "qa_json":    "your path",
        "video_root": "your path",
        "qa_types":   None,
        "train_scenes": [
            "Merom_0_int", "Merom_1_int",
            "Pomaria_0_int", "Pomaria_1_int", "Pomaria_2_int",
            "Rs_int",
            "Wainscott_0_int", "Wainscott_1_int",
        ],
        "val_scenes": [
            "Ihlen_0_int", "Ihlen_1_int",
        ],
        "test_scenes": [
            "Beechwood_0_int", "Beechwood_1_int",
            "Benevolence_0_int", "Benevolence_1_int",
        ],
    },

    "igbson_four": {
        "label":      "4-robot (iGibson)",
        "source":     "igibson",
        "n_robots":   4,
        "qa_json":    "your path",
        "video_root": "your path",
        "qa_types":   None,
        "train_scenes": [
            "Merom_0_int", "Merom_1_int",
            "Pomaria_0_int", "Pomaria_1_int", "Pomaria_2_int",
            "Rs_int",
            "Wainscott_0_int", "Wainscott_1_int",
        ],
        "val_scenes": [
            "Ihlen_0_int", "Ihlen_1_int",
        ],
        "test_scenes": [
            "Beechwood_0_int", "Beechwood_1_int",
            "Benevolence_0_int", "Benevolence_1_int",
        ],
    },
}


# All QA type codes across all configurations (QA-01..QA-23).
# qa_types=None in DATASET_CONFIGS means all 23 types are included.
ALL_QA_TYPES = [f"QA-{i:02d}" for i in range(1, 24)]

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

VIDEO_NAME_PATTERNS = [
    re.compile(r"locobot(\d+)[_\-]cam",        re.IGNORECASE),
    re.compile(r"video[_\-]?robot[_\-]?(\d+)", re.IGNORECASE),
    re.compile(r"robot[_\-]?(\d+)[_\-]?cam",   re.IGNORECASE),
    re.compile(r"robot[_\-]?(\d+)",            re.IGNORECASE),
]

SYSTEM_PROMPT = (
    "You are evaluating a multi-robot cooperative spatial reasoning task. "
    "You will be given sensor data from N robots exploring an indoor environment, "
    "optionally accompanied by sampled video frames from each robot's camera. "
    "Answer the multiple-choice question by selecting the single best option.\n\n"
    "RULES:\n"
    "- Respond with ONLY the letter of your answer: A, B, C, or D.\n"
    "- Do not add any explanation, punctuation, or extra text.\n"
    "- If unsure, make your best guess from the given options."
)

SYSTEM_PROMPT_OPEN = (
    "You are evaluating a multi-robot cooperative spatial reasoning task. "
    "You will be given sensor data from N robots exploring an indoor environment, "
    "optionally accompanied by sampled video frames from each robot's camera. "
    "The question asks you to estimate a specific numeric value.\n\n"
    "RULES:\n"
    "- Respond with ONLY a single numeric value (integer or decimal).\n"
    "- Do not include units, labels, explanation, or any other text.\n"
    "- Use a negative sign where appropriate.\n"
    "- If unsure, give your best numeric estimate."
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("coopsr_sft")

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO → FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def find_robot_videos(video_root: Path, scene: str,
                      exploration: str) -> dict[int, Path]:
    exp_dir = video_root / scene / exploration
    if not exp_dir.is_dir():
        return {}
    found: dict[int, Path] = {}
    for f in sorted(exp_dir.iterdir()):
        if f.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        for pattern in VIDEO_NAME_PATTERNS:
            m = pattern.search(f.stem)
            if m:
                rid = int(m.group(1))
                if rid not in found:
                    found[rid] = f
                break
    return found


def extract_frames(video_path: Path, n_frames: int) -> list[Image.Image]:
    """Extract n_frames evenly-spaced frames using sequential grab().

    Sequential grab() is far faster than random seeking (cap.set) because
    H.264/H.265 random seeks must re-decode from the nearest keyframe each
    time.  grab() advances the decoder one frame at a time without full pixel
    decode, so only the frames we actually keep are converted to BGR/RGB.
    """
    try:
        import cv2
    except ImportError:
        log.warning("opencv-python not installed; skipping video frames.")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    # Minimise internal buffering during sequential read
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    n_frames = min(n_frames, total)
    # sorted unique indices (linspace can produce duplicates for small total)
    indices = sorted(set(np.linspace(0, total - 1, n_frames, dtype=int).tolist()))

    frames: list[Image.Image] = []
    cur = 0  # decoder's current position (next frame cap.read() will return)

    for idx in indices:
        skip = idx - cur
        if skip > 0:
            # grab() advances the decoder without full pixel decode — fast
            for _ in range(skip):
                if not cap.grab():
                    break
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cur = idx + 1

    cap.release()
    return frames


def image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_context_text(item: dict, context_mode: str) -> str:
    nl  = item.get("context_nl",      "").strip()
    tab = item.get("context_tabular", "").strip()
    if context_mode == "nl":
        return nl or ("[Tabular context]\n" + tab if tab else "")
    if context_mode == "tabular":
        return tab or ("[NL context]\n" + nl if nl else "")
    if context_mode == "both":
        parts = []
        if nl:  parts.append("=== Natural-Language Context ===\n" + nl)
        if tab: parts.append("=== Tabular Context ===\n" + tab)
        return "\n\n".join(parts)
    return ""


def build_user_text(item: dict, context_mode: str, robot_ids: list[int]) -> str:
    fmt   = item.get("answer_format", "MC4")
    lines: list[str] = []
    ctx = build_context_text(item, context_mode)
    if ctx:
        lines.append(ctx)
        lines.append("")
    if robot_ids:
        id_str = ", ".join(f"Robot {r}" for r in sorted(robot_ids))
        n = len(robot_ids)
        lines.append(
            f"Video frames are provided for {id_str} "
            f"({n} robot{'s' if n > 1 else ''}). "
            "Frames are ordered by robot ID then chronologically."
        )
        lines.append("")
    lines.append(f"Question: {item['question']}")
    lines.append("")
    if fmt == "OPEN":
        unit = item.get("metadata", {}).get("unit", "")
        hint = f" (expected unit: {unit})" if unit else ""
        lines.append(f"Answer{hint} — numeric value only, no units or text:")
    else:
        for ch in (item.get("choices") or []):
            lines.append(ch)
        lines.append("")
        lines.append("Answer (A/B/C/D only):")
    return "\n".join(lines)


def ground_truth_answer(item: dict) -> str:
    """Return the target answer string the model should learn to produce."""
    fmt = item.get("answer_format", "MC4")
    if fmt == "OPEN":
        gt = item.get("metadata", {}).get("answer_value")
        return str(gt) if gt is not None else str(item.get("answer", ""))
    # MC4: just the letter (A / B / C / D)
    return item.get("answer_key", item.get("answer", ""))

# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_split_items(
    cfg_name:        str,
    cfg:             dict,
    split:           str,           # "train", "val", or "test"
    context_mode:    str,
    frames_per_vid:  int,
    qa_types_filter: Optional[list[str]] = None,
) -> list[dict]:
    """
    Load QA items for one config / split, decode videos once per exploration,
    and return a list of dicts:

        {
            "id", "config", "source", "split",
            "system",        # system prompt text
            "text",          # full user-turn text (context + question + choices)
            "answer",        # ground-truth answer (letter or number)
            "answer_format", # "MC4" | "OPEN"
            "images",        # list[PIL.Image] — may be empty if no videos
            "n_robots", "qa_type", "qa_type_name", "difficulty", "axis",
        }
    """
    qa_json    = Path(cfg["qa_json"])
    video_root = Path(cfg["video_root"]) if cfg.get("video_root") else None
    source     = cfg.get("source", "non_igibson")
    n_robots   = cfg.get("n_robots", 2)
    scene_set  = set(cfg.get(f"{split}_scenes") or [])
    # "test" uses test_scenes; "train"/"val" use their own keys

    if not qa_json.exists():
        log.warning("[%s/%s] QA JSON not found: %s — skipping.", cfg_name, split, qa_json)
        return []

    if not scene_set:
        log.warning("[%s/%s] No scenes configured — skipping.", cfg_name, split)
        return []

    with open(qa_json, encoding="utf-8") as f:
        all_items: list[dict] = json.load(f)

    # Log QA types present in file
    qt_in_file = sorted(set(it["qa_type"] for it in all_items),
                        key=lambda k: (0, int(m.group(1)), "")
                        if (m := re.match(r"QA-(\d+)", k)) else (1, 0, k))
    log.info("[%s/%s] QA types in file: %s", cfg_name, split, qt_in_file)

    # Scene filter
    all_items = [it for it in all_items if it.get("scene") in scene_set]
    log.info("[%s/%s] %d items after scene filter (%d scenes configured)",
             cfg_name, split, len(all_items), len(scene_set))
    missing = scene_set - {it["scene"] for it in all_items}
    if missing:
        log.warning("[%s/%s] Scene IDs not in QA file: %s", cfg_name, split, sorted(missing))

    # Config-level QA-type filter
    cfg_qt = cfg.get("qa_types")
    if cfg_qt is not None:
        all_items = [it for it in all_items if it["qa_type"] in set(cfg_qt)]

    # CLI-level QA-type filter
    if qa_types_filter:
        all_items = [it for it in all_items if it["qa_type"] in set(qa_types_filter)]

    log.info("[%s/%s] %d QA items to process", cfg_name, split, len(all_items))
    if not all_items:
        return []

    # Group by (scene, exploration) — decode each video exactly once
    exploration_groups: OrderedDict[tuple, list[dict]] = OrderedDict()
    for item in all_items:
        key = (item["scene"], item["exploration"])
        exploration_groups.setdefault(key, []).append(item)

    results: list[dict] = []
    n_exp = len(exploration_groups)

    for exp_idx, ((scene, exploration), exp_items) in enumerate(
            tqdm(exploration_groups.items(),
                 desc=f"[{cfg_name}/{split}]", unit="exp"), start=1):

        # Decode all robot videos for this exploration exactly once
        robot_frame_map: dict[int, list[Image.Image]] = {}
        if video_root and frames_per_vid > 0:
            for rid, vpath in sorted(
                    find_robot_videos(video_root, scene, exploration).items()):
                frames = extract_frames(vpath, frames_per_vid)
                if frames:
                    robot_frame_map[rid] = frames
            if robot_frame_map:
                log.debug("[%s/%s] exp %d/%d %s/%s — %d robot(s) decoded",
                          cfg_name, split, exp_idx, n_exp, scene, exploration,
                          len(robot_frame_map))

        all_frames = [fr for rid in sorted(robot_frame_map)
                      for fr in robot_frame_map[rid]]
        robot_ids  = sorted(robot_frame_map.keys())

        for item in exp_items:
            fmt    = item.get("answer_format", "MC4")
            sys_p  = SYSTEM_PROMPT_OPEN if fmt == "OPEN" else SYSTEM_PROMPT
            text   = build_user_text(item, context_mode, robot_ids)
            answer = ground_truth_answer(item)

            results.append({
                "id":            item["id"],
                "config":        cfg_name,
                "source":        source,
                "split":         split,
                "system":        sys_p,
                "text":          text,
                "answer":        answer,
                "answer_format": fmt,
                "images":        list(all_frames),   # same PIL objects reused
                "n_robots":      n_robots,
                "qa_type":       item["qa_type"],
                "qa_type_name":  item.get("qa_type_name", ""),
                "difficulty":    item["difficulty"],
                "axis":          item["axis"],
            })

        del all_frames, robot_frame_map   # release before next exploration

    log.info("[%s/%s] Built %d items", cfg_name, split, len(results))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# JSONL WRITER  (OpenAI / TRL / LLaMA-Factory compatible)
# ─────────────────────────────────────────────────────────────────────────────

def write_jsonl_base64(items: list[dict], path: Path):
    """
    Write items as chat-format JSONL with base64-embedded images.

    Each line:
        {"messages": [
            {"role": "system",    "content": "<system prompt>"},
            {"role": "user",      "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                ...
                {"type": "text", "text": "<user text>"}
            ]},
            {"role": "assistant", "content": "<answer>"}
        ]}

    Compatible with: OpenAI fine-tuning API, TRL SFTTrainer, LLaMA-Factory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            user_content: list[dict] = []
            for img in item.get("images", []):
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_to_base64(img)}"
                    },
                })
            user_content.append({"type": "text", "text": item["text"]})

            record = {
                "messages": [
                    {"role": "system",    "content": item["system"]},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": item["answer"]},
                ],
                # Metadata fields — ignored during training, useful for debugging
                "_id":            item["id"],
                "_config":        item["config"],
                "_source":        item["source"],
                "_split":         item["split"],
                "_qa_type":       item["qa_type"],
                "_answer_format": item["answer_format"],
                "_difficulty":    item["difficulty"],
                "_axis":          item["axis"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Wrote %d items → %s", len(items), path)

# ─────────────────────────────────────────────────────────────────────────────
# OPENAI FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────

class OpenAITrainer:
    """
    Upload merged train JSONL and create an OpenAI fine-tuning job.
    Polls until the job finishes or fails.
    env: OPENAI_API_KEY
    """
    def __init__(self, model: str, api_key: Optional[str] = None,
                 api_base: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key  = api_key  or os.environ.get("OPENAI_API_KEY"),
            base_url = api_base or os.environ.get("OPENAI_API_BASE") or None,
        )
        self.model = model

    def train(self, train_jsonl: Path, val_jsonl: Optional[Path],
              output_dir: Path, epochs: int = 3, **kwargs):
        log.info("Uploading train file: %s", train_jsonl)
        with open(train_jsonl, "rb") as fh:
            train_file = self.client.files.create(file=fh, purpose="fine-tune")

        val_file_id: Optional[str] = None
        if val_jsonl and val_jsonl.exists():
            log.info("Uploading val file: %s", val_jsonl)
            with open(val_jsonl, "rb") as fh:
                val_file = self.client.files.create(file=fh, purpose="fine-tune")
            val_file_id = val_file.id

        ft_kwargs: dict = {
            "training_file":   train_file.id,
            "model":           self.model,
            "hyperparameters": {"n_epochs": epochs},
        }
        if val_file_id:
            ft_kwargs["validation_file"] = val_file_id

        job = self.client.fine_tuning.jobs.create(**ft_kwargs)
        log.info("Fine-tuning job created: %s  status=%s", job.id, job.status)

        poll_interval = 60  # seconds
        while job.status not in ("succeeded", "failed", "cancelled"):
            time.sleep(poll_interval)
            job = self.client.fine_tuning.jobs.retrieve(job.id)
            log.info("Job %s  status=%s", job.id, job.status)

        result = {
            "job_id":            job.id,
            "status":            job.status,
            "fine_tuned_model":  getattr(job, "fine_tuned_model", None),
        }
        out = output_dir / "openai_job_result.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2)
        log.info("Job result → %s", out)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION CALLBACK
# Runs greedy-decode accuracy on val and test items at the end of every
# eval_every_n_epochs epochs, then writes a JSON report alongside the adapter.
# ─────────────────────────────────────────────────────────────────────────────

import re as _re
_ANSWER_PAT = _re.compile(r"\b([A-D])\b")
_NUMERIC_PAT = _re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _extract_answer(raw: str) -> str:
    if not raw: return "X"
    s = raw.strip()
    if s.upper() in {"A","B","C","D"}: return s.upper()
    m = _ANSWER_PAT.search(s.upper())
    return m.group(1) if m else "X"


def _extract_numeric(raw: str) -> Optional[float]:
    if not raw: return None
    try:   return float(raw.strip())
    except ValueError: pass
    m = _NUMERIC_PAT.search(raw.strip())
    return float(m.group()) if m else None


def _score_items(items: list[dict], model_obj, proc_obj, device) -> dict:
    """
    Run greedy generation on each item and compute accuracy (MC4)
    or MSE/MAE (OPEN).  Returns a metrics dict with per-breakdown stats.
    """
    import torch
    mc4_correct = 0; mc4_total = 0; mc4_invalid = 0
    open_sq: list[float] = []; open_ab: list[float] = []
    open_fail = 0

    # Per-item records for breakdown analysis
    records: list[dict] = []

    model_obj.eval()
    with torch.inference_mode():
        for item in items:
            fmt = item["answer_format"]
            # Build input — text only (images already stripped from JSONL for
            # eval items; if PIL images are present use them)
            images = item.get("images") or []
            prompt_text = item["system"] + "\n\n" + item["text"]

            if hasattr(proc_obj, "apply_chat_template"):
                msgs = [{"role":"system","content":item["system"]},
                        {"role":"user","content":item["text"]}]
                try:
                    prompt_text = proc_obj.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    pass

            try:
                if images and hasattr(proc_obj, "__call__"):
                    enc = proc_obj(text=prompt_text, images=images or None,
                                   return_tensors="pt").to(device)
                else:
                    enc = proc_obj(text=prompt_text,
                                   return_tensors="pt").to(device)
            except Exception:
                enc = proc_obj(prompt_text, return_tensors="pt").to(device)

            max_new = 32 if fmt == "OPEN" else 4
            out = model_obj.generate(**enc, max_new_tokens=max_new, do_sample=False)
            decoded = proc_obj.decode(
                out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            rec = {
                "id":         item.get("id", ""),
                "qa_type":    item.get("qa_type", ""),
                "n_robots":   item.get("n_robots", 0),
                "source":     item.get("source", ""),
                "difficulty": item.get("difficulty", ""),
                "axis":       item.get("axis", ""),
                "config":     item.get("config", ""),
                "fmt":        fmt,
            }

            if fmt == "OPEN":
                pred = _extract_numeric(decoded)
                gt   = item.get("metadata_answer_value")
                if gt is None:
                    try: gt = float(item["answer"])
                    except: pass
                if pred is not None and gt is not None:
                    diff = pred - float(gt)
                    open_sq.append(diff**2); open_ab.append(abs(diff))
                    rec.update({"sq_err": diff**2, "abs_err": abs(diff),
                                "failed": False})
                else:
                    open_fail += 1
                    rec.update({"sq_err": None, "abs_err": None, "failed": True})
            else:
                pred_key = _extract_answer(decoded)
                gt_key   = item.get("answer_key") or item["answer"][:1].upper()
                mc4_total += 1
                correct = pred_key == gt_key
                if correct: mc4_correct += 1
                if pred_key == "X": mc4_invalid += 1
                rec.update({"correct": correct, "invalid": pred_key == "X"})

            records.append(rec)

    def _acc(c, t): return round(c / t * 100, 2) if t else 0.0

    # ── Breakdown helpers ──────────────────────────────────────────────────────
    def _breakdown_mc4(recs, key):
        groups: dict = {}
        for r in recs:
            if r["fmt"] != "MC4": continue
            groups.setdefault(r.get(key, ""), []).append(r)
        return {
            gk: {"total": len(gv),
                 "correct": sum(1 for r in gv if r.get("correct")),
                 "accuracy": _acc(sum(1 for r in gv if r.get("correct")), len(gv))}
            for gk, gv in sorted(groups.items()) if gk != ""
        }

    def _breakdown_open(recs, key):
        groups: dict = {}
        for r in recs:
            if r["fmt"] != "OPEN" or r.get("failed"): continue
            groups.setdefault(r.get(key, ""), []).append(r)
        result = {}
        for gk, gv in sorted(groups.items()):
            if gk == "": continue
            sqs  = [r["sq_err"]  for r in gv if r["sq_err"]  is not None]
            abs_ = [r["abs_err"] for r in gv if r["abs_err"] is not None]
            mse  = sum(sqs) / len(sqs) if sqs else None
            result[gk] = {
                "n":    len(sqs),
                "mse":  round(mse, 6)        if mse  is not None else None,
                "rmse": round(mse**0.5, 6)   if mse  is not None else None,
                "mae":  round(sum(abs_) / len(abs_), 6) if abs_ else None,
            }
        return result

    result: dict = {}
    if mc4_total:
        result["mc4"] = {
            "total":         mc4_total,
            "correct":       mc4_correct,
            "accuracy":      _acc(mc4_correct, mc4_total),
            "invalid":       mc4_invalid,
            "by_qa_type":    _breakdown_mc4(records, "qa_type"),
            "by_n_robots":   _breakdown_mc4(records, "n_robots"),
            "by_source":     _breakdown_mc4(records, "source"),
            "by_difficulty": _breakdown_mc4(records, "difficulty"),
            "by_axis":       _breakdown_mc4(records, "axis"),
            "by_config":     _breakdown_mc4(records, "config"),
        }
    if open_sq or open_fail:
        n   = len(open_sq)
        mse = sum(open_sq) / n if n else None
        result["open"] = {
            "total":          n,
            "mse":            round(mse, 6)      if mse is not None else None,
            "rmse":           round(mse**0.5, 6) if mse is not None else None,
            "mae":            round(sum(open_ab) / n, 6) if n else None,
            "parse_failures": open_fail,
            "by_qa_type":     _breakdown_open(records, "qa_type"),
            "by_n_robots":    _breakdown_open(records, "n_robots"),
            "by_source":      _breakdown_open(records, "source"),
        }
    return result


def _print_eval_summary(metrics: dict, split_name: str, epoch: int,
                        out_dir: Path) -> None:
    """
    Print and save a human-readable evaluation summary for one split.
    Writes to <out_dir>/epoch_{epoch:02d}_{split_name}_summary.txt.
    """
    lines: list[str] = []
    w = 72

    def _add(s: str = ""): lines.append(s)

    _add("=" * w)
    _add(f"  CoopSR-Bench SFT — Epoch {epoch}  [{split_name.upper()}]")
    _add("=" * w)

    mc4 = metrics.get("mc4", {})
    op  = metrics.get("open", {})

    # ── Overall ───────────────────────────────────────────────────────────────
    if mc4:
        _add(f"  MC4  overall : Acc={mc4['accuracy']:.1f}%  "
             f"({mc4['correct']}/{mc4['total']})  "
             f"invalid={mc4['invalid']}")
    if op and op.get("total", 0) > 0:
        _add(f"  OPEN overall : MSE={op['mse'] or 0:.4f}  "
             f"RMSE={op['rmse'] or 0:.4f}  MAE={op['mae'] or 0:.4f}  "
             f"n={op['total']}  fail={op['parse_failures']}")

    # ── By Source ─────────────────────────────────────────────────────────────
    mc4_src  = mc4.get("by_source", {})
    open_src = op.get("by_source", {})
    if mc4_src or open_src:
        _add()
        _add("  By Source:")
        for k in sorted(set(list(mc4_src) + list(open_src))):
            parts = [f"    {k:<18}"]
            if k in mc4_src:
                v = mc4_src[k]
                parts.append(f"MC4  Acc={v['accuracy']:5.1f}% ({v['correct']}/{v['total']})")
            if k in open_src:
                v = open_src[k]
                parts.append(f"OPEN MAE={v['mae'] or 0:.4f} (n={v['n']})")
            _add("  ".join(parts))

    # ── By Team Size ──────────────────────────────────────────────────────────
    mc4_n  = mc4.get("by_n_robots", {})
    open_n = op.get("by_n_robots", {})
    if mc4_n or open_n:
        _add()
        _add("  By Team Size:")
        for k in sorted(set(list(mc4_n) + list(open_n))):
            parts = [f"    N={k}  "]
            if k in mc4_n:
                v = mc4_n[k]
                bar = "█" * int(v["accuracy"] / 5)
                parts.append(f"MC4  Acc={v['accuracy']:5.1f}% ({v['correct']}/{v['total']})  {bar}")
            if k in open_n:
                v = open_n[k]
                parts.append(f"OPEN MAE={v['mae'] or 0:.4f} (n={v['n']})")
            _add("  ".join(parts))

    # ── By Difficulty ─────────────────────────────────────────────────────────
    mc4_diff = mc4.get("by_difficulty", {})
    if mc4_diff:
        _add()
        _add("  By Difficulty (MC4):")
        for k in sorted(mc4_diff):
            v = mc4_diff[k]
            bar = "█" * int(v["accuracy"] / 5)
            _add(f"    {k}  Acc={v['accuracy']:5.1f}% ({v['correct']}/{v['total']})  {bar}")

    # ── By Axis ───────────────────────────────────────────────────────────────
    mc4_axis = mc4.get("by_axis", {})
    if mc4_axis:
        _add()
        _add("  By Reasoning Axis (MC4):")
        for k in sorted(mc4_axis):
            v = mc4_axis[k]
            bar = "█" * int(v["accuracy"] / 5)
            _add(f"    {k:<30}  Acc={v['accuracy']:5.1f}% ({v['correct']}/{v['total']})  {bar}")

    # ── By Config ─────────────────────────────────────────────────────────────
    mc4_cfg = mc4.get("by_config", {})
    if mc4_cfg:
        _add()
        _add("  By Config (MC4):")
        for k in sorted(mc4_cfg):
            v = mc4_cfg[k]
            bar = "█" * int(v["accuracy"] / 5)
            _add(f"    {k:<20}  Acc={v['accuracy']:5.1f}% ({v['correct']}/{v['total']})  {bar}")

    # ── By QA Type ────────────────────────────────────────────────────────────
    mc4_qt  = mc4.get("by_qa_type", {})
    open_qt = op.get("by_qa_type", {})
    if mc4_qt or open_qt:
        _add()
        _add("  By QA Type:")
        for k in sorted(set(list(mc4_qt) + list(open_qt))):
            parts = [f"    {k}"]
            if k in mc4_qt:
                v = mc4_qt[k]
                bar = "█" * int(v["accuracy"] / 5)
                parts.append(f"[MC4 ] Acc={v['accuracy']:5.1f}% ({v['correct']}/{v['total']})  {bar}")
            if k in open_qt:
                v = open_qt[k]
                parts.append(f"[OPEN] MAE={v['mae'] or 0:.4f} (n={v['n']})")
            _add("  ".join(parts))

    _add("=" * w)

    text = "\n".join(lines)
    print("\n" + text + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"epoch_{epoch:02d}_{split_name}_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write(text + "\n")
    log.info("  [%s] summary → %s", split_name, txt_path)


class CoopSREvalCallback:
    """
    Hugging Face TrainerCallback that runs accuracy evaluation on val and test
    items at the end of every ``eval_every_n_epochs`` epochs.

    Results are written to:
        <output_dir>/eval_results/epoch_{N:02d}_val.json
        <output_dir>/eval_results/epoch_{N:02d}_test.json
    and accumulated in:
        <output_dir>/eval_results/history.json
    """

    def __init__(self, val_items: list[dict], test_items: list[dict],
                 output_dir: Path, eval_every_n_epochs: int = 1):
        from transformers import TrainerCallback
        self._base_class = TrainerCallback
        self.val_items          = val_items
        self.test_items         = test_items
        self.output_dir         = Path(output_dir) / "eval_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_every         = max(1, eval_every_n_epochs)
        self.history: list[dict] = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        if epoch % self.eval_every != 0:
            return
        import torch
        device = next(model.parameters()).device if model is not None else "cpu"
        proc   = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if model is None or proc is None:
            log.warning("CoopSREvalCallback: model or processor not available at epoch %d",
                        epoch)
            return

        log.info("CoopSREvalCallback: evaluating at epoch %d …", epoch)
        record: dict = {"epoch": epoch}

        for split_name, items in [("val", self.val_items),
                                   ("test", self.test_items)]:
            if not items:
                log.info("  [%s] no items — skipping.", split_name)
                continue
            metrics = _score_items(items, model, proc, device)
            record[split_name] = metrics

            out_path = self.output_dir / f"epoch_{epoch:02d}_{split_name}.json"
            with open(out_path, "w") as fh:
                json.dump({"epoch": epoch, "split": split_name,
                           "metrics": metrics}, fh, indent=2)

            _print_eval_summary(metrics, split_name=split_name,
                                epoch=epoch, out_dir=self.output_dir)

        self.history.append(record)
        hist_path = self.output_dir / "history.json"
        with open(hist_path, "w") as fh:
            json.dump(self.history, fh, indent=2)
        log.info("  history → %s", hist_path)


