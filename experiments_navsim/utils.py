"""
utils.py -- shared IO helpers (config loading, dump read/write, path handling).
Pure-stdlib + numpy + pyyaml. No model / devkit dependency.
"""
from __future__ import annotations

import os
import glob

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str = None) -> dict:
    path = path or os.path.join(HERE, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(p: str) -> str:
    """Resolve a config path relative to this folder unless already absolute."""
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(HERE, p)


# ---------------------------------------------------------------------------
# Scene dump IO. A dump is one .npz per scene with keys:
#   modes        (K,T,3)  per-anchor refined trajectory (x,y,heading)
#   logits       (K,)     classification logits
#   pred_agents  (A,5)    predicted boxes [cx,cy,heading,length,width] (or (A,2) pts)
#   pred_agent_probs (A,) predicted existence probabilities (optional)
#   drivable_bev (H,W)    predicted drivable-area binary raster
#   gt_agents    (A',5)   ground-truth boxes (for perception-sensitivity)
#   gt_drivable  (H,W)    ground-truth drivable raster
#   scene_token  str      scene identifier
#   mode_cov     (K,...)  optional per-mode posterior covariance (for M>1)
#   gt_traj      (T,3)    optional expert/GT trajectory (for calibration)
# ---------------------------------------------------------------------------
def save_dump(path: str, dump: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_kw = {}
    for k, v in dump.items():
        if v is None:
            continue
        if isinstance(v, str):
            save_kw[k] = np.array(v)
        else:
            save_kw[k] = np.asarray(v)
    np.savez(path, **save_kw)


def load_dump(path: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        d = {}
        for k in z.files:
            v = z[k]
            if v.dtype.kind in ("U", "S") and v.ndim == 0:
                d[k] = str(v)
            else:
                d[k] = v
    return d


def list_dumps(dump_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(dump_dir, "*.npz")))


def iter_dumps(dump_dir: str):
    for p in list_dumps(dump_dir):
        yield p, load_dump(p)
