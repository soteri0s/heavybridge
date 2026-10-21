#!/usr/bin/env python3
"""
run_dump.py -- single-inference dump driver. Runs a trained checkpoint ONCE
over a NAVSIM split and writes one .npz per scene with the K per-anchor
trajectories + logits (+ predicted perception, from DumpAgent), plus the
ground-truth agent boxes and expert trajectory from the navsim Scene. The
argmax plan and every forced-anchor rank are then scored on CPU from these
dumps (select_base.py / select_forced_anchor.py + ReplayAgent).

It iterates tokens directly (not through the navsim eval harness) so it can
attach scene-side fields that AgentInput lacks. It fails loudly on a missing
checkpoint, devkit symbol or empty split, and resumes an interrupted dump.

Usage (with the HeavyBridge patch active):
  python3 hb_run.py run_dump.py --split navtest --out dumps --config ../config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)
for _p in (_EXP, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import navsim_agent as NA                              # noqa: E402

utils = NA.utils


def _require_devkit():
    if not NA._NAVSIM_OK:
        raise ImportError(
            "navsim/torch are not importable on this machine. run_dump.py needs "
            f"the devkit installed. Original import error: {NA._IMPORT_ERROR!r}"
        )


def _load_scene_filter(split):
    """Return the SceneFilter of the NAMED split, identical to the one the PDM
    scorer uses (Hydra train_test_split=<split> -> scene_filter/<split>.yaml).
    The bare default SceneFilter() does not equal navtest (navtest pins an
    explicit list of 12,146 tokens), so we load the devkit's yaml."""
    from pathlib import Path
    import navsim
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    cfg_dir = (Path(navsim.__file__).parent
               / "planning" / "script" / "config" / "common"
               / "train_test_split" / "scene_filter")
    yaml_path = cfg_dir / f"{split}.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"scene_filter yaml for split={split!r} not found at {yaml_path}. "
            f"Available: {sorted(p.stem for p in cfg_dir.glob('*.yaml'))}"
        )
    sf = instantiate(OmegaConf.load(yaml_path))
    ntok = len(getattr(sf, "tokens", None) or [])
    nlog = len(getattr(sf, "log_names", None) or [])
    print(f"[dump] scene_filter={yaml_path.name}: "
          f"{ntok or '<all>'} tokens across {nlog or '<all>'} logs")
    return sf


def _scene_gt_boxes(scene) -> np.ndarray:
    """Ground-truth agent boxes (A',5) [cx,cy,heading,length,width], ego frame,
    current frame."""
    try:
        frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    except Exception:
        frame = scene.frames[-1]
    anns = getattr(frame, "annotations", None)
    if anns is None:
        return np.zeros((0, 5))
    centers = np.asarray(getattr(anns, "boxes", getattr(anns, "gt_boxes", [])), dtype=float)
    if centers.size == 0:
        return np.zeros((0, 5))
    if centers.shape[1] >= 7:     # [x,y,z,length,width,height,yaw] -> 5 cols
        out = np.zeros((centers.shape[0], 5))
        out[:, 0] = centers[:, 0]; out[:, 1] = centers[:, 1]
        out[:, 2] = centers[:, 6]
        out[:, 3] = centers[:, 3]; out[:, 4] = centers[:, 4]
        return out
    return NA._agent_states_to_boxes(centers)


def _scene_gt_traj(scene, cfg) -> np.ndarray:
    """Expert future trajectory (T,3) in ego frame."""
    T = cfg["trajectory"]["num_poses"]
    fut = scene.get_future_trajectory(num_trajectory_frames=T)
    poses = np.asarray(getattr(fut, "poses", fut), dtype=float)
    return poses[:, :3] if poses.shape[1] >= 3 else NA._append_heading(poses[None])[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="navtest | navmini")
    ap.add_argument("--out", required=True, help="dump dir")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N scenes")
    args = ap.parse_args()

    _require_devkit()
    from pathlib import Path
    from navsim.common.dataloader import SceneLoader

    cfg = utils.load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    os.environ.setdefault("MC_DUMP_DIR", args.out)
    agent = NA.DumpAgent(dump_dir=args.out, config_path=args.config)
    agent.initialize()

    # remote_run.sh exports the per-split log and sensor dirs.
    _log_path = os.environ.get("NAVSIM_LOG_PATH", os.environ["OPENSCENE_DATA_ROOT"])
    _sensor_path = os.environ.get("NAVSIM_SENSOR_ROOT", _log_path)
    loader = SceneLoader(
        data_path=Path(_log_path),
        sensor_blobs_path=Path(_sensor_path),
        scene_filter=_load_scene_filter(args.split),
        sensor_config=agent.get_sensor_config(),
    )
    tokens = list(loader.tokens)
    if not tokens:
        raise RuntimeError(
            f"SceneLoader returned 0 tokens for split={args.split}; the dataset is "
            f"probably not downloaded or the env vars point at the wrong root."
        )
    if args.limit:
        tokens = tokens[: args.limit]

    print(f"[dump] {len(tokens)} scenes in split={args.split}")
    n_skipped = 0
    for i, token in enumerate(tokens):
        if os.path.exists(os.path.join(args.out, f"{token}.npz")):
            continue                                   # resume
        try:
            scene = loader.get_scene_from_token(token)
            agent_input = scene.get_agent_input()
        except FileNotFoundError as e:                 # partially downloaded split
            n_skipped += 1
            if n_skipped <= 20 or n_skipped % 50 == 0:
                print(f"  [skip] missing sensor/data for {token}: {e}")
            continue

        agent._mc_token = token
        try:
            _ = agent.compute_trajectory(agent_input)  # writes <token>.npz
        except FileNotFoundError as e:
            n_skipped += 1
            if n_skipped <= 20 or n_skipped % 50 == 0:
                print(f"  [skip] missing sensor/data during forward for {token}: {e}")
            continue

        dump_path = os.path.join(args.out, f"{token}.npz")
        dump = utils.load_dump(dump_path)
        try:
            dump["gt_agents"] = _scene_gt_boxes(scene)
        except Exception as e:
            print(f"  [warn] no GT agents for {token}: {e}")
        try:
            dump["gt_traj"] = _scene_gt_traj(scene, cfg)
        except Exception as e:
            print(f"  [warn] no GT traj for {token}: {e}")
        utils.save_dump(dump_path, dump)

        if (i + 1) % 50 == 0:
            print(f"  dumped {i + 1}/{len(tokens)}")

    n = len(utils.list_dumps(args.out))
    if n == 0:
        raise RuntimeError("dump produced 0 .npz files; aborting.")
    print(f"[dump] wrote {n} scene dumps to {args.out}; skipped {n_skipped} "
          f"scene(s) with missing data.")


if __name__ == "__main__":
    main()
