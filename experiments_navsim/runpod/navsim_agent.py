#!/usr/bin/env python3
"""
navsim_agent.py -- the two navsim AbstractAgent subclasses that let us run the
(HeavyBridge-patched) DiffusionDrive planner ONCE per scene and then score any
anchor selection with NO GPU re-inference.

    DumpAgent   : wraps a trained checkpoint, exposes its K per-anchor refined
                  trajectories + classifier logits (+ predicted perception),
                  DUMPS one .npz per scene, and returns the deployed plan
                  (the argmax-logit anchor), so a single inference pass is a
                  real evaluation.

    ReplayAgent : reads a precomputed per-scene SELECTION (token -> chosen
                  (T,3) plan; produced by select_base.py or
                  select_forced_anchor.py from the dumps) and returns it.
                  NO model is loaded, so the navsim PDM scorer can score the
                  argmax plan and every forced-anchor rank on CPU only.

All knowledge of the DiffusionDrive output layout lives in
`extract_diffusiondrive_outputs()`; everything else is plumbing.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

# utils.py / config.yaml live one level up (experiments_navsim/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)
for _p in (_EXP, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load OUR utils.py by explicit file path and register it in sys.modules:
# navsim's Hydra workers may already hold a foreign top-level `utils` module.
import importlib.util as _ilu       # noqa: E402


def _load_local(_name, _path):
    _spec = _ilu.spec_from_file_location(_name, _path)
    _mod = _ilu.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
    return _mod


utils = _load_local("utils", os.path.join(_EXP, "utils.py"))

# --- navsim imports -------------------------------------------------------
try:
    import torch                                                          # noqa: E402
    from navsim.agents.abstract_agent import AbstractAgent                # noqa: E402
    from navsim.common.dataclasses import Trajectory, AgentInput, SensorConfig  # noqa: E402
    _NAVSIM_OK = True
except Exception as _e:            # pragma: no cover - only importable with the devkit
    # Allow this module to be imported without the devkit (e.g. for unit checks);
    # the agents themselves fail loudly when instantiated.
    AbstractAgent = object
    Trajectory = AgentInput = SensorConfig = object
    torch = None
    _NAVSIM_OK = False
    _IMPORT_ERROR = _e


# =============================================================================
# DiffusionDrive output -> canonical per-scene fields.
# =============================================================================
def extract_diffusiondrive_outputs(model_output: dict, cfg: dict) -> dict:
    """Map a DiffusionDrive forward() output dict to the per-scene dump fields.

    Returns a dict with:
        modes        (K,T,3)  per-anchor refined trajectory (x,y,heading)
        logits       (K,)     anchor classification logits (poses_cls)
        pred_agents  (A,5)    [cx,cy,heading,length,width]
        pred_agent_probs (A,) existence probabilities (sigmoid of labels)
        drivable_bev (H,W)    predicted drivable-area binary raster
        bev_semantic (H,W)    full per-pixel BEV class map
    """
    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    # --- multi-mode trajectories + logits -------------------------------
    if "all_poses_reg" in model_output:
        modes = _np(model_output["all_poses_reg"])      # (1,K,T,3) or (K,T,3)
        logits = _np(model_output["all_poses_cls"])     # (1,K) or (K,)
    elif "trajectory" in model_output and "trajectory_logits" in model_output:
        modes = _np(model_output["trajectory"])
        logits = _np(model_output["trajectory_logits"])
    else:
        raise KeyError(
            "Could not find multi-mode trajectory outputs in the DiffusionDrive "
            "forward() result. Expected 'all_poses_reg'/'all_poses_cls'. "
            f"Available keys: {list(model_output.keys())}."
        )
    modes = np.asarray(modes, dtype=float)
    logits = np.asarray(logits, dtype=float)
    if modes.ndim == 4:      # drop the batch dim
        modes = modes[0]
    if logits.ndim == 2:
        logits = logits[0]
    if modes.shape[-1] == 2:  # ensure (K,T,3)
        modes = _append_heading(modes)

    # --- predicted agents ------------------------------------------------
    pred_agents = np.zeros((0, 5))
    pred_probs = np.zeros((0,))
    if "agent_states" in model_output:
        ag = _np(model_output["agent_states"])
        if ag.ndim == 3:
            ag = ag[0]
        pred_agents = _agent_states_to_boxes(ag)
        if "agent_labels" in model_output:
            lab = _np(model_output["agent_labels"])
            if lab.ndim == 2:
                lab = lab[0]
            pred_probs = _sigmoid(lab.astype(float))
        else:
            pred_probs = np.ones((pred_agents.shape[0],))

    # --- predicted drivable BEV (stored for diagnostics only) -------------
    drivable = None
    bev_arg = None
    if "bev_semantic_map" in model_output:
        bev = _np(model_output["bev_semantic_map"])
        if bev.ndim == 4:    # (1,C,H,W) logits
            bev = bev[0]
        if bev.ndim == 3:    # (C,H,W) -> argmax over classes
            bev = bev.argmax(0)
        bev_arg = np.asarray(bev).astype(np.int16)
        classes = cfg["bev"].get("drivable_classes", [1, 3])
        drivable = np.isin(bev_arg, classes).astype(float)

    return dict(modes=modes, logits=logits,
                pred_agents=pred_agents, pred_agent_probs=pred_probs,
                drivable_bev=drivable, bev_semantic=bev_arg)


def _agent_states_to_boxes(ag: np.ndarray) -> np.ndarray:
    """(A,>=5) agent_states -> (A,5) [cx,cy,heading,length,width]."""
    ag = np.asarray(ag, dtype=float)
    if ag.shape[0] == 0:
        return np.zeros((0, 5))
    if ag.shape[1] >= 5:
        return ag[:, :5].copy()
    out = np.zeros((ag.shape[0], 5))
    out[:, :ag.shape[1]] = ag
    out[:, 3] = np.where(out[:, 3] > 0, out[:, 3], 4.6)
    out[:, 4] = np.where(out[:, 4] > 0, out[:, 4], 2.0)
    return out


def _append_heading(modes: np.ndarray) -> np.ndarray:
    K, T, _ = modes.shape
    out = np.zeros((K, T, 3))
    out[..., :2] = modes
    for k in range(K):
        for t in range(T):
            if t < T - 1:
                d = modes[k, t + 1] - modes[k, t]
            else:
                d = modes[k, t] - modes[k, t - 1] if T > 1 else np.zeros(2)
            out[k, t, 2] = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-9 else 0.0
    return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _to_trajectory(poses: np.ndarray, cfg: dict):
    """Wrap a (T,3) [x,y,heading] ego-frame plan as a navsim Trajectory
    (NAVSIM: 8 poses @ 2 Hz = 4 s)."""
    poses = np.asarray(poses, dtype=np.float32)
    try:
        from navsim.common.dataclasses import TrajectorySampling
        sampling = TrajectorySampling(
            num_poses=cfg["trajectory"]["num_poses"],
            interval_length=cfg["trajectory"]["dt"],
        )
        return Trajectory(poses, sampling)
    except TypeError:
        return Trajectory(poses)


# =============================================================================
# DumpAgent -- single-inference dump + deployed (argmax) plan.
# =============================================================================
class DumpAgent(AbstractAgent):
    """Trained (HeavyBridge-patched) DiffusionDrive + per-scene dump.

    On each scene it (1) runs the model once, (2) extracts the K per-anchor
    trajectories and logits, (3) dumps them to $MC_DUMP_DIR/<token>.npz, and
    (4) returns the argmax-logit anchor's trajectory as the deployed plan.

    The sampler is stochastic (bridge churn, per-step scales); the torch seed
    is set from the scene token, so a dump is reproducible scene by scene.

    Env knobs (set by remote_run.sh):
      MC_CHECKPOINT  : path to the checkpoint (.pth)
      MC_DUMP_DIR    : where to write the per-scene .npz dumps
      MC_CONFIG      : path to experiments_navsim/config.yaml
      MC_PLAN_ANCHOR : path to the K=20 k-means anchor vocabulary (.npy)
    """

    requires_scene = False

    def __init__(self, checkpoint_path: str = None, config_path: str = None,
                 dump_dir: str = None):
        super().__init__()
        if not _NAVSIM_OK:
            raise ImportError(
                f"navsim/torch not importable; this agent needs the devkit. "
                f"Original import error: {_IMPORT_ERROR!r}"
            )
        self._ckpt = checkpoint_path or os.environ["MC_CHECKPOINT"]
        self._cfg = utils.load_config(config_path or os.environ.get("MC_CONFIG"))
        self._dump_dir = dump_dir or os.environ["MC_DUMP_DIR"]
        os.makedirs(self._dump_dir, exist_ok=True)
        self._model = None
        self._device = f"cuda:{os.environ.get('MC_CUDA_DEVICE', '0')}"

    def name(self) -> str:
        return "DumpAgent"

    def get_sensor_config(self) -> "SensorConfig":
        try:
            if getattr(self, "_dd", None) is None:
                self.initialize()
            return self._dd.get_sensor_config()
        except Exception:
            return SensorConfig.build_all_sensors()

    def initialize(self) -> None:
        """Construct DiffusionDrive (TransfuserAgent) once and load the checkpoint.
        With HB_MIXING set, apply_patch.py (imported first via hb_run.py) has
        already swapped in HeavyBridgeTrajectoryHead."""
        from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        cfg = TransfuserConfig(
            trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5)
        )
        anchor = os.environ.get("MC_PLAN_ANCHOR", "")
        if anchor:
            cfg.plan_anchor_path = anchor

        self._dd = TransfuserAgent(config=cfg, lr=6e-4, checkpoint_path=self._ckpt)
        self._dd.eval()
        self._dd = self._dd.to(self._device)

        # The eval forward returns only the selected mode; wrap the diffusion
        # decoder to stash its final multi-mode output (poses_reg, poses_cls).
        head = self._dd._transfuser_model._trajectory_head
        self._traj_head = head
        _orig_dec = head.diff_decoder.forward

        def _dec_capture(*a, **k):
            preg_list, pcls_list = _orig_dec(*a, **k)
            head._mc_last_reg = preg_list[-1].detach()   # (B, K, T, 3)
            head._mc_last_cls = pcls_list[-1].detach()   # (B, K)
            return preg_list, pcls_list

        head.diff_decoder.forward = _dec_capture
        self._model = self._dd

    def _forward_canon(self, feats: dict) -> dict:
        with torch.no_grad():
            out = dict(self._dd.forward(feats))
        out["all_poses_reg"] = self._traj_head._mc_last_reg
        out["all_poses_cls"] = self._traj_head._mc_last_cls
        return extract_diffusiondrive_outputs(out, self._cfg)

    def compute_trajectory(self, agent_input: "AgentInput") -> "Trajectory":
        """Run inference once, dump, and return the argmax-logit plan."""
        if getattr(self, "_dd", None) is None:
            self.initialize()

        fb = self._dd.get_feature_builders()[0]
        raw = fb.compute_features(agent_input)
        feats = {k: (v[None].to(self._device) if torch.is_tensor(v) else v)
                 for k, v in raw.items()}

        # run_dump.py sets _mc_token (navsim's AgentInput has no token field).
        token = getattr(self, "_mc_token", None) or _agent_input_token(agent_input)
        torch.manual_seed(_token_seed(token))
        canon = self._forward_canon(feats)             # pass 0: the deployed modes
        dump = dict(canon)
        dump["scene_token"] = token

        # Repeated samples of the same scene (fresh sampler noise, same anchors),
        # used only for the within-anchor diversity metric of select_forced_anchor.py.
        # Pass 0 above is unaffected, so the deployed plan and all forced-anchor
        # selections do not depend on this setting.
        M = int(self._cfg.get("dump", {}).get("samples", 1))
        if M > 1:
            mc = [np.asarray(canon["modes"], dtype=float)]
            for m in range(1, M):
                torch.manual_seed(_token_seed(token) + m)
                mc.append(np.asarray(self._forward_canon(feats)["modes"], dtype=float))
            mc = np.stack(mc, axis=1)                  # (K, M, T, 3)
            if not np.allclose(mc, mc[:, :1], atol=1e-6):
                dump["modes_mc"] = mc
        utils.save_dump(os.path.join(self._dump_dir, f"{token}.npz"), dump)

        k = int(np.argmax(canon["logits"]))
        return _to_trajectory(canon["modes"][k], self._cfg)


# =============================================================================
# ReplayAgent -- CPU-only scoring of a precomputed selection.
# =============================================================================
class ReplayAgent(AbstractAgent):
    """Returns, per token, a precomputed plan (JSON map produced by
    select_base.py / select_forced_anchor.py). Loads NO model.

    Env knobs (set by remote_run.sh):
      MC_SELECTION : path to a JSON { token: [[x,y,h], ...] } of chosen plans.
    """

    # requires_scene=True => navsim's run_pdm_score passes the scene (and so the
    # token) to compute_trajectory; AgentInput carries no token.
    def __init__(self, selection_path: str = None, config_path: str = None):
        super().__init__(requires_scene=True)
        if not _NAVSIM_OK:
            raise ImportError("navsim not importable; ReplayAgent needs the devkit.")
        import json
        self._cfg = utils.load_config(config_path or os.environ.get("MC_CONFIG"))
        sel_path = selection_path or os.environ["MC_SELECTION"]
        with open(sel_path) as f:
            raw = json.load(f)
        self._sel = {k: np.asarray(v, dtype=float) for k, v in raw.items()}
        if not self._sel:
            raise RuntimeError(f"empty selection map at {sel_path}; nothing to replay")

    def name(self) -> str:
        return "ReplayAgent"

    def get_sensor_config(self) -> "SensorConfig":
        return SensorConfig.build_no_sensors()

    def initialize(self) -> None:
        return None

    def compute_trajectory(self, agent_input: "AgentInput", scene=None) -> "Trajectory":
        token = None
        if scene is not None:
            token = getattr(getattr(scene, "scene_metadata", None), "initial_token", None)
        if token is None:
            token = _agent_input_token(agent_input)
        if token not in self._sel:
            raise KeyError(
                f"ReplayAgent has no selected trajectory for token {token!r} "
                f"(selection has {len(self._sel)} tokens). navsim marks this scene "
                f"invalid and continues; this happens for scenes skipped at dump "
                f"time (missing sensors)."
            )
        return _to_trajectory(self._sel[token], self._cfg)


# =============================================================================
# Small helpers.
# =============================================================================
def _token_seed(token: str) -> int:
    """Deterministic non-negative 32-bit seed from a scene token."""
    return int(hashlib.md5(str(token).encode()).hexdigest()[:8], 16)


def _agent_input_token(agent_input) -> str:
    for attr in ("token", "scene_token"):
        if hasattr(agent_input, attr):
            return str(getattr(agent_input, attr))
    meta = getattr(agent_input, "metadata", None)
    if meta is not None and hasattr(meta, "token"):
        return str(meta.token)
    raise AttributeError(
        "Could not read the scene token off the AgentInput; pass it via "
        "agent._mc_token (as run_dump.py does)."
    )
