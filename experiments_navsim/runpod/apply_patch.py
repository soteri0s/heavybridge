"""Monkeypatch: swap DiffusionDrive's TrajectoryHead for HeavyBridgeTrajectoryHead.

Import this ONCE, before the agent/model is constructed (train_stage.sh does
`python -c "import apply_patch" ...` via PYTHONSTARTUP-style wrapper, or the
training wrapper imports it first). Variant selection is via env:

    HB_MIXING = none | ig | ig-learned | stable | tempered
    HB_NU, HB_ALPHA, HB_KAPPA, HB_SAMPLE_STEPS, HB_CHURN_ETA, HB_GRID_N

`HB_MIXING=upstream` leaves DiffusionDrive untouched (true upstream control).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # heavybridge_* importable

if os.environ.get("HB_MIXING", "upstream") != "upstream":
    import navsim.agents.diffusiondrive.transfuser_model_v2 as m
    from heavybridge_head import HeavyBridgeTrajectoryHead
    m.TrajectoryHead = HeavyBridgeTrajectoryHead
    print(f"[heavybridge] TrajectoryHead patched (HB_MIXING="
          f"{os.environ['HB_MIXING']}, nu={os.environ.get('HB_NU','-')}, "
          f"alpha={os.environ.get('HB_ALPHA','-')}, kappa={os.environ.get('HB_KAPPA','-')})")
else:
    print("[heavybridge] HB_MIXING=upstream — no patch applied")

# --- resume support: PL 2.x has no resume_from_checkpoint Trainer arg; navsim's
# run_training calls trainer.fit() without ckpt_path. Inject it via env.
# (added 2026-07-15 after the epoch-18 NaN crash — resume from epoch=17 ckpt)
_resume = os.environ.get("HB_RESUME_CKPT", "")
if _resume:
    import pytorch_lightning as _pl
    _orig_fit = _pl.Trainer.fit
    def _fit_with_resume(self, *args, **kwargs):
        if "ckpt_path" not in kwargs or kwargs["ckpt_path"] is None:
            kwargs["ckpt_path"] = _resume
            print(f"[heavybridge] resuming training from {_resume}", flush=True)
        return _orig_fit(self, *args, **kwargs)
    _pl.Trainer.fit = _fit_with_resume

# --- warm-start (fine-tune mode): load the public DiffusionDrive checkpoint
# into the freshly built module, strict=False (HB-only params stay at init —
# with HB_ZERO_INIT_LAM the model then starts as the upstream denoiser).
# Ignored when HB_RESUME_CKPT is set (a lightning resume takes precedence).
_ws = os.environ.get("HB_WARMSTART_CKPT", "")
if _ws and not os.environ.get("HB_RESUME_CKPT"):
    from navsim.planning.training.agent_lightning_module import AgentLightningModule as _ALM
    _orig_alm_init = _ALM.__init__
    def _alm_init_warmstart(self, *a, **k):
        _orig_alm_init(self, *a, **k)
        import torch as _t2
        sd = _t2.load(_ws, map_location="cpu")["state_dict"]
        res = self.load_state_dict(sd, strict=False)
        print(f"[heavybridge] warm-start {_ws}: "
              f"loaded={len(sd) - len(res.unexpected_keys)} "
              f"missing(HB-new)={len(res.missing_keys)} "
              f"unexpected(upstream-only)={len(res.unexpected_keys)}", flush=True)
    _ALM.__init__ = _alm_init_warmstart

# --- bf16 workaround: grid_sampler_2d has no BFloat16 CUDA kernel in torch 2.1
# (hit 2026-07-15, transfuser BEV projection). Run that op in fp32, cast back.
import torch as _torch
import torch.nn.functional as _F
_orig_grid_sample = _F.grid_sample
def _grid_sample_bf16safe(input, grid, *a, **k):
    if input.dtype == _torch.bfloat16:
        return _orig_grid_sample(input.float(), grid.float(), *a, **k).to(_torch.bfloat16)
    return _orig_grid_sample(input, grid, *a, **k)
_F.grid_sample = _grid_sample_bf16safe

# --- loud NaN-guard on the Hungarian matching cost matrix: a single bad batch
# must not kill a multi-hour run. Sanitizes + WARNS (visible in train.log);
# systematic warnings still indicate real divergence (check before trusting).
try:
    import navsim.agents.diffusiondrive.transfuser_loss as _tl
    import numpy as _np
    _orig_lsa = _tl.linear_sum_assignment
    def _safe_lsa(cost, *a, **k):
        c = _np.asarray(cost)
        if not _np.isfinite(c).all():
            print("[heavybridge][WARN] non-finite assignment cost matrix — "
                  "sanitized (nan_to_num). Investigate if this repeats.", flush=True)
            c = _np.nan_to_num(c, nan=1e6, posinf=1e6, neginf=-1e6)
        return _orig_lsa(c, *a, **k)
    _tl.linear_sum_assignment = _safe_lsa
except (ImportError, AttributeError) as e:
    print(f"[heavybridge][WARN] NaN-guard not installed: {e}", flush=True)
