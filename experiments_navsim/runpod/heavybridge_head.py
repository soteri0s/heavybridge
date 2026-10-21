"""HeavyBridge TrajectoryHead — drop-in replacement for
navsim.agents.diffusiondrive.transfuser_model_v2.TrajectoryHead.

Same __init__/forward contract (verified against upstream HEAD):
    __init__(num_poses, d_ffn, d_model, plan_anchor_path, config)
    forward(ego_query(bs,1,256), agents_query(bs,30,256), bev_feature(bs,256,H,W),
            bev_spatial_shape, status_encoding(bs,1,256), targets=None, global_img=None)
    -> {"trajectory": (bs,8,3)} (+ {"trajectory_loss", "trajectory_loss_dict",
        "diffusion_loss"} in training)

What changes vs. upstream (the paper's colored diffs, Algs. 1-3):
  * forward process = DDBM bridge anchor<->GT (Eqs. (3)-(4)), NOT truncated DDPM:
    x_t = a_t * y_anchor + b_t * x0 + sqrt(lambda) * c_t * eps   (normalized x/y space)
  * mixing variable lambda per VARIANT (env HB_MIXING):
      none | ig (HB_NU) | ig-learned (amortized IG head, ELBO KL) |
      stable (HB_ALPHA) | tempered (HB_ALPHA, HB_KAPPA)
  * lambda/noise-level conditioning added to the FiLM time embedding
  * sampler = DBIM bridge steps with churn (Eq. (5)/(23)), from t=T (pure anchor)
Everything else (decoder layers, anchor encoder, LossComputer, cls loss) is
REUSED from the upstream package, so the nu=inf/none variant isolates
"bridge vs truncated" and the rest isolate the mixing law.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict

from navsim.agents.diffusiondrive.transfuser_model_v2 import (  # reused blocks
    CustomTransformerDecoderLayer, CustomTransformerDecoder,
)
from navsim.agents.diffusiondrive.modules.conditional_unet1d import SinusoidalPosEmb
from navsim.agents.diffusiondrive.modules.blocks import (
    linear_relu_ln, gen_sineembed_for_position,
)
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer

# numpy reference implementation (tested); used for the offline table + samplers
from heavybridge_mixing import precompute, sample_kanter


def _env(name, default):
    return type(default)(os.environ.get(name, default))


class HeavyBridgeTrajectoryHead(nn.Module):
    def __init__(self, num_poses, d_ffn, d_model, plan_anchor_path, config):
        super().__init__()
        self._num_poses = num_poses
        self._d_model = d_model
        self.ego_fut_mode = 20
        self.mixing = _env("HB_MIXING", "none")          # none|ig|ig-learned|stable|tempered
        self.nu = _env("HB_NU", 5.0)
        self.alpha = _env("HB_ALPHA", 1.7)
        self.kappa = _env("HB_KAPPA", 0.0)               # 0 => untempered stable
        # step-dependent tempering (phase-3, R3k): kappa ramps geometrically from
        # kappa (anchor side, i=N-1) to kappa*END_MULT (trajectory side, i=0) —
        # wild jumps early / tight tempering late, where TTC & comfort are made.
        self.kappa_end_mult = _env("HB_KAPPA_END_MULT", 1.0)
        self.n_grid = _env("HB_GRID_N", 20)
        self.sample_steps = _env("HB_SAMPLE_STEPS", 2)   # matches upstream NFE budget
        self.churn_eta = _env("HB_CHURN_ETA", 0.5)       # varsigma_s = eta * c_{s|t}

        # ---- offline table (Alg. 3 Precompute; asserts run at init) ----------
        ts = np.linspace(0.0, 1.0, self.n_grid + 1)
        tab = precompute(ts, alpha=self.alpha)
        for k, v in [("tb_m", tab.m), ("tb_sig2", tab.sig2), ("tb_u", tab.u),
                     ("tb_rho", tab.rho), ("tb_clap", tab.c_lap)]:
            self.register_buffer(k, torch.tensor(v, dtype=torch.float32))
        self._np_tab = tab

        # ---- reused upstream modules -----------------------------------------
        plan_anchor = np.load(plan_anchor_path)                       # (20,8,2)
        self.plan_anchor = nn.Parameter(torch.tensor(plan_anchor, dtype=torch.float32),
                                        requires_grad=False)
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 512), nn.Linear(d_model, d_model))
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model), nn.Linear(d_model, d_model * 4),
            nn.Mish(), nn.Linear(d_model * 4, d_model))
        layer = CustomTransformerDecoderLayer(num_poses=num_poses, d_model=d_model,
                                              d_ffn=d_ffn, config=config)
        self.diff_decoder = CustomTransformerDecoder(layer, 2)
        self.loss_computer = LossComputer(config)

        # ---- HeavyBridge additions -------------------------------------------
        # noise-level embedding (log lambda, or sigma_{t,lambda} for per-step)
        self.lam_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model), nn.Linear(d_model, d_model), nn.Mish(),
            nn.Linear(d_model, d_model))
        # zero-init the last layer so the lambda-conditioning starts NEUTRAL:
        # essential for warm-starting from the upstream checkpoint (the model
        # then begins as (approximately) the upstream denoiser), harmless
        # otherwise. Disable with HB_ZERO_INIT_LAM=0.
        if int(os.environ.get("HB_ZERO_INIT_LAM", "1")):
            _last = [m for m in self.lam_mlp.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(_last.weight)
            nn.init.zeros_(_last.bias)
        if self.mixing in ("tempered-learned", "ig-bounded"):   # R23 / R2b:
            # amortized per-scene multiplier m(z) in [1/4, 4]; zero-init => m=1.
            # tempered-learned (R23): m(z) scales the per-step profile (Alg.2∘Alg.3).
            # ig-bounded (R2b, reviewer-m3J8 control, 2026-09): m(z) scales ONE
            # per-rollout IG(nu/2,nu/2) draw -> the per-rollout analogue of R23's
            # bounded amortization; starts exactly as the fixed-nu R1a/R1b model.
            self.m_head = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.Mish(), nn.Linear(d_model, 1))
            _ml = [m for m in self.m_head.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(_ml.weight); nn.init.zeros_(_ml.bias)
            self.m_reg = _env("HB_M_REG", 1.0)           # (log m)^2 prior weight
        if self.mixing == "ig-learned":                  # amortized IG head (Alg. 2)
            self.ig_head = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.Mish(), nn.Linear(d_model, 2))
        self.kl_weight = _env("HB_KL_WEIGHT", 1e-3)

    # ---- normalization: identical constants to upstream ---------------------
    @staticmethod
    def norm_xy(p):
        x = 2 * (p[..., 0:1] + 1.2) / 56.9 - 1
        y = 2 * (p[..., 1:2] + 20.0) / 46.0 - 1
        return torch.cat([x, y], dim=-1)

    @staticmethod
    def denorm_xy(p):
        x = (p[..., 0:1] + 1) / 2 * 56.9 - 1.2
        y = (p[..., 1:2] + 1) / 2 * 46.0 + 0.0 - 20.0
        return torch.cat([x, y], dim=-1)

    # ---- bridge coefficients at grid index i (horizon = index N) ------------
    def _abc(self, i):
        m, s2 = self.tb_m, self.tb_sig2
        g2 = (m[-1] ** 2 * s2[i]) / (m[i] ** 2 * s2[-1] + 1e-12)
        a = m[i] * g2 / m[-1]
        b = m[i] * (1 - g2)
        c2 = s2[i] * (1 - g2)
        return a, b, c2

    def _abc_sub(self, i, j):                    # horizon j (restriction, Lemma 1)
        m, s2 = self.tb_m, self.tb_sig2
        g2 = (m[j] ** 2 * s2[i]) / (m[i] ** 2 * s2[j] + 1e-12)
        return m[i] * g2 / m[j], m[i] * (1 - g2), s2[i] * (1 - g2)

    # ---- lambda draws --------------------------------------------------------
    def _draw_lambda(self, bs, device, ego_query=None, status=None):
        """Returns (lam (bs,), kl (scalar or 0)). Global mixing only here;
        per-step profile handled in _draw_profile."""
        if self.mixing == "none":
            return torch.ones(bs, device=device), 0.0
        if self.mixing == "ig":
            g = torch.distributions.Gamma(self.nu / 2, self.nu / 2)
            return 1.0 / g.sample((bs,)).to(device), 0.0
        if self.mixing == "ig-bounded":                  # R2b control (2026-09):
            # bounded amortized per-rollout scale: lam = m(z) * lam_prior with
            # lam_prior ~ IG(nu/2, nu/2) (fixed prior, as R1a/R1b) and
            # m(z) = exp(tanh(head) * log 4) in [1/4, 4], regularized by (log m)^2
            # exactly as R23's m(z). The learned part can at most quadruple or
            # quarter the committed scale; it cannot concentrate it.
            g = torch.distributions.Gamma(self.nu / 2, self.nu / 2)
            lam_prior = 1.0 / g.sample((bs,)).to(device)
            h = torch.cat([ego_query.squeeze(1), status.squeeze(1)], dim=-1)
            m = torch.exp(torch.tanh(self.m_head(h)) * float(np.log(4.0))).squeeze(-1)
            reg = (torch.log(m) ** 2).mean() * self.m_reg
            self._last_M = (m.detach(), lam_prior.detach())            # for HB_LOG_LAMBDA
            return lam_prior * m, reg
        if self.mixing == "ig-learned":
            h = torch.cat([ego_query.squeeze(1), status.squeeze(1)], dim=-1)
            AB = torch.nn.functional.softplus(self.ig_head(h)) + 1.0   # A,B > 1
            A, B = AB[:, 0], AB[:, 1]
            g = torch.distributions.Gamma(A, B)                        # 1/lam ~ Ga(A,B)
            lam = 1.0 / g.rsample()                                    # implicit-reparam
            self._last_AB = (A.detach(), B.detach())                   # for HB_LOG_LAMBDA
            # KL( IG(A,B) || IG(nu/2,nu/2) )  (Lemma S6, via Gamma form)
            a0 = b0 = self.nu / 2
            kl = ((A - a0) * torch.digamma(A) - torch.lgamma(A) + torch.lgamma(
                torch.tensor(a0, device=device)) + a0 * (torch.log(B) - np.log(b0))
                + A * (b0 - B) / B).mean()
            return lam, kl
        raise ValueError(f"global draw called for mixing={self.mixing}")

    def _draw_profile(self, bs, device):
        """Per-step profile (Alg. 3), numpy samplers (non-differentiable noise)."""
        beta = self.alpha / 2.0
        rng = np.random.default_rng(torch.randint(0, 2**31, (1,)).item())
        scale = self._np_tab.c_lap ** (1.0 / beta)
        lam = scale[None, :] * sample_kanter(beta, (bs, self.n_grid), rng)
        if self.mixing in ("tempered", "tempered-learned") and self.kappa > 0:
            # tilt-rejection with per-step kappa: geometric ramp kappa*mult^(1-i/(N-1))
            # (END_MULT=1 => uniform, identical to the original R3t behaviour)
            idx = np.arange(self.n_grid, dtype=float)
            kv = self.kappa * (self.kappa_end_mult ** (1.0 - idx / max(self.n_grid - 1, 1)))
            bad = np.ones_like(lam, dtype=bool)
            # cap 256 (was 64): the deeper temperings (kappa~2-3, R3g2/R3gk) have
            # per-step acceptance ~0.02-0.08; 64 left ~25% of draws insufficiently tempered.
            for _ in range(int(os.environ.get("HB_TILT_RETRIES", "256"))):
                acc = rng.uniform(size=lam.shape) < np.exp(-kv[None, :] * lam)
                bad &= ~acc
                if not bad.any():
                    break
                lam[bad] = (scale[None, :] * sample_kanter(
                    beta, (bs, self.n_grid), rng))[bad]
        return torch.tensor(lam, dtype=torch.float32, device=device)

    # ---- shared denoiser call -------------------------------------------------
    def _denoise(self, noisy_xy_metric, lvl_embed, bev, shape, agents_q, ego_q,
                 t_scalar, status, global_img):
        bs = noisy_xy_metric.shape[0]
        noisy_xy_metric = noisy_xy_metric.float()   # cached targets may arrive f64
        pe = gen_sineembed_for_position(noisy_xy_metric, hidden_dim=64).flatten(-2)
        feat = self.plan_anchor_encoder(pe).view(bs, self.ego_fut_mode, -1)
        t_embed = self.time_mlp(t_scalar.expand(bs)) .view(bs, 1, -1)
        t_embed = t_embed + lvl_embed.view(bs, 1, -1)          # lambda conditioning
        return self.diff_decoder(feat, noisy_xy_metric, bev, shape,
                                 agents_q, ego_q, t_embed, status, global_img)

    # ---- train ---------------------------------------------------------------
    def forward_train(self, ego_q, agents_q, bev, shape, status, targets, gimg):
        bs, device = ego_q.shape[0], ego_q.device
        targets = {**targets, "trajectory": targets["trajectory"].float()}  # cache may hold f64
        anchors = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)   # (bs,20,8,2)
        y = self.norm_xy(anchors)
        x0 = self.norm_xy(targets["trajectory"][..., :2]).unsqueeze(1)  # (bs,1,8,2)

        i = torch.randint(1, self.n_grid, (bs,), device=device)       # grid index
        per_step = self.mixing in ("stable", "tempered", "tempered-learned")
        if per_step:
            prof = self._draw_profile(bs, device)                     # (bs,N)
            kl = 0.0
            if self.mixing == "tempered-learned":                     # R23 hybrid
                h = torch.cat([ego_q.squeeze(1), status.squeeze(1)], dim=-1)
                m = torch.exp(torch.tanh(self.m_head(h)) * float(np.log(4.0)))  # (bs,1) in [1/4,4]
                prof = prof * m
                kl = (torch.log(m).squeeze(-1) ** 2).mean() * self.m_reg
            s2lam = torch.zeros(bs, self.n_grid + 1, device=device)
            for k in range(self.n_grid):                              # Eq. (19)
                s2lam[:, k + 1] = (self.tb_m[k + 1] / self.tb_m[k]) ** 2 \
                    * s2lam[:, k] + prof[:, k] * self.tb_u[k]
            g2 = (self.tb_m[i] ** 0 * 0 + (self.tb_m[-1] ** 2)) \
                * s2lam[torch.arange(bs), i] / (self.tb_m[i] ** 2 * s2lam[:, -1])
            a = self.tb_m[i] * g2 / self.tb_m[-1]
            b = self.tb_m[i] * (1 - g2)
            c2 = s2lam[torch.arange(bs), i] * (1 - g2)
            lam_for_scale = torch.ones(bs, device=device)             # inside c2
            lvl = self.lam_mlp(torch.sqrt(s2lam[torch.arange(bs), i]) + 1e-6)
            # (kl already set above: 0.0, or the R23 (log m)^2 regularizer)
        else:
            a, b, c2 = self._abc(i)
            lam_for_scale, kl = self._draw_lambda(bs, device, ego_q, status)
            lvl = self.lam_mlp(torch.log(lam_for_scale + 1e-12))

        eps = torch.randn_like(y)
        std = torch.sqrt(lam_for_scale.view(bs, 1, 1, 1) * c2.view(bs, 1, 1, 1))
        x_t = a.view(bs, 1, 1, 1) * y + b.view(bs, 1, 1, 1) * x0 + std * eps
        x_t = torch.clamp(x_t, -1, 1)
        noisy_metric = self.denorm_xy(x_t)

        t_scalar = (1000.0 * i.float() / self.n_grid).long()
        regs, clss = self._denoise(noisy_metric, lvl, bev, shape, agents_q,
                                   ego_q, t_scalar, status, gimg)

        loss_dict, total = {}, 0.0
        for idx, (pr, pc) in enumerate(zip(regs, clss)):
            l = self.loss_computer(pr, pc, targets, anchors)
            loss_dict[f"trajectory_loss_{idx}"] = l
            total = total + l
        out = {"trajectory": self._pick(regs[-1], clss[-1]),
               "trajectory_loss": total, "trajectory_loss_dict": loss_dict}
        if isinstance(kl, torch.Tensor):
            out["diffusion_loss"] = self.kl_weight * kl                # -> *20.0 cfg
        return out

    # ---- test: DBIM bridge sampler with churn (Eq. (5)/(23)) -----------------
    @torch.no_grad()
    def forward_test(self, ego_q, agents_q, bev, shape, status, gimg):
        bs, device = ego_q.shape[0], ego_q.device
        anchors = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        y = self.norm_xy(anchors)
        per_step = self.mixing in ("stable", "tempered", "tempered-learned")
        if per_step:
            prof = self._draw_profile(bs, device)
            if self.mixing == "tempered-learned":                     # R23 hybrid
                h = torch.cat([ego_q.squeeze(1), status.squeeze(1)], dim=-1)
                m = torch.exp(torch.tanh(self.m_head(h)) * float(np.log(4.0)))
                prof = prof * m
            s2lam = torch.zeros(bs, self.n_grid + 1, device=device)
            for k in range(self.n_grid):
                s2lam[:, k + 1] = (self.tb_m[k + 1] / self.tb_m[k]) ** 2 \
                    * s2lam[:, k] + prof[:, k] * self.tb_u[k]
        else:
            lam, _ = self._draw_lambda(bs, device, ego_q, status)
            # reviewer-m3J8 control (2026-09): record the realized per-rollout
            # scale (and the amortized head's IG parameters) at inference, one
            # line per scene: "lam[,A,B]". Set HB_LOG_LAMBDA=/path/to/file.
            _logf = os.environ.get("HB_LOG_LAMBDA", "")
            if _logf:
                _l = lam.detach().float().cpu().numpy().ravel()
                _ab = getattr(self, "_last_AB", None)
                _m = getattr(self, "_last_M", None)      # ig-bounded: "lam,m,lam_prior"
                with open(_logf, "a") as _f:
                    for _i, _v in enumerate(_l):
                        if _ab is not None:
                            _f.write(f"{_v:.6g},{float(_ab[0][_i]):.6g},{float(_ab[1][_i]):.6g}\n")
                        elif _m is not None:
                            _f.write(f"{_v:.6g},{float(_m[0][_i]):.6g},{float(_m[1][_i]):.6g}\n")
                        else:
                            _f.write(f"{_v:.6g}\n")

        idxs = np.linspace(self.n_grid, 0, self.sample_steps + 1).round().astype(int)
        x = y.clone()                                                  # t=T: anchor
        regs = clss = None
        for i_t, i_s in zip(idxs[:-1], idxs[1:]):
            if per_step:
                sl = s2lam
                g2t = (self.tb_m[-1]**2*sl[:, i_t])/(self.tb_m[i_t]**2*sl[:, -1]+1e-12)
                a_t = self.tb_m[i_t]*g2t/self.tb_m[-1]; b_t = self.tb_m[i_t]*(1-g2t)
                c2t = sl[:, i_t]*(1-g2t)
                g2s = (self.tb_m[-1]**2*sl[:, i_s])/(self.tb_m[i_s]**2*sl[:, -1]+1e-12)
                a_s = self.tb_m[i_s]*g2s/self.tb_m[-1]; b_s = self.tb_m[i_s]*(1-g2s)
                c2s = sl[:, i_s]*(1-g2s)
                gss = (self.tb_m[i_t]**2*sl[:, i_s])/(self.tb_m[i_s]**2*sl[:, i_t]+1e-12)
                c2sub = sl[:, i_s]*(1-gss)
                lvl = self.lam_mlp(torch.sqrt(sl[:, i_t]) + 1e-6)
                lam_eff = torch.ones(bs, device=device)
            else:
                a_t, b_t, c2t = self._abc(i_t); a_s, b_s, c2s = self._abc(i_s)
                _, _, c2sub = self._abc_sub(i_s, i_t)
                a_t = a_t.expand(bs); b_t = b_t.expand(bs)
                c2t = c2t.expand(bs); a_s = a_s.expand(bs); b_s = b_s.expand(bs)
                c2s = c2s.expand(bs); c2sub = c2sub.expand(bs)
                lvl = self.lam_mlp(torch.log(lam + 1e-12)); lam_eff = lam

            t_scalar = torch.tensor(int(1000 * i_t / self.n_grid), device=device)
            regs, clss = self._denoise(self.denorm_xy(torch.clamp(x, -1, 1)),
                                       lvl, bev, shape, agents_q, ego_q,
                                       t_scalar, status, gimg)
            x0_hat = self.norm_xy(regs[-1][..., :2])
            V = lam_eff.view(bs, 1, 1, 1)
            eps_hat = (x - a_t.view(bs,1,1,1)*y - b_t.view(bs,1,1,1)*x0_hat) \
                / torch.sqrt(V * c2t.view(bs,1,1,1) + 1e-12)
            vs2 = (self.churn_eta ** 2) * c2sub.view(bs,1,1,1)
            det = torch.sqrt(torch.clamp(c2s.view(bs,1,1,1) - vs2, min=0.0) * V)
            x = a_s.view(bs,1,1,1)*y + b_s.view(bs,1,1,1)*x0_hat \
                + det*eps_hat + torch.sqrt(vs2*V)*torch.randn_like(x)
        return {"trajectory": self._pick(regs[-1], clss[-1]),
                "all_modes": regs[-1], "mode_logits": clss[-1]}

    def _pick(self, reg, cls):
        idx = cls.argmax(dim=-1)[..., None, None, None].repeat(1, 1, self._num_poses, 3)
        return torch.gather(reg, 1, idx).squeeze(1)

    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,
                                      bev_spatial_shape, status_encoding,
                                      targets, global_img)
        return self.forward_test(ego_query, agents_query, bev_feature,
                                 bev_spatial_shape, status_encoding, global_img)
