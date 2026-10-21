"""GPU compile-smoke for HeavyBridgeTrajectoryHead — no data needed.

Builds the patched V2TransfuserModel head with dummy tensors, runs one
forward_train (loss + backward) and one forward_test per mixing variant.
Exit code 0 = the head compiles and runs on this GPU; anything else fails
remote_run.sh loudly. Runtime: seconds.
"""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

VARIANTS = [("none", {}), ("ig", {"HB_NU": "5"}),
            ("ig-learned", {}), ("tempered", {"HB_ALPHA": "1.7", "HB_KAPPA": "0.5"})]


def main():
    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
    cfg = TransfuserConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bs, H, W = 2, 32, 64
    anchors = np.random.randn(20, 8, 2).astype(np.float32) * 5
    apath = "/tmp/hb_smoke_anchors.npy"; np.save(apath, anchors)

    for mixing, env in VARIANTS:
        os.environ["HB_MIXING"] = mixing
        for k, v in env.items():
            os.environ[k] = v
        from heavybridge_head import HeavyBridgeTrajectoryHead
        head = HeavyBridgeTrajectoryHead(8, cfg.tf_d_ffn, cfg.tf_d_model,
                                         apath, cfg).to(dev)
        args = dict(
            ego_query=torch.randn(bs, 1, 256, device=dev),
            agents_query=torch.randn(bs, 30, 256, device=dev),
            bev_feature=torch.randn(bs, 256, H, W, device=dev),
            bev_spatial_shape=(H, W),
            status_encoding=torch.randn(bs, 1, 256, device=dev))
        head.train()
        out = head(**args, targets={"trajectory": torch.randn(bs, 8, 3, device=dev)})
        out["trajectory_loss"].backward()
        assert out["trajectory"].shape == (bs, 8, 3)
        head.eval()
        out = head(**args)
        assert out["trajectory"].shape == (bs, 8, 3)
        assert out["mode_logits"].shape == (bs, 20)
        print(f"[smoke] {mixing}: OK (train fwd+bwd, test fwd)")
    print("[smoke] ALL VARIANTS OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
