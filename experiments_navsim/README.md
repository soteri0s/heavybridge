# NAVSIM pipeline

See the top-level README for the overview and the list of drivers.

One run of `runpod/remote_run.sh` does, for the variant set in `runpod/config.env`:

1. installs the DiffusionDrive fork of the NAVSIM devkit in a Python 3.9 conda env;
2. downloads the maps and the evaluation split (gated: needs `HF_TOKEN`) and the public DiffusionDrive checkpoint;
3. trains the variant (`train_stage.sh`; warm start from the public checkpoint, HeavyBridge head patched in by `apply_patch.py`);
4. dumps the K = 20 refined anchors and logits of every scene in one inference pass (`hb_run.py run_dump.py`);
5. scores the deployed argmax plan with the NAVSIM PDM scorer on CPU (`select_base.py`, `ReplayAgent`);
6. writes the forced-anchor selections k = 1..20 (`select_forced_anchor.py`), which the drivers then score the same way.

`runpod/smoke_test.py` builds the patched head on dummy tensors and runs one training and one test forward per mixing law; `remote_run.sh` runs it before the dump when `HB_SMOKE=1`.

`config.yaml` holds the trajectory conventions and the number of sampler passes per scene in the dump (pass 0 is the deployed plan; the others feed only the diversity metric).
