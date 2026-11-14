# UniVTAC 50-episode ACT/DP campaign

This campaign trains `insert_HDMI`, `insert_hole`, and `pull_out_key`, with RGB and RGB+tactile for each of ACT and DP (12 jobs). No simulation success rates are claimed by these training runs.

## Data and split

Each task uses original episodes 0–49 from `data/isaac51/<task>/hdf5/`. The split is NumPy RandomState(1).permutation(50): 40 training episodes, 10 validation episodes. This exactly matches upstream ACT's seed before load_data. Both policies and both modalities share the split. Data are in `data/official50/<task>/`, independently of the earlier 30-episode experiment. Episodes 0–29 reference the earlier converted files after joint/action equality checks; episodes 30–49 use OpenCV imdecode and INTER_LINEAR resize to 256×256. No channel swap is added. State is qpos(t)[:8], action is qpos(t+1)[:8]; observations exclude the final frame.

Following the user's request for official defaults, normalization statistics use all 50 episodes, as upstream does, including validation episodes. This differs from the earlier local 30-episode experiment which used training-only statistics. ACT uses mean/std and DP uses min/max. Zero DP ranges are clamped to 1e-6 to avoid division by zero.

## ACT

`run_act.py` invokes the existing `policy/ACT/imitate_episodes.py` with runpy, without rewriting its training loop. `train_config_all.yml` / `train_config_vision_all.yml` retain batch 64, lr 1e-5, weight decay 1e-4, chunk 50, KL weight 10, 4000 configured steps, save frequency 1000. Both RGB cameras are used. Tactile input is left/right rgb_marker, and uses `data/checkpoints/encoder.pth` (not task policy weights). The YAML adds the unused `num_epochs: 6000` key to bypass the upstream nested argparse mismatch. The data registry and output paths are isolated under `outputs/official50`.

The upstream loop's `step_count > num_steps` condition executes 4001 updates. This behavior is retained and explicitly recorded. Validation occurs after each training pass. Official checkpoint and plot saving behavior is retained. The existing user change to `act_policy.py` is untouched.

## DP

The user explicitly selected the upstream-supported `resnet18` initialization from scratch because the default ViTAL CLIP checkpoints are absent. This is an adapted upstream implementation, not an unchanged ready-to-run official UniVTAC DP baseline.

`dp_adapter.py` directly imports the upstream `get_resnet`, `replace_bn_with_gn`, and `ConditionalUnet1D` from `policy/ViTAL/diffusion/network.py`. Each selected RGB/tactile view has a separate encoder, global pooling to 512 features, and concatenation with the 8-dimensional state. RGB uses ImageNet normalization; tactile rgb_marker is scaled to [0,1]. Both tactile views use the generic per-view path. The legacy loop's ABLATE_GEL=True only disables its separate single-depth/strain branch; it does NOT remove the tactile views in camera_names.

`build_dp_loop.py` extracts the upstream train function into the reviewable `dp_train_loop.py`. Default AdamW lr 1e-4, weight decay 1e-6, cosine schedule, 500 warmup steps, batch 8 (validation 2), 3500 epochs, horizon 20, observation horizon 1, 100 DDPM steps, squaredcos_cap_v2, epsilon/MSE objective and no EMA are retained. Dataset length remains one item per episode with random time sampling, matching upstream: 5 batches/epoch, 17,500 optimizer updates total. Validation is every 10 epochs (including epoch 0), with random time/noise sampling as upstream.

Required changes:

- 6 RGB + 1 depth/strain image, 4-DoF becomes 2 RGB + optional 2 tactile RGB images, 8-DoF.
- State/action normalization, final-action repeat padding and frame sampling are retained; tactile observation decoding is adapted.
- Fix upstream failure to restore train() after validation; reset per-epoch loss accumulation rather than accumulating across epochs.
- Remove real-robot four-dimensional visualization code and import-time hardcoded path/copy operations.
- Add finite-loss checks, JSON progress, best-validation weight saving and state-dict checkpoints.
- Keep the 500-epoch periodic saving interval, rotating `policy_periodic.ckpt` to limit disk use; also retain best and final weights. Optimizer-state resumption is not implemented, matching upstream's weight-only checkpoint behavior.
- Seed model and sampling with 42; dataset split is fixed to ACT's seed-1 split.

The project venv now contains diffusers 0.32.2. Dependency and source hashes are recorded in the output manifest.

## Resource measurements and queue

Short real-data training+validation smoke runs measured peak CUDA reserved memory: ACT RGB 7.84 GiB, ACT RGB+tactile 9.82 GiB, DP RGB 2.18 GiB, DP RGB+tactile 3.03 GiB. These exclude some CUDA process overhead and are not full-run upper bounds. GPU: RTX PRO 6000 Blackwell, 97,887 MiB total. The durable queue was increased from four to twelve concurrent jobs at the user’s request; it starts at most twelve jobs, records failures and does not silently retry or overwrite completed runs. It stops launching new jobs below 8 GiB free disk.

```bash
cd /root/autodl-tmp/gujie/UniVTAC
.venv/bin/python scripts/official50/status.py
```

Outputs: `outputs/official50/{act,dp}/<task>/<rgb|rgb_tactile>/`. Queue state: `outputs/official50/status.json`; scheduler PID: `campaign.pid`; each job writes `train.log`. Do not start a second scheduler against the same directory.

Preflight results are in `outputs/official50/preflight.json`: all splits verified, DP RGB/tactile encoders updated, saved weights strictly reloaded, and 100-step DDPM inference returned finite [1,20,8] actions. This is not a physics rollout. Simulation evaluation has not been launched. Use the project's unified `scripts/eval_policy.py` when integrating the checkpoints; the legacy ACT eval.sh has a stale `script/` path and the bundled DP robot runner is not a UniVTAC simulation adapter.
