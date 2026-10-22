# KalSea for HunyuanVideo

KalSea inference for the official [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) text-to-video model: the SEA gate
decides **when** to skip the transformer blocks, a Kalman residual forecaster decides **what** to write on skipped steps,
and predictor-corrector gap smoothing (PCGS) corrects the skipped steps at the next computed step.

## Installation
1. Clone the official [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) repository and install its dependencies and
   checkpoints (`ckpts/`) as described there.
2. Copy `kalsea_generate.py`, `kalsea_hunyuan.py`, `kalsea_forecaster.py` and `util_seacache.py` into the HunyuanVideo
   repository root.

Tested with HunyuanVideo commit 4eda9cb, torch 2.7.1, flash-attn 2.8.3 and one B200 GPU. Nothing in the `hyvideo` package is
modified: the transformer's forward is replaced at run time, as SeaCache's script does.

## Usage
Run from the HunyuanVideo repository root (single GPU):

```bash
python kalsea_generate.py \
    --video-size 480 640 \
    --video-length 65 \
    --infer-steps 50 \
    --flow-reverse \
    --seed 42 \
    --prompt "A cat walks on the grass, realistic style." \
    --save-path ./kalsea_results \
    --delta 0.2575
```

Variants:

```bash
python kalsea_generate.py ... --no_pcgs        # KalSea forecaster only
python kalsea_generate.py ... --reuse          # SeaCache: reuse the last residual
python kalsea_generate.py ... --stale_sigma    # gate on the load-time scheduler, as SeaCache's released script
python kalsea_generate.py ... --no_cache       # official sampler, every step computed
```

All flags of the official `sample_video.py` are accepted (`--model-base`, `--dit-weight`, `--flow-shift`,
`--embedded-cfg-scale`, `--use-cpu-offload`, ...). Videos are written to `--save-path` as `<index>-<prompt>-seed<seed>.mp4`
(24 fps), and every video appends a line to `stats.jsonl` with the latency, the number of full steps and the computed steps.

KalSea arguments (underscore or dash spelling):
- `--prompt_file`: one prompt per line, instead of `--prompt`
- `--delta` (alias `--seacache_thresh`, default 0.2575): gate threshold. 480x640, 65 frames, 50 steps, flow shift 7:
  `δ` 0.2575 -> 24 full steps, 0.48 -> 16
- `--no_pcgs`, `--reuse`, `--no_cache` (see above)
- `--stale_sigma`: see below
- `--kalsea_tau 0.5` (damping time constant of the derivative), `--kalsea_r_obs 1e-3` (observation noise, relative to the
  residual's power)

## Noise level of the gate (`--stale_sigma`)
The SEA filter of the gate needs the noise level `sigma` of the current step. SeaCache's released HunyuanVideo script binds
the scheduler that exists when the model is loaded, but `HunyuanVideoSampler.predict()` samples with a fresh scheduler, so
the filter reads `sigma = 1 - step/1000` instead of the sampling schedule (shift 7). By default this script reads the
`sigma` of the scheduler that `predict()` samples with. `--stale_sigma` reproduces the released behaviour. The two settings
give different schedules for the same `δ`: the operating points above are for the default (live `sigma`).

## How it works
**Gate** (SeaCache's HunyuanVideo gate). The modulated image tokens at the input of `double_blocks[0]` are SEA-filtered (a
spectral Wiener filter, exponent 3 over frames, height and width) and their relative L1 change is accumulated; the blocks are
skipped while the sum stays below `δ`. The first and last steps are always computed. On a skipped step the image tokens get
the residual of the whole block stack.

**Forecaster.** One Kalman filter per element of the residual, state = (value, damped derivative), time coordinate = step
index, calibration-free. It is updated with the true residual on computed steps and predicted on skipped steps. The update
runs in deterministic mode (its two scalar reductions), so a generation is bit-reproducible.

**PCGS.** At a computed step `b` that ends a gap `a+1 ... b-1`:
1. the forecasts of the gap are smoothed with the innovation of the Kalman update at `b` (Rauch-Tung-Striebel gains of the
   forecaster's own model);
2. the Euler steps of the gap are re-integrated from the latent used at `a+1` with `img_in`, `final_layer` and `unpatchify`
   only (no transformer block runs);
3. the output at `b` becomes `Out(img_in(x_b') + r_b) + (x_b' - x_b) / dt_b`, so the sampler continues from the corrected
   latent `x_b'`;
4. the gate's stored input is refreshed from `x_b'`.

## Notes
- Single GPU, one video per `predict()` call (`--batch-size 1 --num-videos 1`): the gate's distance is a mean over the batch.
- The `δ` values above were measured at 480x640 and 65 frames; for other settings, read the number of full steps from
  `stats.jsonl` and adjust `δ`.
