# KalSea for Wan2.1 (text-to-video)

KalSea inference for the official [Wan2.1](https://github.com/Wan-Video/Wan2.1) text-to-video models: the SEA gate decides
**when** to skip the transformer blocks, Kalman residual forecasters decide **what** to write on skipped steps, and
predictor-corrector gap smoothing (PCGS) corrects the skipped steps at the next computed step.

## Installation
1. Clone the official [Wan2.1](https://github.com/Wan-Video/Wan2.1) repository and install its dependencies and the
   checkpoint (e.g. `Wan2.1-T2V-1.3B`) as described there.
2. Copy `kalsea_generate.py`, `kalsea_wan.py`, `kalsea_forecaster.py` and `util_seacache.py` into the Wan2.1 repository root.

Tested with Wan2.1 commit 9737cba, torch 2.7.1 and one B200 GPU. Nothing in the `wan` package is modified: the cached
`generate` and `forward` are bound to the loaded pipeline.

## Usage
Run from the Wan2.1 repository root (single GPU):

```bash
python kalsea_generate.py \
    --task t2v-1.3B \
    --size 832*480 \
    --frame_num 65 \
    --ckpt_dir ./Wan2.1-T2V-1.3B \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --base_seed 42 \
    --sample_steps 50 \
    --delta 0.2
```

Variants:

```bash
python kalsea_generate.py ... --no_pcgs     # KalSea forecaster only
python kalsea_generate.py ... --reuse       # SeaCache: reuse the last residual
python kalsea_generate.py ... --no_cache    # official sampler, every step computed
```

Videos are written to `--output_dir` (default `./kalsea_outputs`) as `<index>-<prompt>.mp4` (16 fps), and every video
appends a line to `stats.jsonl` with the latency, the number of full steps and the computed steps.

Arguments:
- `--prompt` or `--prompt_file` (one prompt per line), `--output_dir`, `--ckpt_dir`
- `--task {t2v-1.3B,t2v-14B}` (default `t2v-1.3B`), `--size` (default `832*480`), `--frame_num` (default 65)
- `--base_seed` (default 42, the same seed for every prompt), `--sample_steps` (default 50), `--sample_shift` (default 5.0),
  `--sample_guide_scale` (default 5.0), `--sample_solver {unipc,dpm++}` (default `unipc`; PCGS needs `unipc`)
- `--offload_model`, `--t5_cpu`: lower GPU memory, as in the official `generate.py`
- `--delta` (alias `--seacache_thresh`, default 0.2): gate threshold. Wan2.1-T2V-1.3B, 832x480, 65 frames, 50 steps:
  `δ` 0.2 -> 24 full steps, 0.35 -> 17. A full step is one conditional and one unconditional transformer pass.
- `--no_pcgs`, `--reuse`, `--no_cache` (see above)
- `--kalsea_tau 0.5` (damping time constant of the derivative), `--kalsea_r_obs 1e-3` (observation noise, relative to the
  residual's power)

## How it works
**Gate** (SeaCache's Wan gate). The modulated input of block 0 is SEA-filtered (a spectral Wiener filter, exponent 3 over
frames, height and width, for the noise level of the live sampling schedule) and its relative L1 change is accumulated; the
blocks are skipped while the sum stays below `δ`. The conditional and the unconditional pass keep separate gate states. The
first and last steps are always computed.

**Forecaster.** One Kalman filter per element of the block-stack residual, state = (value, damped derivative), time
coordinate = step index, calibration-free. There is one forecaster for the conditional and one for the unconditional stream;
each is updated with the true residual on computed steps and predicted on skipped steps. The forecaster's two scalar
reductions run in deterministic mode, so a generation is bit-reproducible.

**PCGS.** At a computed step `b` that ends a gap `a+1 ... b-1`:
1. the forecasts of the gap are smoothed, per stream, with the innovation of the Kalman update at `b`
   (Rauch-Tung-Striebel gains of the forecaster's own model);
2. the UniPC multistep state saved after step `a` is restored and steps `a+1 ... b-1` are replayed exactly with the corrected
   guided outputs `u' + g (c' - u')`, where `c'` and `u'` come from the patch embedding of the replayed latent plus the
   smoothed residuals, passed through the head (no transformer block runs);
3. sampling continues from the corrected latent and the replayed solver state; the outputs at `b` are recomputed from the
   corrected latent with the residuals observed at `b`;
4. the gate's stored input of both streams is refreshed from the corrected latent.

## Notes
- Text-to-video only, single GPU (no FSDP / sequence parallelism).
- One prompt per `generate()` call: the gate's distance is a mean over the batch.
- The `δ` values above were measured with Wan2.1-T2V-1.3B; for other models, resolutions or frame counts, read the number
  of full steps from `stats.jsonl` and adjust `δ`.
