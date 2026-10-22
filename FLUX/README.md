# KalSea for FLUX

KalSea inference for FLUX.1-dev / FLUX.1-schnell with diffusers: the SEA gate decides **when** to skip the transformer
blocks, a Kalman residual forecaster decides **what** to write on skipped steps, and predictor-corrector gap smoothing
(PCGS) corrects the skipped steps at the next computed step.

## Installation

```bash
pip install --upgrade \
    "diffusers[torch]>=0.35" \
    transformers \
    protobuf \
    tokenizers \
    sentencepiece \
    accelerate \
    safetensors
```

Needs diffusers >= 0.35 (the patched forward uses the newer single-stream block signature). Tested with diffusers 0.41.0,
torch 2.14.1 and transformers 5.20.2. FLUX.1-dev is gated on Hugging Face: accept the license and log in
(`huggingface-cli login`).

## Usage

```bash
python kalsea_generate.py \
  --prompt "a high-resolution photo of a panda drinking coffee in a cozy cafe" \
  --output_dir ./outputs \
  --delta 0.37
```

Variants (same gate; the gate refresh of PCGS can move a few computed steps):

```bash
python kalsea_generate.py --prompt "..." --output_dir out_kalsea --delta 0.37 --no_pcgs     # forecaster only
python kalsea_generate.py --prompt "..." --output_dir out_seacache --delta 0.37 --reuse     # SeaCache residual reuse
python kalsea_generate.py --prompt "..." --output_dir out_full --no_cache                  # every step computed
```

Multi-GPU (one process per GPU; seeds and file names do not depend on the sharding):

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python kalsea_generate.py --prompt_file my_prompts.txt --output_dir out \
      --delta 0.37 --num_shards 8 --shard_id $i --quiet &
done; wait
```

Images are saved as `KalSea_<index>-<prompt>.png`. Every image also appends a line to `stats_shard<i>.jsonl` with the
latency, the number of full transformer evaluations and the computed steps.

Arguments:
- `--prompt` / `--prompt_file`, `--output_dir`, `--width` / `--height` (default 1024), `--num_inference_steps` (default 50),
  `--guidance` (default 3.5), `--seed` (base seed; the prompt index is added), `--num_images_per_prompt`,
  `--model_name {flux-dev,flux-schnell}` / `--model_id`, `--offload`, `--dtype {bf16,fp16}`
- `--delta` (alias `--seacache_thresh`, default 0.37): gate threshold. FLUX.1-dev, 50 steps, 1024x1024: `δ` 0.2 -> 27 full
  evaluations, 0.3 -> 21, 0.37 -> 18, 0.5 -> 15, 0.6 -> 13, 0.8 -> 11, 1.0 -> 9. The gate is almost prompt independent.
- `--no_pcgs`, `--reuse`, `--no_cache` (see above), `--num_shards` / `--shard_id`, `--quiet`

Forecaster options (the defaults are the settings of the method):
- `--kalsea_tau 0.5`: damping time constant of the derivative.
- `--kalsea_r_obs 1e-3`: observation noise, relative to the residual's power.
- `--kalsea_time {warped,logsnr,step}` (default `warped`), `--kalsea_Tw 2.0`: time coordinate of the filters.
- `--kalsea_calib` (default `kalsea_calib_flux.json` next to the script, or `none`), `--kalsea_q_mult 1.0`,
  `--kalsea_no_adapt_q`.
- `--kalsea_no_dct`: one filter per element instead of per DCT band. With `--kalsea_calib none --kalsea_time step` it needs
  no calibration (the setting of the video models).

## How it works
**Gate.** The first block's modulated input is SEA-filtered (a spectral Wiener filter for the current noise level), its
relative L1 change is accumulated, and the step is skipped while the sum stays below `δ`. The first and last steps are always
computed.

**Forecaster.** On a computed step the residual `r = h_out - h_in` of the whole block stack is observed. On a skipped step
the blocks are bypassed and `h_in + forecast(r)` goes through `norm_out` / `proj_out`. The forecaster tracks `r` with Kalman
filters, state = (value, derivative), damped derivative `nu <- exp(-ds/tau) nu`, predicted at **every** step and corrected
only on computed steps, so the irregular gate schedule is handled through the step size. In the default mode the 4096 image
tokens of every channel are laid out on their 64x64 grid, 2D-DCT'd and split into 7 radial bands (band-shared covariance and
a band-specific time coordinate from the noise schedule, as in SpecKCast). `kalsea_calib_flux.json` holds the offline
calibration of that mode (band powers and process noise from 16 FLUX.1-dev trajectories, 50 steps, 1024x1024); the
per-element mode uses its values pooled over the bands unless `--kalsea_calib none` is given.

**PCGS.** When a computed step `b` ends a gap of skipped steps `a+1 ... b-1` (`pcgs.py`):
1. The forecasts of the gap are smoothed with the innovation of the Kalman update at `b`:
   `r~_i = r^_i + g_i e_b`, with the Rauch-Tung-Striebel gains `g_i` of the forecaster's own model (one gain per DCT band).
2. The Euler steps of the gap are re-integrated from the exact latent `x_{a+1}` with `x_embedder`, `norm_out` and `proj_out`
   only: `x_{i+1} = x_i + (sigma_{i+1} - sigma_i) Out(x_embedder(x_i) + r~_i)`.
3. The output at `b` becomes `Out(x_embedder(x_b') + r_b) + (x_b' - x_b) / (sigma_{b+1} - sigma_b)`, so the scheduler's Euler
   step continues from the corrected latent `x_b'`.
4. The gate's stored input is recomputed from `x_b'`.

No transformer block runs in PCGS. Its cost is one `x_embedder` and one `norm_out` / `proj_out` evaluation per skipped
step and one more at `b`, plus the gate refresh (block-0 `norm1` and the SEA filter).

## Notes
- The DCT mode is not bit-reproducible run to run: the forecaster's band reductions use `scatter_add_` (CUDA atomic adds),
  so two runs of the same configuration differ slightly. The per-element mode (`--kalsea_no_dct`) and `--reuse` are
  bit-reproducible.
