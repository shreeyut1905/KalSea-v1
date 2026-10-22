# KalSea for MiniMax-H3 (text-to-video + audio)

KalSea inference for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), on top of the MiniMax-H3 modular pipeline of
diffusers: a SeaCache-style spectrally aligned gate (**when** to skip a transformer evaluation) plus Kalman residual
forecasters (**what** to write on skipped steps).

> **Status: text-to-video only; one real-checkpoint demo, no evaluation.** The code has been run on the real 33B checkpoint
> (62 GB transformer + 62 GB Qwen3-VL conditioner) once: a demo of 3 prompts with one seed each (see
> [Real-checkpoint demo](#real-checkpoint-demo)). That first run found one bug, in model loading only, which is fixed. What was
> verified is listed under [Verification](#verification). The gate threshold `δ` and the forecaster settings are not calibrated
> for H3; see [Tuning](#tuning).

Supported workflow: `t2va` (text -> video + audio). The keyframe (`fl2va`) and reference (`ref2va`) workflows are not
supported; a conditioned layout raises a clear error.

## Installation

```bash
pip install --upgrade "diffusers>=0.41" transformers accelerate safetensors protobuf sentencepiece av
```

`av` (PyAV) is used by diffusers' `encode_video` to write the mp4 with the soundtrack. The MiniMax-H3 weights are fetched from
the Hub on first use (`MiniMaxAI/MiniMax-H3`, `transformer/` + shared components, about 135 GB). The weights are under the
[MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).

`t2va` needs only `transformer`, `text_encoder`, `processor`, `tokenizer`, `vae`, `audio_vae`, `scheduler` and `audio_scheduler`; the
`transformer_ref/`, `FL2VA/` and `Ref2VA/` folders are not used. **Running offline** (`HF_HUB_OFFLINE=1`): every component is loaded from
the cached snapshot directory, so a partial download of only those components works. This was the bug fixed after the first real run:
before, diffusers still contacted the Hub for the sharded checkpoint, `transformer`, `vae` and `processor` failed to load with only a
warning, and the script crashed later. A component that fails to load now raises a clear error.

## Usage

```bash
python kalsea_generate.py \
  --prompt "A red fox trotting through a snowy pine forest, snow crunching underfoot" \
  --output_dir ./outputs \
  --num_frames 124 \
  --seacache_thresh 0.3
```

It writes `outputs/KalSea_00000-<prompt>.mp4` (24 fps, 32 kHz stereo soundtrack) and appends a line to `outputs/stats.jsonl`
(latency, number of full evaluations, which steps were computed). The plain model for comparison: add `--no_cache`.

Arguments:
- `--prompt` / `--prompt_file` (one prompt per line), `--output_dir`, `--seed` (base seed; the prompt index is added)
- `--num_frames` (default 124 = 5.2 s at 24 fps; snapped up to `17*n+5`, 5 to 15 s), `--height` / `--width` (multiples of 32,
  default the 16:9 768p canvas; a smaller canvas such as 960x544 is the biggest speed lever)
- `--num_inference_steps` (default 50 sigma grid points, i.e. 49 model evaluations without caching)
- `--offload {auto,none,two_gpu}`: `auto` (default) registers the components in diffusers' `ComponentsManager` and offloads
  as needed (one GPU, recipe of the diffusers docs); `none` keeps everything on one large GPU (the two big components need about 124 GB
  in bf16); `two_gpu` puts the text encoder on `cuda:1` and the rest on `cuda:0`. `--memory_reserve_margin` (default `12GB`)
  and `--attention_backend` (e.g. `_flash_3_hub` on Hopper) are passed through.
  On the real checkpoint (B200, 960x544, 124 frames) all three modes worked: `none` was used for the demo (peak allocated about 134.7 GiB
  with KalSea, up to 143,148 MiB in `nvidia-smi`); `auto` ran once and gave bitwise-identical output; `two_gpu` ran once (same computed steps,
  identical audio; 72,422 / 65,448 MiB in `nvidia-smi` on the two GPUs). With `auto` / `two_gpu` the first call includes moving the weights
  to the GPU; their steady-state speed was not measured.
- `--no_cache`: plain MiniMax-H3 (the original forward runs unchanged)

KalSea knobs:
- `--seacache_thresh` (`δ`, default 0.3): larger skips more
- `--sea_power_exp 3.0`: exponent of the SEA filter's power-law prior (SeaCache: 2 for images, 3 for videos)
- `--warmup_steps 3`: first evaluations that are always computed (the last evaluation is always computed too)
- `--kalsea_tau 0.5`, `--kalsea_r_obs 1e-3`, `--kalsea_q_mult 1.0`, `--kalsea_no_adapt_q`

## How it works

MiniMax-H3 runs one 50-block transformer over one packed sequence `[text | audio | video]`; the output heads read only the
audio and video rows, and the checkpoint is guidance-distilled (one forward per step, no unconditional pass). Video and audio
step down two different schedules (`shift` 12 and 3), and a schedule of `N` sigma points drives `N-1` evaluations.

- **Gate.** First block's AdaLN-modulated input of the video rows, laid out on their (frames, h, w) grid (recovered from the
  rotary coordinates and checked to be frame-major / row-major), filtered with the SEA Wiener filter for the current noise
  level (`a = t`, `b = 1 - t`, H3's `t = 1 - sigma` convention; frequency exponent `--sea_power_exp 3`, the value SeaCache uses for videos),
  relative L1 change accumulated; the step is skipped while the sum
  stays below `δ`. If the grid cannot be recovered, a 1-D filter along the token axis is used.
- **Residual.** `h_out - h_in` of the 50-block stack, for the video rows and for the audio rows only (the rows the heads read).
  Each stream has its own per-element Kalman forecaster (value + damped derivative, predicted every step, corrected on computed
  steps; no grid, no calibration). Text rows are never read by the heads, so on a skipped step they are not computed at all.
- **Skipped step.** The 50 blocks are bypassed: the video / audio input embeddings plus the forecast residual go through
  `norm_out` and the two output heads.
- **Patching.** The patch is installed on the *class* `forward` so that the accelerate / `ComponentsManager` offload hooks, which
  wrap the instance's forward, keep working. With no KalSea state attached, or with `--no_cache`, the original forward runs.
- **Memory.** The state is two fp32 copies of the residual plus the previous gate input. At the default 1344x768 / 124 frames there
  are 37,296 video rows of width 5376, i.e. about 2 x 0.8 GB (fp32) + 0.4 GB (bf16) resident and a few GB of transient
  buffers during the FFT filter and the update (estimated from the shapes, not measured). It is freed before the VAE decode.
  Measured at 960x544 / 124 frames: peak allocated 134.4 to 134.7 GiB with KalSea vs 133.1 to 133.3 GiB for plain H3.

## Tuning
- **`δ` is not calibrated for H3.** On FLUX.1-dev `δ=0.37` gives 18 of 50 evaluations and `δ=0.6` gives 13, but H3 has a different
  noise schedule (video `shift=12`), so those numbers do not transfer. Start with the default, look at the printed
  `evaluations N/49 computed at steps [...]`, then raise `δ` for more speed or lower it for more fidelity.
  Measured on the real checkpoint at 960x544 / 124 frames (one prompt): `δ` = 0.1 / 0.2 / 0.3 / 0.45 / 0.6 gave 46 / 30 / 23 / 17 / 14 of 49
  evaluations; at 0.3 and 0.6 all three demo prompts computed the same steps. Other canvases and frame counts were not tried.
- The Kalman settings (`tau=0.5`, `r_obs=1e-3`, large process noise so that the trend is re-estimated at every update) come from
  the FLUX study (`../FLUX`), where the same per-element filter was evaluated; on recorded FLUX residuals the built-in
  calibration-free noise values behave like the calibrated filter. None of this was tuned on H3.

## Real-checkpoint demo
The script was run once on the real checkpoint (2026-10-20): 3 prompts with sound, one seed each, 960x544, 124 frames
(5.2 s), 50 sigma points (49 evaluations), default KalSea settings, `--offload none`, one B200 per run. Plain H3 was
deterministic (repeated reference runs were bitwise identical). This was a functional check, not an evaluation.

## Verification
Done (without the real checkpoint), using a tiny random-weight `MiniMaxH3Transformer3DModel` built from the real diffusers
classes and driven with the pipeline's own layout builder, row-timestep plan and the two `MiniMaxH3Scheduler`s:
- with the gate never skipping, every output of the patched forward is **bitwise identical** to the original forward over a
  whole denoising trajectory, and with no state attached it is bitwise identical too;
- skipping works: outputs are finite, the warm-up steps and the last step are computed, the compute steps are increasing, and the
  filter state is released after the last step; the state object can be reused for a second generation;
- layout / grid inference, the 1-D gate fallback, `return_dict`, and the `attention_kwargs` LoRA-scale path;
- a keyframe-conditioned layout is rejected with a clear error;
- the diffusers APIs used by the loader exist with the signatures used, and `encode_video` writes video + stereo audio.

The tiny-model test was re-run after the loading fix and still passes (it does not exercise loading).

Done on the real checkpoint: the [demo](#real-checkpoint-demo) above, with `--offload none`, plus one run each with `auto` and
`two_gpu` (see `--offload`).

Not done: any quality or speed evaluation, steady-state latency with `auto` / `two_gpu`, other canvases
and frame counts, and tuning `δ` or the Kalman settings for H3.
