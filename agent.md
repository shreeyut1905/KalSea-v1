# agent.md: everything an agent needs to continue the KalSea project

Written for an AI agent that has **none** of the original conversation. Read it fully before acting.
Owner: Shreeyut Maheshwari (GitHub `shreeyut1905`). Repo: https://github.com/shreeyut1905/KalSea-v1 (branch `main`).
State described here: 2026-10-20. Facts below are things that were measured or checked; anything not verified says so.

---------------------------------------------------------------------------------------------------------------

## 0. The task this file was written for

The owner asked for a **technical blog post about KalSea**, with these requirements (their words, paraphrased only for grammar):

1. Write a good technical blog for the method.
2. **Push it to Hugging Face Spaces** so the owner can see it.
3. Keep **a markdown file of the exact same blog in this repo**.
4. The blog should be good; **animations, GIFs and videos** would be better.
5. For the blog, **use models like MiniMax-H3 and FLUX.2 [dev]**.

Nothing of the blog exists yet. Section 6 is the spec and the plan. Sections 1 to 5 are the facts you need to write it
honestly. **Do not skip section 9 (credentials).**

---------------------------------------------------------------------------------------------------------------

## 1. What KalSea is (one paragraph)

KalSea is a training-free cache for diffusion transformers. Per denoising step it decides **whether to run the full transformer**
(a SeaCache-style *spectrally aligned gate*) and, when it skips, it **forecasts the transformer's residual** with per-element
**Kalman filters** instead of copying a cached value. It sits inside the framework of the owner's earlier method **ReSPect**
(SeaCache gate + residual-space Chebyshev/Taylor forecasting, https://github.com/shreeyut1905/ReSpect) and replaces ReSPect's
forecaster (taken from **Spectrum**, Han et al.) with a Kalman forecaster adapted from **SpecKCast** (Rajabi et al.,
arXiv:2610.26837, which forecasts the final *velocity* in the DCT domain; KalSea forecasts the *block-stack residual*).
Title: "KalSea: Kalman Residual Forecasting with Spectrally Aligned Skipping for Training-Free Acceleration of Diffusion
Transformers". Backbones in the repo: FLUX.1-dev (evaluated) and MiniMax-H3 text-to-video (inference code only, never run on the real checkpoint).

---------------------------------------------------------------------------------------------------------------

## 2. Method details (enough to write the technical sections)

### 2.1 Per-step control flow (identical idea in FLUX and H3)
```
embed inputs -> gate input (first block modulated input) -> SEA filter -> accumulate rel-L1 -> compare with delta
  compute : run ALL blocks;  r = h_out - h_in (residual of the whole block stack);  Kalman update with r
  skip    : bypass ALL blocks; h_out = h_in + Kalman_forecast(r);  norm_out / output heads still run
```
The first steps (warm-up) and the **last step are always computed** (the last observation is never reused).

### 2.2 Gate (from SeaCache; unchanged from ReSPect)
- Gate signal = relative L1 change of the **first block's modulated input** (`norm1` + AdaLN shift/scale), after a **SEA filter**.
- SEA filter (`util_seacache.py::apply_sea_from_ab`): a separable Wiener gain per spatial axis,
  `H1(f) = a*S(f) / (a^2*S(f) + b^2)` with `S(f) = 1/|f|^2` (`power_exp=2`), multiplied over axes, normalised to mean 1.
  `a = 1 - sigma`, `b = sigma` (flow-matching noise level), so the filter changes with the noise level.
  FLUX filters over `(H, W)` of the token grid (dims `(-2,-3)` on `[B,H,W,C]`); H3 filters over `(frames, h, w)` (dims `(-4,-3,-2)`
  on `[B,F,H,W,D]`) with H3's `t = 1 - sigma` convention (`a = t`, `b = 1 - t`).
- Distances are accumulated; the step is **skipped while the sum stays below delta**, otherwise it is computed and the sum resets.
- On FLUX the gate is almost prompt independent (same compute steps for every prompt), so NFE is the same for every forecaster.
  FLUX.1-dev, 50 steps: delta -> evaluations: 0.2 -> 27, 0.3 -> 20.8, 0.37 -> 18.1, 0.5 -> 15, 0.6 -> 13, 0.8 -> 11, 1.0 -> 9.
  Compute steps at 0.37: `[0,1,2,3,4,5,7,9,11,14,18,22,27,32,37,42,46,49]`; at 0.6: `[0,1,2,4,6,9,13,18,25,33,41,47,49]`.

### 2.3 Residual and forecast target
- FLUX: residual = (image-token hidden state after all double+single blocks) minus (image tokens after `x_embedder`), shape
  `[1, 4096, 3072]`; added back to the *current* step's input on a skip, then `norm_out` + `proj_out`.
- H3: one packed sequence `[text | audio | video]`, 50 blocks; the residual is taken **only on the video rows and the audio rows**
  (the only rows the output heads read), one forecaster per stream.

### 2.4 Kalman forecaster (`kalsea_forecaster.py`)
Per element (or per DCT coefficient) state `(m, nu)` = (value, temporal derivative), time coordinate `s_j` in `[-1, 1]`, step `ds_j = s_j - s_{j-1}`:
```
predict (EVERY step, also skipped ones):  m <- m + ds*nu ;  nu <- exp(-ds/tau)*nu            (damped trend, tau = 0.5)
                                          P <- A P A^T + Q ,  A = [[1, ds],[0, rho]], Q = q*[[ds^3/3, ds^2/2],[ds^2/2, ds]]
update  (computed steps only):           innovation e = z - m ;  K = P[:,0]/(P00 + R) ;  m += K0*e ; nu += K1*e ;  P = (I - K H)P ,  H=[1,0], R=r_obs=1e-3
online process-noise adaptation:         qscale <- clip(qscale * clip(err/(P00+R), 0.1, 10)^0.2, 0.1, 10)
```
- Everything is in "relative units" (values divided by sqrt of the signal power; `p_scale`).
- Because the step size `ds` enters `A` and `Q`, the irregular compute pattern of the gate is handled natively (no fixed-step assumption).
- Modes: `use_dct=False`: one filter per element, one shared covariance, works on any token set (used for H3).
  `use_dct=True` (FLUX default): tokens laid on the 64x64 grid per channel, orthonormal 2D DCT, 7 radial bands (edges
  `[0,1/64,1/32,1/16,1/8,1/4,1/2,1]`, radius `sqrt((a/H)^2+(b/W)^2)/sqrt(2)`), band-shared covariance, band-specific time warp
  `s_l(j) = rescale_[-1,1](sigmoid((lambda_j + log p_l)/Tw))`, `lambda = 2 log((1-sigma)/sigma)`, `Tw = 2`.
- Calibration (`FLUX/kalsea_calib_flux.json`, from 16 COCO prompts; the scripts that made it are NOT in this repo, see section 7): `p_scale` (band power of the residual),
  `p_warp` (band power of the clean-sample estimate x0_hat = x - sigma*v in latent space, mapped onto token-grid bands: token DCT index (a,b) <->
  latent DCT index (a,b)), `q` (median over steps of mean[(d nu)^2]/ds in warped time), `p22_init = 2q`.
- Calibration-free defaults: `q = 7.4e3`, `p22 = 1.5e4` (the FLUX calibration pooled over bands), band power from the first observation.
  They are large on purpose: reasoned from the filter equations (not ablated), a large process noise makes the trend be re-estimated at every update.
  Empirical hint only: the `nocalib` FLUX variant (q=1, p22=2, plus online band power and an unwarped time axis) was ~2.5 dB worse; the cause was not isolated.
- tau: SpecKCast uses 0.1 for velocities; **0.5** is better for residuals (broad plateau for tau in 0.2 to 3, tuned on 64 COCO prompts).

### 2.5 What came from where (credit correctly in the blog)
SeaCache (gate, https://github.com/jiwoogit/SeaCache), ReSPect (framework; owner's repo), Spectrum (Chebyshev baseline,
https://github.com/hanjq17/Spectrum), SpecKCast (Kalman forecaster + band warp idea), MiniMax-H3 and FLUX (backbones), Diffusers.

---------------------------------------------------------------------------------------------------------------

## 3. Repo map and how to run

```
KalSea-v1/
  README.md                     overview (ReSPect-style)
  agent.md                      this file
  assets/                       overview.png (schematic; panel b = toy signal through the real forecaster), fidelity_vs_nfe.png, qualitative_delta0.6.png
  FLUX/    kalsea_generate.py   FLUX.1 inference (default --forecaster kalsea; --forecaster spectrum = ReSPect baseline; --no_cache = 50-step reference)
           kalsea_forecaster.py the forecaster (identical file in MiniMax-H3/)
           spectrum_forecaster.py, util_seacache.py, kalsea_calib_flux.json, README.md
  MiniMax-H3/ kalsea_generate.py  H3 text-to-video(+audio) inference; patches MiniMaxH3Transformer3DModel.forward at CLASS level
           kalsea_forecaster.py, util_seacache.py, README.md
```
Run (venv and env vars: section 7):
```bash
# FLUX (verified)
python FLUX/kalsea_generate.py --prompt "..." --output_dir out --seacache_thresh 0.37
python FLUX/kalsea_generate.py --prompt_file p.txt --output_dir ref --no_cache          # reference
# multi-GPU: one process per GPU:  CUDA_VISIBLE_DEVICES=$i ... --num_shards 8 --shard_id $i --quiet
# H3 (code verified only on a tiny random-weight model)
python MiniMax-H3/kalsea_generate.py --prompt "..." --output_dir out --num_frames 124 --seacache_thresh 0.3   # --no_cache = plain H3
```
Per image/video a stats line is written (latency, NFE, compute steps): `stats_shard<i>.jsonl` (FLUX), `stats.jsonl` (H3).

---------------------------------------------------------------------------------------------------------------

## 4. Results you may quote (FLUX.1-dev, DrawBench200, n=200, 50 steps, 1024x1024, 8x B200)

Fidelity = PSNR / SSIM (skimage defaults) / LPIPS-Alex against the **50-step image of the same seed** (reference latency 5.27 s).
Latency measured with 8 concurrent single-GPU processes. "extra mem" = peak allocated above the no-cache run.
Only the forecaster differs between rows; the gate, hence NFE, is identical.

| delta | NFE | forecaster | latency s | PSNR | SSIM | LPIPS | extra mem GiB |
|---|---|---|---|---|---|---|---|
| 0.37 | 18.1 | ReSPect (Spectrum), repo default w=0.5 | 2.161 | 26.18 | 0.8968 | 0.0908 | 0.47 |
| 0.37 | 18.1 | ReSPect, Chebyshev weight w=0 (control) | 2.161 | 26.66 | 0.9049 | 0.0812 | 0.47 |
| 0.37 | 18.1 | KalSea, tau=0.1 (SpecKCast value) | 2.113 | 26.62 | 0.9024 | 0.0856 | 0.14 |
| 0.37 | 18.1 | **KalSea tau=0.5 (DCT, warped)** | 2.114 | **26.91** | 0.9066 | 0.0801 | 0.14 |
| 0.37 | 18.1 | KalSea no-DCT tau=0.5 | 2.104 | 26.87 | 0.9067 | 0.0799 | 0.14 |
| 0.6 | 13.0 | ReSPect default | 1.619 | 21.55 | 0.8196 | 0.1826 | 0.35 |
| 0.6 | 13.0 | ReSPect w=0 | 1.619 | 21.80 | 0.8268 | 0.1708 | 0.35 |
| 0.6 | 13.0 | KalSea tau=0.1 | 1.585 | 21.74 | 0.8171 | 0.1860 | 0.14 |
| 0.6 | 13.0 | **KalSea tau=0.5** | 1.585 | **21.91** | 0.8256 | 0.1716 | 0.14 |
| 0.6 | 13.0 | KalSea no-DCT tau=0.5 | 1.575 | 21.94 | 0.8292 | 0.1693 | 0.14 |

Paired differences over the 200 prompts (95% bootstrap CI):
- KalSea(tau=0.5) minus ReSPect default: delta 0.37 -> PSNR **+0.72 dB [+0.45, +1.01]**, LPIPS -0.0108 [-0.0142, -0.0074];
  delta 0.6 -> PSNR **+0.37 dB [+0.23, +0.51]**, LPIPS -0.0110 [-0.0152, -0.0071].
- KalSea(tau=0.5) minus ReSPect **w=0**: delta 0.37 -> +0.25 dB [+0.13, +0.36] (significant); delta 0.6 -> +0.11 dB [-0.02, +0.25] (**not** significant).
  No-DCT minus w=0: +0.20 dB [+0.14, +0.27] at 0.37 and +0.14 dB [+0.04, +0.23] at 0.6.
- Sweep (PSNR, ReSPect default / ReSPect w=0 / KalSea): delta 0.2 (27 NFE): 30.75 / 32.27 / **32.89**; 0.3 (20.8): 27.79 / 28.51 / **28.79**;
  0.5 (15): 23.38 / 23.58 / **23.79**; 0.8 (11): 19.77 / 19.61 / **20.30**; 1.0 (9): 19.05 / 18.70 / **19.36**.
  KalSea has higher PSNR and lower LPIPS than the default at all 7 thresholds; SSIM is higher at 6 of 7 (lower at 1.0: 0.7376 vs 0.7433).
- Reference-free: CLIP-L 27.6 to 27.8 and PickScore 22.85 to 22.89 for every method, no significant difference.

### What these results do and do NOT show (the blog must say this)
- **Most of the gain over the repo default is that ReSPect's Chebyshev term hurts here** (PSNR falls monotonically as the Chebyshev weight goes 0 -> 1).
  Against the w=0 control KalSea's edge is small.
- **The spectral parts do not help in residual space:** the no-DCT variant matches or beats the DCT/band-warp variant; the time warp is a wash.
  The only truly spectral component is the SEA gate. Do not market "spectral Kalman" as the source of the gain.
- Open-loop check on recorded FLUX residuals: a plain **damped two-point linear extrapolation** had *lower* raw residual error than the Kalman filter
  (0.191 vs 0.202 at delta 0.37), and ReSPect's default had the lowest raw error of all while being worse in image space. So raw residual error does **not**
  rank methods like image fidelity, and **a closed-loop comparison against damped linear extrapolation was never run**. Do not claim the Kalman machinery is necessary.
- tau and ablations were chosen on 64 COCO prompts disjoint from DrawBench and from the 16 calibration prompts; DrawBench is unseen.
- One model, one seed per prompt, fidelity vs the 50-step sample. **Runs are not bit-reproducible** (atomic float adds, GPU kernels; two runs of one config differed by
  39.8 dB PSNR on one image). Do not interpret differences below ~0.1 dB.
- The calibration-free FLUX setting (`--kalsea_no_dct --kalsea_calib none --kalsea_time step`) matched the calibrated no-DCT filter in open-loop forecast error
  (0.202 vs 0.204 at 0.37; 0.275 vs 0.279 at 0.6); its image quality was **not** evaluated.

---------------------------------------------------------------------------------------------------------------

## 5. Verification status

| item | status |
|---|---|
| FLUX KalSea / ReSPect baseline | run and evaluated (section 4); after the final restructure one image per forecaster re-run; baseline output bit-identical to before, KalSea at the 39.8 dB noise floor |
| `kalsea_forecaster.py` DCT path vs the old implementation that produced the results | max relative difference 0.0 on a synthetic trajectory (no-DCT path 3e-5) |
| H3 patched forward | tested **only on a tiny random-weight `MiniMaxH3Transformer3DModel`** built from real diffusers classes, driven with the real layout builder, row-timestep plan and the two `MiniMaxH3Scheduler`s. With the gate never skipping, every output is **bitwise equal** to the original forward over a whole trajectory; skipping, state release/reuse, grid inference, 1-D gate fallback, `return_dict`, LoRA-scale kwarg and rejection of conditioned layouts all pass. Test file: `/home/ubuntu/KalSea-dev/h3_tiny_model_test.py` (outside the repo). |
| H3 on the real checkpoint | **never run.** Speed, memory, quality, the right delta, `two_gpu` / `none` loading paths: unknown |
| H3 delta / tau defaults | delta=0.3 is a guess; Kalman settings were transferred from FLUX |
| FLUX.2 [dev] | **not supported by the code** (see 6.5) |
| Wan2.1 / HunyuanVideo / LTX | not in this repo (the owner removed them from scope) |

---------------------------------------------------------------------------------------------------------------

## 6. The blog: spec and plan

### 6.1 Deliverables
1. `blog/kalsea.md` in this repo: the blog, plus `blog/media/` (GIFs, PNGs, short MP4s). Relative media links.
2. A **Hugging Face Space** (static) whose page is generated **from that same markdown file** (same text, same relative media paths) so the two never diverge;
   also upload the `.md` itself. Do not write a second, different version.
3. Tell the owner the Space URL and what is on it.

### 6.2 Decisions to make or confirm with the owner first
- **Which HF account/namespace** to publish under (the machine's own token file belongs to HF user `Shreeyut`; the key the owner pasted in chat may be another account; check with `whoami`).
- **Private vs public:** the method/paper may be unpublished. **Default to a private Space** and tell the owner how to switch it public.
- Whether to credit the AI assistant in the blog (the repo README already does).

### 6.3 Suggested structure
1. TL;DR + hero GIF (reference vs plain cache vs KalSea at the same NFE) and the headline number (with the honesty caveat in the next line).
2. Why cache diffusion transformers; reuse vs forecast.
3. Background: SeaCache gate, ReSPect, Spectrum (one short, correct paragraph each).
4. KalSea: (a) residual-space framework, (b) the Kalman forecaster (state, damping, why the step size entering A/Q matters for an irregular schedule), (c) optional DCT bands and band-specific time warp, (d) calibration-free defaults.
5. Results on FLUX.1-dev: table, the trade-off curve, best/median/worst qualitative, latency and memory.
6. **What actually matters** (ablations, the w=0 control, no-DCT, tau plateau, nondeterminism). This section is what makes the post credible.
7. MiniMax-H3 (design: packed sequence, two streams, per-stream forecasters; demo; clearly labelled as a first real run if that is what it is).
8. FLUX.2 [dev] (only if actually ported and run; otherwise one honest sentence).
9. Limitations. 10. Reproduce. 11. Credits.

### 6.4 Animation / GIF / video plan (what to build and how)
Everything marked **(no GPU)** can be made from data already on disk. Keep each GIF under about 6 MB, MP4 under about 15 MB (GitHub limit 100 MB/file); GitHub does not
play `<video>` tags from repo files, so use **GIFs in the markdown and link the MP4** (the Space can additionally embed `<video>`, but then keep the markdown the source of truth).

1. **Blink/flip comparison GIFs (no GPU):** reference vs ReSPect vs KalSea at delta 0.6 (and 0.37) for ~6 prompts. All 200 prompts x 26 configs are on disk
   (section 7). Include the best, median and **worst** cases (`assets/qualitative_delta0.6.png` already picks them by PSNR gain: prompts 76, 57, 187).
2. **SEA filter animation (no GPU):** plot the Wiener gain `H(f)` of `apply_sea_from_ab` for sigma going 1 -> 0 (it changes with the noise level).
3. **Gate sawtooth animation:** accumulated rel-L1 vs step with the delta line and compute/skip markers. Needs the per-step distances: monkeypatch `util_seacache.rel_l1`
   to log its return values during one FLUX run (cheap), or reconstruct from the compute steps in section 2.2.
4. **Kalman predict/update toy animation (no GPU):** run `KalSeaForecaster(calib=None, time_mode="step", use_dct=False)` on a synthetic 1-D trajectory (as `assets/overview.png` panel b does) and animate predict vs update. Label it a toy.
5. **Denoising timeline GIF (GPU, FLUX.1):** decode the predicted clean image `x0_hat = x_t - sigma*v` (or the latents) at every step for reference vs KalSea side by side with a green/red computed/skipped strip. Use `callback_on_step_end`.
6. **Trade-off plot:** `assets/fidelity_vs_nfe.png` exists.
7. **H3 videos (GPU, never run before):** reference (`--no_cache`) vs KalSea, side by side, **with audio** (`encode_video` already muxes it; use `ffmpeg` to build the side-by-side and a short GIF preview).
   Use a smaller canvas (960x544, 124 frames = 5.2 s, the minimum duration) for speed; estimate about 1.5 to 2 s per evaluation per B200 and 49 evaluations for the reference
   (estimates from FLOP counts, not measured). Measure and report real latency and evaluations. This will be the first real test of the H3 code: **expect to debug**, and to tune delta
   (try 0.1 to 0.6 and read the printed evaluation count) because the default is a guess. This is a demo, not an evaluation: no metric claims from one or two videos.
8. **Real-data illustrations are fine, cherry-picked ones are not.** A forecast-vs-truth plot of one element looked bad for KalSea on the one element tried; if you show such a plot choose the element by a stated rule (e.g. median error) or use a toy and label it.

### 6.5 Models named by the owner
- **MiniMax-H3** (`MiniMaxAI/MiniMax-H3`, not gated, license "MiniMax-H3 Community License"): 33B single-stream transformer, two checkpoint partitions; the diffusers repo holds both.
  t2va needs `transformer/` (61.7 GB) + `text_encoder/` (Qwen3-VL-32B, 62.1 GB) + VAEs (about 10 GB) = about 135 GB download. 49 evaluations at the default 50 sigma points
  (a schedule of N points drives N-1 evaluations). Accessible with the machine token. Docs: https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3.
- **FLUX.2 [dev]** (`black-forest-labs/FLUX.2-dev`, gated "auto", 165 GB, license "other"; accessible with the machine token): diffusers 0.41 has `Flux2Pipeline` and `Flux2Transformer2DModel`
  (`diffusers/models/transformers/transformer_flux2.py`). **KalSea does not support it today**: `FLUX/kalsea_generate.py` patches `FluxTransformer2DModel.forward` (FLUX.1).
  To support it, follow exactly how the H3 port was done: read the FLUX.2 forward, patch the **class** forward, gate input = first block's modulated input of the image tokens,
  residual = image-token stream before `norm_out`, then (1) a tiny random-weight model test that the patched forward is bitwise equal to the original when nothing is skipped, (2) a real run,
  (3) report honestly. **If it is not ported, use FLUX.1-dev in the blog and say so; never invent FLUX.2 numbers or images.** Disk: 165 + 135 GB for both models vs about 394 GB free, so clear caches you do not need.

### 6.6 Publishing to a Hugging Face static Space
```python
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])            # token ONLY from the environment, never printed, never written to a file
user = api.whoami()["name"]                          # also tells you whether the token is valid; check it can write
repo_id = f"{user}/KalSea-blog"
api.create_repo(repo_id, repo_type="space", space_sdk="static", private=True, exist_ok=True)
api.upload_folder(folder_path="space_build", repo_id=repo_id, repo_type="space", commit_message="KalSea blog")
```
`space_build/` = `index.html` (generated from `blog/kalsea.md`, e.g. with the `markdown` package + a small CSS file), `blog/kalsea.md`, `media/`, and a `README.md` with the Space front matter:
```
---
title: KalSea
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---
```
Binary media is stored through LFS/Xet automatically by `upload_folder`. Do **not** run `huggingface-cli login` (it overwrites `~/.cache/huggingface/token`, see section 7); pass the token per process.
After uploading, open the Space URL (or `api.space_info`) to check it built, then tell the owner.

### 6.7 Honesty rules for the text
Report the w=0 control and the no-DCT result prominently; state that most of the gain over the default is the Chebyshev term hurting; label schematics and toy signals; show worst cases;
say H3 / FLUX.2 results are single demos, not evaluations; state nondeterminism; credit SeaCache, ReSPect, Spectrum, SpecKCast, MiniMax-H3, FLUX; never claim anything section 5 marks unverified.

---------------------------------------------------------------------------------------------------------------

## 7. Environment on this machine (valid only on the original box)

- Hardware: 8x NVIDIA B200 (183 GB each), about 1.7 TB RAM, disk about 394 GB free of 774 GB.
- **Use the venv `~/spectrum_env/rv`** (`source ~/spectrum_env/rv/bin/activate`): torch 2.14.1+cu130, diffusers 0.41.0, transformers 5.20.2, lpips, scikit-image, matplotlib, av. (`~/spectrum_env/sv` has diffusers 0.34 and is only for the old Spectrum repo.) System Python has no torch and is PEP-668 locked; create uv venvs.
- **Env var trap:** the shell has `HF_TOKEN` set but it is **invalid (401)**. Always `unset HF_TOKEN HUGGING_FACE_HUB_TOKEN` and rely on `~/.cache/huggingface/token` (valid; HF user `Shreeyut`; has FLUX.1-dev, FLUX.2-dev and MiniMax-H3 access). Use `HF_HUB_OFFLINE=1` for models that are already cached (FLUX.1-dev, 32 GB, is cached; H3 and FLUX.2 are not).
- Progress bars flood the logs: use `--quiet` / `pipe.set_progress_bar_config(disable=True)` and `| tail`.
- GitHub: this machine's SSH key authenticates as `shreeyut1905`; `origin` of this repo is the SSH URL. The repo's git identity is `Shreeyut Maheshwari <149173158+shreeyut1905@users.noreply.github.com>` (GitHub noreply address, so the real email is not exposed). Only force-push (`--force-with-lease`) when you deliberately rewrite history.
- Directories (none of these are in the KalSea repo):
  - `/home/ubuntu/ReSpect`: clone of the owner's ReSPect repo with my **uncommitted, unpushed** additions in `FLUX/`: `speckcast_forecaster.py` (old name of `kalsea_forecaster.py`), `speckcast_record.py`, `speckcast_calibrate.py` (calibration), `benchmark.py` (8-GPU multi-config runner), `eval_metrics.py`, `quality_scores.py`, `cfg/*.json`, `prompts/` (drawbench200, coco_calib16, coco_tune64), `results/*.txt|png`, `runs/drawbench/` (6.2 GB), `calib_data/` (3.2 GB), and a modified `respect_generate.py`.
  - `/home/ubuntu/ReSpect/FLUX/runs/drawbench/<config>/`: 200 images per config + `stats_shard*.jsonl`; also `metrics_main.txt|json`, `metrics_sweep.txt|json`, `quality_main.txt|json`. **Directory names predate the rename:** `speckcast_0.37` = KalSea tau=0.5 (DCT); `speckcast_paper_0.37` = tau 0.1; `speckcast_nodct_0.37`; `respect_0.37` = ReSPect default; `respect_0.37_w0` = Chebyshev weight 0; `ref` = 50-step reference. File names `ReSPect_<idx>-<prompt>.png`, idx = DrawBench prompt index = seed.
  - `/home/ubuntu/Spectrum`: clone of the Spectrum repo with my earlier **velocity-space SpecKCast** implementation (`src/utils/speckcast.py`, uncommitted): there the forecaster alone was on par to slightly better than Spectrum, the adaptive scheduler worse. Background only.
  - `/home/ubuntu/KalSea-dev/`: `h3_tiny_model_test.py` (the H3 verification), `SpecKCast_paper_2610.26837.txt` (text of the paper). `/tmp/*` is ephemeral.
- To run the evaluation scripts again, copy them from `/home/ubuntu/ReSpect/FLUX/` (they import the old `speckcast_*` names).

---------------------------------------------------------------------------------------------------------------

## 8. Pitfalls and lessons learned

1. **H3 patching must be at the class level.** diffusers' `ComponentsManager` offload hooks wrap the *instance's* `forward`; an instance-level patch made after hooks are attached bypasses the offloading. The original forward is decorated with `apply_lora_scale` (functools.wraps), so call `forward.__wrapped__` for the unchanged path.
2. H3 timesteps are `t = 1 - sigma` (1 = clean); video (`shift=12`) and audio (`shift=3`) have different schedules; one forward serves both; N sigma points = N-1 evaluations; the checkpoint is guidance-distilled (one forward per step).
3. ReSPect's FLUX forward used to log the final step as `-1` (counter reset before the step index was read). Fixed in KalSea (`step_idx`); no effect on ReSPect's outputs.
4. FLUX gate: in practice steps 0 to 2 were always computed (in both schedules above). Probably because ReSPect's first comparison is filtered vs unfiltered input (reasoned from the code, not tested); in the H3 port the warm-up is made explicit (`--warmup_steps 3`).
5. Do not rank forecasters by raw residual error (see section 4); use closed-loop image fidelity.
6. GitHub's contributors graph is cached and can show a stale author for hours after a history rewrite; the contributors API is the truth.
7. Latency numbers depend on 8 concurrent processes; compare only configs run in the same session (the benchmark interleaves configs per prompt).
8. The `nocalib` FLUX variant in `results/tune_coco64.txt` (q=1, p22=2) is not a clean calibration-free ablation; it only shows that a *small* noise default is bad.
9. Never trust a claim just because it is in a README: section 5 is the source of truth for what was verified.

---------------------------------------------------------------------------------------------------------------

## 9. Credentials and security (important)

- **No secret may appear in this repo, in `blog/`, in the Space, in logs or in commit messages.** Tokens go through environment variables for a single process only.
- The owner pasted a Hugging Face token into the chat on 2026-10-20. That makes it **exposed**; it should be rotated at https://huggingface.co/settings/tokens once the Space is published. It is deliberately **not** reproduced here. The owner has to supply it (or a fresh one with *write* scope) to you as `HF_TOKEN`.
- Never `huggingface-cli login` / `HfApi().set_access_token` on this machine: it would overwrite `~/.cache/huggingface/token`, which other tools (and FLUX/H3 downloads) depend on.
- Creating a Space is a public-facing action on the owner's account: confirm namespace and visibility (6.2), default private.

---------------------------------------------------------------------------------------------------------------

## 10. Open TODOs (rough priority)

1. Write the blog (section 6) and publish the Space + markdown copy.
2. First real run of the H3 code; fix what breaks; choose delta; produce the H3 demo videos.
3. Decide on FLUX.2 [dev]: port (6.5) or leave out.
4. Closed-loop comparison against a plain damped two-point linear extrapolator, and an image-quality run of the calibration-free no-DCT setting (these decide how much of KalSea's "Kalman" is needed).
5. Add a `LICENSE` file (none yet) before making the repo public; note the MiniMax-H3 community license for H3 users.
6. Optionally move the evaluation/calibration scripts into the repo (the owner asked for inference code only, so they were left out).
