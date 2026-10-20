# KalSea: Kalman Residual Forecasting with Spectrally Aligned Skipping for Training-Free Acceleration of Diffusion Transformers

Training-free cache for FLUX and MiniMax-H3. A spectrally aligned gate decides **when** to skip a transformer
evaluation; a **Kalman residual forecaster** decides **what** to write on skipped steps.

<p align="center">
  <img src="assets/overview.png" width="900" />
</p>
<p align="center">
  <b>Figure 1:</b> KalSea. (a) Spectrally aligned skip gate (SeaCache). (b) Kalman forecast of the transformer
  residual on skipped steps (schematic: a toy signal run through the real forecaster code).
</p>

KalSea keeps the SeaCache-style gate and the residual-space framework of ReSPect and replaces ReSPect's
Chebyshev + Taylor forecaster by per-element Kalman filters (value + damped derivative), adapted from
[SpecKCast](https://arxiv.org/abs/2610.26837).

## Practical notes
- **Training-free.** No extra weights. The transformer forward is patched; run the generate script of a backbone.
- **One threshold.** Sweep `δ` (`--seacache_thresh`) for the speed-quality trade-off. Forecaster defaults (`τ=0.5`,
  `r_obs=1e-3`, always-computed warm-up steps and last step) are fixed.
- **FLUX.1-dev operating points (50 steps).** `δ` -> full evaluations: `0.2` -> 27, `0.3` -> 21, `0.37` -> 18, `0.5` -> 15,
  `0.6` -> 13, `0.8` -> 11, `1.0` -> 9. The gate is almost prompt independent, so the schedule is the same for every prompt.
- **MiniMax-H3: `δ` is not calibrated.** Start around `0.3`, read the printed number of evaluations and tune.

## Supported models
#### Text-to-Image
- [FLUX](./FLUX/README.md): implemented and evaluated (DrawBench200, 8 GPUs).

#### Text-to-Video (+ audio)
- [MiniMax-H3](./MiniMax-H3/README.md): text-to-video only (`t2va`). **Inference code only: it has not been run on the real
  33B checkpoint** (it is verified against the original forward on a tiny random-weight model, see its README).

## Result on FLUX.1-dev (what is and is not shown)

DrawBench200, 50 steps, fidelity to the 50-step image of the same seed. Only the forecaster changes between rows; the gate
(and so the number of full evaluations) is identical.

| gate δ | NFE | forecaster | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| 0.37 | 18.1 | ReSPect (Spectrum), repo default | 26.18 | 0.8968 | 0.0908 |
| 0.37 | 18.1 | **KalSea** | **26.91** | 0.9066 | 0.0801 |
| 0.6 | 13.0 | ReSPect (Spectrum), repo default | 21.55 | 0.8196 | 0.1826 |
| 0.6 | 13.0 | **KalSea** | **21.91** | 0.8256 | 0.1716 |

<p align="center"><img src="assets/fidelity_vs_nfe.png" width="900" /></p>

KalSea is better than the default ReSPect forecaster at every threshold tried (PSNR +0.3 to +2.1 dB) with about 3x less
forecaster memory. **But most of that gain comes from ReSPect's own Chebyshev term hurting in this setup**: with its Chebyshev
weight set to 0 the Spectrum forecaster already closes most of the gap, and KalSea's remaining edge is small (+0.25 dB at
δ=0.37, significant; +0.11 dB at δ=0.6, not significant). Details, the ablations and all caveats are in
[FLUX/README.md](./FLUX/README.md).

## Layout
Same layout as ReSPect: one folder per backbone with a generate script you run.

```
KalSea-v1/
  FLUX/kalsea_generate.py
  MiniMax-H3/kalsea_generate.py
```
Evaluation / calibration scripts are not part of this release (inference code only).

## Acknowledgement
Built on [SeaCache](https://github.com/jiwoogit/SeaCache) (gate), ReSPect (residual-space framework),
[Spectrum](https://github.com/hanjq17/Spectrum) (baseline forecaster), [SpecKCast](https://arxiv.org/abs/2610.26837)
(Kalman forecasting), [FLUX](https://github.com/black-forest-labs/flux),
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and [Diffusers](https://github.com/huggingface/diffusers).
The implementation was written with the help of an AI coding assistant (Claude, run through the pi agent harness).
MiniMax-H3 weights are under the MiniMax-H3 Community License, which you must follow when using that backbone.

## Citation
```bibtex
@inproceedings{kalsea,
  title={KalSea: Kalman Residual Forecasting with Spectrally Aligned Skipping for Training-Free Acceleration of Diffusion Transformers}
}
```
