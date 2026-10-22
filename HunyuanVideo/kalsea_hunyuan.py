"""KalSea for the official HunyuanVideo code (hyvideo): SEA gate + Kalman residual forecast + PCGS.

Methods (`install` patches HYVideoDiffusionTransformer.forward, as SeaCache's script does; hyvideo is not modified):
  seacache     SeaCache's HunyuanVideo gate: double_blocks[0] modulated image tokens, SEA filter (beta 3 over T, H, W),
               accumulated relative L1 distance against the stored input, first and last step always computed. Skipped
               steps reuse the last block-stack residual of the image tokens.
  kalsea       same gate; a skipped step adds the forecast of a per-element, calibration-free Kalman filter of the
               block-stack residual (updated with the true residual on computed steps).
  kalsea_pcgs  kalsea + predictor-corrector gap smoothing (PCGS). At a computed step b that closes a gap a+1..b-1:
                 1. smooth the forecasts of the gap with the innovation of the update at b (Rauch-Tung-Striebel gains of
                    the forecaster's own damped-trend model);
                 2. re-integrate the gap's Euler steps from the latent actually used at a+1 with the output layers only:
                    x_{i+1} = x_i + dt_i * unpatchify(final_layer(img_in(x_i) + smoothed residual_i, vec_i));
                 3. continue from the corrected latent x_b': the step-b output becomes
                    unpatchify(final_layer(img_in(x_b') + r_b, vec_b)) + (x_b' - x_b) / dt_b;
                 4. refresh the gate's stored input from x_b'.
               No extra transformer-block calls.

Noise level of the gate: the SEA filter needs the noise level sigma of the current step. SeaCache's released script binds
the scheduler that exists at load time, but `HunyuanVideoSampler.predict` samples with a fresh scheduler, so the filter
sees sigma = 1 - step/1000 instead of the sampling schedule ("stale" sigma). By default (live_sigma=True) the filter
reads the sigma of the scheduler that predict() samples with; live_sigma=False reproduces the released behaviour.
"""
from contextlib import contextmanager
from typing import Dict, Optional, Union

import numpy as np
import torch

from hyvideo.modules.attenion import get_cu_seqlens
from hyvideo.modules.modulate_layers import modulate
from kalsea_forecaster import KalSeaForecaster
from util_seacache import apply_sea_with_scheduler, rel_l1

METHODS = ("seacache", "kalsea", "kalsea_pcgs")


@contextmanager
def deterministic():
    """The forecaster's two scalar reductions use scatter_add_ (CUDA atomics); deterministic mode fixes their summation
    order, so a generation is bitwise reproducible."""
    prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def cache_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        ks = self._kalsea
        x_in = x                                   # latent input of this step (PCGS re-integrates from it)
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # ---- SEA gate ----
        step = int(self.cnt)
        inp = img.clone()
        vec_ = vec.clone()
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
        normed_inp = self.double_blocks[0].img_norm1(inp)
        modulated_inp = modulate(
            normed_inp, shift=img_mod1_shift, scale=img_mod1_scale
        )
        cur_5d = modulated_inp.reshape(
            modulated_inp.shape[0], tt, th, tw, modulated_inp.shape[-1]
        )
        cur_5d = apply_sea_with_scheduler(
            cur_5d,
            ks.gate_scheduler(),
            self.cnt,
            power_exp=3.0,
            dims=(-2, -3, -4),
            norm_mode="mean",
        )
        modulated_inp = cur_5d.reshape(cur_5d.shape[0], -1, cur_5d.shape[-1])
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            self.accumulated_rel_l1_distance += rel_l1(
                modulated_inp, self.previous_modulated_input
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

        # ---- skip (reuse / forecast) or compute ----
        if not should_calc:
            pred = ks.on_skip(step, x_in, vec, img)
            if pred is None:                       # seacache: reuse the last residual
                img += self.previous_residual
            else:                                  # kalsea: forecast residual
                img += pred
        else:
            ori_img = img.clone()
            # --------------------- Pass through DiT blocks ------------------------
            for _, block in enumerate(self.double_blocks):
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                ]

                img, txt = block(*double_block_args)

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                    ]

                    x = block(*single_block_args)

            img = x[:, :img_seq_len, ...]
            self.previous_residual = img - ori_img
            ks.on_compute(step, self.previous_residual, tt, th, tw)

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        img = ks.on_output(step, should_calc, x_in, vec, img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img


class KalSeaHunyuan:
    """Per-video state of the gate, the forecaster and PCGS. Call before_video() before every predict()."""

    def __init__(self, sampler, method: str, delta: float, num_steps: int = 50, kalsea_tau: float = 0.5,
                 kalsea_r_obs: float = 1e-3, live_sigma: bool = True):
        assert method in METHODS, method
        self.sampler = sampler
        self.pipeline = sampler.pipeline
        self.method = method
        self.delta = float(delta)
        self.num_steps = int(num_steps)
        self.tau, self.r_obs = float(kalsea_tau), float(kalsea_r_obs)
        self.live_sigma = bool(live_sigma)
        self.use_fc = method in ("kalsea", "kalsea_pcgs")
        self.use_pcgs = method == "kalsea_pcgs"
        self.tr = None
        self.load_time_scheduler = None
        self.fc = None
        self.compute_steps = []
        self.gap, self.hist, self.prior_P, self.upd, self.cur = [], [], {}, None, None

    def install(self, tr):
        self.tr = tr
        # predict() replaces pipeline.scheduler by a fresh sampling scheduler; this is the load-time one
        self.load_time_scheduler = self.pipeline.scheduler
        tr.__class__.forward = cache_forward
        tr._kalsea = self
        return self

    def gate_scheduler(self):
        """Scheduler whose sigmas the SEA filter reads: the live sampling scheduler, or the load-time one (stale)."""
        return self.pipeline.scheduler if self.live_sigma else self.load_time_scheduler

    # ------------------------------------------------------------------ per video
    def before_video(self):
        tr = self.tr
        tr.cnt = 0
        tr.accumulated_rel_l1_distance = 0
        tr.previous_modulated_input = None
        tr.previous_residual = None
        tr.num_steps = self.num_steps
        tr.rel_l1_thresh = self.delta
        self.compute_steps = []
        self.gap, self.hist, self.prior_P, self.upd, self.cur = [], [], {}, None, None
        self.fc = self._make_forecaster() if self.use_fc else None

    def _make_forecaster(self):
        from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
        a = self.sampler.args
        s = FlowMatchDiscreteScheduler(shift=a.flow_shift, reverse=a.flow_reverse, solver=a.flow_solver)
        s.set_timesteps(self.num_steps)           # the sampling schedule (only its length matters for time_mode 'step')
        # calibration-free, one filter per element, time coordinate = step index
        return KalSeaForecaster(calib=None, scheduler=None, num_steps=self.num_steps, sigmas=s.sigmas.double().numpy(),
                                grid=None, tau=self.tau, r_obs=self.r_obs, time_mode="step", use_dct=False, adapt_q=True)

    def after_video(self):
        """Returns the steps that ran the transformer blocks; drops every tensor reference of the video."""
        steps = list(self.compute_steps)
        assert int(self.tr.cnt) == 0, "the step counter was not reset: did predict() use num_steps steps?"
        self.gap, self.hist, self.prior_P, self.upd, self.cur = [], [], {}, None, None
        self.fc = None
        return steps

    # ------------------------------------------------------------------ forward callbacks
    @torch.no_grad()
    def on_skip(self, step, x_in, vec, img):
        if not self.use_fc or self.fc is None or self.fc.n_obs < 1:
            return None
        fc = self.fc
        pred = fc.predict(step).to(dtype=img.dtype, device=img.device)
        if self.use_pcgs:
            self.prior_P[step] = fc.P.copy()       # P_i^- (predict() advanced the filter to step i)
            self.gap.append(dict(step=step, x=x_in, vec=vec, pred=pred))
        return pred

    @torch.no_grad()
    def on_compute(self, step, residual, tt, th, tw):
        self.compute_steps.append(step)
        if not self.use_fc:
            return
        fc = self.fc
        if fc.grid is None:
            fc.grid = (tt * th, tw)                 # token layout for the forecaster's reshape (per element: any grid)
        if self.use_pcgs and self.gap and fc._ready and fc.m is not None:
            fc._advance_to(step)                    # the first call of fc.update (idempotent afterwards)
            Pb = fc.P.copy()
            Z = fc._fwd(fc._to_grid(residual))
            self.upd = dict(step=step, Pb=Pb, e_rel=Z * fc.scale - fc.m)   # innovation, relative units
        with deterministic():
            fc.update(step, residual)
        if self.use_pcgs:
            self.cur = (step, residual)

    @torch.no_grad()
    def on_output(self, step, computed, x_in, vec, out, tt, th, tw):
        if not self.use_pcgs or not computed:
            return out
        r_b = self.cur[1] if (self.cur is not None and self.cur[0] == step) else None
        if self.gap and self.hist and r_b is not None:
            out = self._correct(step, x_in, vec, r_b, out, tt, th, tw)
        self.gap, self.prior_P, self.upd = [], {}, None
        if r_b is not None:
            self.hist = [(step, r_b)]
        self.cur = None
        return out

    # ------------------------------------------------------------------ PCGS
    def _kalman_gains(self, a, b, steps):
        """g_i = [P_i^- Phi(b,i)^T]_00 / (P_b^-_00 + R), Phi(b,i) = A_b ... A_{i+1}."""
        fc = self.fc
        Pb = self.upd["Pb"]
        S = Pb[:, 0, 0] + fc.r_obs
        L = Pb.shape[0]
        Phi = np.tile(np.eye(2), (L, 1, 1))
        gains = {}
        for i in range(b - 1, a, -1):
            k = i + 1
            dl = fc.delta[:, min(k, fc.N - 1)]
            rho = np.exp(-dl / fc.tau)
            A = np.zeros((L, 2, 2))
            A[:, 0, 0], A[:, 0, 1], A[:, 1, 1] = 1.0, dl, rho
            Phi = Phi @ A
            if i in steps:
                Pi = self.prior_P[i]
                gains[i] = (Pi[:, 0, :] * Phi[:, 0, :]).sum(-1) / S
        return gains

    def _smooth(self, a, b):
        assert self.upd is not None and self.upd["step"] == b, "PCGS: no forecaster update captured at step b"
        fc = self.fc
        gains = self._kalman_gains(a, b, {rec["step"] for rec in self.gap})
        E = self.upd["e_rel"] / fc.scale                       # innovation in element coordinates, absolute units
        out = []
        for rec in self.gap:
            corr = fc._from_grid(fc._inv(fc._band_vec_to_map(gains[rec["step"]]) * E))
            out.append((rec["pred"].float() + corr).to(rec["pred"].dtype))
        return out

    def _out(self, h, vec, tt, th, tw):
        tr = self.tr
        return tr.unpatchify(tr.final_layer(h, vec), tt, th, tw)

    def _sampling_sigmas(self):
        sig = self.pipeline.scheduler.sigmas
        assert sig.numel() == self.num_steps + 1, "PCGS needs the sampling scheduler (num_steps + 1 sigmas)"
        return sig

    def _correct(self, b, x_b, vec_b, r_b, out_b, tt, th, tw):
        tr = self.tr
        a, _ = self.hist[-1]
        steps = [rec["step"] for rec in self.gap]
        assert steps == list(range(a + 1, b)), f"PCGS gap bookkeeping: a={a} b={b} steps={steps}"
        rt = self._smooth(a, b)
        sg = self._sampling_sigmas()
        x = self.gap[0]["x"]                                   # x_{a+1}: the latent actually used at step a+1
        for rec, r_i in zip(self.gap, rt):
            i = rec["step"]
            h = tr.img_in(x)
            v = self._out(h + r_i, rec["vec"], tt, th, tw)
            dt = sg[i + 1] - sg[i]                             # = FlowMatchDiscreteScheduler.step (same tensors/ops)
            x = x.to(torch.float32) + v.to(torch.float32) * dt
        if torch.equal(x, x_b):
            return out_b
        dxf = x.float() - x_b.float()
        dt_b = sg[b + 1] - sg[b]
        h_new = tr.img_in(x)                                   # residual of step b held fixed: img_in(x_b') + r_b
        v_new = self._out(h_new + r_b, vec_b, tt, th, tw)
        out = v_new.float() + dxf / dt_b                       # the scheduler adds dt_b * out to x_b
        if b < self.num_steps - 1 and tr.previous_modulated_input is not None:
            self._gate_refresh(b, h_new, vec_b, tt, th, tw)
        return out

    def _gate_refresh(self, b, h_new, vec_b, tt, th, tw):
        """Recompute the gate's stored (filtered) modulated input from the corrected latent; same ops as the gate."""
        tr = self.tr
        (shift, scale, _g1, _s2, _c2, _g2) = tr.double_blocks[0].img_mod(vec_b).chunk(6, dim=-1)
        mod = modulate(tr.double_blocks[0].img_norm1(h_new), shift=shift, scale=scale)
        cur = mod.reshape(mod.shape[0], tt, th, tw, mod.shape[-1])
        cur = apply_sea_with_scheduler(cur, self.gate_scheduler(), b, power_exp=3.0, dims=(-2, -3, -4),
                                       norm_mode="mean")
        tr.previous_modulated_input = cur.reshape(cur.shape[0], -1, cur.shape[-1])


def install(sampler, method, delta, num_steps=50, kalsea_tau=0.5, kalsea_r_obs=1e-3, live_sigma=True):
    """Patch the transformer of a loaded HunyuanVideoSampler. Must run before the first predict()."""
    return KalSeaHunyuan(sampler, method, delta, num_steps, kalsea_tau, kalsea_r_obs, live_sigma).install(
        sampler.pipeline.transformer)
