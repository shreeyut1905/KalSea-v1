"""PCGS: predictor-corrector gap smoothing for KalSea on FLUX (zero extra transformer-block evaluations).

KalSea uses its Kalman filter as a PREDICTOR on the skipped steps a+1..b-1: the residual r = h_out - h_in is extrapolated
(r_hat_i) and the sampler integrates the latent with it.  When the next computed step b arrives, the true residual r_b is
observed and PCGS closes the loop:

  1. SMOOTH    the residuals of the skipped steps with the innovation e_b = z_b - m_b^- of the update at b
               (Rauch-Tung-Striebel smoother of the forecaster's own damped-trend model):
                   r~_i = r_hat_i + g_i * e_b,   g_i = [P_i^- Phi(b,i)^T]_00 / (P_b^-_00 + R),   Phi(b,i) = A_b ... A_{i+1}
               (a gain per DCT band, or a scalar for the per-element filter; P, A and R are the forecaster's own).
  2. RE-INTEGRATE the Euler steps of the gap from the latent actually used at a+1 (exact, step a was computed), with the
               output layers only:  x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * Out(x_embedder(x_i) + r~_i, temb_i),
               Out(h, temb) = proj_out(norm_out(h, temb)), same arithmetic as FlowMatchEulerDiscreteScheduler.step.
  3. CONTINUE  from the corrected latent x_b': the step-b output becomes
                   v_b' = Out(x_embedder(x_b') + r_b, temb_b) + (x_b' - x_b) / (sigma_{b+1} - sigma_b),
               so the scheduler's Euler step yields x_{b+1} = x_b' + dt_b * Out(x_embedder(x_b') + r_b, temb_b).
  4. REFRESH   the SEA gate's stored (filtered) modulated input from x_b', so the gate measures the change along the
               trajectory actually followed (block-0 norm1 + SEA filter, no block call).

Only tensors of the current and past steps of the trajectory are used.  The extra work is one x_embedder + norm_out +
proj_out evaluation per skipped step and one more at the computed step b; no transformer block is called.
"""
from __future__ import annotations

import numpy as np
import torch

from util_seacache import apply_sea_with_scheduler


class PCGS:
    """Per-generation PCGS state; created by `reset_transformer_state` and driven by the patched forward."""

    def __init__(self, tr, scheduler, num_steps: int, gate_refresh: bool = True):
        self.tr, self.sched, self.num_steps = tr, scheduler, int(num_steps)
        self.gate_refresh = bool(gate_refresh)
        self.gap = []        # records of the skipped steps since the last computed step
        self.hist = None     # (step, residual) of the last computed step
        self.prior_P = {}    # skipped step -> forecaster prior covariance P_i^-  [L, 2, 2]
        self.upd = None      # (P_b^-, innovation) captured before the forecaster update at the computed step b
        self.st = dict(n_corr=0, n_out_calls=0, n_emb_calls=0, n_gate_refresh=0, gap_len=[])

    # ------------------------------------------------------------------------------------------- forward hooks
    def on_skip(self, step, x_in, temb, pred, fc):
        """Skipped step: `pred` is the forecast residual about to be added to x_embedder(x_in)."""
        self.prior_P[step] = fc.P.copy()          # P_i^- (predict() advanced the filter to step i)
        self.gap.append(dict(step=step, x=x_in, temb=temb, pred=pred))

    @torch.no_grad()
    def before_update(self, step, fc, residual):
        """Computed step, right before fc.update(step, residual): record P_b^- and the innovation (relative units)."""
        if self.gap and fc._ready and fc.m is not None:
            fc._advance_to(step)                  # the first call of fc.update (idempotent afterwards)
            Z = fc._fwd(fc._to_grid(residual))
            self.upd = dict(step=step, Pb=fc.P.copy(), e_rel=Z * fc.scale - fc.m)

    @torch.no_grad()
    def on_computed_output(self, step, x_in, temb, output, residual, img_ids):
        """Computed step, after norm_out / proj_out: correct the gap that this step closes, return the velocity to use."""
        out = output
        if self.gap and self.hist is not None:
            out = self._correct(step, x_in, temb, residual, output, img_ids)
        self.gap, self.prior_P, self.upd = [], {}, None
        self.hist = (step, residual)
        return out

    def stats(self) -> dict:
        return dict(self.st)

    # ------------------------------------------------------------------------------------------- smoother
    def _kalman_gains(self, a, b, steps):
        fc = self.tr.kalsea_forecaster
        Pb = self.upd["Pb"]
        S = Pb[:, 0, 0] + fc.r_obs                            # innovation variance per band  [L]
        L = Pb.shape[0]
        Phi = np.tile(np.eye(2), (L, 1, 1))
        gains = {}
        for i in range(b - 1, a, -1):
            k = i + 1
            dl = fc.delta[:, min(k, fc.N - 1)]
            rho = np.exp(-dl / fc.tau)
            A = np.zeros((L, 2, 2))
            A[:, 0, 0], A[:, 0, 1], A[:, 1, 1] = 1.0, dl, rho
            Phi = Phi @ A                                     # Phi(b, i) = Phi(b, i+1) A_{i+1}
            if i in steps:
                Pi = self.prior_P[i]
                gains[i] = (Pi[:, 0, :] * Phi[:, 0, :]).sum(-1) / S   # [P_i Phi^T]_00 / S
        return gains

    def _smooth(self, a, b):
        """Smoothed residuals (bf16) of the gap records."""
        assert self.upd is not None and self.upd["step"] == b, "PCGS: no forecaster update captured at the computed step"
        fc = self.tr.kalsea_forecaster
        gains = self._kalman_gains(a, b, {rec["step"] for rec in self.gap})
        E = self.upd["e_rel"] / fc.scale                      # innovation in DCT / element coordinates, absolute units
        out = []
        for rec in self.gap:
            corr = fc._from_grid(fc._inv(fc._band_vec_to_map(gains[rec["step"]]) * E))
            out.append((rec["pred"].float() + corr).to(rec["pred"].dtype))
        return out

    # ------------------------------------------------------------------------------------------- correction
    def _out(self, h, temb):
        self.st["n_out_calls"] += int(h.shape[0])
        return self.tr.proj_out(self.tr.norm_out(h, temb))

    def _correct(self, b, x_b, temb_b, r_b, output, img_ids):
        tr = self.tr
        a = self.hist[0]
        steps = [rec["step"] for rec in self.gap]
        assert steps == list(range(a + 1, b)), f"PCGS gap bookkeeping: a={a} b={b} steps={steps}"
        rt = self._smooth(a, b)
        sg = self.sched.sigmas
        x = self.gap[0]["x"]                                  # x_{a+1}: exact (step a was computed)
        for rec, r_i in zip(self.gap, rt):
            i = rec["step"]
            h = tr.x_embedder(x)
            self.st["n_emb_calls"] += 1
            v = self._out(h + r_i, rec["temb"])
            dt = sg[i + 1] - sg[i]                            # same tensors / arithmetic as FlowMatchEulerDiscreteScheduler.step
            x = (x.to(torch.float32) + dt * v).to(v.dtype)
        x_corr = x
        self.st["gap_len"].append(len(steps))
        if torch.equal(x_corr, x_b):
            return output
        self.st["n_corr"] += 1
        dxf = x_corr.float() - x_b.float()
        dt_b = float(sg[b + 1] - sg[b])
        h_new = tr.x_embedder(x_corr)                         # residual held fixed: h_out' = x_embedder(x_b') + r_b
        self.st["n_emb_calls"] += 1
        v_new = self._out(h_new + r_b, temb_b)
        out = (v_new.float() + dxf / dt_b).to(output.dtype)
        if self.gate_refresh and b < self.num_steps - 1 and getattr(tr, "enable_seacache", False) \
                and tr.previous_modulated_input is not None:
            self._refresh_gate(b, h_new, temb_b, img_ids)
        return out

    def _refresh_gate(self, b, h_new, temb_b, img_ids):
        """Recompute the SEA gate's reference (previous filtered modulated input) from the corrected latent."""
        tr = self.tr
        mod = tr.transformer_blocks[0].norm1(h_new, emb=temb_b)[0]
        Hg, Wg = int(img_ids[:, 1].max().item() + 1), int(img_ids[:, 2].max().item() + 1)
        mod = mod.reshape(mod.shape[0], Hg, Wg, mod.shape[-1])
        mod = apply_sea_with_scheduler(mod, tr.scheduler, b, power_exp=2.0, dims=(-2, -3), norm_mode="mean")
        tr.previous_modulated_input = mod.reshape(mod.shape[0], -1, mod.shape[-1])
        self.st["n_gate_refresh"] += 1
