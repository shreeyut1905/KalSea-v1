"""KalSea residual forecaster.

The forecaster is adapted from SpecKCast (Rajabi et al., arXiv:2610.26837), which forecasts the final velocity with
per-DCT-coefficient Kalman filters.  Here it forecasts the *transformer residual* (block-stack output minus block-stack
input, shape [B, N_tokens, D]) that is written on skipped steps; *when* to skip is still decided by the SeaCache/SEA
gate.

Every element of the residual is tracked by a Kalman filter with state (value m, temporal derivative nu):

    m_j  = m_{j-1} + ds_j * nu_{j-1},      nu_j = exp(-ds_j / tau) * nu_{j-1}          (damped trend)
    P_j  = A P_{j-1} A^T + Q(ds_j)                                                    (predict at EVERY step)
    update with the observed residual on computed steps only                           (Kalman correction)

so the irregular compute pattern produced by the gate is handled natively through the step size ds entering A and Q.
On a skipped step the predicted values m are returned.

Two modes
  * use_dct=False (default for video / grid-free token sets): one filter per element, one shared covariance.
    No grid and no calibration needed.
  * use_dct=True (FLUX): the N_tokens image tokens are laid out on their (Hg, Wg) grid per channel, 2D-DCT'd and split
    into L=7 radial bands; the covariance, noise and time coordinate (a band-specific warp of the log-SNR, paper
    Eq. 8) are shared inside a band.  Needs the offline calibration json for the warp.

All filter statistics are in "relative units" (values divided by the sqrt of the signal power).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

BAND_EDGES = [0.0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0 + 1e-6]

# Calibration-free filter noise (relative units, time coordinate s in [-1, 1]): the values of the FLUX calibration pooled
# over the 7 bands.  They are *large* compared with the observation noise on purpose: the trend is then re-estimated at
# every update, which makes the filter a damped two-point extrapolator that is robust to the unknown scale of a model.
DEFAULT_Q = 7.4e3
DEFAULT_P22 = 1.5e4


# ----------------------------------------------------------------------------------------------
# DCT / bands
# ----------------------------------------------------------------------------------------------
def dct_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix D (n x n); Z = D x."""
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    D = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    D[0] = D[0] / math.sqrt(2.0)
    return D.to(device=device, dtype=dtype)


def band_index_map(H: int, W: int, edges=BAND_EDGES) -> torch.Tensor:
    """Radial band of every (a, b) DCT coefficient: r = sqrt((a/H)^2 + (b/W)^2)/sqrt(2) in [0, 1]."""
    a = torch.arange(H, dtype=torch.float64).view(-1, 1) / H
    b = torch.arange(W, dtype=torch.float64).view(1, -1) / W
    r = torch.sqrt(a ** 2 + b ** 2) / math.sqrt(2.0)
    return torch.bucketize(r, torch.tensor(edges[1:-1], dtype=torch.float64), right=True).long()


# ----------------------------------------------------------------------------------------------
# Time coordinate (paper Eq. 8 / 17)
# ----------------------------------------------------------------------------------------------
def log_snr(sigmas, clamp: float = 1e-4) -> np.ndarray:
    s = np.clip(np.asarray(sigmas, dtype=np.float64), clamp, 1.0 - clamp)
    return 2.0 * np.log((1.0 - s) / s)


def band_time_coords(sigmas, p_warp, Tw: float = 2.0, mode: str = "warped", num_steps: Optional[int] = None) -> np.ndarray:
    """s[l, j] in [-1, 1].  mode: 'warped' (sigmoid of the band's own log-SNR), 'logsnr' (plain rescaled log-SNR for all
    bands), 'step' (uniform in the step index; needs no noise schedule)."""
    L = len(p_warp)
    if mode == "step":
        N = int(num_steps if num_steps is not None else len(sigmas))
        t = np.arange(N, dtype=np.float64)
        return np.tile((2.0 * (t - t[0]) / (t[-1] - t[0] + 1e-12) - 1.0)[None], (L, 1))
    lam = log_snr(sigmas)
    out = np.zeros((L, len(lam)))
    for l in range(L):
        if mode == "warped":
            t = 1.0 / (1.0 + np.exp(-(lam + math.log(p_warp[l])) / Tw))
        elif mode == "logsnr":
            t = lam
        else:
            raise ValueError(mode)
        out[l] = 2.0 * (t - t[0]) / (t[-1] - t[0] + 1e-12) - 1.0
    return out


# ----------------------------------------------------------------------------------------------
# Calibration container
# ----------------------------------------------------------------------------------------------
@dataclass
class KalSeaCalib:
    p_warp: List[float]                 # band signal power used for the time warp (paper B.4)
    q: List[float]                      # process-noise intensity per band, relative units
    p22_init: List[float]               # initial trend variance per band
    p_scale: Optional[List[float]] = None  # per-band power of the residual (relative-unit scale); None = estimate online
    meta: dict = field(default_factory=dict)

    @staticmethod
    def load(path: str) -> "KalSeaCalib":
        with open(path) as f:
            return KalSeaCalib(**json.load(f))

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=1)

    @staticmethod
    def default(L: int = len(BAND_EDGES) - 1) -> "KalSeaCalib":
        """Calibration-free fallback: no band-specific warp, diffuse trend prior, band power estimated online."""
        return KalSeaCalib(p_warp=[1.0] * L, q=[DEFAULT_Q] * L, p22_init=[DEFAULT_P22] * L, p_scale=None)


# ----------------------------------------------------------------------------------------------
# Forecaster
# ----------------------------------------------------------------------------------------------
class KalSeaForecaster:
    """Interface: update(step, residual), predict(step), n_obs.

    `update` is called on computed steps (observation), `predict` on skipped steps; the filter is advanced through every
    intermediate step automatically, so any mixture of computed / skipped steps is fine.
    """

    def __init__(
        self,
        calib: Optional[KalSeaCalib] = None,
        scheduler=None,
        num_steps: int = 50,
        sigmas=None,
        grid=None,
        tau: float = 0.5,
        Tw: float = 2.0,
        r_obs: float = 1e-3,
        q_mult: float = 1.0,
        time_mode: str = "warped",
        use_dct: bool = True,
        adapt_q: bool = True,
        device=None,
    ):
        self.calib = calib if calib is not None else KalSeaCalib.default()
        self.scheduler = scheduler
        self.num_steps = int(num_steps)
        self._sigmas = sigmas
        self.grid = tuple(grid) if grid is not None else None   # (Hg, Wg), only used with use_dct (square if None)
        self.tau, self.Tw, self.r_obs, self.q_mult = float(tau), float(Tw), float(r_obs), float(q_mult)
        self.time_mode, self.use_dct, self.adapt_q = time_mode, bool(use_dct), bool(adapt_q)
        self.device_ref = device
        self._ready = False
        self._n_obs = 0
        self.m = None            # values in relative units, [C, Hg, Wg] (DCT) or the input shape [B, Nt, D]
        self.nu = None           # derivatives
        self.j_last = -1

    # ------------------------------------------------------------------ setup
    @property
    def n_obs(self) -> int:
        return self._n_obs

    def _setup(self, h: torch.Tensor):
        B, Nt, D = h.shape
        self.B, self.Nt, self.D = B, Nt, D
        self.dev = h.device
        cal = self.calib
        p_warp = np.array(cal.p_warp, dtype=np.float64)
        q = np.array(cal.q, dtype=np.float64) * self.q_mult
        p22 = np.array(cal.p22_init, dtype=np.float64)
        L_full = len(p_warp)
        if self.use_dct:
            if self.grid is None:
                g = int(round(math.sqrt(Nt)))
                self.grid = (g, g)
            Hg, Wg = self.grid
            assert Hg * Wg == Nt, f"token grid {self.grid} does not match {Nt} tokens"
            self.C = B * D
            bm = band_index_map(Hg, Wg)
            cnt = np.array([(bm == l).sum().item() for l in range(L_full)], dtype=np.float64)
            self.L = L_full
            self.bm = bm.to(self.dev)
            self.d = np.maximum(cnt * self.C, 1.0)
            self.dct = (dct_matrix(Hg, self.dev), dct_matrix(Wg, self.dev))
        else:  # one band, no frequency decomposition; works for any token count (no grid needed)
            if self.grid is not None and self.grid[0] * self.grid[1] == Nt:
                bm = band_index_map(*self.grid)
                w = np.array([(bm == l).sum().item() for l in range(L_full)], dtype=np.float64)
            else:
                w = np.ones(L_full)
            w = w / w.sum()
            self.L = 1
            p_warp = np.array([float((w * p_warp).sum())])
            q = np.array([float((w * q).sum())])
            p22 = np.array([float((w * p22).sum())])
            self.bm, self.dct = None, None
        self.q0, self.p22_init = q, p22
        self.qscale = np.ones(self.L)
        self.P = np.zeros((self.L, 2, 2))
        # time coordinate per band
        if self.time_mode == "step":
            sig = None
        elif self._sigmas is not None:
            sig = np.asarray(self._sigmas, dtype=np.float64)[: self.num_steps]
        else:
            sig = self.scheduler.sigmas[: self.num_steps].detach().float().cpu().numpy().astype(np.float64)
        self.s = band_time_coords(sig, p_warp, self.Tw, self.time_mode, num_steps=self.num_steps)
        self.N = self.s.shape[1]
        self.delta = np.zeros_like(self.s)
        self.delta[:, 1:] = np.diff(self.s, axis=1)
        self._p_scale_cfg = cal.p_scale
        self._ready = True

    # ------------------------------------------------------------------ layout helpers
    def _to_state(self, h: torch.Tensor) -> torch.Tensor:
        x = h.float()
        if self.dct is None:
            return x
        Hg, Wg = self.grid
        x = x.reshape(self.B, Hg, Wg, self.D).permute(0, 3, 1, 2).reshape(self.C, Hg, Wg)
        DH, DW = self.dct
        return DH @ x @ DW.T

    def _from_state(self, z: torch.Tensor) -> torch.Tensor:
        if self.dct is None:
            return z
        Hg, Wg = self.grid
        DH, DW = self.dct
        x = DH.T @ z @ DW
        return x.reshape(self.B, self.D, Hg, Wg).permute(0, 2, 3, 1).reshape(self.B, self.Nt, self.D)

    def _band_vec(self, v: np.ndarray):
        """Per-band vector -> broadcastable factor: a python float for a single band, a [Hg, Wg] map for DCT bands."""
        if self.bm is None:
            return float(v[0])
        return torch.tensor(v, dtype=torch.float32, device=self.dev)[self.bm]

    def _band_err(self, e: torch.Tensor) -> np.ndarray:
        """Mean squared innovation per band."""
        if self.bm is None:
            return np.array([float((e ** 2).mean())])
        return torch.zeros(self.L, device=self.dev).scatter_add_(
            0, self.bm.reshape(-1), (e ** 2).sum(0).reshape(-1)).cpu().numpy() / self.d

    # ------------------------------------------------------------------ filter
    def _predict_step(self, j: int):
        """Advance the prior from step j-1 to j (paper Eq. 9-11)."""
        self.j_last = j
        if j == 0 or self.m is None:
            return
        dl = self.delta[:, min(j, self.N - 1)]
        rho = np.exp(-dl / self.tau)
        self.m = self.m + self._band_vec(dl) * self.nu
        self.nu = self._band_vec(rho) * self.nu
        q = self.q0 * self.qscale
        for l in range(self.L):
            A = np.array([[1.0, dl[l]], [0.0, rho[l]]])
            Q = q[l] * np.array([[dl[l] ** 3 / 3, dl[l] ** 2 / 2], [dl[l] ** 2 / 2, dl[l]]])
            self.P[l] = A @ self.P[l] @ A.T + Q

    def _advance_to(self, j: int):
        while self.j_last < j:
            self._predict_step(self.j_last + 1)

    @torch.no_grad()
    def update(self, t, h: torch.Tensor) -> None:
        """Observation at a computed step: Kalman correction of (value, derivative) of every element."""
        j = int(round(float(t)))
        if not self._ready:
            self._setup(h)
        self._advance_to(j)
        Z = self._to_state(h)
        if self.m is None:
            if self._p_scale_cfg is not None and self.bm is not None:
                p_scale = np.array(self._p_scale_cfg, dtype=np.float64)
            elif self._p_scale_cfg is not None:
                p_scale = np.array([float(np.mean(self._p_scale_cfg))])
            elif self.bm is None:  # online estimate of the signal power from the first observation
                p_scale = np.array([float((Z ** 2).mean())])
            else:
                p_scale = torch.zeros(self.L, device=self.dev).scatter_add_(
                    0, self.bm.reshape(-1), (Z ** 2).sum(0).reshape(-1)).cpu().numpy() / self.d
            p_scale = np.maximum(p_scale, 1e-12)
            self.p_scale = p_scale
            self.scale = self._band_vec(1.0 / np.sqrt(p_scale))
            self.m = Z * self.scale
            self.nu = torch.zeros_like(self.m)
            for l in range(self.L):
                self.P[l] = np.diag([self.r_obs, self.p22_init[l]])
        else:
            z = Z * self.scale
            e = z - self.m
            Sv = self.P[:, 0, 0] + self.r_obs
            K = self.P[:, :, 0] / Sv[:, None]                              # [L, 2]  (H = [1, 0])
            if self.adapt_q:
                err = self._band_err(e)
                g = np.clip(err / np.maximum(self.P[:, 0, 0] + self.r_obs, 1e-9), 0.1, 10.0)
                self.qscale = np.clip(self.qscale * g ** 0.2, 0.1, 10.0)
            self.m = self.m + self._band_vec(K[:, 0]) * e
            self.nu = self.nu + self._band_vec(K[:, 1]) * e
            for l in range(self.L):
                IKH = np.eye(2) - np.outer(K[l], np.array([1.0, 0.0]))
                self.P[l] = IKH @ self.P[l]
        self._n_obs += 1

    @torch.no_grad()
    def predict(self, t) -> torch.Tensor:
        """Forecast at a skipped step: predicted values -> (IDCT) -> [B, Nt, D] (bf16)."""
        j = int(round(float(t)))
        assert self._ready and self.m is not None, "predict() called before any observation"
        self._advance_to(j)
        z = self.m / self.scale
        return self._from_state(z).to(torch.bfloat16)
