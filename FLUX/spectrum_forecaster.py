"""Spectrum Chebyshev + discrete-Taylor feature forecaster.

Faithful reimplementation of Spectrum's `src/utils/basis_utils.py` so we can
forecast transformer features instead of copying a cached residual.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

DTYPE = torch.bfloat16


def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    shape = x.shape
    return x.reshape(1, -1), shape


def _unflatten(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return x_flat.reshape(shape)


def _chunk_ranges(n: int, n_chunks: int):
    """Yield (start, end) index ranges splitting n items into n_chunks parts."""
    base, rem = divmod(n, n_chunks)
    s = 0
    for i in range(n_chunks):
        e = s + base + (1 if i < rem else 0)
        if e > s:
            yield s, e
        s = e


class BaseForecaster(nn.Module):
    def __init__(
        self,
        M: int = 3,
        K: int = 10,
        lam: float = 1e-3,
        device: Optional[torch.device] = None,
        feature_shape=None,
        t_min: float = 0.0,
        t_max: float = 50.0,
        fit_chunks: int = 1,
    ):
        super().__init__()
        assert K >= M + 2, "K should exceed basis size for stability"
        assert fit_chunks >= 1, "fit_chunks must be >= 1"
        self.fit_chunks = int(fit_chunks)
        self.M = M
        self.K = K
        self.lam = lam
        self.register_buffer("t_buf", torch.empty(0))
        self._H_buf: Optional[torch.Tensor] = None
        self._shape: Optional[torch.Size] = None
        self._coef: Optional[torch.Tensor] = None
        self._XtX_fac: Optional[torch.Tensor] = None
        self._tau_cache: Optional[torch.Tensor] = None
        self._X_cache: Optional[torch.Tensor] = None
        self._last_delta_norm: Optional[torch.Tensor] = None
        self.device_ref = device
        self.feature_shape = feature_shape
        self.t_min_val = float(t_min)
        self.t_max_val = float(t_max)

    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        """Map scalar times to tau in [-1, 1] using a fixed (t_min, t_max) window."""
        assert self.t_buf.numel() >= 1
        t_min = torch.ones(1, device=t.device, dtype=t.dtype) * self.t_min_val
        t_max = torch.ones(1, device=t.device, dtype=t.dtype) * self.t_max_val
        if torch.isclose(t_max, t_min):
            return torch.zeros_like(t)
        mid = 0.5 * (t_min + t_max)
        rng = t_max - t_min
        return (t - mid) * 2.0 / rng

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def P(self) -> int:
        raise NotImplementedError
    def update(self, t: float | torch.Tensor, h: torch.Tensor) -> None:
        device = self.device_ref or h.device
        t = torch.as_tensor(t, dtype=DTYPE, device=device)
        h_flat, shape = _flatten(h)
        h_flat = h_flat.to(device=device, dtype=DTYPE)
        if self._shape is None:
            self._shape = shape
        else:
            assert shape == self._shape, "Feature shape must remain constant"

        if self.t_buf.numel() == 0:
            self.t_buf = t[None]
            self._H_buf = h_flat
        else:
            delta = h_flat - self._H_buf[-1]
            self._last_delta_norm = delta.norm(p=2)
            self.t_buf = torch.cat([self.t_buf, t[None]], dim=0)
            self._H_buf = torch.cat([self._H_buf, h_flat], dim=0)
            if self.t_buf.numel() > self.K:
                self.t_buf = self.t_buf[-self.K :]
                self._H_buf = self._H_buf[-self.K :]
        self._coef = None
        self._XtX_fac = None
        self._tau_cache = None
        self._X_cache = None

    def last_delta(self) -> torch.Tensor:
        if self._last_delta_norm is None:
            return torch.tensor(
                1e-6,
                device=self.t_buf.device if self.t_buf.numel() else "cpu",
            )
        return self._last_delta_norm

    def ready(self) -> bool:
        return True

    @property
    def n_obs(self) -> int:
        return int(self.t_buf.numel()) if self.t_buf is not None else 0

    def _fit_if_needed(self) -> None:
        if self._coef is not None:
            return
        assert self.ready()
        taus = self._taus(self.t_buf)
        X = self._build_design(taus).to(torch.float32)
        H = self._H_buf
        _, P = X.shape
        assert P == self.P
        lamI = self.lam * torch.eye(P, device=X.device, dtype=X.dtype)
        Xt = X.transpose(0, 1)
        XtX = Xt @ X + lamI
        try:
            L = torch.linalg.cholesky(XtX.to(torch.float32))
        except Exception:
            jitter = 1e-6 * XtX.diag().mean()
            L = torch.linalg.cholesky(XtX + jitter * torch.eye(P, device=X.device))

        if self.fit_chunks <= 1:
            XtH = Xt @ H.to(torch.float32)
            C = torch.cholesky_solve(XtH.to(torch.float32), L).to(DTYPE)
        else:
            # Chunked solve over the feature dim: each column-block is an
            # independent ridge solve with the same Cholesky factor, so the
            # result matches the unchunked fit while peak memory drops by
            # ~fit_chunks (the full fp32 upcast of H never materializes).
            D = H.shape[1]
            n_chunks = min(self.fit_chunks, D)
            blocks = []
            for s, e in _chunk_ranges(D, n_chunks):
                Hb = H[:, s:e].to(torch.float32)
                XtHb = Xt @ Hb
                Cb = torch.cholesky_solve(XtHb.to(torch.float32), L).to(DTYPE)
                blocks.append(Cb)
                del Hb, XtHb, Cb
            C = torch.cat(blocks, dim=1)
            del blocks
        self._coef = C
        self._XtX_fac = L
        self._tau_cache = taus
        self._X_cache = X.to(DTYPE)

    @torch.no_grad()
    def predict(self, t_star: float | torch.Tensor) -> torch.Tensor:
        assert self._shape is not None
        device = self.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=DTYPE, device=device)
        self._fit_if_needed()
        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star[None])
        h_flat = x_star @ self._coef
        return _unflatten(h_flat, self._shape)

class ChebyshevForecaster(BaseForecaster):
    """Chebyshev T-polynomials on tau in [-1, 1]: T_0..T_M via recurrence."""

    def __init__(
        self,
        M: int = 4,
        K: int = 10,
        lam: float = 1e-3,
        device: Optional[torch.device] = None,
        feature_shape=None,
        t_min: float = 0.0,
        t_max: float = 50.0,
        fit_chunks: int = 1,
    ):
        super().__init__(M, K, lam, device, feature_shape, t_min=t_min, t_max=t_max,
                         fit_chunks=fit_chunks)

    @property
    def P(self) -> int:
        return self.M + 1

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1)
        k = taus.shape[0]
        t0 = torch.ones((k, 1), device=taus.device, dtype=taus.dtype)
        if self.M == 0:
            return t0
        t1 = taus
        cols = [t0, t1]
        for _ in range(2, self.M + 1):
            tm = 2 * taus * cols[-1] - cols[-2]
            cols.append(tm)
        return torch.cat(cols[: self.M + 1], dim=1)


class Spectrum(nn.Module):
    def __init__(
        self,
        cheb_like,
        taylor_order: int = 1,
        enable_blend: bool = True,
        prefer: str = "cheb",
        w: float = None,
        alpha: float = 6.0,
        ema_beta: float = 0.9,
    ):
        super().__init__()
        assert taylor_order in (1, 2, 3)
        assert prefer in ("auto", "taylor", "cheb")
        self.cheb = cheb_like
        self.taylor_order = taylor_order
        self.enable_blend = enable_blend
        self.prefer = prefer
        self.alpha = alpha
        self.ema_beta = ema_beta
        self._delta_ref = None
        self.w = w

    @torch.no_grad()
    def _local_taylor_discrete(self, t_star: torch.Tensor) -> torch.Tensor:
        H = self.cheb._H_buf
        t = self.cheb.t_buf
        h_i = H[-1]
        t_i = t[-1]
        if t.numel() < 2:
            return h_i.clone()
        h_im1 = H[-2]
        t_im1 = t[-2]
        dh1 = h_i - h_im1
        dt_last = (t_i - t_im1).clamp_min(1e-8)
        k = ((t_star - t_i) / dt_last).to(h_i.dtype)
        out = h_i + k * dh1
        if self.taylor_order >= 2 and t.numel() >= 3:
            h_im2 = H[-3]
            d2 = h_i - 2 * h_im1 + h_im2
            out = out + 0.5 * k * (k - 1.0) * d2
        if self.taylor_order >= 3 and t.numel() >= 4:
            h_im3 = H[-4]
            d3 = h_i - 3 * h_im1 + 3 * h_im2 - h_im3
            out = out + (k * (k - 1.0) * (k - 2.0) / 6.0) * d3
        return out

    @torch.no_grad()
    def predict(self, t_star: float | torch.Tensor, return_weight: bool = False):
        device = self.cheb.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=DTYPE, device=device)
        h_cheb = self.cheb.predict(t_star)
        h_taylor = self._local_taylor_discrete(t_star)
        assert self.w is not None
        w = self.w
        h_mix = (1 - w) * h_taylor + w * h_cheb
        return (h_mix, float(w)) if return_weight else h_mix

    def update_w(self, new_w: float):
        self.w = new_w

    def update(self, t, h):
        return self.cheb.update(t, h)

    def last_delta(self):
        return self.cheb.last_delta()

    def ready(self):
        return self.cheb.ready()

    @property
    def n_obs(self) -> int:
        return self.cheb.n_obs


class SpectrumResidualForecaster:
    """Forecast a feature tensor with Spectrum (Chebyshev + discrete Taylor).

    Matches Spectrum defaults: M=4, K=100, lam=0.1, w=0.5, taylor_order=1.
    Chebyshev is mixed in only after enough observations so a 1-point ridge
    fit cannot pollute early skipped steps (those fall back to Taylor, which
    is last-value copy with 1 obs and linear extrapolation with 2+).
    """

    def __init__(
        self,
        m: int = 4,
        k: int = 100,
        lam: float = 0.1,
        w: float = 0.5,
        taylor_order: int = 1,
        device=None,
        t_max: float = 50.0,
        min_cheb_obs: int = 4,
        fit_chunks: int = 1,
    ):
        cheb = ChebyshevForecaster(
            M=m,
            K=k,
            lam=lam,
            device=device,
            t_max=t_max,
            fit_chunks=fit_chunks,
        )
        self.spectrum = Spectrum(
            cheb,
            taylor_order=taylor_order,
            enable_blend=True,
            prefer="cheb",
            w=w,
        )
        self.min_cheb_obs = int(min_cheb_obs)
        self._feat_shape = None

    @property
    def n_obs(self) -> int:
        return self.spectrum.n_obs

    def update(self, t, h: torch.Tensor) -> None:
        self._feat_shape = tuple(h.shape)
        self.spectrum.update(t, h.reshape(-1))

    @torch.no_grad()
    def predict(self, t) -> torch.Tensor:
        device = self.spectrum.cheb.t_buf.device
        t_star = torch.as_tensor(t, dtype=DTYPE, device=device)
        h_taylor = self.spectrum._local_taylor_discrete(t_star)
        if self.n_obs >= self.min_cheb_obs:
            h_cheb = self.spectrum.cheb.predict(t_star)
            w = float(self.spectrum.w)
            out = (1.0 - w) * h_taylor + w * h_cheb
        else:
            out = h_taylor
        return out.reshape(self._feat_shape).to(dtype=DTYPE)

