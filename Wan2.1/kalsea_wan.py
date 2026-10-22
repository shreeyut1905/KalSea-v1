"""KalSea for the official Wan2.1 text-to-video code (WanT2V): SEA gate + Kalman residual forecast + PCGS.

Methods (bound per instance by `install`; nothing in the `wan` package is modified):
  seacache     SeaCache's Wan gate: block-0 modulated input, SEA filter (beta 3 over F, H, W) on the live sampling sigma,
               accumulated relative L1 distance against the stored input, separate gate state for the cond and the
               uncond call of a step, first and last step always computed. Skipped calls reuse the last block-stack
               residual (x += previous residual).
  kalsea       same gate; a skipped call adds the forecast of a per-element, calibration-free Kalman filter of the
               block-stack residual (one filter per CFG stream, updated with the true residual on computed calls).
  kalsea_pcgs  kalsea + predictor-corrector gap smoothing (PCGS). At a computed step b that closes a gap a+1..b-1:
                 1. smooth the forecasts of the gap with the innovation of the update at b (Rauch-Tung-Striebel gains of
                    the forecaster's own damped-trend model), per stream;
                 2. restore the UniPC state saved after step a and replay steps a+1..b-1 exactly with the corrected
                    guided outputs u' + g (c' - u'), c' / u' = head(patch_embedding(x') + smoothed residual), i.e. with
                    the output layers only (no transformer-block call);
                 3. continue from the corrected latent and scheduler state; the step-b outputs are recomputed from the
                    corrected latent with the observed residuals of step b;
                 4. refresh the gate's stored input of both streams from the corrected latent.
"""
import math
import random
import sys
import types
from contextlib import contextmanager

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from kalsea_forecaster import KalSeaForecaster
from util_seacache import apply_sea_with_scheduler, rel_l1
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

METHODS = ('seacache', 'kalsea', 'kalsea_pcgs')


# ======================================================================================================================
# UniPC scheduler state: FlowUniPCMultistepScheduler.step() mutates exactly these fields (the history lists in place,
# never the tensors themselves), so a shallow copy of the lists is a full snapshot.
# ======================================================================================================================
def sched_state(s):
    return dict(model_outputs=list(s.model_outputs), timestep_list=list(s.timestep_list),
                lower_order_nums=s.lower_order_nums, last_sample=s.last_sample,
                this_order=getattr(s, 'this_order', None), _step_index=s._step_index)


def sched_restore(s, st):
    s.model_outputs = list(st['model_outputs'])
    s.timestep_list = list(st['timestep_list'])
    s.lower_order_nums = st['lower_order_nums']
    s.last_sample = st['last_sample']
    if st['this_order'] is None:
        if hasattr(s, 'this_order'):
            del s.this_order
    else:
        s.this_order = st['this_order']
    s._step_index = st['_step_index']


# ======================================================================================================================
# model pieces shared by the forward and the PCGS replay (same ops as WanModel.forward)
# ======================================================================================================================
def _embed(model, x, seq_len):
    x = [model.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])
    return x, grid_sizes, seq_lens


def _head_out(model, x, e, grid_sizes):
    x = model.head(x, e)
    x = model.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def _gate_input(model, x, e0):
    e_ = (model.blocks[0].modulation + e0).chunk(6, dim=1)
    return model.blocks[0].norm1(x).float() * (1 + e_[1]) + e_[0]


def _sea(model, modulated_inp, grid_sizes, step):
    modulated_inp = modulated_inp.reshape(modulated_inp.shape[0], grid_sizes[0, 0], grid_sizes[0, 1], grid_sizes[0, 2],
                                          modulated_inp.shape[-1])
    modulated_inp = apply_sea_with_scheduler(modulated_inp, model.scheduler, step, dims=(-2, -3, -4), mode="flow",
                                             power_exp=3.0, norm_mode="mean")
    return modulated_inp.reshape(modulated_inp.shape[0], -1, modulated_inp.shape[-1])


@contextmanager
def deterministic_algorithms():
    """The forecaster's two scalar reductions use scatter_add_ (CUDA atomics); deterministic mode fixes their summation
    order, so a generation is bitwise reproducible."""
    prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


# ======================================================================================================================
# PCGS (two CFG streams, exact UniPC multistep replay)
# ======================================================================================================================
class PCGSWan:
    def __init__(self, gate_refresh=True):
        self.gate_refresh = bool(gate_refresh)

    def begin(self, kc):
        self.kc = kc
        self.gap = []               # skipped steps since the last computed step
        self.snap = None            # {'step': a, 'x': x_{a+1}, 'sched': scheduler state after step a}
        self.prior_P = [{}, {}]     # per stream: skipped step -> prior covariance P_i^- of the forecaster
        self.upd = [None, None]     # per stream: (P_b^-, innovation) of the update at the current computed step
        self.n_corr = 0

    def end(self):
        self.gap, self.snap, self.prior_P, self.upd = [], None, [{}, {}], [None, None]

    # ------------------------------------------------------------------------------------- forecaster taps
    def after_predict(self, s, step, fc):
        self.prior_P[s][step] = fc.P.copy()      # P_i^- (predict() advanced the filter to step i)

    def before_update(self, s, step, fc, h):
        """Right before fc.update(step, h): record P_b^- and the innovation in the filter's relative units."""
        if fc._ready and fc.m is not None:
            fc._advance_to(step)                 # the first call of fc.update (idempotent afterwards)
            Z = fc._fwd(fc._to_grid(h))
            self.upd[s] = dict(step=step, Pb=fc.P.copy(), e_rel=Z * fc.scale - fc.m)

    # ------------------------------------------------------------------------------------- sampling-loop hooks
    @torch.no_grad()
    def on_step(self, i, t, latents, c, u, sched):
        """After both forwards of step i; returns the (cond, uncond, latents) to use for the scheduler step."""
        kc = self.kc
        comp = kc.cur['computed']
        assert comp[0] is not None and comp[0] == comp[1], f'step {i}: cond/uncond decisions differ {comp}'
        if not comp[0]:
            self.gap.append(dict(step=i, t=t, x=latents[0], e=kc.cur['e'], pred=[kc.cur['pred'][0], kc.cur['pred'][1]]))
            return c, u, latents
        out = (c, u, latents)
        if self.gap:
            assert self.snap is not None and self.snap['step'] == i - len(self.gap) - 1, 'gap bookkeeping'
            out = self._correct(i, latents, c, u, sched)
        self.gap = []
        self.prior_P = [{}, {}]
        self.upd = [None, None]
        return out

    @torch.no_grad()
    def after_step(self, i, latents, sched):
        if self.kc.cur['computed'][0]:
            self.snap = dict(step=i, x=latents[0], sched=sched_state(sched))

    # ------------------------------------------------------------------------------------- smoother
    def _kalman_gains(self, s, a, b, steps):
        fc = self.kc.fc[s]
        Pb = self.upd[s]['Pb']
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
            Phi = Phi @ A                        # Phi(b, i) = Phi(b, i+1) A_{i+1}
            if i in steps:
                Pi = self.prior_P[s][i]
                gains[i] = (Pi[:, 0, :] * Phi[:, 0, :]).sum(-1) / S     # [P_i Phi^T]_00 / S
        return gains

    def _smooth(self, s, a, b):
        upd = self.upd[s]
        assert upd is not None and upd['step'] == b, f'PCGS: no forecaster update captured at step {b} (stream {s})'
        fc = self.kc.fc[s]
        gains = self._kalman_gains(s, a, b, {rec['step'] for rec in self.gap})
        E = upd['e_rel'] / fc.scale              # innovation, absolute units
        out = []
        for rec in self.gap:
            corr = fc._from_grid(fc._inv(fc._band_vec_to_map(gains[rec['step']]) * E))
            p = rec['pred'][s]
            out.append((p.float() + corr).to(p.dtype))
        return out

    # ------------------------------------------------------------------------------------- correction
    def _correct(self, b, latents, c_b, u_b, sched):
        kc = self.kc
        model = kc.model
        a = self.snap['step']
        steps = [rec['step'] for rec in self.gap]
        assert steps == list(range(a + 1, b)), f'gap bookkeeping: a={a} b={b} steps={steps}'
        r_b = [kc.prev_res[0], kc.prev_res[1]]
        rt = [self._smooth(s, a, b) for s in (0, 1)]
        sched_restore(sched, self.snap['sched'])
        x = self.snap['x']                       # x_{a+1}: exact (step a was computed)
        g = kc.guide_scale
        for k, rec in enumerate(self.gap):
            h, grid_sizes, _ = _embed(model, [x], kc.seq_len)
            hc = h.clone()
            hc += rt[0][k]                       # same in-place add as the forward's skip path
            h += rt[1][k]
            ci = _head_out(model, hc, rec['e'], grid_sizes)[0]
            ui = _head_out(model, h, rec['e'], grid_sizes)[0]
            vi = ui + g * (ci - ui)              # the pipeline's CFG combination
            x = sched.step(vi.unsqueeze(0), rec['t'], x.unsqueeze(0), return_dict=False, generator=kc.seed_g)[0]
            x = x.squeeze(0)
        if torch.equal(x, latents[0]):
            return c_b, u_b, latents
        self.n_corr += 1
        h, grid_sizes, _ = _embed(model, [x], kc.seq_len)
        hf = h.float()                           # block residuals of step b held fixed, input moved to x_b'
        c_new = _head_out(model, hf + r_b[0], kc.cur['e'], grid_sizes)[0]
        u_new = _head_out(model, hf + r_b[1], kc.cur['e'], grid_sizes)[0]
        if self.gate_refresh and kc.ret_steps <= 2 * b < kc.cutoff_steps:
            mod = _sea(model, _gate_input(model, h, kc.cur['e0']), grid_sizes, b)
            kc.prev_e0[0] = mod.clone()
            kc.prev_e0[1] = mod.clone()
        return c_new, u_new, [x]


# ======================================================================================================================
# per-video state
# ======================================================================================================================
class WanCacheController:
    def __init__(self, model, method, delta, sampling_steps=50, kalsea_tau=0.5, kalsea_r_obs=1e-3):
        assert method in METHODS, method
        self.model = model
        self.method = method
        self.thresh = float(delta)
        self.sampling_steps = int(sampling_steps)
        self.num_steps = 2 * self.sampling_steps          # one cond and one uncond call per step
        self.ret_steps, self.cutoff_steps = 2, self.sampling_steps * 2 - 2   # first and last step always computed
        self.tau, self.r_obs = float(kalsea_tau), float(kalsea_r_obs)
        self.pcgs = PCGSWan() if method == 'kalsea_pcgs' else None
        self.reset()

    def reset(self):
        self.cnt = 0
        self.acc = [0, 0]
        self.prev_e0 = [None, None]
        self.prev_res = [None, None]
        self.fc = [None, None]
        self.cur = None
        self.compute_steps = []
        self.seed_g = None
        self.guide_scale = None
        self.seq_len = None

    def _new_forecaster(self, scheduler):
        # calibration-free, one filter per element, time coordinate = step index
        return KalSeaForecaster(calib=None, scheduler=scheduler, num_steps=self.sampling_steps, tau=self.tau,
                                r_obs=self.r_obs, time_mode='step', use_dct=False, adapt_q=True)

    def begin_video(self, scheduler, sampling_steps, seq_len, guide_scale, seed_g):
        assert int(sampling_steps) == self.sampling_steps, 'install() with the sampling steps of the generation'
        self.reset()
        self.seq_len, self.guide_scale, self.seed_g = seq_len, guide_scale, seed_g
        if self.method in ('kalsea', 'kalsea_pcgs'):
            self.fc = [self._new_forecaster(scheduler), self._new_forecaster(scheduler)]
        if self.pcgs is not None:
            self.pcgs.begin(self)

    def begin_step(self, i):
        assert self.cnt == 2 * i, f'forward counter {self.cnt} out of sync with step {i}'
        self.cur = dict(step=i, computed=[None, None], pred=[None, None], e=None, e0=None)

    def end_step(self, i):
        if self.cur['computed'][0]:
            self.compute_steps.append(i)

    def end_video(self):
        if self.pcgs is not None:
            self.pcgs.end()
        self.cur = None
        self.prev_res = [None, None]
        self.prev_e0 = [None, None]
        self.fc = [None, None]


# ======================================================================================================================
# forward: WanModel.forward (T2V) with the SEA gate and the residual source of the method
# ======================================================================================================================
def cache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    kc = self._kc
    assert self.model_type == 't2v' and clip_fea is None and y is None, 'text-to-video only'
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    # embeddings
    x, grid_sizes, seq_lens = _embed(self, x, seq_len)

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    cnt = kc.cnt
    s = cnt % 2                                  # 0 = even call = cond, 1 = odd call = uncond
    step = cnt // 2
    cur = kc.cur
    if s == 0:
        cur['e'], cur['e0'] = e, e0
    modulated_inp = _gate_input(self, x, e0)
    if cnt < kc.ret_steps or cnt >= kc.cutoff_steps:
        should_calc = True
        kc.acc[s] = 0
    else:
        modulated_inp = _sea(self, modulated_inp, grid_sizes, cnt // 2)
        kc.acc[s] += rel_l1(modulated_inp, kc.prev_e0[s])
        if kc.acc[s] < kc.thresh:
            should_calc = False
        else:
            should_calc = True
            kc.acc[s] = 0
    kc.prev_e0[s] = modulated_inp.clone()
    cur['computed'][s] = should_calc

    if not should_calc:
        fc = kc.fc[s]
        if fc is not None and fc.n_obs >= 1:     # kalsea / kalsea_pcgs: forecast residual
            with deterministic_algorithms():
                pred = fc.predict(step).to(dtype=x.dtype, device=x.device)
            if kc.pcgs is not None:
                kc.pcgs.after_predict(s, step, fc)
        else:                                    # seacache: reuse the last residual
            pred = kc.prev_res[s]
        cur['pred'][s] = pred
        x += pred
    else:
        ori_x = x.clone()
        for block in self.blocks:
            x = block(x, **kwargs)
        kc.prev_res[s] = x - ori_x
        fc = kc.fc[s]
        if fc is not None:
            if getattr(fc, 'grid', 0) is None:   # per-element filter: any (Hg, Wg) with Hg * Wg = tokens
                g = [int(v) for v in grid_sizes[0].tolist()]
                fc.grid = (g[0] * g[1], g[2])
            with deterministic_algorithms():
                if kc.pcgs is not None:
                    kc.pcgs.before_update(s, step, fc, kc.prev_res[s])
                fc.update(step, kc.prev_res[s])

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    kc.cnt += 1
    if kc.cnt >= kc.num_steps:
        kc.cnt = 0
    return [u.float() for u in x]


# ======================================================================================================================
# WanT2V.generate with the cache hooks (single GPU)
# ======================================================================================================================
def cache_t2v_generate(self,
                       input_prompt,
                       size=(1280, 720),
                       frame_num=81,
                       shift=5.0,
                       sample_solver='unipc',
                       sampling_steps=50,
                       guide_scale=5.0,
                       n_prompt="",
                       seed=-1,
                       offload_model=True):
    kc = self.model._kc
    # preprocess
    F = frame_num
    target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    # evaluation mode
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        self.model.scheduler = sample_scheduler   # the SEA filter reads the live sampling sigmas
        assert kc.pcgs is None or sample_solver == 'unipc', 'the PCGS replay is implemented for the UniPC solver'
        kc.begin_video(sample_scheduler, sampling_steps, seq_len, guide_scale, seed_g)

        # sample videos
        latents = noise

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        for i, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            timestep = [t]

            timestep = torch.stack(timestep)

            self.model.to(self.device)
            kc.begin_step(i)
            noise_pred_cond = self.model(
                latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = self.model(
                latent_model_input, t=timestep, **arg_null)[0]
            if kc.pcgs is not None:              # PCGS correction at the computed step that closes a gap
                noise_pred_cond, noise_pred_uncond, latents = kc.pcgs.on_step(
                    i, t, latents, noise_pred_cond, noise_pred_uncond, sample_scheduler)

            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]
            if kc.pcgs is not None:              # scheduler snapshot after computed steps
                kc.pcgs.after_step(i, latents, sample_scheduler)
            kc.end_step(i)

        x0 = latents
        kc.end_video()
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents
    del sample_scheduler
    if offload_model:
        import gc
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None


def install(pipe, method, delta, sampling_steps=50, kalsea_tau=0.5, kalsea_r_obs=1e-3):
    """Bind the cached generate / forward to one loaded WanT2V pipeline (instance level). Returns the controller, whose
    `compute_steps` lists the steps of the last video that ran the transformer blocks."""
    kc = WanCacheController(pipe.model, method, delta, sampling_steps, kalsea_tau, kalsea_r_obs)
    pipe.model._kc = kc
    pipe.generate = types.MethodType(cache_t2v_generate, pipe)
    pipe.model.forward = types.MethodType(cache_forward, pipe.model)
    return kc
