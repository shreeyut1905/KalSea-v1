#!/usr/bin/env python3
"""KalSea inference for MiniMax-H3 (joint video + audio generation), on top of diffusers' MiniMax-H3 modular pipeline.

KalSea = SeaCache's spectrally aligned skip gate (*when* to skip a transformer evaluation) + a Kalman residual forecaster
(*what* to write on a skipped step).  MiniMax-H3 runs ONE 50-block transformer over one packed sequence
`[text | audio | video]` (t2va) and predicts the video and the audio velocity from it, so:

  * gate      : first block's AdaLN-modulated input of the video rows, laid out on their (frames, h, w) grid and
                filtered with the SEA Wiener filter; skip while the accumulated relative L1 change stays below delta.
  * residual  : (block-stack output - block-stack input) of the video rows and of the audio rows, i.e. of the only
                rows the output heads read.  Each stream has its own per-element Kalman forecaster (no grid, no
                calibration needed).
  * skipped step : the 50 blocks are bypassed; input embeddings + forecast residual go through the output norm / heads.

The patch is installed on the *class* `forward` (like ReSPect does for FLUX) so accelerate / ComponentsManager offload
hooks, which wrap the instance's forward, keep working.  Without a KalSea state attached (or with `--no_cache`) the
original diffusers forward runs unchanged.

Text-to-video(+audio) only (`t2va` workflow): the keyframe (`fl2va`) and reference (`ref2va`) workflows are not supported.
"""
import argparse
import json
import os
import re
import time
from types import SimpleNamespace

import numpy as np
import torch
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers import transformer_minimax_h3 as h3
from diffusers.utils import apply_lora_scale

from kalsea_forecaster import KalSeaForecaster
from util_seacache import apply_sea_from_ab, rel_l1

MiniMaxH3Transformer3DModel = h3.MiniMaxH3Transformer3DModel
MODALITY_NUM = h3.MINIMAX_H3_MODALITY_NUM

# the undecorated original forward (`apply_lora_scale` uses functools.wraps); used when KalSea is switched off
_ORIG_FORWARD = getattr(MiniMaxH3Transformer3DModel.forward, "__wrapped__", MiniMaxH3Transformer3DModel.forward)


# ----------------------------------------------------------------------------------------------
# per-generation state
# ----------------------------------------------------------------------------------------------
class KalSeaH3State:
    """Gate + forecaster state of one generation, attached to the transformer as `transformer.kalsea`."""

    def __init__(self, thresh=0.3, warmup=3, tau=0.5, r_obs=1e-3, q_mult=1.0, adapt_q=True, num_steps=None,
                 scheduler=None, enabled=True, sea_power_exp=3.0):
        self.enabled = enabled
        self.thresh, self.warmup = float(thresh), int(warmup)
        self.sea_power_exp = float(sea_power_exp)   # SEA prior |f|^-beta: SeaCache uses beta=2 for images, 3 for videos
        self.tau, self.r_obs, self.q_mult, self.adapt_q = tau, r_obs, q_mult, adapt_q
        self.num_steps_arg = None if num_steps is None else int(num_steps)
        self.scheduler = scheduler            # video scheduler; its `timesteps` give the number of model evaluations
        self.reset()

    def reset(self):
        self.cnt = 0
        self.acc = 0.0
        self.prev_mod = None
        self.fc_v = self.fc_a = None
        self.nfe = 0
        self.compute_steps = []
        self.layout = None
        self.num_steps = self.num_steps_arg

    # -- called on the first evaluation of a generation ------------------------------------------------------------
    def begin(self, video_indices, audio_indices, position_ids):
        sch = self.scheduler
        if sch is not None and getattr(sch, "timesteps", None) is not None:
            self.num_steps = int(sch.timesteps.numel())        # = num_inference_steps - 1 for MiniMax-H3
        if self.num_steps is None:
            raise RuntimeError("KalSea needs the number of model evaluations: pass `num_steps` or a scheduler.")
        self.acc, self.prev_mod, self.nfe, self.compute_steps = 0.0, None, 0, []

        # t2va layout: [text | audio | video]; every video row is a generated row
        if video_indices.numel() == 0 or audio_indices.numel() == 0 or int(video_indices.min()) <= int(audio_indices.max()):
            raise RuntimeError("KalSea supports the text-to-video (t2va) layout [text | audio | video] only; "
                               "keyframe / reference conditioning rows were found.")
        target = video_indices
        rows = torch.cat([target, audio_indices])

        # (frames, h, w) grid of the video rows from their rotary coordinates; verified to be frame-major row-major
        pos = position_ids[target]
        uniq = [torch.unique(pos[:, k]) for k in range(3)]
        F, Hp, Wp = (u.numel() for u in uniq)
        grid = None
        if F * Hp * Wp == target.numel():
            ranks = [torch.searchsorted(uniq[k], pos[:, k]) for k in range(3)]
            lin = (ranks[0] * Hp + ranks[1]) * Wp + ranks[2]
            if bool((lin == torch.arange(target.numel(), device=lin.device)).all()):
                grid = (F, Hp, Wp)
        self.layout = SimpleNamespace(target=target, rows=rows, nv=int(target.numel()), na=int(audio_indices.numel()),
                                      grid=grid)
        mk = lambda: KalSeaForecaster(calib=None, num_steps=self.num_steps, tau=self.tau, r_obs=self.r_obs,
                                      q_mult=self.q_mult, time_mode="step", use_dct=False, adapt_q=self.adapt_q)
        self.fc_v, self.fc_a = mk(), mk()   # video and audio residuals have different statistics

    def finish(self):
        """After the last evaluation: free the filter state (it is ~2 fp32 copies of the residual) before decoding."""
        self.cnt = 0
        self.fc_v = self.fc_a = None
        self.prev_mod = None


# ----------------------------------------------------------------------------------------------
# patched forward
# ----------------------------------------------------------------------------------------------
def _gate_input(self, st, video_embeds, temb, adaln_indices, dtype):
    """First-block AdaLN-modulated input of the video rows (the input of the SEA gate)."""
    lay = st.layout
    blk = self.transformer_blocks[0]
    shift_msa, scale_msa = blk.adaln_proj(temb)[:2]
    idx = adaln_indices.index_select(0, lay.target)
    x = video_embeds.to(dtype)
    mod = blk.norm1(x) * (1.0 + scale_msa.index_select(0, idx)) + shift_msa.index_select(0, idx)
    return mod


@apply_lora_scale("attention_kwargs")
def kalsea_forward(
    self,
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    timestep,
    timestep_indices,
    token_tags,
    position_ids,
    video_indices,
    audio_indices,
    text_indices,
    attention_kwargs=None,
    return_dict=True,
):
    st = getattr(self, "kalsea", None)
    if st is None or not st.enabled:
        return _ORIG_FORWARD(
            self, hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices, token_tags,
            position_ids, video_indices, audio_indices, text_indices, attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

    B, S = hidden_states.shape[0], position_ids.shape[0]
    pack_dtype = get_parameter_dtype(self.context_embedder)

    # 1. input embeddings of the rows the output heads read (video + audio); same casts as the original forward
    video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
    audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
    temb = self.time_embedder(self.time_proj(timestep).to(get_parameter_dtype(self.time_embedder)))
    adaln_indices = timestep_indices * MODALITY_NUM + token_tags

    if st.cnt >= (st.num_steps or 0):   # more evaluations than expected (e.g. a second generation w/o reset)
        st.cnt = 0
    if st.cnt == 0:
        st.begin(video_indices, audio_indices, position_ids)
    j, N, lay = st.cnt, st.num_steps, st.layout

    # 2. SEA gate (always compute the warm-up steps and the last step)
    should_calc = True
    if j < N - 1:
        mod = _gate_input(self, st, video_embeds, temb, adaln_indices, pack_dtype)
        t_v = float(timestep[timestep_indices[lay.target[0]]])          # H3 convention: t = 1 - sigma, 1 = clean
        a, b = min(max(t_v, 1e-6), 1 - 1e-6), min(max(1.0 - t_v, 1e-6), 1 - 1e-6)
        if lay.grid is not None:
            x = mod.reshape(B, *lay.grid, mod.shape[-1])
            f = apply_sea_from_ab(x, a, b, power_exp=st.sea_power_exp, dims=(-4, -3, -2), norm_mode="mean")
        else:  # unexpected grid: 1-D filter along the token axis
            f = apply_sea_from_ab(mod, a, b, power_exp=st.sea_power_exp, dims=(-2,), norm_mode="mean")
        if j >= st.warmup and st.prev_mod is not None:
            st.acc += rel_l1(f, st.prev_mod)
            if st.acc < st.thresh:
                should_calc = False
            else:
                st.acc = 0.0
        else:
            st.acc = 0.0
        st.prev_mod = f

    rows_in = torch.cat([video_embeds, audio_embeds], dim=1).to(pack_dtype)
    if should_calc:
        # 3a. full evaluation: pack text / video / audio rows and run the 50 blocks
        text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
        text_embeds = self.token_refiner(text_embeds)
        hidden = text_embeds.new_zeros((B, S, text_embeds.shape[-1]))
        hidden = hidden.index_copy(1, text_indices, text_embeds)
        hidden = hidden.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden = hidden.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))
        rows_in = hidden.index_select(1, lay.rows)
        rotary_emb = self.rope(position_ids)
        for block in self.transformer_blocks:
            hidden = block(hidden, temb, adaln_indices, rotary_emb)
        if j < N - 1:  # the last evaluation is never reused
            res = hidden.index_select(1, lay.rows) - rows_in
            st.fc_v.update(j, res[:, : lay.nv])
            st.fc_a.update(j, res[:, lay.nv:])
        st.nfe += 1
        st.compute_steps.append(j)
    else:
        # 3b. skipped step: input embeddings + forecast residual; text rows are never read by the heads
        pred = torch.cat([st.fc_v.predict(j), st.fc_a.predict(j)], dim=1).to(pack_dtype)
        hidden = rows_in.new_zeros((B, S, rows_in.shape[-1]))
        hidden = hidden.index_copy(1, video_indices, video_embeds.to(pack_dtype))
        hidden = hidden.index_copy(1, audio_indices, audio_embeds.to(pack_dtype))
        hidden = hidden.index_copy(1, lay.rows, rows_in + pred)

    # 4. output norm + heads: identical to the original forward
    hidden = self.norm_out(hidden, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
    video_output = self.proj_out(hidden).index_select(1, video_indices)
    audio_output = self.audio_proj_out(hidden).index_select(1, audio_indices)

    st.cnt += 1
    if st.cnt >= N:
        st.finish()

    if not return_dict:
        return (video_output, audio_output)
    return h3.MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


MiniMaxH3Transformer3DModel.forward = kalsea_forward


def attach_state(transformer, scheduler, args):
    """Fresh gate / forecaster state for the next generation (or `enabled=False` for the plain reference run)."""
    transformer.kalsea = KalSeaH3State(
        thresh=args.seacache_thresh, warmup=args.warmup_steps, tau=args.kalsea_tau, r_obs=args.kalsea_r_obs,
        q_mult=args.kalsea_q_mult, adapt_q=not args.kalsea_no_adapt_q, num_steps=args.num_inference_steps - 1,
        scheduler=scheduler, enabled=not args.no_cache, sea_power_exp=args.sea_power_exp,
    )
    return transformer.kalsea


# ----------------------------------------------------------------------------------------------
# pipeline loading (the recipes of the diffusers MiniMax-H3 documentation)
# ----------------------------------------------------------------------------------------------
def load_pipeline(args):
    from diffusers import ComponentsManager, ModularPipeline

    repo = args.model_id
    dtype = torch.bfloat16
    if args.offload == "two_gpu":
        # text encoder (62 GB) on cuda:1, everything else on cuda:0 -- nothing is evicted to host memory
        workflow = ModularPipeline.from_pretrained(repo).blocks.get_workflow("t2va")
        text_manager = ComponentsManager()
        text_manager.enable_auto_cpu_offload(device="cuda:1")
        conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(repo, components_manager=text_manager)
        conditioner.load_components(dtype=dtype)
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda:0")
        rest = workflow.init_pipeline(repo, components_manager=manager)
        rest.load_components(dtype=dtype)
        return conditioner, rest
    if args.offload == "auto":
        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(repo, components_manager=manager)
        pipe.load_components(workflow="t2va", dtype=dtype)
        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=args.memory_reserve_margin)
    else:  # "none": all components resident on one large GPU (transformer 62 GB + text encoder 62 GB in bf16)
        pipe = ModularPipeline.from_pretrained(repo)
        pipe.load_components(workflow="t2va", dtype=dtype)
        pipe.to("cuda")
    return None, pipe


# ----------------------------------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------------------------------
def read_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def safe_filename(name):
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^0-9A-Za-z._-]", "", name)
    return name or "video"


def main():
    p = argparse.ArgumentParser(description="MiniMax-H3 text-to-video + KalSea (SEA gate + Kalman residual forecaster).")
    p.add_argument("--prompt", type=str, default="A red fox trotting through a snowy pine forest, snow crunching underfoot")
    p.add_argument("--prompt_file", type=str, default=None, help="one prompt per line")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_id", type=str, default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--num_frames", type=int, default=124, help="24 fps; snapped up to 17*n+5; 5..15 seconds")
    p.add_argument("--height", type=int, default=None, help="multiple of 32 (default: the 16:9 768p canvas)")
    p.add_argument("--width", type=int, default=None, help="multiple of 32")
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="sigma grid points incl. the terminal 0, i.e. N-1 model evaluations without caching")
    p.add_argument("--seed", type=int, default=42, help="base seed (+ prompt index)")
    p.add_argument("--offload", type=str, default="auto", choices=["auto", "none", "two_gpu"],
                   help="auto: ComponentsManager offload (one GPU); none: all on one large GPU; two_gpu: text encoder on cuda:1")
    p.add_argument("--memory_reserve_margin", type=str, default="12GB")
    p.add_argument("--attention_backend", type=str, default=None,
                   help="optional diffusers attention backend for the transformer, e.g. _flash_3_hub (Hopper)")

    g = p.add_argument_group("KalSea")
    g.add_argument("--no_cache", action="store_true", help="plain MiniMax-H3, no gate / forecaster (reference run)")
    g.add_argument("--seacache_thresh", type=float, default=0.3,
                   help="SEA gate threshold delta. NOT calibrated for MiniMax-H3 -- check the printed NFE and tune it.")
    g.add_argument("--sea_power_exp", type=float, default=3.0,
                   help="exponent beta of the SEA filter's power-law prior; SeaCache uses 2 for images and 3 for videos")
    g.add_argument("--warmup_steps", type=int, default=3, help="first evaluations that are always computed")
    g.add_argument("--kalsea_tau", type=float, default=0.5, help="derivative damping time constant")
    g.add_argument("--kalsea_r_obs", type=float, default=1e-3, help="observation noise (relative units)")
    g.add_argument("--kalsea_q_mult", type=float, default=1.0, help="multiplier on the process noise")
    g.add_argument("--kalsea_no_adapt_q", action="store_true", help="disable the online process-noise adaptation")
    args = p.parse_args()

    from diffusers.utils.export_utils import encode_video

    prompts = read_prompts(args.prompt_file) if args.prompt_file else [args.prompt]
    os.makedirs(args.output_dir, exist_ok=True)
    conditioner, pipe = load_pipeline(args)
    tr = pipe.transformer
    if args.attention_backend:
        tr.set_attention_backend(args.attention_backend)

    for i, prompt in enumerate(prompts):
        st = attach_state(tr, pipe.scheduler, args)
        gen = torch.Generator().manual_seed(args.seed + i)
        call = dict(num_frames=args.num_frames, num_inference_steps=args.num_inference_steps, generator=gen,
                    output=["videos", "audio", "sampling_rate"])
        if args.height and args.width:
            call.update(height=args.height, width=args.width)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if conditioner is not None:   # two_gpu recipe
            results = pipe(state=conditioner(prompt=prompt), **call)
        else:
            results = pipe(prompt=prompt, **call)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        name = f"KalSea_{i:05d}-{safe_filename(prompt)[:80]}"
        encode_video(results["videos"][0], fps=24, output_path=os.path.join(args.output_dir, name + ".mp4"),
                     audio=results["audio"][0], audio_sample_rate=results["sampling_rate"])
        total = args.num_inference_steps - 1
        rec = dict(idx=i, file=name + ".mp4", prompt=prompt, seed=args.seed + i, latency=latency,
                   nfe=(total if args.no_cache else st.nfe), total_evaluations=total,
                   compute_steps=(None if args.no_cache else st.compute_steps),
                   seacache_thresh=(None if args.no_cache else args.seacache_thresh))
        with open(os.path.join(args.output_dir, "stats.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i + 1}/{len(prompts)}] {name}.mp4  latency {latency:.1f}s  "
              f"evaluations {rec['nfe']}/{total}" + ("" if args.no_cache else f"  computed at steps {st.compute_steps}"))


if __name__ == "__main__":
    main()
