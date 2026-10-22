#!/usr/bin/env python3
"""KalSea for Wan2.1 text-to-video. Copy the .py files of this folder into the root of the official Wan2.1 repository
and run from there (single GPU):

    python kalsea_generate.py --ckpt_dir ./Wan2.1-T2V-1.3B --prompt "A cat walks on the grass." --delta 0.2
    python kalsea_generate.py ... --no_pcgs      # KalSea forecast only
    python kalsea_generate.py ... --reuse        # SeaCache residual reuse
    python kalsea_generate.py ... --no_cache     # official sampler, every step computed
"""
import argparse
import json
import os
import re
import time
import warnings

warnings.filterwarnings('ignore')

import torch  # noqa: E402

import wan  # noqa: E402
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS  # noqa: E402
from wan.utils.utils import cache_video  # noqa: E402

import kalsea_wan  # noqa: E402


def read_prompts(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def safe_name(text, n=80):
    return re.sub(r'[^0-9A-Za-z._-]', '', re.sub(r'\s+', '_', text.strip()))[:n] or 'video'


def method_name(args):
    if args.no_cache:
        return 'reference'
    if args.reuse:
        return 'seacache'
    return 'kalsea' if args.no_pcgs else 'kalsea_pcgs'


def parse_args():
    p = argparse.ArgumentParser(description='Wan2.1 text-to-video with KalSea (SEA gate + Kalman residual forecast + PCGS).')
    p.add_argument('--task', default='t2v-1.3B', choices=['t2v-1.3B', 't2v-14B'])
    p.add_argument('--size', default='832*480', choices=list(SIZE_CONFIGS.keys()), help='width*height')
    p.add_argument('--frame_num', type=int, default=65, help='number of frames (4n+1)')
    p.add_argument('--ckpt_dir', required=True, help='checkpoint directory, e.g. ./Wan2.1-T2V-1.3B')
    p.add_argument('--prompt', default=None, help='text prompt')
    p.add_argument('--prompt_file', default=None, help='text file with one prompt per line')
    p.add_argument('--output_dir', default='./kalsea_outputs')
    p.add_argument('--base_seed', type=int, default=42, help='seed (the same for every prompt)')
    p.add_argument('--sample_solver', default='unipc', choices=['unipc', 'dpm++'], help='PCGS needs unipc')
    p.add_argument('--sample_steps', type=int, default=50)
    p.add_argument('--sample_shift', type=float, default=5.0)
    p.add_argument('--sample_guide_scale', type=float, default=5.0)
    p.add_argument('--offload_model', action='store_true', help='move models to the CPU between uses (less memory)')
    p.add_argument('--t5_cpu', action='store_true', help='run the T5 text encoder on the CPU')
    p.add_argument('--delta', '--seacache_thresh', dest='delta', type=float, default=0.2,
                   help='SEA gate threshold (Wan2.1-T2V-1.3B, 50 steps: 0.2 -> 24 full steps, 0.35 -> 17)')
    p.add_argument('--no_pcgs', action='store_true', help='KalSea without predictor-corrector gap smoothing')
    p.add_argument('--reuse', action='store_true', help='SeaCache: reuse the last residual (no forecaster, no PCGS)')
    p.add_argument('--no_cache', action='store_true', help='official sampler, every step computed')
    p.add_argument('--kalsea_tau', type=float, default=0.5, help='damping time constant of the derivative')
    p.add_argument('--kalsea_r_obs', type=float, default=1e-3, help='observation noise (relative units)')
    args = p.parse_args()
    if (args.prompt is None) == (args.prompt_file is None):
        p.error('give exactly one of --prompt / --prompt_file')
    if args.size not in SUPPORTED_SIZES[args.task]:
        p.error(f'--size {args.size} is not supported by {args.task}: {SUPPORTED_SIZES[args.task]}')
    if method_name(args) == 'kalsea_pcgs' and args.sample_solver != 'unipc':
        p.error('PCGS replays the UniPC solver: use --sample_solver unipc, or --no_pcgs')
    return args


def main():
    args = parse_args()
    prompts = read_prompts(args.prompt_file) if args.prompt_file else [args.prompt]
    method = method_name(args)
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = WAN_CONFIGS[args.task]

    pipe = wan.WanT2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
                      use_usp=False, t5_cpu=args.t5_cpu)
    kc = None
    if method != 'reference':
        kc = kalsea_wan.install(pipe, method, args.delta, sampling_steps=args.sample_steps,
                                kalsea_tau=args.kalsea_tau, kalsea_r_obs=args.kalsea_r_obs)
    print(f'method={method} delta={None if kc is None else args.delta} task={args.task} size={args.size} '
          f'frames={args.frame_num} steps={args.sample_steps} prompts={len(prompts)}', flush=True)

    stats_path = os.path.join(args.output_dir, 'stats.jsonl')
    for idx, prompt in enumerate(prompts):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        video = pipe.generate(
            prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        steps = list(range(args.sample_steps)) if kc is None else list(kc.compute_steps)
        fn = f'{idx:04d}-{safe_name(prompt)}.mp4'
        cache_video(tensor=video[None], save_file=os.path.join(args.output_dir, fn), fps=cfg.sample_fps, nrow=1,
                    normalize=True, value_range=(-1, 1))
        rec = dict(idx=idx, file=fn, prompt=prompt, seed=args.base_seed, method=method,
                   delta=None if kc is None else args.delta, latency=round(latency, 3), nfe=len(steps),
                   compute_steps=steps)
        with open(stats_path, 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'[{idx + 1}/{len(prompts)}] {fn} | full steps: {len(steps)}/{args.sample_steps} | {latency:.1f} s',
              flush=True)


if __name__ == '__main__':
    main()
