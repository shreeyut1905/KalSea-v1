#!/usr/bin/env python3
"""KalSea for HunyuanVideo text-to-video. Copy the .py files of this folder into the root of the official HunyuanVideo
repository and run from there (single GPU). All flags of the official sample_video.py are accepted; the KalSea options
are listed in kalsea_parser().

    python kalsea_generate.py --video-size 480 640 --video-length 65 --infer-steps 50 --flow-reverse --seed 42 \
        --prompt "A cat walks on the grass, realistic style." --save-path ./kalsea_results --delta 0.2575
    python kalsea_generate.py ... --no_pcgs        # KalSea forecast only
    python kalsea_generate.py ... --reuse          # SeaCache residual reuse
    python kalsea_generate.py ... --no_cache       # official sampler, every step computed
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from loguru import logger

from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.utils.file_utils import save_videos_grid

import kalsea_hunyuan


def kalsea_parser():
    """KalSea options (both spellings, e.g. --no_pcgs / --no-pcgs); everything else goes to hyvideo's parser."""
    p = argparse.ArgumentParser(description="KalSea options (all other flags are those of hyvideo)", add_help=False,
                                allow_abbrev=False)
    p.add_argument("--delta", "--seacache_thresh", "--seacache-thresh", dest="delta", type=float, default=0.2575,
                   help="SEA gate threshold (480x640, 65 frames, 50 steps: 0.2575 -> 24 full steps, 0.48 -> 16)")
    p.add_argument("--no_pcgs", "--no-pcgs", dest="no_pcgs", action="store_true",
                   help="KalSea without predictor-corrector gap smoothing")
    p.add_argument("--reuse", action="store_true", help="SeaCache: reuse the last residual (no forecaster, no PCGS)")
    p.add_argument("--no_cache", "--no-cache", dest="no_cache", action="store_true",
                   help="official sampler, every step computed")
    p.add_argument("--stale_sigma", "--stale-sigma", dest="stale_sigma", action="store_true",
                   help="SEA filter on the load-time scheduler, as SeaCache's released script (default: live sigma)")
    p.add_argument("--prompt_file", "--prompt-file", dest="prompt_file", default=None,
                   help="text file with one prompt per line (instead of --prompt)")
    p.add_argument("--kalsea_tau", "--kalsea-tau", dest="kalsea_tau", type=float, default=0.5,
                   help="damping time constant of the derivative")
    p.add_argument("--kalsea_r_obs", "--kalsea-r-obs", dest="kalsea_r_obs", type=float, default=1e-3,
                   help="observation noise (relative units)")
    return p


def method_name(kargs):
    if kargs.no_cache:
        return "reference"
    if kargs.reuse:
        return "seacache"
    return "kalsea" if kargs.no_pcgs else "kalsea_pcgs"


def safe_name(text, n=80):
    return re.sub(r"[^0-9A-Za-z._-]", "", re.sub(r"\s+", "_", text.strip()))[:n] or "video"


def main():
    kp = kalsea_parser()
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        kp.print_help()                            # then hyvideo's parser prints its own options and exits
        print()
    kargs, rest = kp.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    args = parse_args()
    if kargs.prompt_file:
        with open(kargs.prompt_file, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        raise SystemExit("give --prompt or --prompt_file")
    if args.batch_size != 1 or args.num_videos != 1:
        raise SystemExit("KalSea runs one video per predict() call (--batch-size 1 --num-videos 1)")
    method = method_name(kargs)

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    save_path = args.save_path if args.save_path_suffix == "" else f"{args.save_path}_{args.save_path_suffix}"
    os.makedirs(save_path, exist_ok=True)

    sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = sampler.args
    ctrl = None
    if method != "reference":                      # before the first predict(): records the load-time scheduler
        ctrl = kalsea_hunyuan.install(sampler, method, kargs.delta, num_steps=args.infer_steps,
                                      kalsea_tau=kargs.kalsea_tau, kalsea_r_obs=kargs.kalsea_r_obs,
                                      live_sigma=not kargs.stale_sigma)
    logger.info(f"KalSea: method={method} delta={None if ctrl is None else kargs.delta} "
                f"sigma={'stale' if kargs.stale_sigma else 'live'} prompts={len(prompts)}")

    stats_path = os.path.join(save_path, "stats.jsonl")
    for idx, prompt in enumerate(prompts):
        if ctrl is not None:
            ctrl.before_video()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = sampler.predict(
            prompt=prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
        )
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        steps = list(range(args.infer_steps)) if ctrl is None else ctrl.after_video()
        sample = outputs["samples"][0].unsqueeze(0)
        fn = f"{idx:04d}-{safe_name(prompt)}-seed{outputs['seeds'][0]}.mp4"
        save_videos_grid(sample, os.path.join(save_path, fn), fps=24)
        rec = dict(idx=idx, file=fn, prompt=prompt, seed=outputs["seeds"][0], method=method,
                   delta=None if ctrl is None else kargs.delta, live_sigma=None if ctrl is None else not kargs.stale_sigma,
                   latency=round(latency, 3), nfe=len(steps), compute_steps=steps)
        with open(stats_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"[{idx + 1}/{len(prompts)}] {fn} | full steps: {len(steps)}/{args.infer_steps} | {latency:.1f} s")


if __name__ == "__main__":
    main()
