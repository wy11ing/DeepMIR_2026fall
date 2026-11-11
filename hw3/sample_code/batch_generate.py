"""Generate audio for every caption in a CSV (columns: id, caption), optionally sweeping guidance parameters.

Plain CFG, one setting:
    python batch_generate.py --csv test.csv --out_dir outputs
CFG, sweep the scale:
    python batch_generate.py --out_dir sweep --scales 4,7,10
APG, sweep everything (grid = product of all lists):
    python batch_generate.py --out_dir sweep --apg --scales 7,10 --etas 0,0.5,1 --momentums 0,-0.5 --norm_thresholds 0

Each setting gets its own folder <out_dir>/<tag>/<id>.wav plus a config.json, e.g. apg_s7_m-0.5_e0_n0.
Existing files are skipped, so a Colab disconnect can be resumed by re-running the same command.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import soundfile as sf
import torch
from diffusers import StableAudioPipeline

from generate import generate


def floats(s):
    return [float(x) for x in s.split(",")]


def fmt(x):
    return f"{x:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="test.csv")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--apg", action="store_true", help="use adaptive projected guidance instead of CFG")
    parser.add_argument("--scales", type=floats, default=[7.0], help="comma-separated guidance scales")
    parser.add_argument("--etas", type=floats, default=[0.0], help="APG only")
    parser.add_argument("--momentums", type=floats, default=[0.0], help="APG only")
    parser.add_argument("--norm_thresholds", type=floats, default=[0.0], help="APG only; 0 disables clipping")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="regenerate files that already exist")
    args = parser.parse_args()

    out_root = Path(args.out_dir)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.apg:
        grid = list(itertools.product(args.scales, args.momentums, args.etas, args.norm_thresholds))
    else:
        grid = [(s, 0.0, 0.0, 0.0) for s in args.scales]
    print(f"{len(rows)} prompts x {len(grid)} settings = {len(rows) * len(grid)} clips")

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    sr = pipe.vae.config.sampling_rate

    for g, (scale, momentum, eta, thr) in enumerate(grid, 1):
        if args.apg:
            tag = f"apg_s{fmt(scale)}_m{fmt(momentum)}_e{fmt(eta)}_n{fmt(thr)}"
        else:
            tag = f"cfg_s{fmt(scale)}"
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        config = dict(apg=args.apg, scale=scale, momentum=momentum, eta=eta, norm_threshold=thr,
                      steps=args.steps, seconds=args.seconds, seed=args.seed)
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
        print(f"=== [{g}/{len(grid)}] {tag}")

        for i, row in enumerate(rows, 1):
            out_path = out_dir / f"{row['id']}.wav"
            if out_path.exists() and not args.overwrite:
                continue
            print(f"[{tag}] {i}/{len(rows)} id={row['id']}")
            audio = generate(pipe, row["caption"], scale, args.steps, args.seconds, args.seed,
                             use_apg=args.apg, momentum=momentum, eta=eta, norm_threshold=thr)
            sf.write(out_path, audio.T.float().cpu().numpy(), sr)

    print(f"Done. Audio saved under {out_root}/")


if __name__ == "__main__":
    main()
