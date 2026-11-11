"""Generate audio for every caption in a CSV (columns: id, caption) using generate.py.

    python batch_generate.py --csv test.csv --out_dir outputs
    python batch_generate.py --csv test.csv --out_dir outputs --scale 7 --steps 100 --seconds 10

Files are saved as <out_dir>/<id>.wav. Existing files are skipped, so a Colab disconnect
can be resumed by re-running the same command.
"""

import argparse
import csv
from pathlib import Path

import soundfile as sf
import torch
from diffusers import StableAudioPipeline

from generate import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="test.csv")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="regenerate files that already exist")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} prompts from {args.csv}")

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    sr = pipe.vae.config.sampling_rate

    for i, row in enumerate(rows, 1):
        out_path = out_dir / f"{row['id']}.wav"
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(rows)}] skip {out_path} (exists)")
            continue
        print(f"[{i}/{len(rows)}] id={row['id']}: {row['caption'][:80]}...")
        audio = generate(pipe, row["caption"], args.scale, args.steps, args.seconds, args.seed)
        sf.write(out_path, audio.T.float().cpu().numpy(), sr)

    print(f"Done. Audio saved in {out_dir}/")


if __name__ == "__main__":
    main()
