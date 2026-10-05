"""Offline source separation with Demucs (run once before training).

For every clip in the manifest, writes
    <out_dir>/<sample_id>/vocals.wav
    <out_dir>/<sample_id>/no_vocals.wav
as mono 24 kHz WAV, i.e. the same format as the original dataset, so the dataloader can read
stems exactly like the mix (no 44.1 kHz resampling or MP3 decoding per training step).
"""
import argparse
import os

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from demucs.apply import apply_model
from demucs.pretrained import get_model

from dataset import TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="market", choices=list(TASKS))
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--out_dir", default=None, help="Default: the task's stem_dir")
    parser.add_argument("--shifts", type=int, default=1, help="Demucs test-time shifts (more = better, slower)")
    args = parser.parse_args()

    task = TASKS[args.task]
    out_dir = args.out_dir or task["stem_dir"]
    data_dir = task["data_dir"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = get_model(args.model).to(device).eval()
    vocals_idx = model.sources.index("vocals")
    df = pd.read_csv(task["manifest"])

    for i, row in enumerate(df.itertuples()):
        clip_dir = os.path.join(out_dir, row.sample_id)
        if os.path.exists(os.path.join(clip_dir, "no_vocals.wav")):
            continue

        wav, sr = torchaudio.load(data_dir + row.audio_path)
        to_model, from_model = T.Resample(sr, model.samplerate), T.Resample(model.samplerate, sr)
        # Demucs expects stereo at 44.1 kHz; normalise like the demucs CLI does
        mix = to_model(wav.mean(0, keepdim=True)).repeat(model.audio_channels, 1)
        mean, std = mix.mean(), mix.std() + 1e-8
        with torch.no_grad():
            sources = apply_model(model, ((mix - mean) / std)[None], device=device, shifts=args.shifts,
                                  split=True, overlap=0.25)[0]
        sources = (sources * std + mean).cpu()  # (n_sources, channels, samples)

        vocals = sources[vocals_idx]
        no_vocals = sources.sum(0) - vocals
        os.makedirs(clip_dir, exist_ok=True)
        for name, stem in [("vocals", vocals), ("no_vocals", no_vocals)]:
            torchaudio.save(os.path.join(clip_dir, f"{name}.wav"), from_model(stem.mean(0, keepdim=True)), sr)
        print(f"[{i + 1}/{len(df)}] {row.sample_id}")


if __name__ == "__main__":
    main()
