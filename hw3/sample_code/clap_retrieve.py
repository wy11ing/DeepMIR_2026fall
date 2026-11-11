"""Text-to-music retrieval with LAION-CLAP: for every caption in a CSV, find the closest audio file.

    pip install laion_clap
    python clap_retrieve.py --csv test.csv --audio_dir path/to/audio --ckpt music_audioset_epoch_15_esc_90.14.pt

Output: retrieval.json, {"<caption id>": "<top-1 audio id>", ...} where the audio id is the file name
without its extension (path relative to --audio_dir, extension stripped).
Audio embeddings are cached in <audio_dir>/clap_audio_emb.npz, so re-runs are fast.
"""

import argparse
import csv
import json
from pathlib import Path

import laion_clap
import numpy as np
import torch

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _np(x):
    """Some laion_clap versions return tensors, others numpy arrays."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="test.csv")
    parser.add_argument("--audio_dir", required=True, help="folder with the candidate music files (searched recursively)")
    parser.add_argument("--ckpt", default="music_audioset_epoch_15_esc_90.14.pt")
    parser.add_argument("--out", default="retrieval.json")
    parser.add_argument("--batch", type=int, default=16, help="audio files per forward pass")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(f"No audio files found in {audio_dir}")
    rel = [str(p.relative_to(audio_dir)) for p in files]
    print(f"{len(files)} candidate audio files")

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} prompts")

    # The music checkpoint uses the HTSAT-base audio encoder without feature fusion.
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(args.ckpt)

    # Audio embeddings (cached)
    cache = audio_dir / "clap_audio_emb.npz"
    audio_emb = None
    if cache.exists() and not args.no_cache:
        data = np.load(cache, allow_pickle=True)
        if list(data["names"]) == rel:
            audio_emb = data["emb"]
            print(f"Loaded cached audio embeddings from {cache}")
    if audio_emb is None:
        chunks = []
        for i in range(0, len(files), args.batch):
            batch = [str(p) for p in files[i : i + args.batch]]
            with torch.no_grad():
                chunks.append(_np(model.get_audio_embedding_from_filelist(x=batch)))
            print(f"  audio {min(i + args.batch, len(files))}/{len(files)}")
        audio_emb = np.concatenate(chunks)
        np.savez(cache, names=np.array(rel), emb=audio_emb)
    audio_emb = normalize(audio_emb)

    # Text embeddings
    text_emb = normalize(_np(model.get_text_embedding([r["caption"] for r in rows])))

    # Cosine similarity and top-k
    sims = text_emb @ audio_emb.T  # (num_prompts, num_audio)
    best = sims.argmax(axis=1)

    results = {row["id"]: str(Path(rel[j]).with_suffix("")) for row, j in zip(rows, best)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
