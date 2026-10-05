"""Test-set predictions from a train_features.py checkpoint (frozen Whisper / MERT features + dense head).

Reads the pre-extracted feature files (extract_features.py) listed in the checkpoint, averages the
softmax of every fold model in it, and writes the top-3 classes per test clip in the submission format.

    python inference_features.py --ckpt checkpoints/market_lang.pt
    python inference_features.py --ckpt checkpoints/market_lang.pt \
        --features features/market/large-v3_vocals.npz features/market/MERT-v1-330M_mix.npz
"""
import argparse
import json

import numpy as np
import torch

from dataset import TASKS
from train_features import load_checkpoint, load_data


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="e.g. ./checkpoints/market_lang.pt")
    parser.add_argument("--features", nargs="+", default=None,
                        help="Default: the feature files the checkpoint was trained on (same order)")
    parser.add_argument("--output", default=None, help="Default: ./predictions_<task>_features.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, ckpt = load_checkpoint(args.ckpt, device)
    task, classes = ckpt["task"], ckpt["classes"]
    print(f"Loaded {args.ckpt}: {len(models)} model(s), epoch {ckpt['epoch']}, inputs {ckpt['inputs']}")
    print(f"Task: {task} ({len(classes)} classes: {classes}), validation {ckpt['metrics']}")

    branches, names, ids, _, splits = load_data(args.features or ckpt["feature_files"], task,
                                                ckpt["args"]["lang_probs"])
    assert names == ckpt["inputs"], f"feature inputs {names} do not match the checkpoint's {ckpt['inputs']}"

    test = np.where(splits == "test")[0]
    xs = [torch.from_numpy(b[test]).to(device) for b in branches]
    probs = torch.stack([model(xs).softmax(dim=-1) for model in models]).mean(dim=0)
    top3 = probs.topk(3, dim=1).indices.cpu().numpy()

    predictions = {sid: [classes[i] for i in row] for sid, row in zip(ids[test], top3)}
    output = args.output or f"./predictions_{task}_features.json"
    with open(output, "w") as f:
        json.dump({TASKS[task]["submission_key"]: predictions}, f, indent=2)
    print(f"Saved {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
