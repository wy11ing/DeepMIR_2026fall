import argparse
import json

import torch
from torch.utils.data import DataLoader

from model import build_model
from dataset import TASKS

import pandas as pd


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    all_top3 = []

    for mel, _ in loader:
        batch_size, num_chunks = mel.shape[:2]
        mel = mel.flatten(0, 1).to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(mel)

        # Clip-level prediction: average logits over chunks (same as validation in train.py)
        logits = logits.float().view(batch_size, num_chunks, -1).mean(dim=1)
        all_top3.append(logits.topk(3, dim=1).indices.cpu())

    # (num_clips, 3) class indices, best first
    return torch.cat(all_top3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="e.g. ./checkpoints/market/best_combined.pt")
    parser.add_argument("--manifest", default=None, help="Default: the checkpoint task's manifest")
    parser.add_argument("--output", default=None, help="Default: ./predictions_<task>.json")
    parser.add_argument("--stem_dir", default=None, help="Default: the one used in training")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f"Loaded {args.ckpt} (epoch {ckpt['epoch']}, best so far: {ckpt['best']})")

    # The checkpoint knows which task it was trained for (older checkpoints are decade models)
    train_args = ckpt["args"]
    task_name = train_args.get("task", "year")
    task = TASKS[task_name]
    classes = task["classes"]
    print(f"Task: {task_name} ({len(classes)} classes: {classes})")

    stems = train_args.get("stems", ["mix"])  # older checkpoints were trained on the mix only
    # Width is not stored in args (some checkpoints use n_channels=128), so read it off the weights
    n_channels = next(v.shape[0] for k, v in ckpt["model"].items() if k.endswith("layer1.conv.weight"))
    model = build_model(len(classes), stems, train_args.get("fusion", "early"), n_channels).to(device)
    model.load_state_dict(ckpt["model"])

    # Rebuild the dataset with the same crop settings the model was trained with
    df = pd.read_csv(args.manifest or task["manifest"])
    test_set = task["dataset"](
        df[df["split"] == "test"],
        train=False,
        num_chunks=train_args["num_chunks"],
        stems=stems,
        stem_dir=args.stem_dir or train_args.get("stem_dir") or task["stem_dir"],
        activity_crop=train_args.get("activity_crop", False),
        random_crop=train_args["random_crop"],
        crop_seconds=train_args["crop_seconds"],
        eval_crops=train_args["eval_crops"],
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    top3 = test(model, test_loader, device)

    # shuffle=False, so row i of top3 belongs to test_set.df row i
    sample_ids = test_set.df["sample_id"].tolist()
    assert len(sample_ids) == len(top3)
    predictions = {
        sample_id: [classes[i] for i in row.tolist()]
        for sample_id, row in zip(sample_ids, top3)
    }

    output = args.output or f"./predictions_{task_name}.json"
    with open(output, "w") as f:
        json.dump({task["submission_key"]: predictions}, f, indent=2)
    print(f"Saved {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
