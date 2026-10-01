import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from dataset import TASKS
from model import ShortChunkCNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS), help="year = decade, market = release market")
    parser.add_argument("--manifest", default=None, help="Default: the task's manifest")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_chunks", type=int, default=3, help="Chunks/crops per clip per training step")
    parser.add_argument("--random_crop", action=argparse.BooleanOptionalAction, default=False,
                        help="Random crops in train, evenly spaced overlapping crops in eval")
    parser.add_argument("--crop_seconds", type=float, default=4.0)
    parser.add_argument("--eval_crops", type=int, default=10, help="Crops per clip at eval (with --random_crop)")
    parser.add_argument("--spec_augment", action=argparse.BooleanOptionalAction, default=False,
                        help="Frequency/time masking on training spectrograms")
    parser.add_argument("--freq_mask_param", type=int, default=24)
    parser.add_argument("--time_mask_param", type=int, default=40)
    parser.add_argument("--num_masks", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", default=None, help="Default: ./checkpoints/<task>")
    parser.add_argument("--resume", action="store_true", help="Resume from <ckpt_dir>/last.pt")
    parser.add_argument("--wandb_project", default="deepmir-hw1")
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, warmup_steps, total_steps):
    # Linear warmup then cosine decay to 0, stepped per iteration
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, global_step):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for mel, label in loader:
        # (B, C, 1, M, T) -> (B*C, 1, M, T); each chunk inherits its clip's label
        num_chunks = mel.shape[1]
        mel = mel.flatten(0, 1).to(device, non_blocking=True)
        label = label.repeat_interleave(num_chunks).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(mel)
            loss = criterion(logits, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        total_loss += loss.item() * label.size(0)
        total_correct += (logits.argmax(dim=1) == label).sum().item()
        total_count += label.size(0)

        wandb.log({"train/step_loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

    return total_loss / total_count, total_correct / total_count, global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels = [], []

    for mel, label in loader:
        batch_size, num_chunks = mel.shape[:2]
        mel = mel.flatten(0, 1).to(device, non_blocking=True)
        # Clip-level prediction: average logits over chunks
        logits = model(mel).view(batch_size, num_chunks, -1).mean(dim=1)
        all_logits.append(logits.float().cpu())
        all_labels.append(label)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss = criterion(logits, labels).item()

    top3 = logits.topk(3, dim=1).indices
    top1_acc = (top3[:, 0] == labels).float().mean().item()
    top3_acc = (top3 == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return {
        "loss": loss,
        "top1": top1_acc,
        "top3": top3_acc,
        "combined": top1_acc + 0.5 * top3_acc,
        "preds": top3[:, 0].numpy(),
        "labels": labels.numpy(),
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step, best, args, run_id):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best": best,
        "args": vars(args),
        "wandb_run_id": run_id,
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    task = TASKS[args.task]
    classes = task["classes"]
    args.manifest = args.manifest or task["manifest"]
    args.ckpt_dir = args.ckpt_dir or f"./checkpoints/{args.task}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    df = pd.read_csv(args.manifest)
    data_kwargs = dict(
        num_chunks=args.num_chunks,
        random_crop=args.random_crop,
        crop_seconds=args.crop_seconds,
        eval_crops=args.eval_crops,
        spec_augment=args.spec_augment,
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
        num_masks=args.num_masks,
    )
    train_set = task["dataset"](df[df["split"] == "train"], train=True, **data_kwargs)
    val_set = task["dataset"](df[df["split"] == "validation"], train=False, **data_kwargs)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    # Model, loss, optimizer, scheduler
    model = ShortChunkCNN(n_class=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, args.warmup_epochs * steps_per_epoch, args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    # Resume
    start_epoch, global_step = 1, 0
    best = {"top1": -1.0, "combined": -1.0}
    run_id = None
    last_path = os.path.join(args.ckpt_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best = ckpt["best"]
        run_id = ckpt["wandb_run_id"]
        print(f"Resumed from epoch {ckpt['epoch']}")

    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        tags=[args.task],
        config=vars(args),
        id=run_id,
        resume="allow",
    )
    wandb.define_metric("epoch")
    for key in ["train/loss", "train/acc", "val/*", "best/*"]:
        wandb.define_metric(key, step_metric="epoch")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, global_step
        )
        val = evaluate(model, val_loader, criterion, device)

        # Checkpointing
        ckpt_args = (model, optimizer, scheduler, scaler, epoch, global_step)
        if val["top1"] > best["top1"]:
            best["top1"] = val["top1"]
            save_checkpoint(os.path.join(args.ckpt_dir, "best_top1.pt"), *ckpt_args, best, args, run.id)
        if val["combined"] > best["combined"]:
            best["combined"] = val["combined"]
            save_checkpoint(os.path.join(args.ckpt_dir, "best_combined.pt"), *ckpt_args, best, args, run.id)
        save_checkpoint(last_path, *ckpt_args, best, args, run.id)

        # Logging
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val["loss"],
            "val/top1": val["top1"],
            "val/top3": val["top3"],
            "val/combined": val["combined"],
            "best/top1": best["top1"],
            "best/combined": best["combined"],
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=val["labels"].tolist(), preds=val["preds"].tolist(), class_names=classes
            ),
        }, step=global_step)
        print(
            f"Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val['loss']:.4f} top1 {val['top1']:.3f} top3 {val['top3']:.3f} "
            f"combined {val['combined']:.3f}"
        )

    wandb.summary["best_top1"] = best["top1"]
    wandb.summary["best_combined"] = best["combined"]
    wandb.finish()


if __name__ == "__main__":
    main()
