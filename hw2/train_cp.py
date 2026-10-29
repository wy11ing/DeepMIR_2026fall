import argparse
import math
from itertools import islice
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from miditok import CPWord, TokenizerConfig
from transformers import get_cosine_schedule_with_warmup

from cp_model import EMB_SIZES, CPTransformer
from dataset import RandomCropDataset
from train import AUGMENT_PARAMS, TOKENIZER_PARAMS, get_device, split_by_song, tokenize_songs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="POP909/training", help="folder of song folders with .mid files")
    parser.add_argument("--no_augment", action="store_true", help="train on the original songs only")
    parser.add_argument("--out_dir", default="checkpoints_cp")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=150)
    parser.add_argument("--seq_len", type=int, default=512, help="in compound tokens (each holds a whole note)")
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", default="deepmir-hw2")
    parser.add_argument("--run_name", default=None, help="wandb run name (default: random)")
    parser.add_argument("--log_every", type=int, default=50, help="log train metrics every N steps")
    parser.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--save_every", type=int, default=500, help="save a resumable checkpoint every N steps")
    parser.add_argument("--resume", default=None, help="checkpoint dir to resume from, e.g. checkpoints_cp/last")
    return parser.parse_args()

@torch.no_grad()
def evaluate(model, loader, device, pad_id):
    model.eval()
    total, per_type = 0.0, None
    for batch in loader:
        loss, losses = model(batch["input_ids"].to(device), pad_id)
        total += loss.item()
        losses = torch.stack(losses).cpu()
        per_type = losses if per_type is None else per_type + losses
    model.train()
    return total / len(loader), (per_type / len(loader)).tolist()

def save_checkpoint(path, model, optimizer, scheduler, args, epoch, batches_done, step, best_val):
    """Save everything needed to continue training exactly where it stopped."""
    model.save_pretrained(path)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "batches_done": batches_done,  # batches finished in this epoch (0 = epoch not started)
        "step": step,
        "best_val": best_val,
        "wandb_id": wandb.run.id if wandb.run else None,
    }, Path(path) / "training_state.pt")

def main():
    args = parse_args()
    state = None
    if args.resume:
        state = torch.load(Path(args.resume) / "training_state.pt", weights_only=False)
        # Reuse the original run's settings so the data split, shuffling and LR schedule match
        keep = {k: getattr(args, k) for k in ("resume", "device", "no_wandb")}
        args = argparse.Namespace(**{**state["args"], **keep})
        print(f"resuming from {args.resume}: epoch {state['epoch']}, batch {state['batches_done']}, step {state['step']}")

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same settings as the REMI tokenizer, so the two models see the same musical information
    tokenizer = CPWord(TokenizerConfig(**TOKENIZER_PARAMS))
    tokenizer.save(out_dir / "tokenizer.json")
    type_names = sorted(set(tokenizer.vocab_types_idx) - {"Bar"}, key=tokenizer.vocab_types_idx.get)
    vocab_sizes = [len(v) for v in tokenizer.vocab]
    pad_id = tokenizer.pad_token_id  # PAD is id 0 in every type's vocabulary
    # A BOS/EOS compound token is BOS/EOS in every slot
    bos = [v["BOS_None"] for v in tokenizer.vocab]
    eos = [v["EOS_None"] for v in tokenizer.vocab]

    # Split by song *before* augmenting, so no version of a validation song is ever trained on
    train_songs, val_songs = split_by_song(args.data_dir, args.val_ratio, args.seed)
    # CPWord.encode returns one TokSequence whose ids are lists of K ids (one compound token each)
    train_seqs = tokenize_songs(train_songs, tokenizer, augment=not args.no_augment, desc="tokenizing train")
    val_seqs = tokenize_songs(val_songs, tokenizer, augment=False, desc="tokenizing val")  # validate on real songs only
    # Random bar-aligned windows (a new start every epoch), padded to seq_len; Bar shares the Position slot
    bar_slot, bar_id = tokenizer.vocab_types_idx["Bar"], tokenizer.vocab[tokenizer.vocab_types_idx["Bar"]]["Bar_None"]
    is_bar = lambda t: t[bar_slot] == bar_id
    pad = [pad_id] * len(vocab_sizes)
    train_set = RandomCropDataset(train_seqs, args.seq_len, is_bar, pad, bos, eos, seed=args.seed)
    val_set = RandomCropDataset(val_seqs, args.seq_len, is_bar, pad, bos, eos, deterministic=True, seed=args.seed)
    print(f"train: {len(train_songs)} songs / {len(train_seqs)} sequences / {len(train_set)} windows per epoch, "
          f"val: {len(val_songs)} songs / {len(val_set)} windows")

    # All windows are padded to seq_len, so the default collate just stacks them into (B, L, K)
    shuffle_gen = torch.Generator()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=shuffle_gen)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    if state:
        model = CPTransformer.from_pretrained(args.resume).to(device)
    else:
        model = CPTransformer(
            vocab_sizes=vocab_sizes,
            emb_sizes=[EMB_SIZES[name] for name in type_names],
            n_positions=args.seq_len,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}, params: {n_params / 1e6:.1f}M, types: {dict(zip(type_names, vocab_sizes))}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epoch * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        id=state["wandb_id"] if state else None,
        resume="allow",
        config={**vars(args), "model": "cp", "tokenizer": TOKENIZER_PARAMS, "augment": AUGMENT_PARAMS,
                "emb_sizes": EMB_SIZES, "n_params": n_params,
                "chunking": "random_crop", "train_windows": len(train_set), "val_windows": len(val_set)},
        mode="disabled" if args.no_wandb else None,
    )

    start_epoch = state["epoch"] if state else 1
    skip_batches = state["batches_done"] if state else 0
    best_val = state["best_val"] if state else math.inf
    step = state["step"] if state else 0
    running_loss = 0.0
    running_types = torch.zeros(len(type_names))
    for epoch in range(start_epoch, args.epoch + 1):
        shuffle_gen.manual_seed(args.seed + epoch)
        pbar = tqdm(islice(train_loader, skip_batches, None), desc=f"epoch {epoch}",
                    initial=skip_batches, total=len(train_loader))
        for batch_idx, batch in enumerate(pbar, start=skip_batches + 1):
            loss, losses = model(batch["input_ids"].to(device), pad_id)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            running_loss += loss.item()
            running_types += torch.stack(losses).detach().cpu()
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if step % args.log_every == 0:
                wandb.log({
                    "train/loss": running_loss / args.log_every,
                    **{f"train/loss_{n}": l / args.log_every for n, l in zip(type_names, running_types.tolist())},
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "epoch": epoch,
                }, step=step)
                running_loss = 0.0
                running_types.zero_()

            if step % args.save_every == 0:
                save_checkpoint(out_dir / "last", model, optimizer, scheduler, args, epoch, batch_idx, step, best_val)
        skip_batches = 0

        val_loss, val_types = evaluate(model, val_loader, device, pad_id)
        print(f"epoch {epoch}: val loss {val_loss:.4f} | "
              + " ".join(f"{n}={l:.3f}" for n, l in zip(type_names, val_types)))
        wandb.log({"val/loss": val_loss, **{f"val/loss_{n}": l for n, l in zip(type_names, val_types)},
                   "epoch": epoch}, step=step)
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out_dir / "best")
            print(f"  new best, saved to {out_dir / 'best'}")
            wandb.run.summary["best_val_loss"] = best_val
        model.save_pretrained(out_dir / f"epoch_{epoch:03d}")
        # Epoch finished: a resume from here starts at the beginning of the next epoch
        save_checkpoint(out_dir / "last", model, optimizer, scheduler, args, epoch + 1, 0, step, best_val)

    wandb.finish()

if __name__ == "__main__":
    main()
