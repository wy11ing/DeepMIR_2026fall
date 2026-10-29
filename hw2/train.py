import argparse
import json
import math
import random
from itertools import islice
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from miditok import REMI, TokenizerConfig
from miditok.data_augmentation import augment_score_multiple_offsets
from miditok.pytorch_data import DataCollator
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from dataset import ChunkMidiDataset, load_beat_aligned_score

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": True,  # all tracks in one token stream
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}

# Each offset makes one extra copy of a song (offsets are not combined)
AUGMENT_PARAMS = {
    "pitch_offsets": [-12, 12],  # semitones; skipped if notes would leave pitch_range
    "velocity_offsets": [-4, 5],
    "duration_offsets": [-0.5, 1],  # in beats
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="POP909/training", help="folder of song folders with .mid files")
    parser.add_argument("--tok_dir", default=None,
                        help="use pretokenized JSON folders instead of --data_dir (e.g. POP909/tok_merged); "
                             "must be tokenized with the same TOKENIZER_PARAMS")
    parser.add_argument("--no_augment", action="store_true", help="train on the original songs only")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", default="deepmir-hw2")
    parser.add_argument("--run_name", default=None, help="wandb run name (default: random)")
    parser.add_argument("--log_every", type=int, default=50, help="log train metrics every N steps")
    parser.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--save_every", type=int, default=1000, help="save a resumable checkpoint every N steps")
    parser.add_argument("--resume", default=None, help="checkpoint dir to resume from, e.g. checkpoints/last")
    return parser.parse_args()

def get_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def split_by_song(data_dir, val_ratio, seed):
    songs = sorted(p for p in Path(data_dir).iterdir() if p.is_dir())
    random.Random(seed).shuffle(songs)
    n_val = max(1, int(len(songs) * val_ratio))
    return songs[n_val:], songs[:n_val]

def tokenize_songs(song_dirs, tokenizer, augment, desc):
    """Load each song's MIDI, optionally add augmented copies, and return a list of token id lists."""
    sequences = []
    for song_dir in tqdm(song_dirs, desc=desc):
        for midi_path in sorted(song_dir.glob("*.mid")):
            score = load_beat_aligned_score(midi_path)  # bar lines from beat_midi.txt, not the MIDI's own grid
            scores = [score]
            if augment:
                scores += [s for _, s in augment_score_multiple_offsets(score, **AUGMENT_PARAMS) if len(s.tracks) > 0]
            # use_programs=True -> encode returns a single TokSequence; .ids are the token ids
            sequences += [tokenizer.encode(s).ids for s in scores]
    return sequences

def load_pretokenized(song_dirs, augment, desc):
    """Load token ids from JSON files written by tokenizer.tokenize_dataset (e.g. POP909/tok_merged)."""
    sequences = []
    for song_dir in tqdm(song_dirs, desc=desc):
        for json_path in sorted(song_dir.glob("*.json")):
            if not augment and "#" in json_path.stem:  # augmented files are named like 001#p12.json
                continue
            sequences.append(json.load(open(json_path))["ids"])
    return sequences

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**batch).loss.item()
    model.train()
    return total / len(loader)

def save_checkpoint(path, model, optimizer, scheduler, args, epoch, batches_done, step, best_val):
    """Save everything needed to continue training exactly where it stopped."""
    model.save_pretrained(path)  # weights + config.json, loadable with from_pretrained
    torch.save({
        "optimizer": optimizer.state_dict(),  # AdamW's running averages of gradients
        "scheduler": scheduler.state_dict(),  # where we are on the warmup/cosine curve
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

    tokenizer = REMI(TokenizerConfig(**TOKENIZER_PARAMS))
    tokenizer.save(out_dir / "tokenizer.json")
    bos_id, eos_id = tokenizer["BOS_None"], tokenizer["EOS_None"]

    # Split by song *before* augmenting, so no version of a validation song is ever trained on
    # Both folders use the same song folder names, so the same seed gives the same validation songs
    if args.tok_dir:
        train_songs, val_songs = split_by_song(args.tok_dir, args.val_ratio, args.seed)
        train_seqs = load_pretokenized(train_songs, augment=not args.no_augment, desc="loading train")
        val_seqs = load_pretokenized(val_songs, augment=False, desc="loading val")  # validate on real songs only
    else:
        train_songs, val_songs = split_by_song(args.data_dir, args.val_ratio, args.seed)
        train_seqs = tokenize_songs(train_songs, tokenizer, augment=not args.no_augment, desc="tokenizing train")
        val_seqs = tokenize_songs(val_songs, tokenizer, augment=False, desc="tokenizing val")  # validate on real songs only
    train_set = ChunkMidiDataset(train_seqs, args.seq_len, bos_id, eos_id)
    val_set = ChunkMidiDataset(val_seqs, args.seq_len, bos_id, eos_id)
    print(f"train: {len(train_songs)} songs / {len(train_seqs)} sequences / {len(train_set)} chunks, "
          f"val: {len(val_songs)} songs / {len(val_set)} chunks")

    # copy_inputs_as_labels: labels = input_ids; GPT2LMHeadModel shifts them by one internally
    collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    # The shuffle order comes from this generator, re-seeded every epoch, so a resumed run
    # sees the same order and can skip exactly the batches it already trained on
    shuffle_gen = torch.Generator()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collator, generator=shuffle_gen)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    if state:
        model = GPT2LMHeadModel.from_pretrained(args.resume).to(device)
    else:
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=args.seq_len,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = GPT2LMHeadModel(config).to(device)  # random init, no pretrained weights
    print(f"device: {device}, params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epoch * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    # mode="disabled" turns every wandb call into a no-op, so the loop needs no if-checks
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        id=state["wandb_id"] if state else None,  # same id + resume="allow" continues the old run's charts
        resume="allow",
        config={**vars(args), "tokenizer": TOKENIZER_PARAMS, "augment": AUGMENT_PARAMS,
                "n_params": sum(p.numel() for p in model.parameters()),
                "train_chunks": len(train_set), "val_chunks": len(val_set)},
        mode="disabled" if args.no_wandb else None,  # None: online, or WANDB_MODE env var if set
    )

    start_epoch = state["epoch"] if state else 1
    skip_batches = state["batches_done"] if state else 0
    best_val = state["best_val"] if state else math.inf
    step = state["step"] if state else 0
    running_loss = 0.0
    for epoch in range(start_epoch, args.epoch + 1):
        shuffle_gen.manual_seed(args.seed + epoch)
        # islice skips batches already trained on (they are still loaded, but not run through the model)
        pbar = tqdm(islice(train_loader, skip_batches, None), desc=f"epoch {epoch}",
                    initial=skip_batches, total=len(train_loader))
        for batch_idx, batch in enumerate(pbar, start=skip_batches + 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # norm before clipping
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every  # averaged so the curve is less noisy
                running_loss = 0.0
                wandb.log({
                    "train/loss": avg_loss,
                    "train/perplexity": math.exp(avg_loss),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "epoch": epoch,
                }, step=step)

            if step % args.save_every == 0:
                save_checkpoint(out_dir / "last", model, optimizer, scheduler, args, epoch, batch_idx, step, best_val)
        skip_batches = 0

        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: val loss {val_loss:.4f}, val ppl {math.exp(val_loss):.2f}")
        wandb.log({"val/loss": val_loss, "val/perplexity": math.exp(val_loss), "epoch": epoch}, step=step)
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out_dir / "best")
            print(f"  new best, saved to {out_dir / 'best'}")
            wandb.run.summary["best_val_loss"] = best_val
            wandb.run.summary["best_val_perplexity"] = math.exp(best_val)
        # weights only (no optimizer state), so each epoch can be listened to with generate.py --ckpt_dir
        model.save_pretrained(out_dir / f"epoch_{epoch:03d}")
        # Epoch finished: a resume from here starts at the beginning of the next epoch
        save_checkpoint(out_dir / "last", model, optimizer, scheduler, args, epoch + 1, 0, step, best_val)

    wandb.finish()

if __name__ == "__main__":
    main()
