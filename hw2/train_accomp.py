"""
Train a melody -> piano accompaniment model (bonus task). Two architectures on the same tokens (accomp_data.py):

  --arch dec   decoder-only GPT-2 on bar-interleaved sequences (melody bar i, then piano bar i);
               loss only on the piano tokens, the melody is given at generation time
  --arch s2s   BART encoder-decoder: the encoder reads the melody, the decoder writes the piano
               (--bar_emb and --hist_dropout: fixes for the decoder ignoring the melody, see accomp_model.py)

python train_accomp.py --arch dec --out_dir checkpoints_accomp_dec
python train_accomp.py --arch s2s --out_dir checkpoints_accomp_s2s
python train_accomp.py --arch s2s --bar_emb --hist_dropout 0.2 --out_dir checkpoints_accomp_s2s_bar_hd02
python train_accomp.py --arch dec --chord --out_dir checkpoints_accomp_dec_chord   (either arch: predict a chord per bar)
"""
import argparse
import math
import pickle
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (BartConfig, GPT2Config, GPT2LMHeadModel,
                          get_cosine_schedule_with_warmup)

from accomp_model import AccompBart
from accomp_data import BOS, EOS, PAD, VOCAB, AccompDataset, collate, load_song
from train import get_device, split_by_song

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["dec", "s2s"], required=True)
    parser.add_argument("--data_dir", default="POP909/training")
    parser.add_argument("--cache", default="POP909/accomp_songs.pkl", help="parsed songs, rebuilt if missing")
    parser.add_argument("--out_dir", default=None, help="default: checkpoints_accomp_{arch}")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--epoch", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--n_bars", type=int, default=16, help="bars per training window (the prompts have 16)")
    parser.add_argument("--windows_per_song", type=int, default=8, help="random windows per song per epoch")
    parser.add_argument("--max_len", type=int, default=2048, help="decoder / interleaved sequence length")
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=8, help="dec: transformer layers")
    parser.add_argument("--enc_layers", type=int, default=4, help="s2s: encoder layers")
    parser.add_argument("--dec_layers", type=int, default=4, help="s2s: decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bar_emb", action="store_true", help="s2s: add a bar-index embedding to every token (fix A)")
    parser.add_argument("--chord", action="store_true",
                        help="dec / s2s: the model predicts each bar's chord before its piano notes")
    parser.add_argument("--hist_dropout", type=float, default=0.0,
                        help="s2s: fraction of decoder input tokens replaced by PAD in training (fix B, e.g. 0.2)")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", default="deepmir-hw2")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", default=None, help="checkpoint dir to resume from, e.g. checkpoints_accomp_dec/last")
    args = parser.parse_args()
    if args.arch == "dec" and (args.bar_emb or args.hist_dropout > 0):
        parser.error("--bar_emb and --hist_dropout are for --arch s2s")
    return args

def build_model(args):
    if args.arch == "dec":
        return GPT2LMHeadModel(GPT2Config(
            vocab_size=len(VOCAB), n_positions=args.max_len, n_layer=args.n_layer, n_head=args.n_head,
            n_embd=args.n_embd, resid_pdrop=args.dropout, embd_pdrop=args.dropout, attn_pdrop=args.dropout,
            bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD, chord=getattr(args, "chord", False),
        ))
    return AccompBart(BartConfig(
        vocab_size=len(VOCAB), max_position_embeddings=args.max_len, d_model=args.n_embd,
        encoder_layers=args.enc_layers, decoder_layers=args.dec_layers,
        encoder_attention_heads=args.n_head, decoder_attention_heads=args.n_head,
        encoder_ffn_dim=4 * args.n_embd, decoder_ffn_dim=4 * args.n_embd,
        dropout=args.dropout, attention_dropout=args.dropout,
        pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, decoder_start_token_id=BOS,
        forced_eos_token_id=None,
        bar_emb=getattr(args, "bar_emb", False), hist_dropout=getattr(args, "hist_dropout", 0.0),
        chord=getattr(args, "chord", False),
    ))

def load_model(path, arch):
    cls = GPT2LMHeadModel if arch == "dec" else AccompBart
    return cls.from_pretrained(path)

def load_songs(song_dirs, cache):
    """Parse every song once; later runs read the pickle (songs cached before chord labels existed are re-parsed)."""
    cache = Path(cache)
    songs = pickle.load(open(cache, "rb")) if cache.exists() else {}
    missing = [d for d in song_dirs if d.name not in songs or (songs[d.name] and "chord" not in songs[d.name][0])]
    for d in tqdm(missing, desc="parsing songs"):
        songs[d.name] = load_song(d)
    if missing:
        pickle.dump(songs, open(cache, "wb"))
    return [songs[d.name] for d in song_dirs]

def forward(model, batch, device):
    # Same call for both: GPT-2 shifts the labels itself (-100 = melody / header tokens, given rather than
    # predicted); BART builds decoder_input_ids from the labels (shifted right, starting with BOS)
    batch = {k: v.to(device) for k, v in batch.items()}
    return model(**batch).loss

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = sum(forward(model, batch, device).item() for batch in loader)
    model.train()
    return total / len(loader)

def main():
    args = parse_args()
    state = None
    if args.resume:
        state = torch.load(Path(args.resume) / "training_state.pt", weights_only=False)
        keep = {k: getattr(args, k) for k in ("resume", "device", "no_wandb")}
        args = argparse.Namespace(**{**state["args"], **keep})
        print(f"resuming from {args.resume}: epoch {state['epoch']}, step {state['step']}")
    args.out_dir = args.out_dir or f"checkpoints_accomp_{args.arch}"

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dirs, val_dirs = split_by_song(args.data_dir, args.val_ratio, args.seed)
    chord = getattr(args, "chord", False)
    train_set = AccompDataset(load_songs(train_dirs, args.cache), args.arch, args.n_bars, args.max_len,
                              windows_per_song=args.windows_per_song, chord=chord)
    val_set = AccompDataset(load_songs(val_dirs, args.cache), args.arch, args.n_bars, args.max_len,
                            windows_per_song=4, deterministic=True, chord=chord)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f"train: {len(train_set.songs)} songs / {len(train_set)} windows per epoch, "
          f"val: {len(val_set.songs)} songs / {len(val_set)} windows")

    model = (load_model(args.resume, args.arch) if state else build_model(args)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    extra = f", chord: {getattr(model.config, 'chord', False)}"
    if args.arch == "s2s":
        extra += f", bar_emb: {model.config.bar_emb}, hist_dropout: {model.config.hist_dropout}"
    print(f"arch: {args.arch}, device: {device}, params: {n_params / 1e6:.1f}M{extra}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epoch * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    wandb.init(project=args.wandb_project, name=args.run_name, id=state["wandb_id"] if state else None,
               resume="allow", config={**vars(args), "model": f"accomp_{args.arch}", "n_params": n_params},
               mode="disabled" if args.no_wandb else None)

    start_epoch = state["epoch"] if state else 1
    best_val = state["best_val"] if state else math.inf
    step = state["step"] if state else 0
    running = 0.0
    for epoch in range(start_epoch, args.epoch + 1):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            loss = forward(model, batch, device)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if step % args.log_every == 0:
                wandb.log({"train/loss": running / args.log_every, "train/lr": scheduler.get_last_lr()[0],
                           "train/grad_norm": grad_norm.item(), "epoch": epoch}, step=step)
                running = 0.0

        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: val loss {val_loss:.4f}")
        wandb.log({"val/loss": val_loss, "epoch": epoch}, step=step)
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out_dir / "best")
            print(f"  new best, saved to {out_dir / 'best'}")
            wandb.run.summary["best_val_loss"] = best_val
        # a resume from here starts at the next epoch
        model.save_pretrained(out_dir / "last")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "args": vars(args),
                    "epoch": epoch + 1, "step": step, "best_val": best_val,
                    "wandb_id": wandb.run.id if wandb.run else None}, out_dir / "last" / "training_state.pt")

    wandb.finish()

if __name__ == "__main__":
    main()
