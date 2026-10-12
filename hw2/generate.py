import argparse
from pathlib import Path

import torch
from miditok import REMI
from transformers import GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList

from train import get_device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="checkpoints/best")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    parser.add_argument("--out_dir", default="generated")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--n_bars", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

class BarStoppingCriteria(StoppingCriteria):
    """Stops generate() once the whole sequence contains more than n_bars Bar tokens."""
    def __init__(self, bar_id, n_bars, bars_outside_window):
        self.bar_id = bar_id
        self.n_bars = n_bars
        self.bars_outside_window = bars_outside_window  # bars already cut off by the sliding window

    def __call__(self, input_ids, scores, **kwargs):
        n = (input_ids == self.bar_id).sum(dim=1) + self.bars_outside_window
        # REMI puts Bar at the *start* of each bar, so the (n_bars+1)-th Bar means n_bars are complete
        return n > self.n_bars

@torch.no_grad()
def generate_song(model, tokenizer, args, device):
    bar_id = tokenizer["Bar_None"]
    special_ids = [tokenizer[t] for t in ["PAD_None", "BOS_None", "EOS_None", "MASK_None"]]
    max_len = model.config.n_positions
    keep = max_len // 2  # how many recent tokens to keep as context when the window is full

    ids = torch.tensor([[tokenizer["BOS_None"]]], device=device)
    while (ids == bar_id).sum() <= args.n_bars:
        ctx = ids[:, -keep:]
        bars_outside = int((ids == bar_id).sum() - (ctx == bar_id).sum())
        out = model.generate(
            ctx,
            attention_mask=torch.ones_like(ctx),
            max_new_tokens=max_len - ctx.shape[1],
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            suppress_tokens=special_ids,  # never sample PAD/BOS/EOS/MASK, so the song can't end early
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([BarStoppingCriteria(bar_id, args.n_bars, bars_outside)]),
        )
        ids = torch.cat([ids, out[:, ctx.shape[1]:]], dim=1)

    ids = ids[0].tolist()[1:]  # drop BOS
    bar_positions = [i for i, t in enumerate(ids) if t == bar_id]
    if len(bar_positions) > args.n_bars:
        ids = ids[:bar_positions[args.n_bars]]  # cut right before the (n_bars+1)-th Bar
    return ids

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = REMI(params=Path(args.tokenizer))
    model = GPT2LMHeadModel.from_pretrained(args.ckpt_dir).to(device).eval()

    for i in range(args.n_samples):
        ids = generate_song(model, tokenizer, args, device)
        score = tokenizer.decode(ids)  # token ids -> symusic Score
        path = out_dir / f"{i:03d}.mid"
        score.dump_midi(path)
        print(f"{path}: {len(ids)} tokens, {ids.count(tokenizer['Bar_None'])} bars")

if __name__ == "__main__":
    main()
