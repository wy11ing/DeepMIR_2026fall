import argparse
from pathlib import Path

import torch
from miditok import REMI
from transformers import GPT2LMHeadModel, StoppingCriteriaList

from train import get_device
from dataset import load_beat_aligned_score
from generate import BarStoppingCriteria

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dir", default="POP909/prompts/basic-b", help="Folder searched recursively for prompt MIDIs")
    parser.add_argument("--checkpoint", default="checkpoints/best", help="Model's checkpoint")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json", help="Model's tokenizer")
    parser.add_argument("--output", default="./cont")
    parser.add_argument("--prompt_bars", type=int, default=None, help="Keep only the first N bars of each prompt (default: whole prompt)")
    parser.add_argument("--n_bars", type=int, default=24, help="How many bars to generate after the prompt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def load_prompt(tokenizer, path, prompt_bars=None):
    """Tokenize the (re-timed) prompt MIDI, optionally keeping only its first prompt_bars bars."""
    ids = tokenizer.encode(load_beat_aligned_score(path)).ids
    bar_id = tokenizer["Bar_None"]
    bar_positions = [i for i, t in enumerate(ids) if t == bar_id]
    if prompt_bars is not None and len(bar_positions) > prompt_bars:
        ids = ids[:bar_positions[prompt_bars]]  # cut right before the (prompt_bars+1)-th Bar
    return ids

@torch.no_grad()
def generate_song(model, tokenizer, prompt_ids, args, device):
    bar_id = tokenizer["Bar_None"]
    special_ids = [tokenizer[t] for t in ["PAD_None", "BOS_None", "EOS_None", "MASK_None"]]
    max_len = model.config.n_positions
    keep = max_len // 2  # how many recent tokens to keep as context when the window is full

    total_bars = prompt_ids.count(bar_id) + args.n_bars

    # BOS + prompt, the same way songs start during training
    ids = torch.tensor([[tokenizer["BOS_None"]] + prompt_ids], device=device)
    while (ids == bar_id).sum() <= total_bars:
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
            stopping_criteria=StoppingCriteriaList([BarStoppingCriteria(bar_id, total_bars, bars_outside)]),
        )
        ids = torch.cat([ids, out[:, ctx.shape[1]:]], dim=1)

    ids = ids[0].tolist()[1:]  # drop BOS
    bar_positions = [i for i, t in enumerate(ids) if t == bar_id]
    if len(bar_positions) > total_bars:
        ids = ids[:bar_positions[total_bars]]  # cut right before the (total_bars+1)-th Bar
    return ids

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = REMI(params=Path(args.tokenizer))
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint).to(device).eval()

    prompt_paths = sorted(Path(args.prompt_dir).rglob("*.mid"))
    if not prompt_paths:
        raise FileNotFoundError(f"no .mid files under {args.prompt_dir}")
    for prompt_path in prompt_paths:
        prompt_ids = load_prompt(tokenizer, prompt_path, args.prompt_bars)
        ids = generate_song(model, tokenizer, prompt_ids, args, device)
        score = tokenizer.decode(ids)  # prompt bars + generated bars
        path = out_dir / f"{prompt_path.stem}_cont.mid"
        score.dump_midi(path)
        n_prompt = prompt_ids.count(tokenizer["Bar_None"])
        print(f"{path}: {len(ids)} tokens, {n_prompt} prompt bars + {ids.count(tokenizer['Bar_None']) - n_prompt} generated")

if __name__ == "__main__":
    main()
