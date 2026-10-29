"""
Generate MIDI with the Compound Word model (train_cp.py), for evaluation with eval_midi.py.

Unconditional (like generate.py):  python generate_cp.py --out_dir generated_cp
Continuation (like continue.py):   python generate_cp.py --prompt_dir POP909/prompts/basic-b --out_dir cont_cp
Then:                              python eval_midi.py --midi_dir generated_cp --output_csv generated_cp_metrics.csv
"""
import argparse
from pathlib import Path

import torch
from miditok import CPWord

from cp_model import CPTransformer
from dataset import load_beat_aligned_score
from train import get_device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="checkpoints_cp/best")
    parser.add_argument("--tokenizer", default="checkpoints_cp/tokenizer.json")
    parser.add_argument("--out_dir", default="generated_cp")
    parser.add_argument("--prompt_dir", default=None,
                        help="continue every prompt MIDI under this folder instead of generating from scratch")
    parser.add_argument("--prompt_bars", type=int, default=8,
                        help="Bars of each prompt to use; shorter prompts are padded with empty bars (-1 = whole prompt)")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--n_samples", type=int, default=20, help="unconditional mode only")
    parser.add_argument("--n_bars", type=int, default=None,
                        help="bars to generate (default: 32 unconditional, 24 after a prompt)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--type_temperature", nargs="*", default=[], metavar="TYPE=T",
                        help="per-type temperature overrides, e.g. Duration=2.0 Velocity=1.5 "
                             "(the original repo samples each type with its own temperature)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

class CPGrammar:
    """
    Which values each slot may take, given what kind of compound token is being generated.
    In the training data every compound token is one of three kinds:
      Note:     Pitch / Velocity / Duration / Program set, everything else Ignore
      Bar:      everything except Family is Ignore
      Position: Tempo set, Chord set or Ignore, note slots Ignore
    The model predicts all non-Family slots from the Family alone, so e.g. it can't know whether the
    Metric token it is producing is a Bar or a Position when it picks the Tempo slot. Sampling inside
    these masks keeps every token decodable (miditok reads the Tempo of every Position token).
    """
    NOTE_TYPES = ("Pitch", "Velocity", "Duration", "Program")

    def __init__(self, tokenizer, device):
        self.names = sorted(set(tokenizer.vocab_types_idx) - {"Bar"}, key=tokenizer.vocab_types_idx.get)
        self.vocab = tokenizer.vocab
        self.pos_slot = self.names.index("Position")
        self.note_family = self.vocab[0]["Family_Note"]
        self.bar_id = self.vocab[self.pos_slot]["Bar_None"]
        self.ignore = [v.get("Ignore_None") for v in self.vocab]
        self.device = device
        self.family_mask = self._mask(0, ["Family"])
        self.metric_pos_mask = self._mask(self.pos_slot, ["Bar", "Position"])
        # Position_x -> x, so positions can be kept increasing inside a bar
        self.pos_value = torch.full((len(self.vocab[self.pos_slot]),), -1, device=device)
        for tok, i in self.vocab[self.pos_slot].items():
            if tok.startswith("Position_"):
                self.pos_value[i] = int(tok.split("_")[1])
        self.masks = {kind: [self._mask(k, self._allowed(kind, name)) for k, name in enumerate(self.names)]
                      for kind in ("Note", "Bar", "Position")}

    def _allowed(self, kind, name):
        if kind == "Note":
            return [name] if name in self.NOTE_TYPES else ["Ignore"]
        if kind == "Bar":
            return ["Ignore"]
        return {"Chord": ["Chord", "Ignore"], "Tempo": ["Tempo"]}.get(name, ["Ignore"])

    def _mask(self, slot, prefixes):
        mask = torch.zeros(len(self.vocab[slot]), dtype=torch.bool, device=self.device)
        for tok, i in self.vocab[slot].items():
            mask[i] = tok.split("_")[0] in prefixes
        return mask

    def bar_token(self):
        return [self.vocab[0]["Family_Metric"], self.bar_id] + self.ignore[2:]

    def is_bar(self, token):
        return token[self.pos_slot] == self.bar_id

def sample(logits, mask, temperature, top_k, top_p):
    """Temperature + top-k + nucleus sampling over the allowed ids of one slot."""
    logits = (logits / temperature).masked_fill(~mask, float("-inf"))
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, int(mask.sum()))).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
        remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p  # always keeps the most likely id
        logits[order[remove]] = float("-inf")
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1))

@torch.no_grad()
def generate_song(model, grammar, prefix, n_prefix_bars, args, device):
    """Extend prefix (BOS [+ prompt ending with a Bar token]) until n_prefix_bars + n_bars bars are complete."""
    max_len = model.config["n_positions"]
    temps = [args.type_temperatures.get(name, args.temperature) for name in grammar.names]
    total_bars = n_prefix_bars + args.n_bars
    ids = [list(t) for t in prefix]
    n_bars = sum(grammar.is_bar(t) for t in ids)
    last_pos = -1  # last Position in the current bar; prompts end with a Bar, so it starts empty

    while True:
        # Sliding window: the model only ever sees the last max_len compound tokens
        h = model.hidden(torch.tensor([ids[-max_len:]], device=device))[:, -1]
        family = sample(model.family_logits(h)[0], grammar.family_mask, temps[0], args.top_k, args.top_p)
        logits = [lg[0] for lg in model.other_logits(h, torch.tensor([family], device=device))]
        token = [family] + [None] * (len(grammar.names) - 1)

        if family == grammar.note_family:
            kind = "Note"
        else:
            # Bar or Position first, since it decides what the other slots may be
            k = grammar.pos_slot
            pos_mask = grammar.metric_pos_mask & ((grammar.pos_value > last_pos) | (grammar.pos_value < 0))
            token[k] = sample(logits[k - 1], pos_mask, temps[k], args.top_k, args.top_p)
            kind = "Bar" if token[k] == grammar.bar_id else "Position"
        for k in range(1, len(token)):
            if token[k] is None:
                token[k] = sample(logits[k - 1], grammar.masks[kind][k], temps[k], args.top_k, args.top_p)

        if kind == "Bar":
            n_bars += 1
            last_pos = -1
            if n_bars > total_bars:  # the (total_bars+1)-th Bar opens a bar we don't want
                break
        elif kind == "Position":
            last_pos = int(grammar.pos_value[token[grammar.pos_slot]])
        ids.append(token)
    return ids[1:]  # drop BOS

def load_prompt(tokenizer, grammar, path, prompt_bars=8):
    """
    Tokenize the (re-timed) prompt MIDI as exactly prompt_bars bars (None = however many bars it has),
    ending with the Bar token that opens the first generated bar, like continue.py.
    Returns (compound ids, number of prompt bars).
    """
    ids = tokenizer.encode(load_beat_aligned_score(path)).ids
    bar_positions = [i for i, t in enumerate(ids) if grammar.is_bar(t)]
    if prompt_bars is None:
        prompt_bars = len(bar_positions)
    if len(bar_positions) > prompt_bars:
        ids = ids[:bar_positions[prompt_bars]]  # cut right before the (prompt_bars+1)-th Bar
    n_missing = prompt_bars + 1 - sum(grammar.is_bar(t) for t in ids)
    ids = ids + [grammar.bar_token() for _ in range(n_missing)]  # empty bars, then open bar prompt_bars+1
    return ids, prompt_bars

def main():
    args = parse_args()
    args.type_temperatures = {k: float(v) for k, v in (s.split("=") for s in args.type_temperature)}
    if args.n_bars is None:
        args.n_bars = 24 if args.prompt_dir else 32
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CPWord(params=Path(args.tokenizer))
    model = CPTransformer.from_pretrained(args.ckpt_dir, map_location=device).to(device).eval()
    grammar = CPGrammar(tokenizer, device)
    unknown = set(args.type_temperatures) - set(grammar.names)
    if unknown:
        raise ValueError(f"unknown types {unknown}, choose from {grammar.names}")
    bos = [v["BOS_None"] for v in tokenizer.vocab]

    if args.prompt_dir:
        prompt_paths = sorted(Path(args.prompt_dir).rglob("*.mid"))
        if not prompt_paths:
            raise FileNotFoundError(f"no .mid files under {args.prompt_dir}")
        jobs = []
        for p in prompt_paths:
            prompt, n_prompt = load_prompt(tokenizer, grammar, p, None if args.prompt_bars < 0 else args.prompt_bars)
            jobs.append((out_dir / f"{p.stem}_cont.mid", [bos] + prompt, n_prompt))
    else:
        jobs = [(out_dir / f"{i:03d}.mid", [bos], 0) for i in range(args.n_samples)]

    for path, prefix, n_prefix_bars in jobs:
        ids = generate_song(model, grammar, prefix, n_prefix_bars, args, device)
        tokenizer.decode(ids).dump_midi(path)
        n_bars = sum(grammar.is_bar(t) for t in ids)
        print(f"{path}: {len(ids)} compound tokens, {n_prefix_bars} prompt bars + {n_bars - n_prefix_bars} generated")

if __name__ == "__main__":
    main()
