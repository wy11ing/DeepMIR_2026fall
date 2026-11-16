"""APG generation + cross-attention word boosting (prompt-to-prompt style reweighting, training-free).

For every prompt, the T5 tokens of chosen tag words (default: the instrument tags of that caption) get
their cross-attention weight multiplied by --weight in every DiT block. This is done by adding log(weight)
to those keys' attention logits (the softmax renormalizes), for the conditional half of the batch only.

    python boost_generate.py --out_dir ../generated/apg_s16_boost_inst_w2 --weight 2 --attrs instrument
Sampling is generate.generate (APG from cfg.py, s16, x0 space, seed 0, 100 steps, 10 s).
"""

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path

import soundfile as sf
import torch
from diffusers import StableAudioPipeline
from diffusers.models.attention_processor import StableAudioAttnProcessor2_0

from generate import generate


class BoostProcessor(StableAudioAttnProcessor2_0):
    """Cross-attention processor; self.bias (batch, 1, n_keys) is added to the attention logits."""

    bias = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        if BoostProcessor.bias is not None and encoder_hidden_states is not None and attention_mask is None:
            attention_mask = BoostProcessor.bias.to(hidden_states.dtype)
        return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)


def entity(el):
    return el["entity"] if isinstance(el, dict) else el


def find_spans(caption, words, weight):
    """(start, end, text, weight) of each word in the caption; 'electric guitar' also matches 'electricguitar'.
    `words` is a list of words (all get `weight`) or of (word, weight) pairs."""
    spans = []
    for w in words:
        w, wt = (w, weight) if isinstance(w, str) else w
        pat = r"[\s\-]?".join(map(re.escape, w.split()))
        spans += [(m.start(), m.end(), m.group(0), wt) for m in re.finditer(rf"\b{pat}s?\b", caption, flags=re.I)]
    return spans


def token_bias(tokenizer, caption, spans, n_keys):
    """Later spans win on overlapping tokens."""
    enc = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                    return_offsets_mapping=True)
    bias = torch.zeros(2, 1, n_keys)
    hit = []
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        ws = [wt for s, e, _, wt in spans if a < e and b > s and wt != 1]
        if b > a and ws:
            bias[1, 0, t] = math.log(ws[-1])  # row 0 is the (zero) unconditional embedding
            hit.append(f"{tokenizer.convert_ids_to_tokens(enc['input_ids'][t])}x{ws[-1]:g}")
    return bias, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="test.csv")
    ap.add_argument("--tags", default="test_tags_norm.jsonl")
    ap.add_argument("--attrs", default="instrument", help="comma list of tag attributes whose words get boosted")
    ap.add_argument("--weight", type=float, default=2.0)
    ap.add_argument("--weights", default=None, help='jsonl {"id", "words": [[word, weight], ...]}; overrides --attrs/--weight')
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scale", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--deadline", type=float, default=0)
    args = ap.parse_args()
    t0 = time.time()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"apg": True, "scale": args.scale, "momentum": -0.5, "eta": 0.4,
                                                 "norm_threshold": 12.0, "space": "x0", "seed": args.seed,
                                                 "boost_attrs": args.attrs, "boost_weight": args.weight,
                                                 "weights_file": args.weights}, indent=2))
    tags = {d["id"]: d["tags"] for d in map(json.loads, open(args.tags, encoding="utf-8"))}
    per_id = {d["id"]: d["words"] for d in map(json.loads, open(args.weights))} if args.weights else None
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    if args.ids:
        keep = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in keep]
    todo = [r for r in rows if not (out / f"{r['id']}.wav").exists()]
    print(f"{len(todo)} to generate ({len(rows) - len(todo)} exist)", flush=True)

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    assert pipe.tokenizer.is_fast, "need a fast tokenizer for offset mapping"
    for blk in pipe.transformer.transformer_blocks:
        blk.attn2.set_processor(BoostProcessor())
    sr = pipe.vae.config.sampling_rate
    n_keys = pipe.tokenizer.model_max_length + 2  # T5 tokens + seconds_start + seconds_end
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    log_path = out / "boost_tokens.jsonl"
    with open(log_path, "a", encoding="utf-8") as log:
        for i, r in enumerate(todo, 1):
            if args.deadline and time.time() - t0 > args.deadline:
                print(f"deadline reached, stopping ({len(todo) - i + 1} left)", flush=True)
                break
            ts = time.time()
            if per_id is not None:
                words = per_id.get(r["id"], [])
            else:
                words = [entity(el) for a in args.attrs.split(",") for el in tags[r["id"]].get(a, [])]
            spans = find_spans(r["caption"], words, args.weight)
            bias, hit = token_bias(pipe.tokenizer, r["caption"], spans, n_keys)
            BoostProcessor.bias = bias.to("cuda") if hit else None
            a = generate(pipe, r["caption"], args.scale, 100, 10.0, args.seed, use_apg=True,
                         momentum=-0.5, eta=0.4, norm_threshold=12.0, apg_space="x0")
            sf.write(out / f"{r['id']}.wav", a.T.float().cpu().numpy(), sr)
            log.write(json.dumps({"id": r["id"], "words": words, "matched": [[s[2], s[3]] for s in spans], "tokens": hit}) + "\n")
            print(f"[{i}/{len(todo)}] id={r['id']} boosted={hit} {time.time() - ts:.1f}s", flush=True)


if __name__ == "__main__":
    main()
