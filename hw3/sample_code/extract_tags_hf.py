"""Extract MQAScore tags from test.csv captions with plain transformers (no vLLM).

Mirrors MQAScore's scripts/extract_tags.py: same instruction file, same system/user chat layout,
greedy decoding, same JSON parsing. Only the inference engine differs (HF generate vs vLLM), so the
output can differ slightly from the authors' setup.

Run with the MQAScore env's python:
    /path/to/MQAScore/.venv/bin/python extract_tags_hf.py --mqa_repo /path/to/MQAScore \
        --model /path/to/Qwen3-4B-Instruct-2507 --csv test.csv --out test_tags.jsonl

Output JSONL: {"id", "text", "tags": {genre, instrument, mood_theme, rhythm_tempo, vocal,
production_sound_quality, context, others}}. Unparseable items go to <out>.failures.jsonl;
rerunning skips ids already in --out.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mqa_repo", required=True)
    ap.add_argument("--model", required=True, help="local path / HF id of Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--csv", default="test.csv")
    ap.add_argument("--out", default="test_tags.jsonl")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    args = ap.parse_args()

    repo = Path(args.mqa_repo).resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))
    from extract_tags import extract_json  # the repo's own parser
    from mqascore.io import append_jsonl, done_ids
    from mqascore.prompts import tag_extraction_instruction

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [{"id": r["id"], "text": r["caption"]} for r in csv.DictReader(f)]
    out = Path(args.out)
    fail_path = out.with_name(out.name + ".failures.jsonl")
    done = done_ids(out)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} captions, {len(done)} already in {out}, {len(todo)} to extract")
    if not todo:
        return
    fail_path.unlink(missing_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    instruction = tag_extraction_instruction()

    n_fail = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": instruction}, {"role": "user", "content": r["text"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for r in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        for r, g in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
            raw = tok.decode(g, skip_special_tokens=True)
            try:
                append_jsonl(out, {**r, "tags": extract_json(raw)})
            except (ValueError, json.JSONDecodeError) as e:
                n_fail += 1
                append_jsonl(fail_path, {**r, "error": str(e), "raw_output": raw})
        print(f"{min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)
    print(f"{len(todo) - n_fail} ok, {n_fail} failed" + (f" (see {fail_path})" if n_fail else ""))
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
