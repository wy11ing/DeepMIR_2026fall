"""Per-prompt attention weights from a previous run's per-tag MQA scores:
tags already judged present (P >= --high) are turned DOWN (x --down); low instrument tags are boosted (x --up).
Note: uses the judge's scores on the test set, i.e. this tunes on the evaluation set.

    python make_weights.py --scores eval_mqa_tagnorm_apg16.json --out boost_weights_hilo.jsonl
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--scores", default="eval_mqa_tagnorm_apg16.json")
ap.add_argument("--out", default="boost_weights_hilo.jsonl")
ap.add_argument("--high", type=float, default=0.9)
ap.add_argument("--down", type=float, default=0.5)
ap.add_argument("--up", type=float, default=2.0)
ap.add_argument("--down_attrs", default="genre,instrument,mood_theme")
ap.add_argument("--up_attrs", default="instrument")
a = ap.parse_args()
items = json.load(open(a.scores))["per_item"]
n_down = n_up = 0
with open(a.out, "w") as f:
    for i in sorted(items, key=int):
        words = []
        for attr, lst in items[i].get("MQA_tags", {}).items():
            for t in lst:
                if t["score"] >= a.high and attr in a.down_attrs.split(","):
                    words.append([t["tag"], a.down]); n_down += 1
                elif t["score"] < a.high and attr in a.up_attrs.split(","):
                    words.append([t["tag"], a.up]); n_up += 1
        # boosts last so they win where a boosted and a reduced tag share tokens
        words.sort(key=lambda w: w[1] > 1)
        f.write(json.dumps({"id": i, "words": words}) + "\n")
print(f"wrote {a.out}: {n_down} tags x{a.down}, {n_up} tags x{a.up}")
