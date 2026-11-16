"""Normalize tag spellings in a tags JSONL (output of extract_tags_hf.py) so the same concept is asked the same way.

The MQA judge is spelling-sensitive (e.g. 'drum machine' vs 'drummachine' score very differently), and the captions
use run-together spellings that the tag extractor copies. Only spelling / canonical-name changes are made; no tags
are added or removed (except duplicates created by the mapping, which MQAScore de-duplicates anyway).

    python normalize_tags.py --inp test_tags.jsonl --out test_tags_norm.jsonl
"""
import argparse
import json
import re

# lowercase key -> canonical spelling
CANON = {
    # genre
    "hiphop": "hip hop",
    "hardrock": "hard rock",
    "folkrock": "folk rock",
    "bluesrock": "blues rock",
    "popfolk": "pop folk",
    "worldfusion": "world fusion",
    "singersongwriter": "singer-songwriter",
    "dancepop": "dance pop",
    "thrashmetal": "thrash metal",
    "rocknroll": "rock and roll",
    "alternativerock": "alternative rock",
    "edm": "EDM",
    # instrument
    "drummachine": "drum machine",
    "slideguitar": "slide guitar",
    "acousticguitar": "acoustic guitar",
    "electricguitar": "electric guitar",
    "synth pads": "synth pad",
    "ambient synth pads": "ambient synth pad",
    "rhodes": "Rhodes piano",
    "rhodes piano": "Rhodes piano",
    "hammond organ": "Hammond organ",
    "french horn": "French horn",
}
# id -> {attribute: {old: new}} for extraction artifacts that are not spelling issues
FIXES = {
    "88": {"genre": {"music": "70s music"}},  # extractor put the generic word "music" as the genre
}


def canon(tag):
    t = re.sub(r"\s+", " ", tag.strip())
    return CANON.get(t.lower(), t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="test_tags.jsonl")
    ap.add_argument("--out", default="test_tags_norm.jsonl")
    args = ap.parse_args()
    changes = {}
    with open(args.inp, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as g:
        for d in map(json.loads, f):
            for attr, lst in d["tags"].items():
                for k, el in enumerate(lst):
                    old = el["entity"] if isinstance(el, dict) else el
                    new = FIXES.get(d["id"], {}).get(attr, {}).get(old, canon(old))
                    if isinstance(el, dict):
                        el["entity"] = new
                        if "subgenres" in el:
                            el["subgenres"] = [canon(s) for s in el["subgenres"]]
                    else:
                        lst[k] = new
                    if new != old:
                        changes.setdefault((attr, old, new), []).append(d["id"])
            g.write(json.dumps(d, ensure_ascii=False) + "\n")
    for (attr, old, new), ids in sorted(changes.items()):
        print(f"{attr:12s} {old!r:24s} -> {new!r:22s} ids={','.join(ids)}")
    print(f"{sum(map(len, changes.values()))} tag strings changed; wrote {args.out}")


if __name__ == "__main__":
    main()
