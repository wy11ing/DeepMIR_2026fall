"""Rewrite captions into short, tag-style Stable Audio Open prompts (rule-based, deterministic).

Prompt layout:  <genres>, <adjective + instrument>..., <moods>..., <rhythm>..., <energy cue>, <BPM> BPM
  - genres:      tag-extracted genres (normalized) + genre words the extractor missed in the caption
  - instruments: each with its first attribute word ("warm Rhodes piano")
  - energy:      high-energy captions get "high energy, driving rhythm" + a fast BPM; calm ones get "slow tempo" + a slow BPM
Only generation uses these prompts; evaluation still scores against the original caption / tags.

    python rewrite_prompts.py --tags test_tags_norm.jsonl --out test_prompts_tagstyle.csv
"""
import argparse
import csv
import json
import re

# genre words that appear in captions but are sometimes missed by the tag extractor.
# Adjective-like genres only count when they modify a piece noun ("club track", "breakbeat-driven piece"),
# so "ambient settings" or "dances with the percussion" do not add a genre.
CAPTION_GENRES = ["club", "dance", "drone", "breakbeat", "fusion", "electronic", "orchestral", "experimental",
                  "ambient", "symphonic", "rock and roll", "atmospheric", "noise"]
NOUN_GENRES = ["soundtrack", "score", "eurodance", "jazzfunk"]
PIECE = r"(-\w+)? (track|piece|tune|music|soundscape|ballad|number)\b"
GENRE_BPM = {"house": 124, "techno": 130, "trance": 138, "EDM": 128, "dance": 124, "eurodance": 130, "dance pop": 118,
             "dancehall": 100, "hip hop": 90, "rap": 90, "breakbeat": 135, "garage": 130, "chiptune": 140,
             "rock": 120, "hard rock": 130, "rock and roll": 160, "metal": 150, "thrash metal": 180, "latin": 110,
             "club": 124, "drum and bass": 174}
HIGH = {"energetic", "upbeat", "driving", "fast", "fast-paced", "lively", "vibrant", "intense", "powerful", "festive",
        "joyful", "party", "party-loving", "celebratory", "celebration", "energizing", "aggressive", "relentless",
        "exciting", "electrifying", "bold", "fun", "action", "action-themed", "action-oriented", "danceable"}
LOW = {"calm", "soothing", "meditative", "serene", "relaxed", "peaceful", "quiet", "slow", "background", "stillness",
       "somber", "tender"}


def words(text):
    return set(re.findall(r"[a-z][a-z\-]+", text.lower()))


def entity(el):
    return el["entity"] if isinstance(el, dict) else el


def dedup(seq):
    seen, out = set(), []
    for s in seq:
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def rewrite(caption, tags):
    cap = caption.lower().replace("rocknroll", "rock and roll").replace("thrashmetal", "thrash metal")
    genres = []
    for el in tags.get("genre", []):
        subs = el.get("subgenres", []) if isinstance(el, dict) else []
        # prefer a specific subgenre if it is a real style name (e.g. "hard rock"), keep the parent too
        genres += [s for s in subs if len(s.split()) <= 2 and s.lower() not in {"dark", "epic", "fantasy"}] + [entity(el)]
    genres += [g for g in CAPTION_GENRES if re.search(rf"\b{g}{PIECE}", cap)]
    genres += [g for g in NOUN_GENRES if re.search(rf"\b{g}\b", cap)]
    genres = dedup(genres)

    instruments = []
    for el in tags.get("instrument", []):
        name = entity(el)
        attrs = el.get("attributes", []) if isinstance(el, dict) else []
        adj = next((a for a in attrs if len(a.split()) == 1), "")
        instruments.append(f"{adj} {name}".strip())
    moods = [m for m in tags.get("mood_theme", []) if len(m.split()) <= 2]
    rhythm = [r for r in tags.get("rhythm_tempo", []) if len(r.split()) <= 3]

    w = words(cap) | {m.lower() for m in moods}
    high, low = bool(w & HIGH), bool(w & LOW) and not (w & HIGH)
    bpm = next((GENRE_BPM[g] for g in genres if g in GENRE_BPM), None)
    if high:
        energy = ["high energy", "driving rhythm"]  # neutral cue; "upbeat" would push dark/intense prompts toward happy
        bpm = max(bpm or 0, 125)
    elif low:
        energy = ["slow tempo"]
        bpm = min(bpm, 85) if bpm else 75
    else:
        energy = []
    parts = dedup(genres + instruments + moods + rhythm + energy)
    if bpm:
        parts.append(f"{bpm} BPM")
    return ", ".join(parts)


def hybrid(caption, tags):
    """Original caption with only a tempo/energy cue in front, e.g. "High energy, driving rhythm, 110 BPM. <caption>".

    Unlike the tag-style rule, a genre with a known tempo keeps it (hip hop stays 90 BPM even when the caption is
    high-energy); the 125 / 75 BPM defaults apply only when no genre tempo is known.
    """
    cap = caption.lower().replace("rocknroll", "rock and roll").replace("thrashmetal", "thrash metal")
    genres = []
    for el in tags.get("genre", []):
        subs = el.get("subgenres", []) if isinstance(el, dict) else []
        genres += subs + [entity(el)]
    genres += [g for g in CAPTION_GENRES if re.search(rf"\b{g}{PIECE}", cap)]
    genres += [g for g in NOUN_GENRES if re.search(rf"\b{g}\b", cap)]
    genre_bpm = next((GENRE_BPM[g] for g in genres if g in GENRE_BPM), None)

    w = words(cap) | {m.lower() for m in tags.get("mood_theme", [])}
    if w & HIGH:
        cue, bpm = ["High energy", "driving rhythm"], genre_bpm or 125
    elif w & LOW:
        cue, bpm = ["Slow tempo"], min(genre_bpm, 90) if genre_bpm else 75  # never "slow tempo, 120 BPM"
    else:
        cue, bpm = [], genre_bpm
    if bpm:
        cue.append(f"{bpm} BPM")
    return (", ".join(cue) + ". " + caption) if cue else caption


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="test.csv")
    ap.add_argument("--tags", default="test_tags_norm.jsonl")
    ap.add_argument("--out", default="test_prompts_tagstyle.csv")
    ap.add_argument("--mode", choices=["tagstyle", "hybrid"], default="tagstyle")
    args = ap.parse_args()
    fn = rewrite if args.mode == "tagstyle" else hybrid
    tags = {d["id"]: d["tags"] for d in map(json.loads, open(args.tags, encoding="utf-8"))}
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "caption"])
        for r in rows:
            w.writerow([r["id"], fn(r["caption"], tags[r["id"]])])
    print(f"wrote {len(rows)} prompts to {args.out}")


if __name__ == "__main__":
    main()
