"""
Generate piano accompaniment for every bonus melody prompt, for evaluation with HW2_scripts/eval_bonus.py.

python generate_accomp.py --arch dec --ckpt_dir checkpoints_accomp_dec/best --out_dir accomp_dec
python generate_accomp.py --arch s2s --ckpt_dir checkpoints_accomp_s2s/best --out_dir accomp_s2s
python HW2_scripts/eval_bonus.py --prompt_dir POP909/prompts/bonus --gen_midi_dir accomp_dec --output_csv accomp_dec.csv

Sampling sweep (every temperature x top_p x seed, each scored with eval_bonus.py, summary in sweep.csv):
python generate_accomp.py --arch dec --sweep --out_dir sweep_dec
python generate_accomp.py --arch dec --sweep --temperatures 0.8 1.0 --top_ps 0.9 --seeds 0 1 2 --out_dir sweep_dec

Each output MIDI has the prompt melody (track MELODY, copied unchanged) and the generated PIANO track, timed on
the prompt's beat_midi.txt grid, so eval_bonus.py's bars line up with ours.

A model trained with --chord first samples each bar's chord, then the piano. Its chords are written to
chords.csv in the output folder, next to the prompt's labelled chords (used only for this report, never as
input), and the sweep adds their per-bar accuracy as chord_pred_acc.

The chord token can be chosen differently from the notes (chord models only):
  --chord_temperature T   sample the chord at temperature T, no top_p (0 = always the most likely chord);
                          default: same temperature / top_p as the notes
  --chord_mel_weight W    add W * (cosine of the bar's melody chroma with each triad template, the same
                          templates as eval_bonus.py) to the chord logits; 0 = off
The sweep takes lists of both (--chord_temperatures, --chord_mel_weights), e.g.
python generate_accomp.py --arch dec --ckpt_dir checkpoints_accomp_dec_chord/best --sweep --temperatures 0.8 0.9 \
    --top_ps 0.8 0.9 --chord_temperatures 0 0.5 --chord_mel_weights 0 3 6 --out_dir sweep_dec_chord2
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
import pretty_midi
import torch

from accomp_data import (ACC, BOS, CHORD0, EOS, MAX_BARS, MEL, SUBS, VOCAB, bar_chords, bar_header, chord_name,
                         decode_piano, encoder_seq, load_prompt, note_tokens, token_type)
from accomp_model import bar_index
from train import get_device
from train_accomp import load_model

sys.path.insert(0, str(Path(__file__).parent / "HW2_scripts"))
import eval_bonus  # noqa: E402  (the TA's metrics, reused as-is)

# Default sweep grid. Low temperature / top_p tends toward safe, repetitive patterns (stabler chords, narrower
# rhythm and velocity distributions); high values toward more variety and more wrong notes.
# top_p 1.0 = plain sampling, for reference.
SWEEP_TEMPERATURES = [0.7, 0.8, 0.9, 1.0, 1.1]
SWEEP_TOP_PS = [0.8, 0.9, 0.95, 1.0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["dec", "s2s"], required=True)
    parser.add_argument("--ckpt_dir", default=None, help="default: checkpoints_accomp_{arch}/best")
    parser.add_argument("--prompt_dir", default="POP909/prompts/bonus")
    parser.add_argument("--out_dir", default=None, help="default: accomp_{arch} (sweep: one subfolder per setting)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_bar_tokens", type=int, default=400, help="force a bar to end after this many tokens")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true", help="generate + evaluate every combination below")
    parser.add_argument("--temperatures", type=float, nargs="+", default=SWEEP_TEMPERATURES)
    parser.add_argument("--top_ps", type=float, nargs="+", default=SWEEP_TOP_PS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="sweep: seeds per setting (10 prompts are noisy, so average over a few)")
    parser.add_argument("--chord_temperature", type=float, default=None,
                        help="chord token's temperature (0 = argmax); default: the notes' temperature and top_p")
    parser.add_argument("--chord_mel_weight", type=float, default=0.0,
                        help="weight of the melody-template prior added to the chord logits")
    parser.add_argument("--chord_temperatures", type=float, nargs="+", default=None,
                        help="sweep: chord temperatures (-1 = same as the notes); default: --chord_temperature")
    parser.add_argument("--chord_mel_weights", type=float, nargs="+", default=None,
                        help="sweep: melody-prior weights; default: --chord_mel_weight")
    args = parser.parse_args()
    if args.chord_temperature is not None and args.chord_temperature < 0:
        args.chord_temperature = None
    return args

def ids_of(prefix):
    return torch.tensor([i for t, i in VOCAB.items() if t.startswith(prefix + "_")])

class PianoGrammar:
    """
    Which tokens may come next inside a bar of piano notes: (Pos) Pitch Vel Dur, Pos increasing and inside
    the bar, Pitch only after a Pos and rising within one Pos (the training order, sorted by (pos, pitch)).
    The rising pitch also stops a low-temperature model from stacking the same note forever at one position.
    `end_ids` are the tokens that close the bar.
    """
    def __init__(self, device, vocab_size):
        self.pos, self.pitch, self.vel, self.dur = (ids_of(p).to(device) for p in ("Pos", "Pitch", "Vel", "Dur"))
        self.bar = ids_of("Bar").to(device)
        self.n = vocab_size  # the model's, which is smaller than VOCAB for checkpoints from before chords
        self.chord_mask = torch.zeros(self.n, dtype=torch.bool, device=device)
        if self.n == len(VOCAB):
            self.chord_mask[ids_of("Chord")] = True
        self.device = device

    def start_bar(self, beats):
        self.prev, self.last_pos, self.n_pos, self.last_pitch = None, -1, beats * SUBS, -1

    def mask(self, end_ids):
        allowed = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        if self.prev in (None, "Dur"):
            allowed[self.pos[self.last_pos + 1:self.n_pos]] = True
            if self.last_pos >= 0:
                allowed[self.pitch[self.last_pitch + 1:]] = True
            allowed[end_ids] = True
        elif self.prev == "Pos":
            allowed[self.pitch] = True
        else:
            allowed[{"Pitch": self.vel, "Vel": self.dur}[self.prev]] = True
        return allowed

    def update(self, token):
        self.prev = token_type(token)
        if self.prev == "Pos":
            self.last_pos, self.last_pitch = int(token) - int(self.pos[0]), -1
        elif self.prev == "Pitch":
            self.last_pitch = int(token) - int(self.pitch[0])

def sample(logits, mask, temperature, top_p):
    if temperature <= 0:
        return int(logits.masked_fill(~mask, float("-inf")).argmax())
    logits = (logits / temperature).masked_fill(~mask, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        remove = probs.cumsum(-1) - probs > top_p  # always keeps the most likely token
        logits[order[remove]] = float("-inf")
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1))

class Stepper:
    """
    Feed tokens to an HF model with a KV cache and return the next-token logits.

    The model has max_len positions (2048); going past them is an out-of-range index (on CUDA a device-side
    assert). So when a feed would overflow, the context is cut back to the last bars that fit in half the
    window, starting at a Bar token after a fresh BOS, and re-fed without the cache. The decoder-only model
    always saw windows starting at Bar_0, so its kept bars are renumbered from 0 (and later Bar tokens shifted
    to match); the seq2seq decoder keeps its numbers, since they point into the full melody in the encoder.
    """
    def __init__(self, model, device, encoder_ids=None):
        self.model, self.device, self.past = model, device, None
        self.enc = None
        if encoder_ids is not None:
            self.enc = model.encode(torch.tensor([encoder_ids], device=device))
        config = model.config
        self.max_len = getattr(config, "n_positions", None) or config.max_position_embeddings
        self.ids, self.bar_offset = [], 0
        self.bar0 = VOCAB["Bar_0"]

    def is_bar(self, i):
        return self.bar0 <= i < self.bar0 + MAX_BARS

    @torch.no_grad()
    def feed(self, ids):
        if self.bar_offset:
            ids = [i - self.bar_offset if self.is_bar(i) else i for i in ids]
        if len(self.ids) + len(ids) > self.max_len:
            self.truncate(len(ids))
        self.ids += ids
        if self.past is None:
            ids = self.ids  # first call, or the cache was dropped by truncate
        x = torch.tensor([ids], device=self.device)
        if self.enc is None:
            out = self.model(input_ids=x, past_key_values=self.past, use_cache=True)
        else:
            # the new tokens' bar indices come from the whole decoded sequence (used when the model has bar_emb)
            bar_ids = bar_index(torch.tensor([self.ids], device=self.device))[:, -len(ids):]
            out = self.model(encoder_outputs=self.enc, decoder_input_ids=x, decoder_bar_ids=bar_ids,
                             past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        return out.logits[0, -1]

    def truncate(self, n_new):
        budget = self.max_len // 2 - n_new
        bars = [k for k, i in enumerate(self.ids) if self.is_bar(i) and len(self.ids) - k <= budget]
        tail = self.ids[bars[0]:] if bars else self.ids[len(self.ids) - max(budget, 0):]
        if self.enc is None and bars:
            shift = tail[0] - self.bar0
            tail = [i - shift if self.is_bar(i) else i for i in tail]
            self.bar_offset += shift
        self.ids, self.past = [BOS] + tail, None
        print(f"  context full: kept the last {len(tail)} tokens")

TEMPLATES = torch.tensor(eval_bonus.TEMPLATES, dtype=torch.float)  # (24, 12), rows in chord-class order

def melody_prior(mel):
    """(24,) cosine of the bar's duration-weighted melody chroma with each major/minor triad; zeros if silent."""
    chroma = torch.zeros(12)
    for _, pitch, dur in mel:
        chroma[pitch % 12] += dur
    if chroma.norm() == 0:
        return torch.zeros(len(TEMPLATES))
    return TEMPLATES @ chroma / chroma.norm()

def sample_chord(logits, grammar, mel, args):
    """The bar's chord token: model logits + the melody-template prior, sampled with the chord settings."""
    logits = logits.clone()
    if args.chord_mel_weight:
        logits[CHORD0:CHORD0 + len(TEMPLATES)] += args.chord_mel_weight * melody_prior(mel).to(logits.device)
    if args.chord_temperature is None:
        return sample(logits, grammar.chord_mask, args.temperature, args.top_p)
    return sample(logits, grammar.chord_mask, args.chord_temperature, 1.0)

def sample_bar(stepper, logits, grammar, beats, end_ids, args):
    """Sample one bar of piano tokens; returns (tokens, the sampled end token)."""
    grammar.start_bar(beats)
    tokens = []
    while True:
        mask = grammar.mask(end_ids)
        if len(tokens) >= args.max_bar_tokens and grammar.prev in (None, "Dur"):
            mask = torch.zeros_like(mask)
            mask[end_ids] = True
        token = sample(logits, mask, args.temperature, args.top_p)
        if token in end_ids.tolist():
            return tokens, token
        tokens.append(token)
        grammar.update(token)
        logits = stepper.feed([token])

def generate_dec(model, bars, grammar, args, device):
    """
    Bar by bar: feed the bar's header + melody (given), [sample the bar's chord,] then sample piano notes until
    the model closes the bar. Returns (piano tokens, sampled chord classes or None).
    """
    use_chord = getattr(model.config, "chord", False)
    stepper = Stepper(model, device)
    piano, chords, pending = [], [], [BOS]
    for i, bar in enumerate(bars):
        head = bar_header(i, bar)
        given = pending + head + [MEL] + note_tokens(bar["mel"])
        if use_chord:
            logits = stepper.feed(given)
            chord = sample_chord(logits, grammar, bar["mel"], args)
            chords.append(chord - CHORD0)
            logits = stepper.feed([chord, ACC])
        else:
            logits = stepper.feed(given + [ACC])
        # any Bar token closes the bar; the right Bar_{i+1} is then fed with the next header
        tokens, _ = sample_bar(stepper, logits, grammar, bar["beats"], grammar.bar, args)
        piano += [head[0]] + tokens
        pending = []
    return piano, chords if use_chord else None

def generate_s2s(model, bars, grammar, args, device, encoder_ids):
    """
    The encoder reads the whole melody; the decoder writes the piano, with the Bar_i tokens forced in order
    [and each bar's chord sampled right after its Bar token]. Returns (piano tokens, chord classes or None).
    """
    use_chord = getattr(model.config, "chord", False)
    stepper = Stepper(model, device, encoder_ids)
    piano, chords = [], []
    end_ids = torch.cat([grammar.bar, torch.tensor([EOS], device=device)])
    logits = stepper.feed([BOS])
    for i, bar in enumerate(bars):
        bar_id = VOCAB[f"Bar_{i}"]
        logits = stepper.feed([bar_id])
        piano.append(bar_id)
        if use_chord:
            chord = sample_chord(logits, grammar, bar["mel"], args)
            chords.append(chord - CHORD0)
            logits = stepper.feed([chord])
        tokens, _ = sample_bar(stepper, logits, grammar, bar["beats"], end_ids, args)
        piano += tokens
    return piano, chords if use_chord else None

def generate_all(model, grammar, prompts, args, device, out_dir):
    """
    Write one MIDI per prompt into out_dir, sampling with args.temperature / args.top_p / args.seed.
    For a chord model, also writes chords.csv and returns the fraction of bars whose sampled chord equals the
    prompt's labelled chord (same per-bar simplification as CA); None otherwise.
    """
    torch.manual_seed(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    chord_rows = []
    for prompt in prompts:
        bars, grid, melody = load_prompt(prompt)
        if args.arch == "dec":
            piano, chords = generate_dec(model, bars, grammar, args, device)
        else:
            piano, chords = generate_s2s(model, bars, grammar, args, device, encoder_seq(bars))
        if chords is not None:
            truth = bar_chords(prompt / "chord_midi.txt", grid)
            chord_rows += [{"song_name": prompt.name, "bar": k, "predicted": chord_name(c), "labelled": chord_name(t)}
                           for k, (c, t) in enumerate(zip(chords, truth))]
        midi = pretty_midi.PrettyMIDI()
        mel_track = pretty_midi.Instrument(0, name="MELODY")
        mel_track.notes = melody.notes
        piano_track = pretty_midi.Instrument(0, name="PIANO")
        piano_track.notes = decode_piano(piano, grid)
        midi.instruments = [mel_track, piano_track]
        midi.write(str(out_dir / f"{prompt.name}.mid"))
        print(f"{prompt.name}: {len(bars)} bars, {len(piano_track.notes)} piano notes -> {out_dir / f'{prompt.name}.mid'}")
    if not chord_rows:
        return None
    chord_df = pd.DataFrame(chord_rows)
    chord_df.to_csv(out_dir / "chords.csv", index=False)
    acc = float((chord_df["predicted"] == chord_df["labelled"]).mean())
    print(f"sampled chord == labelled chord in {acc:.3f} of bars")
    return acc

def evaluate_dir(prompt_dir, gen_dir):
    """eval_bonus.py on one folder: (per-song rows, metrics pooled over all bars like its summary)."""
    eval_args = argparse.Namespace(prompt_dir=str(prompt_dir), gen_midi_dir=str(gen_dir))
    rows = [{"song_name": f.stem, **eval_bonus.evaluate_song(f.stem, eval_args)}
            for f in sorted(Path(gen_dir).glob("*.mid"))]
    df = pd.DataFrame(rows)
    pooled = {c: (df[c] * df["n_bars"]).sum() / df["n_bars"].sum() for c in eval_bonus.METRIC_COLS}
    return df, pooled

def fmt(x):
    """1.0 -> '1', 0.95 -> '095' (the naming used for the earlier output folders)."""
    return f"{x:g}".replace(".", "")

def sweep(model, grammar, prompts, args, device):
    root = Path(args.out_dir or f"sweep_{args.arch}")
    results = []
    # chord settings: -1 = chord sampled like the notes, 0 = no melody prior (both left out of folder names)
    chord_temps = args.chord_temperatures or [-1 if args.chord_temperature is None else args.chord_temperature]
    mel_weights = args.chord_mel_weights or [args.chord_mel_weight]
    if not getattr(model.config, "chord", False):
        chord_temps, mel_weights = [-1], [0.0]
    combos = list(itertools.product(args.temperatures, args.top_ps, chord_temps, mel_weights, args.seeds))
    for i, (temperature, top_p, chord_temp, mel_weight, seed) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] temperature={temperature} top_p={top_p} chord_temperature={chord_temp} "
              f"chord_mel_weight={mel_weight} seed={seed}")
        run = argparse.Namespace(**{**vars(args), "temperature": temperature, "top_p": top_p, "seed": seed,
                                    "chord_temperature": None if chord_temp < 0 else chord_temp,
                                    "chord_mel_weight": mel_weight})
        name = f"temp{fmt(temperature)}_topp{fmt(top_p)}"
        name += (f"_ctemp{fmt(chord_temp)}" if chord_temp >= 0 else "") + (f"_melw{fmt(mel_weight)}" if mel_weight else "")
        gen_dir = root / f"{name}_seed{seed}"
        chord_acc = generate_all(model, grammar, prompts, run, device, gen_dir)
        per_song, pooled = evaluate_dir(args.prompt_dir, gen_dir)
        per_song.to_csv(gen_dir / "metrics.csv", index=False)
        if chord_acc is not None:
            pooled["chord_pred_acc"] = chord_acc
        results.append({"temperature": temperature, "top_p": top_p, "chord_temperature": chord_temp,
                        "chord_mel_weight": mel_weight, "seed": seed, **pooled})
        # rewritten after every run, so a stopped sweep still leaves its results
        pd.DataFrame(results).to_csv(root / "sweep_runs.csv", index=False)

    # average the seeds of each setting; rank by chord accuracy, then pitch-class OA
    cols = eval_bonus.METRIC_COLS + (["chord_pred_acc"] if "chord_pred_acc" in results[0] else [])
    keys = ["temperature", "top_p", "chord_temperature", "chord_mel_weight"]
    summary = pd.DataFrame(results).groupby(keys)[cols].agg(["mean", "std"])
    summary.columns = [f"{m} {s}" for m, s in summary.columns]
    summary = summary.sort_values(["CA mean", "DP (Pitch-class OA) mean"], ascending=False)
    summary.to_csv(root / "sweep.csv")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        print(f"\n=== {args.arch}: pooled over {len(prompts)} prompts, mean over seeds {args.seeds} ===")
        print(summary[[c for c in summary.columns if c.endswith("mean") and not c.startswith("CA-ceiling")]])
    print(f"\nper-run results: {root / 'sweep_runs.csv'}, summary: {root / 'sweep.csv'}")

def main():
    args = parse_args()
    device = get_device(args.device)
    ckpt = args.ckpt_dir or f"checkpoints_accomp_{args.arch}/best"
    model = load_model(ckpt, args.arch).to(device).eval()
    grammar = PianoGrammar(device, model.config.vocab_size)
    prompts = sorted(p for p in Path(args.prompt_dir).iterdir() if p.is_dir())

    if args.sweep:
        sweep(model, grammar, prompts, args, device)
    else:
        generate_all(model, grammar, prompts, args, device, Path(args.out_dir or f"accomp_{args.arch}"))

if __name__ == "__main__":
    main()
