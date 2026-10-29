"""
Compute the HW2 basic-task metrics (H1, H4, GS) and the Structureness Indicator (SI) directly from MIDI files.

HW2_scripts/eval_basic.py expects songs already in the TA's token format. This script turns each MIDI into
exactly that format (same vocabulary, same beat grid, reusing the helpers in HW2_scripts/tokenization.py),
then computes the same metrics as eval_basic.py / MusDr, so the numbers are comparable to the official script.

Beat grid per MIDI:
  - if a beat_midi.txt sits next to the MIDI (POP909 songs and prompts), it is used, like the TA's tokenizer
  - otherwise (e.g. model outputs) the MIDI's own tempo map is used: one beat per quarter note, 4/4 bars

e.g. python eval_midi.py --midi_dir generated --output_csv generated_metrics.csv
     python eval_midi.py --midi_dir POP909/training --output_csv real_metrics.csv   # real-data reference
"""

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pretty_midi
from tqdm import tqdm

from HW2_scripts.tokenization import create_vocabulary, get_beat_info, time_to_grid

POS_PER_BEAT = 16  # tokenization.py's time_to_grid always uses its default of 16
MAX_POSITIONS = 8 * POS_PER_BEAT  # = len(POS_EVS); GS divides by this, as in eval_basic.py

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", required=True, help="Folder searched recursively for .mid/.midi files")
    parser.add_argument("--output_csv", default="midi_metrics.csv")
    parser.add_argument("--ignore_beat_files", action="store_true",
                        help="Always use the MIDI's own tempo map, even if a beat_midi.txt is next to the MIDI")
    parser.add_argument("--sequences_output", default=None,
                        help="Also save the TA-format tokens, so HW2_scripts/eval_basic.py can be run on them")
    parser.add_argument("--seed", type=int, default=0, help="GS samples 1000 bar pairs when a song has more")
    parser.add_argument("--no_si", action="store_true", help="Skip the Structureness Indicator (the slowest metric)")
    parser.add_argument("--scape_plot_dir", default=None,
                        help="Also save each song's fitness scape plot (the SI's source) as a PNG here; needs matplotlib")
    return parser.parse_args()

def get_beats(midi_data, midi_path, ignore_beat_files):
    """Returns (beat_times, beat_orders) where beat_orders is 1.0 on downbeats, like tokenization.py reads them."""
    beat_path = midi_path.parent / "beat_midi.txt"
    if beat_path.exists() and not ignore_beat_files:
        beat_times, beat_orders = [], []
        with open(beat_path, "r") as f:
            for line in f:
                parts = line.split()  # [start_time(sec), strong/weak beat, beat_order(is it a downbeat?)]
                if not parts: continue
                beat_times.append(round(float(parts[0]), 6))
                beat_orders.append(float(parts[2]) if len(parts) > 2 else 0.0)
        return beat_times, beat_orders

    # No annotation: quarter notes from the MIDI's tempo map, a downbeat every 4 beats
    beat_times = [round(float(t), 6) for t in midi_data.get_beats()]
    beat_orders = [1.0 if i % 4 == 0 else 0.0 for i in range(len(beat_times))]
    return beat_times, beat_orders

def midi_to_ta_tokens(midi_path, ignore_beat_files=False):
    """Same steps as process_pop909_dataset() in HW2_scripts/tokenization.py, for a single MIDI file."""
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    beat_times, beat_orders = get_beats(midi_data, midi_path, ignore_beat_files)
    beat_info = get_beat_info(beat_orders)

    tempo_times, tempi = midi_data.get_tempo_changes()
    tempo_events = [{"time": round(float(t), 6), "qpm": round(qpm)} for t, qpm in zip(tempo_times, tempi)]

    all_events = []
    for inst in midi_data.instruments:
        for note in inst.notes:
            bar, pos, b_len = time_to_grid(note.start, beat_times, beat_info)
            dur = max(1, min(MAX_POSITIONS * 2, int(((note.end - note.start) / b_len) * POS_PER_BEAT)))
            vel = max(1, note.velocity // 4 * 4 + 1)
            all_events.append({"bar": bar, "pos": pos, "type": "note", "pitch": note.pitch, "vel": vel, "dur": dur})

    for t in tempo_events:
        bar, pos, _ = time_to_grid(t["time"], beat_times, beat_info)
        val = max(30, min(210, int(t["qpm"]) // 5 * 5))
        all_events.append({"bar": bar, "pos": pos, "type": "tempo", "val": val})

    bars = {}
    for ev in all_events:
        bars.setdefault(ev["bar"], []).append(ev)
    max_bar = max(bars.keys()) if bars else 0

    # skip silent intro bars
    first_note_bar = next((b for b in range(max_bar + 1) if any(e["type"] == "note" for e in bars.get(b, []))), None)
    if first_note_bar is None:
        return None

    type_order = {"tempo": 0, "note": 1}
    tokens = ["<BOS>"]
    for bar_idx in range(first_note_bar, max_bar + 1):
        tokens.append("Bar")
        for e in sorted(bars.get(bar_idx, []), key=lambda e: (e["pos"], type_order[e["type"]], e.get("pitch", 0))):
            tokens.append(f"Pos_{e['pos']}")
            if e["type"] == "tempo":
                tokens.append(f"Tempo_{e['val']}")
            else:
                tokens.extend([f"Pitch_{e['pitch']}", f"Vel_{e['vel']}", f"Dur_{e['dur']}"])
    tokens.append("<EOS>")
    return tokens

# --- Metrics: same logic as eval_basic.py and the MusDr helpers it imports (musdr/side_utils.py),
# --- rewritten with numpy only so this script doesn't need MusDr, pandas or scipy installed.

def get_bars_crop(ev_seq, start_bar, end_bar, bar_ev_id):
    bar_markers = [i for i, ev in enumerate(ev_seq) if ev == bar_ev_id]
    end = bar_markers[end_bar + 1] if end_bar < len(bar_markers) - 1 else len(ev_seq)
    return ev_seq[bar_markers[start_bar]:end]

def get_pitch_histogram(ev_seq, pitch_evs):
    # like MusDr, pitch classes are taken from the token ids (a constant offset from the pitch, so entropy is the same)
    pitch_set = set(pitch_evs)
    classes = [ev % 12 for ev in ev_seq if ev in pitch_set]
    if not classes:
        return None
    return np.bincount(classes, minlength=12) / len(classes)

def compute_histogram_entropy(hist):
    p = hist[hist > 0]
    return float(-(p * np.log2(p)).sum())

def get_onset_xor_distance(seq_a, seq_b, pos_evs, pitch_evs):
    n_pos = len(pos_evs)
    pos_set, pitch_set = set(pos_evs), set(pitch_evs)
    def make_onset_vec(seq):
        cur_pos = -1
        onset_vec = np.zeros((n_pos,))
        for ev in seq:
            if ev in pos_set:
                cur_pos = ev - pos_evs[0]
            if ev in pitch_set:
                onset_vec[cur_pos] = 1
        return onset_vec
    return np.sum(np.abs(make_onset_vec(seq_a) - make_onset_vec(seq_b))) / n_pos

def compute_piece_pitch_entropy(piece_ev_seq, window_size, bar_ev_id, pitch_evs):
    """For calculating H1 and H4"""
    if piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]
    n_bars = piece_ev_seq.count(bar_ev_id)
    window_size = min(window_size, n_bars)
    pitch_ents = []
    for st_bar in range(0, n_bars - window_size + 1):
        seg_ev_seq = get_bars_crop(piece_ev_seq, st_bar, st_bar + window_size - 1, bar_ev_id)
        pitch_hist = get_pitch_histogram(seg_ev_seq, pitch_evs)
        if pitch_hist is None:
            continue
        pitch_ents.append(compute_histogram_entropy(pitch_hist))
    return np.mean(pitch_ents) if pitch_ents else np.nan

def compute_piece_groove_similarity(piece_ev_seq, bar_ev_id, pos_evs, pitch_evs, max_pairs=1000):
    """For calculating GS"""
    if piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]
    n_bars = piece_ev_seq.count(bar_ev_id)
    bar_seqs = [get_bars_crop(piece_ev_seq, b, b, bar_ev_id) for b in range(n_bars)]
    pairs = list(itertools.combinations(range(n_bars), 2))
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    grv_sims = [1.0 - get_onset_xor_distance(bar_seqs[a], bar_seqs[b], pos_evs, pitch_evs) for a, b in pairs]
    return np.mean(grv_sims) if grv_sims else np.nan

# --- Structureness Indicator (SI), as in MusDr's compute_structure_indicator(): the max of the fitness scape plot
# --- over segments whose duration is within [low, upp] seconds. MusDr reads that scape plot from a .mat made by the
# --- Matlab SM Toolbox on synthesized audio; here it is rebuilt in numpy (following Müller's audio thumbnailing,
# --- same parameters as the FMP notebooks) from the MIDI's own chroma, so no audio rendering or Matlab is needed.

SI_FPS = 2  # feature rate of the scape plot (MusDr's sample_rate=2)
SI_BOUNDS = {"SI_short": (3, 8), "SI_mid": (8, 15), "SI_long": (15, None)}  # seconds, as in the Jazz Transformer paper

def midi_to_cens(midi_data, in_fps=10, win=21):
    """CENS^21_5 chroma: 10 Hz chroma, quantized, Hann-smoothed over ~2 s, downsampled to 2 Hz, L2-normalized."""
    chroma = midi_data.get_chroma(fs=in_fps)
    energy = chroma.sum(axis=0, keepdims=True)
    chroma = np.divide(chroma, energy, out=np.zeros_like(chroma), where=energy > 0)
    quant = sum((chroma >= thr).astype(float) for thr in (0.05, 0.1, 0.2, 0.4))
    hann = np.hanning(win + 2)[1:-1]
    cens = np.stack([np.convolve(q, hann, mode="same") for q in quant])[:, ::in_fps // SI_FPS]
    norm = np.linalg.norm(cens, axis=0, keepdims=True)
    return cens / np.maximum(norm, 1e-12)  # silent frames stay all-zero, so silence never counts as a repetition

def smooth_diagonal(ssm, length, forward=True):
    """Average each cell with the next (or previous) length-1 cells on its diagonal; zero-padded at the edges like FMP."""
    n = len(ssm)
    acc = np.zeros_like(ssm)
    for l in range(length):
        if l >= n:
            break
        if forward:
            acc[:n - l, :n - l] += ssm[l:, l:]
        else:
            acc[l:, l:] += ssm[:n - l, :n - l]
    return acc / length

def compute_ssm(feat, smooth_len=12, thresh=0.15, penalty=-2.0):
    """Transposition-invariant, path-enhanced SSM; keeps the top 15% scaled to [0, 1], the rest gets the penalty."""
    ssm = np.full((feat.shape[1],) * 2, -np.inf)
    for shift in range(12):
        s = feat.T @ np.roll(feat, shift, axis=0)
        ssm = np.maximum(ssm, np.maximum(smooth_diagonal(s, smooth_len, True), smooth_diagonal(s, smooth_len, False)))
    cut, top = np.quantile(ssm, 1 - thresh), ssm.max()
    scaled = np.clip((ssm - cut) / (top - cut), 0, 1) if top > cut else np.ones_like(ssm)
    return np.where(ssm >= cut, scaled, penalty)

def optimal_path_family(sub):
    """
    Optimal path family over sub (k, n, m): for each of k segments, the n x m SSM block of all frames vs. the segment.
    Same DP as FMP's compute_accumulated_score_matrix (steps (1,1), (2,1), (1,2) plus the "elevator" column 0
    that lets one path end and the next one start), vectorized over the k segments. Instead of backtracking,
    the path family's total length (cells) and coverage (rows of its induced segments) are carried along.
    Returns (score, length, coverage), each of shape (k,).
    """
    k, n, m = sub.shape
    def empty():
        return np.full((k, m + 1), -np.inf), np.zeros((k, m + 1)), np.zeros((k, m + 1))
    prev2 = empty()
    d, ln, cov = empty()
    d[:, 0], d[:, 1], ln[:, 1], cov[:, 1] = 0.0, sub[:, 0, 0], 1, 1
    prev = (d, ln, cov)
    for row in range(1, n):
        (pd, pl, pc), (qd, ql, qc) = prev, prev2
        d, ln, cov = empty()
        # elevator: idle, or finish the path that just reached the segment's last column
        take_end = pd[:, m] > pd[:, 0]
        d[:, 0] = np.where(take_end, pd[:, m], pd[:, 0])
        ln[:, 0] = np.where(take_end, pl[:, m], pl[:, 0])
        cov[:, 0] = np.where(take_end, pc[:, m], pc[:, 0])
        # a new path starts at the segment's first column
        d[:, 1], ln[:, 1], cov[:, 1] = d[:, 0] + sub[:, row, 0], ln[:, 0] + 1, cov[:, 0] + 1
        if m > 1:
            best_d, best_l, best_c = pd[:, 1:m], pl[:, 1:m], pc[:, 1:m] + 1           # step (1, 1)
            cands = [(qd[:, 1:m], ql[:, 1:m], qc[:, 1:m] + 2)]                        # step (2, 1)
            if m > 2:  # step (1, 2), from real columns only
                cands.append((np.pad(pd[:, 1:m - 1], ((0, 0), (1, 0)), constant_values=-np.inf),
                              np.pad(pl[:, 1:m - 1], ((0, 0), (1, 0))), np.pad(pc[:, 1:m - 1], ((0, 0), (1, 0))) + 1))
            for cd, cl, cc in cands:
                better = cd > best_d
                best_d, best_l, best_c = np.where(better, cd, best_d), np.where(better, cl, best_l), np.where(better, cc, best_c)
            d[:, 2:], ln[:, 2:], cov[:, 2:] = best_d + sub[:, row, 1:], best_l + 1, best_c
        prev2, prev = prev, (d, ln, cov)
    d, ln, cov = prev
    take_end = d[:, m] > d[:, 0]
    return (np.where(take_end, d[:, m], d[:, 0]), np.where(take_end, ln[:, m], ln[:, 0]),
            np.where(take_end, cov[:, m], cov[:, 0]))

def compute_fitness_scape(ssm, min_len=1):
    """Fitness scape plot, indexed [segment length in frames, start frame] like MusDr's fitness .mat."""
    n = len(ssm)
    fitness = np.zeros((n + 1, n))
    diag_cumsum = np.concatenate([[0.0], np.cumsum(np.diag(ssm))])
    for m in range(max(1, min_len), n + 1):
        starts = n - m + 1
        sub = np.lib.stride_tricks.sliding_window_view(ssm, m, axis=1)[:, :starts].transpose(1, 0, 2)  # (starts, n, m)
        score, length, cover = optimal_path_family(sub)
        # the trivial self-match path is discounted; FMP subtracts m, assuming a diagonal of 1, but silence or the
        # zero-padded edges can lower the diagonal, so subtract what the diagonal actually contributes
        self_score = diag_cumsum[m:m + starts] - diag_cumsum[:starts]
        score_n = (score - self_score) / (length + 1e-16)
        cover_n = (cover - m) / (n + 1e-16)
        both = (score_n > 0) & (cover_n > 0)  # otherwise the harmonic mean is meaningless (and can blow up)
        fitness[m, :starts] = np.where(both, 2 * score_n * cover_n / np.where(both, score_n + cover_n, 1), 0)
    return fitness

def structure_analysis(midi_path):
    """SSM, fitness, SI per band and each band's peak (length, start) in frames; see compute_structure_indicators."""
    feat = midi_to_cens(pretty_midi.PrettyMIDI(str(midi_path)))
    ssm = compute_ssm(feat)
    fitness = compute_fitness_scape(ssm, min_len=min(lo for lo, _ in SI_BOUNDS.values()) * SI_FPS)
    result, peaks = {}, {}
    for name, (lo, hi) in SI_BOUNDS.items():
        band = fitness[lo * SI_FPS: (len(fitness) if hi is None else hi * SI_FPS + 1)]
        result[name] = float(band.max()) if band.size else np.nan
        if band.size and band.max() > 0:  # no marker when nothing repeats at this scale
            m, start = np.unravel_index(band.argmax(), band.shape)
            peaks[name] = (m + lo * SI_FPS, start)
    return ssm, fitness, result, peaks

def compute_structure_indicators(midi_path, plot_path=None):
    """SI_short / SI_mid / SI_long: MusDr's np.max(fitness[low * fps : upp * fps + 1]) for each duration band."""
    ssm, fitness, result, peaks = structure_analysis(midi_path)
    if plot_path is not None:
        plot_scape(ssm, fitness, result, peaks, plot_path, title=Path(midi_path).stem)
    return result

def scape_image(fitness):
    """Fitness as an image: x = segment center (half-frame grid), y = segment length in frames."""
    n = len(fitness) - 1
    # scape[m, c]: fitness of the segment of length m whose center is at c / 2 frames (half-frame grid)
    scape = np.full((n + 1, 2 * n + 1), np.nan)
    for m in range(1, n + 1):
        starts = np.arange(n - m + 1)
        scape[m, 2 * starts + m] = fitness[m, starts]
        scape[m, 2 * starts + m + 1] = fitness[m, starts]  # fill the gap between neighbouring centers
    scape[:min(SI_BOUNDS[k][0] for k in SI_BOUNDS) * SI_FPS] = np.nan  # not computed
    return scape

def plot_scape(ssm, fitness, si, peaks, plot_path, title):
    """SSM (left) and fitness scape plot (right): x = segment center, y = segment duration, both in seconds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(ssm)
    scape = scape_image(fitness)
    fig, (ax_ssm, ax_fit) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})
    dur = n / SI_FPS
    im = ax_ssm.imshow(ssm, origin="lower", cmap="gray_r", vmin=-2, vmax=1, extent=(0, dur, 0, dur))
    ax_ssm.set(title="Self-similarity matrix (thresholded)", xlabel="time (s)", ylabel="time (s)")
    fig.colorbar(im, ax=ax_ssm, fraction=0.046)

    im = ax_fit.imshow(scape, origin="lower", cmap="hot_r", vmin=0, vmax=1, aspect="auto",
                       extent=(0, dur, 0, dur), interpolation="nearest")
    colors = {"SI_short": "tab:blue", "SI_mid": "tab:green", "SI_long": "tab:purple"}
    for name, (lo, hi) in SI_BOUNDS.items():
        ax_fit.axhline(lo, color=colors[name], ls="--", lw=0.8)
        if name in peaks:
            m, start = peaks[name]
            ax_fit.plot((start + m / 2) / SI_FPS, m / SI_FPS, "o", mfc="none", mec=colors[name], ms=10, mew=2,
                        label=f"{name} = {si[name]:.3f}  ({start / SI_FPS:.1f}-{(start + m) / SI_FPS:.1f} s)")
    ax_fit.legend(loc="upper right", fontsize=8)
    ax_fit.set(title="Fitness scape plot", xlabel="segment center (s)", ylabel="segment duration (s)")
    fig.colorbar(im, ax=ax_fit, fraction=0.046, label="fitness")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

def main():
    args = parse_args()
    random.seed(args.seed)

    tok2idx = create_vocabulary(args.midi_dir, MAX_POSITIONS)
    bar_ev = tok2idx["Bar"]
    pos_evs = [tok2idx[f"Pos_{i}"] for i in range(MAX_POSITIONS)]
    pitch_evs = [tok2idx[f"Pitch_{p}"] for p in range(21, 109)]

    midi_paths = sorted(p for p in Path(args.midi_dir).rglob("*") if p.suffix.lower() in (".mid", ".midi"))
    if not midi_paths:
        raise FileNotFoundError(f"no MIDI files under {args.midi_dir}")
    print(f"Evaluating {len(midi_paths)} MIDI files...")
    if args.scape_plot_dir:
        Path(args.scape_plot_dir).mkdir(parents=True, exist_ok=True)

    rows, sequences = [], []
    for midi_path in tqdm(midi_paths):
        tokens = midi_to_ta_tokens(midi_path, args.ignore_beat_files)
        if tokens is None:
            print(f"skipping {midi_path}: no notes")
            continue
        song_name = str(midi_path.relative_to(args.midi_dir).with_suffix(""))
        plot_path = None
        if args.scape_plot_dir:
            plot_path = Path(args.scape_plot_dir) / (song_name.replace("/", "_") + ".png")
        sequences.append({"song_name": song_name, "tokens": tokens})
        seq_ids = [tok2idx[tok] for tok in tokens if tok in tok2idx]
        rows.append((song_name,
                     compute_piece_pitch_entropy(seq_ids, 1, bar_ev, pitch_evs),
                     compute_piece_pitch_entropy(seq_ids, 4, bar_ev, pitch_evs),
                     compute_piece_groove_similarity(seq_ids, bar_ev, pos_evs, pitch_evs),
                     *([] if args.no_si else compute_structure_indicators(midi_path, plot_path).values()),
                     tokens.count("Bar")))

    header = ["song_name", "H1 (Pitch Entropy 1-Bar)", "H4 (Pitch Entropy 4-Bar)", "GS (Groove Similarity)",
              *([] if args.no_si else SI_BOUNDS), "n_bars"]
    with open(args.output_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    print(f"\nEvaluation complete! Results saved to {args.output_csv}")
    for col, name in enumerate(header[1:-1], start=1):
        print(f"Average {name.split()[0]}: {np.nanmean([r[col] for r in rows]):.4f}")

    if args.sequences_output:
        with open(args.sequences_output, "w") as f:
            json.dump(sequences, f)
        print(f"TA-format tokens saved to {args.sequences_output}")

if __name__ == "__main__":
    main()
