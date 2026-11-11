"""Evaluate generated audio: FAD, KL, CLAP score, MQAScore.

    python evaluate.py --gen_dir outputs --ref_dir path/to/candidate_audio --mqa_repo path/to/MQAScore

Inputs
  --csv        test.csv (id, caption)
  --gen_dir    <id>.wav produced by batch_generate.py
  --retrieval  retrieval.json from clap_retrieve.py: {"<caption id>": "<audio id relative to --ref_dir, no extension>"}
  --ref_dir    the folder that was passed to clap_retrieve.py as --audio_dir

Metrics
  FAD   VGGish embeddings (frechet_audio_distance); reference = retrieved audio (one copy per caption id),
        evaluated = generated audio.
  KL    PANNs (Cnn14) AudioSet class distributions; per pair KL(P_retrieved || P_generated), averaged.
        P = softmax over the logit of the 527 sigmoid outputs.
  CLAP  cosine(generated audio, own caption) with the music CLAP checkpoint, averaged.
  MQA   MQAScore, caption protocol (one yes/no question per caption), run through the repo's scripts/score.py.
        Per-tag protocol needs Qwen3-Omni (60-70 GB) for tag extraction, so it is not used here.

Results go to eval_results.json (summary + per-item scores).
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]


def _np(x):
    """Some laion_clap versions return tensors, others numpy arrays."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def find_audio(base: Path, stem: str) -> Path:
    for ext in AUDIO_EXTS:
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no audio for '{stem}' under {base}")


def load_mono(path, sr, seconds=10.0):
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=seconds)
    return y.astype(np.float32)


# ---------------------------------------------------------------- FAD
def eval_fad(pairs, workdir: Path):
    from frechet_audio_distance import FrechetAudioDistance

    gen_dir, ref_dir = workdir / "fad_gen", workdir / "fad_ref"
    for d in (gen_dir, ref_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for cid, gen, ref in pairs:
        (gen_dir / f"{cid}{gen.suffix}").symlink_to(gen.resolve())
        (ref_dir / f"{cid}{ref.suffix}").symlink_to(ref.resolve())
    frechet = FrechetAudioDistance(
        model_name="vggish", sample_rate=16000, use_pca=False, use_activation=False, verbose=False
    )
    return float(frechet.score(str(ref_dir), str(gen_dir)))


# ---------------------------------------------------------------- KL
def eval_kl(pairs):
    from panns_inference import AudioTagging

    tagger = AudioTagging(checkpoint_path=None, device="cuda" if _cuda() else "cpu")  # downloads Cnn14 to ~/panns_data

    def dist(path):
        wav = load_mono(path, 32000)[None]
        p, _ = tagger.inference(wav)
        p = np.clip(p[0].astype(np.float64), 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p))
        e = np.exp(logits - logits.max())
        return e / e.sum()

    kls = {}
    for cid, gen, ref in pairs:
        p_ref, p_gen = dist(ref), dist(gen)
        kls[cid] = float(np.sum(p_ref * (np.log(p_ref + 1e-10) - np.log(p_gen + 1e-10))))
    return kls


# ---------------------------------------------------------------- CLAP
def eval_clap(rows, gen_paths, ckpt):
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    # torch>=2.6 defaults to weights_only=True, which rejects this (trusted, official LAION) checkpoint.
    import torch

    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    # transformers>=5 no longer keeps RoBERTa's `position_ids` buffer, but the checkpoint still stores it.
    # Drop just that key; every other mismatch still raises (strict loading is kept).
    _orig_lsd = model.model.load_state_dict
    model.model.load_state_dict = lambda sd, *a, **kw: _orig_lsd(
        {k: v for k, v in sd.items() if k != "text_branch.embeddings.position_ids"}, *a, **kw
    )
    try:
        model.load_ckpt(ckpt)
    finally:
        torch.load = _orig_load
    ids =[r["id"] for r in rows]
    files = [str(gen_paths[i]) for i in ids]
    with torch.no_grad():  # laion_clap does not disable autograd itself; without this activations pile up
        a = np.concatenate([_np(model.get_audio_embedding_from_filelist(x=files[i : i + 8])) for i in range(0, len(files), 8)])
    t = _np(model.get_text_embedding([r["caption"] for r in rows]))
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    return {i: float(s) for i, s in zip(ids, (a * t).sum(-1))}


# ---------------------------------------------------------------- MQAScore
def eval_mqa(rows, gen_paths, repo: Path, model_name, python, workdir: Path):
    inp, out = workdir / "mqa_in.jsonl", workdir / "mqa_out.jsonl"
    out.unlink(missing_ok=True)
    Path(str(out) + ".meta.json").unlink(missing_ok=True)
    with open(inp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "audio": str(gen_paths[r["id"]].resolve()), "text": r["caption"]}) + "\n")
    cmd = [python, "scripts/score.py", "--model", model_name, "--protocol", "caption",
           "--input", str(inp), "--output", str(out)]
    subprocess.run(cmd, cwd=repo, check=True)
    scores = {}
    with open(out, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            scores[str(d["id"])] = float(d["caption_score"])
    return scores


def _cuda():
    import torch

    return torch.cuda.is_available()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="test.csv")
    ap.add_argument("--gen_dir", default="outputs")
    ap.add_argument("--retrieval", default="retrieval.json")
    ap.add_argument("--ref_dir", required=True, help="folder CLAP retrieved from (the --audio_dir of clap_retrieve.py)")
    ap.add_argument("--ckpt", default="music_audioset_epoch_15_esc_90.14.pt")
    ap.add_argument("--mqa_repo", default=None, help="clone of github.com/Lonian6/MQAScore")
    ap.add_argument("--mqa_model", default="mf", choices=["mf", "af", "qwen3omni"])
    ap.add_argument("--mqa_python", default=sys.executable, help="python of the env that has MQAScore's deps")
    ap.add_argument("--metrics", default="fad,kl,clap", help="MQA is optional")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()
    metrics = set(args.metrics.split(","))
    # laion_clap (also imported by frechet_audio_distance) runs its own argparse on sys.argv; hide our flags.
    sys.argv = sys.argv[:1]

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    retrieval = json.load(open(args.retrieval, encoding="utf-8"))
    gen_paths = {r["id"]: find_audio(Path(args.gen_dir), r["id"]) for r in rows}
    ref_paths = {r["id"]: find_audio(Path(args.ref_dir), retrieval[r["id"]]) for r in rows}
    pairs = [(r["id"], gen_paths[r["id"]], ref_paths[r["id"]]) for r in rows]
    print(f"{len(rows)} items")

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="eval_"))
    workdir.mkdir(parents=True, exist_ok=True)

    summary, per_item = {}, {r["id"]: {} for r in rows}

    def add(name, scores):
        for k, v in scores.items():
            per_item[k][name] = v
        summary[name] = float(np.mean(list(scores.values())))

    if "fad" in metrics:
        summary["FAD"] = eval_fad(pairs, workdir)
        print("FAD", summary["FAD"])
    if "kl" in metrics:
        add("KL", eval_kl(pairs))
        print("KL", summary["KL"])
    if "clap" in metrics:
        add("CLAP", eval_clap(rows, gen_paths, args.ckpt))
        print("CLAP", summary["CLAP"])
    if "mqa" in metrics:
        if not args.mqa_repo:
            raise SystemExit("--mqa_repo is required for MQAScore")
        add("MQAScore", eval_mqa(rows, gen_paths, Path(args.mqa_repo), args.mqa_model, args.mqa_python, workdir))
        print("MQAScore", summary["MQAScore"])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_item": per_item}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
