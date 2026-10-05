"""Which layer knows the label? Linear probe per layer with stratified K-fold CV on train+validation.

For every layer of each feature file: standardise, fit a logistic regression, and score the
out-of-fold predictions with the task metrics (top1 / top3 / combined).

    python probe_layers.py --task market features/market/large-v3_vocals.npz features/market/MERT-v1-330M_no_vocals.npz
    python probe_layers.py --task year features/year/large-v3_vocals.npz features/year/MERT-v1-330M_no_vocals.npz
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import TASKS
from train_features import load_data, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS), help="year = decade, market = release market")
    parser.add_argument("features", nargs="+")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--C", type=float, default=0.01, help="Inverse L2 strength; small = strong regularisation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Probe Whisper's language probabilities too, when a Whisper file is among the inputs
    has_lang = any("lang_probs" in np.load(f) for f in args.features)
    branches, names, _, labels, _ = load_data(args.features, args.task, lang_probs=has_lang)
    labelled = labels >= 0
    y = labels[labelled]
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=3000))

    for name, feats in zip(names, branches):
        print(f"\n=== {name}  ({feats.shape[1]} layers x {feats.shape[2]} dims) ===")
        print("layer   top1   top3   combined")
        scores = []
        for layer in range(feats.shape[1]):
            probs = cross_val_predict(probe, feats[labelled, layer], y, cv=cv, method="predict_proba")
            m = metrics(probs, y)
            scores.append(m["combined"])
            print(f"{layer:5d}  {m['top1']:.3f}  {m['top3']:.3f}  {m['combined']:.3f}  " + "#" * int(40 * m["top1"]))
        print(f"best layer: {int(np.argmax(scores))} (combined {max(scores):.3f})")


if __name__ == "__main__":
    main()
