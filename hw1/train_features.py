"""Frozen Whisper / MERT features -> the same dense head as ShortChunkCNN, for a like-for-like comparison.

Works for both tasks, like train.py: --task year (dataset_A, decades) or --task market (dataset_B).

Each --features file holds every layer of one model. Its layers are standardised (LayerNorm) and
averaged with learned softmax weights (SUPERB style) into one vector. With --lang_probs, Whisper's
language log-probabilities are appended. The concatenated vector goes through the CNN's dense head
(model._DenseHead: Linear -> BatchNorm -> ReLU -> Dropout(0.5) -> Linear), trained with train.py's
recipe (AdamW, warmup + cosine, label smoothing, batch 16). The features replace the conv encoder.

Evaluation:
  --folds K (default 5): stratified K-fold CV on train+validation. The score is at the epoch with the
      best mean out-of-fold score; the test submission averages the K fold models at that epoch.
  --folds 0: the manifest's train/validation split, like train.py, so numbers are directly comparable.

Logs to wandb with train.py's metric names (train/*, val/*, best/*, val/confusion_matrix against
"epoch"), so feature runs and CNN runs share charts. With K folds, val/* is the out-of-fold score
of all folds together at each epoch and train/* is the mean over folds; each fold's own curves are
logged under fold<k>/*, and the learned layer weights as bar charts.

Checkpoint: the selected epoch is only known once every fold has finished, so each fold is then
retrained up to that epoch (same seed and LR schedule, so it retraces the first run exactly; this
is checked) and all fold models are saved together in <ckpt_dir>/<run name>.pt. Reload them with
load_checkpoint().

    python train_features.py --task market --lang_probs \
        --features features/market/large-v3_vocals.npz features/market/MERT-v1-330M_no_vocals.npz
    python train_features.py --task year --lang_probs \
        --features features/year/large-v3_vocals.npz features/year/MERT-v1-330M_no_vocals.npz
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import StratifiedKFold

from dataset import TASKS
from model import _DenseHead
from train import build_scheduler


def load_data(feature_files, task, lang_probs=False):
    """-> branches: list of (N, L, D) float32 arrays, ids, labels (-1 for test), splits."""
    df = pd.read_csv(TASKS[task]["manifest"]).set_index("sample_id")
    branches, names, ids = [], [], None
    lang = None
    for path in feature_files:
        data = np.load(path)
        if ids is None:
            ids = data["ids"]
        assert (data["ids"] == ids).all(), f"{path} is not in the same clip order"
        assert set(ids) == set(df.index), f"{path} does not match the {task} manifest; wrong --task or feature file?"
        branches.append(data["feats"].astype(np.float32))
        names.append(path.split("/")[-1].removesuffix(".npz"))
        if lang_probs and lang is None and "lang_probs" in data:
            # log-probs as a 1-layer input so LayerMix handles it like any other file
            lang = np.log(data["lang_probs"] + 1e-6)[:, None, :].astype(np.float32)
    if lang_probs:
        assert lang is not None, "--lang_probs needs a Whisper feature file"
        branches.append(lang)
        names.append("lang_probs")
    classes = TASKS[task]["classes"]
    labels = df.loc[ids, "label"].map({c: i for i, c in enumerate(classes)}).fillna(-1).astype(int).to_numpy()
    return branches, names, ids, labels, df.loc[ids, "split"].to_numpy()


def metrics(scores, labels):
    top3 = np.argsort(-scores, axis=1)[:, :3]
    top1 = (top3[:, 0] == labels).mean()
    top3_acc = (top3 == labels[:, None]).any(axis=1).mean()
    return {"top1": top1, "top3": top3_acc, "combined": top1 + 0.5 * top3_acc}


class LayerMix(nn.Module):
    """(B, L, D) -> (B, D): learned softmax-weighted average of one model's layers."""

    def __init__(self, n_layers, dim):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))  # zeros = start from a uniform average
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)  # per layer, so scales are comparable

    def forward(self, x):
        w = self.layer_logits.softmax(dim=0)
        return (self.norm(x) * w[:, None]).sum(dim=1)


class FeatureClassifier(_DenseHead):
    """One LayerMix per feature file, concatenated, then ShortChunkCNN's dense head."""

    def __init__(self, shapes, n_class, hidden=256):
        super().__init__()
        self.mixes = nn.ModuleList(LayerMix(L, D) for L, D in shapes)
        self._build_head(sum(D for _, D in shapes), hidden, n_class)

    def forward(self, xs):
        return self._head(torch.cat([mix(x) for mix, x in zip(self.mixes, xs)], dim=1))


def load_checkpoint(path, device="cpu"):
    """-> (list of fold models in eval mode, checkpoint dict) from a train_features.py checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    models = []
    for state in ckpt["models"]:
        model = FeatureClassifier(ckpt["shapes"], len(ckpt["classes"]), ckpt["args"]["hidden"]).to(device)
        model.load_state_dict(state)
        models.append(model.eval())
    return models, ckpt


def train_fold(args, branches, labels, n_class, train_idx, eval_sets, device, seed, stop_epoch=None):
    """Train one model, optionally stopping after stop_epoch epochs (LR schedule unchanged).
    Returns {name: (epochs, n, n_class) logits} for each eval set plus "train_loss" / "train_acc"
    (epochs,), and the model. Seeded per fold, so the same call retraces the same run."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FeatureClassifier([b.shape[1:] for b in branches], n_class, args.hidden).to(device)
    # Layer-mixing logits: no weight decay (it would pull them back to uniform) and a higher LR,
    # since a softmax over ~30 logits otherwise moves very slowly. Everything else is train.py's AdamW.
    layer_params = [mix.layer_logits for mix in model.mixes]
    other_params = [p for n, p in model.named_parameters() if not n.endswith("layer_logits")]
    lrs = [args.lr * args.layer_lr_mult, args.lr]
    optimizer = torch.optim.AdamW([
        {"params": layer_params, "lr": lrs[0], "weight_decay": 0.0},
        {"params": other_params, "lr": lrs[1], "weight_decay": args.weight_decay},
    ])
    steps_per_epoch = int(np.ceil(len(train_idx) / args.batch_size))
    scheduler = build_scheduler(optimizer, args.warmup_epochs * steps_per_epoch, args.epochs * steps_per_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    xs = [torch.from_numpy(b).to(device) for b in branches]
    y = torch.from_numpy(labels).to(device)
    history = {name: [] for name in [*eval_sets, "train_loss", "train_acc"]}
    for _ in range(stop_epoch or args.epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(train_idx)).to(device)
        total_loss, total_correct = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = model([x[idx] for x in xs])
            loss = criterion(logits, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(dim=1) == y[idx]).sum().item()
        history["train_loss"].append(total_loss / len(perm))
        history["train_acc"].append(total_correct / len(perm))

        model.eval()
        with torch.no_grad():
            for name, idx in eval_sets.items():
                idx = torch.from_numpy(idx).to(device)
                history[name].append(model([x[idx] for x in xs]).cpu().numpy())

    return {name: np.stack(h) for name, h in history.items()}, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS), help="year = decade, market = release market")
    parser.add_argument("--features", nargs="+", required=True, help="npz files from extract_features.py, all layers of one model each")
    parser.add_argument("--lang_probs", action=argparse.BooleanOptionalAction, default=False,
                        help="Append Whisper's language log-probabilities to the features")
    parser.add_argument("--folds", type=int, default=5, help="0 = the manifest's train/validation split")
    # Defaults below match train.py, so only the input (features vs conv encoder) differs
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layer_lr_mult", type=float, default=10.0, help="LR multiplier for the layer weights")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=256, help="dense1 width; ShortChunkCNN uses 256")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Default: ./predictions_<task>_features.json")
    parser.add_argument("--ckpt_dir", default=None, help="Default: ./checkpoints/<task>_features")
    parser.add_argument("--wandb_project", default="deepmir-hw1")
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = TASKS[args.task]["classes"]
    args.output = args.output or f"./predictions_{args.task}_features.json"
    branches, names, ids, labels, splits = load_data(args.features, args.task, args.lang_probs)
    test = np.where(splits == "test")[0]
    if args.folds > 0:
        labelled = np.where(labels >= 0)[0]
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        folds = [(labelled[tr], labelled[va]) for tr, va in skf.split(labelled, labels[labelled])]
        protocol = f"{args.folds}-fold CV on {len(labelled)} clips"
    else:
        folds = [(np.where(splits == "train")[0], np.where(splits == "validation")[0])]
        labelled = folds[0][1]  # only validation clips get scored
        protocol = f"train {len(folds[0][0])} / validation {len(labelled)} (same split as train.py)"
    print("Inputs: " + ", ".join(f"{n} {b.shape[1:]}" for n, b in zip(names, branches)))
    print(f"{protocol}, {len(test)} test clips")

    run = wandb.init(project=args.wandb_project, name=args.run_name, tags=[args.task, "features"],
                     config={**vars(args), "inputs": names, "protocol": protocol})
    wandb.define_metric("epoch")
    for key in ["train/*", "val/*", "best/*", "fold*"]:
        wandb.define_metric(key, step_metric="epoch")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    oof = np.zeros((args.epochs, len(ids), len(classes)), dtype=np.float32)
    test_probs = np.zeros((args.epochs, len(test), len(classes)), dtype=np.float32)
    train_curves = []
    for fold, (tr, va) in enumerate(folds):
        history, _ = train_fold(args, branches, labels, len(classes), tr, {"val": va, "test": test}, device, args.seed + fold)
        oof[:, va] = history["val"]
        test_probs += torch.from_numpy(history["test"]).softmax(dim=-1).numpy() / len(folds)
        fold_scores = [metrics(history["val"][e], labels[va])["combined"] for e in range(args.epochs)]
        train_curves.append((history["train_loss"], history["train_acc"]))
        if len(folds) > 1:
            for e in range(args.epochs):
                wandb.log({"epoch": e + 1, f"fold{fold}/train_loss": history["train_loss"][e],
                           f"fold{fold}/val_combined": fold_scores[e]})
        print(f"fold {fold}: best combined {max(fold_scores):.3f} @ epoch {int(np.argmax(fold_scores)) + 1}, "
              f"last {fold_scores[-1]:.3f}")

    # Pick the epoch by mean out-of-fold (or validation) score, then report at that epoch
    curve = [metrics(oof[e, labelled], labels[labelled]) for e in range(args.epochs)]
    best = int(np.argmax([c["combined"] for c in curve]))
    m = curve[best]

    # Same per-epoch keys as train.py
    best_so_far = {"top1": -1.0, "combined": -1.0}
    for e, c in enumerate(curve):
        best_so_far = {k: max(best_so_far[k], c[k]) for k in best_so_far}
        val_loss = criterion(torch.from_numpy(oof[e, labelled]), torch.from_numpy(labels[labelled])).item()
        wandb.log({
            "epoch": e + 1,
            "train/loss": np.mean([loss[e] for loss, _ in train_curves]),
            "train/acc": np.mean([acc[e] for _, acc in train_curves]),
            "val/loss": val_loss,
            **{f"val/{k}": v for k, v in c.items()},
            **{f"best/{k}": v for k, v in best_so_far.items()},
        })
    print(f"\n{'CV' if args.folds > 0 else 'val'} @ epoch {best + 1}: top1 {m['top1']:.3f} | top3 {m['top3']:.3f} | combined {m['combined']:.3f}"
          f"   (last epoch: combined {curve[-1]['combined']:.3f})")

    preds = oof[best, labelled].argmax(axis=1)
    conf = pd.crosstab(pd.Series([classes[i] for i in labels[labelled]], name="true"),
                       pd.Series([classes[i] for i in preds], name="pred"))
    print(f"\n{'Out-of-fold' if args.folds > 0 else 'Validation'} confusion (rows = true, cols = predicted top1):")
    print(conf.reindex(index=classes, columns=classes, fill_value=0).to_string())

    wandb.log({"val/confusion_matrix": wandb.plot.confusion_matrix(
        y_true=labels[labelled].tolist(), preds=preds.tolist(), class_names=classes)})

    # Retrain each fold up to the selected epoch to get the weights behind the reported score
    print(f"\nRetraining {len(folds)} fold model(s) to epoch {best + 1} for the checkpoint...")
    models = []
    for fold, (tr, va) in enumerate(folds):
        history, model = train_fold(args, branches, labels, len(classes), tr, {"val": va}, device,
                                    args.seed + fold, stop_epoch=best + 1)
        drift = np.abs(history["val"][-1] - oof[best, va]).max()
        if drift > 1e-3:
            print(f"  warning: fold {fold} retrain differs from the first run by up to {drift:.4f} in logits "
                  f"(non-deterministic GPU kernels); the saved model is close but not identical")
        models.append(model)

    args.ckpt_dir = args.ckpt_dir or f"./checkpoints/{args.task}_features"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.run_name or run.name}.pt")
    torch.save({
        "models": [{k: v.cpu() for k, v in model.state_dict().items()} for model in models],
        "shapes": [tuple(b.shape[1:]) for b in branches],
        "inputs": names,
        "feature_files": args.features,
        "task": args.task,
        "classes": classes,
        "epoch": best + 1,
        "folds": [{"train": ids[tr].tolist(), "val": ids[va].tolist()} for tr, va in folds],
        "metrics": {k: float(v) for k, v in m.items()},
        "args": vars(args),
        "wandb_run_id": run.id,
    }, ckpt_path)
    print(f"Saved {len(models)} fold model(s) to {ckpt_path}")

    print("\nLearned layer weights at the selected epoch (mean over folds), top 5 layers per feature file:")
    for i, name in enumerate(names):
        w = np.mean([model.mixes[i].layer_logits.softmax(dim=0).detach().cpu().numpy() for model in models], axis=0)
        if len(w) > 1:
            table = wandb.Table(data=[[layer, float(v)] for layer, v in enumerate(w)], columns=["layer", "weight"])
            wandb.log({f"layer_weights/{name}": wandb.plot.bar(table, "layer", "weight", title=f"Layer weights: {name}")})
            top = np.argsort(-w)[:5]
            print(f"  {name}: " + ", ".join(f"L{l} {w[l]:.2f}" for l in top) + f"   (uniform = {1 / len(w):.2f})")

    top3 = np.argsort(-test_probs[best], axis=1)[:, :3]
    predictions = {sid: [classes[i] for i in row] for sid, row in zip(ids[test], top3)}
    with open(args.output, "w") as f:
        json.dump({TASKS[args.task]["submission_key"]: predictions}, f, indent=2)
    print(f"\nSaved {len(predictions)} test predictions ({len(folds)}-model ensemble) to {args.output}")

    wandb.summary.update({"best_top1": m["top1"], "best_top3": m["top3"], "best_combined": m["combined"],
                          "selected_epoch": best + 1, "checkpoint": ckpt_path})
    wandb.finish()


if __name__ == "__main__":
    main()
