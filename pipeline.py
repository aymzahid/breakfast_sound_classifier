"""
Main entry point: reproduces Table 3 and the confusion matrix of the report.

    cd code
    python pipeline.py

Steps: load data/<label>/*.wav -> band-pass -> 1 s frames -> FFT -> features (features.py)
       -> 5-fold grouped cross-validation of every classifier in Table 3 (evaluation.py)
       -> print Table 3, print + save the confusion matrix of the final classifier
       -> train the final classifier on all data and save it for classify.py

Outputs (folder results/):
    table3.csv              accuracy and macro F1 per classifier (mean and std over folds, %)
    confusion_matrix.csv    final classifier, frames summed over all folds (rows = actual)
    confusion_matrix.png    the same as an image
    model.joblib            final classifier trained on all clips (used by classify.py)
"""
from pathlib import Path
import csv
import time

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")                 # save figures to files, never open a window
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay

from features import build_table, CLASSES, FEATURE_NAMES
from evaluation import (SEED, N_FOLDS, make_models, cross_validate_model, clip_votes,
                        confusion, format_confusion)

FINAL = "k-NN (k=5, scaled)"          # the classifier chosen in the report (section 2e)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def main():
    RESULTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    X, y, groups = build_table()
    print(f"Data: {X.shape[0]} frames x {X.shape[1]} features from {len(set(groups))} clips "
          f"({time.perf_counter() - t0:.1f} s)")
    for c in CLASSES:
        print(f"  {c:9s} {np.sum(y == c):4d} frames, {len(set(groups[y == c])):3d} clips")
    print(f"Evaluation: {N_FOLDS}-fold StratifiedGroupKFold grouped by clip, seed {SEED}\n")

    # ---- Table 3 ----
    results = {}
    header = f"{'Classifier (settings)':36s} {'Accuracy %':>14s} {'Macro F1 %':>14s} {'fit s':>8s} {'predict s':>10s}"
    print("Table 3: classifier comparison (mean ± std over folds)")
    print(header)
    print("-" * len(header))
    for name, model in make_models().items():
        r = cross_validate_model(model, X, y, groups)
        results[name] = r
        print(f"{name:36s} {r['acc'].mean():6.1f} ± {r['acc'].std():4.1f}  {r['f1'].mean():6.1f} ± {r['f1'].std():4.1f}"
              f" {r['fit_s']:8.3f} {r['predict_s']:10.4f}")

    with open(RESULTS / "table3.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classifier", "acc_mean", "acc_std", "f1_mean", "f1_std", "fit_s", "predict_s"]
                   + [f"acc_fold{i + 1}" for i in range(N_FOLDS)])
        for name, r in results.items():
            w.writerow([name, f"{r['acc'].mean():.2f}", f"{r['acc'].std():.2f}", f"{r['f1'].mean():.2f}",
                        f"{r['f1'].std():.2f}", f"{r['fit_s']:.4f}", f"{r['predict_s']:.4f}"]
                       + [f"{a:.2f}" for a in r["acc"]])

    # ---- confusion matrix of the final classifier ----
    pred = results[FINAL]["pred"]
    cm = confusion(y, pred, CLASSES)
    print(f"\nConfusion matrix: {FINAL}, frames summed over all {N_FOLDS} folds")
    print(format_confusion(cm, CLASSES))

    with open(RESULTS / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actual \\ predicted"] + CLASSES)
        for c, row in zip(CLASSES, cm):
            w.writerow([c] + list(row))

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"{FINAL}\nframes, summed over {N_FOLDS} folds", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=200)
    fig.savefig(RESULTS / "confusion_matrix.pdf")
    plt.close(fig)

    # ---- one answer per clip (how classify.py decides) ----
    t, v = clip_votes(pred, y, groups)
    print(f"\nClip level (majority vote of a clip's frames): {np.sum(t == v)}/{len(t)} clips correct "
          f"({100 * np.mean(t == v):.1f} %)")
    print(format_confusion(confusion(t, v, CLASSES), CLASSES))

    # ---- final model on all clips, for classify.py ----
    final = clone(make_models()[FINAL]).fit(X, y)
    joblib.dump({"model": final, "name": FINAL, "classes": CLASSES, "features": FEATURE_NAMES},
                RESULTS / "model.joblib")
    print(f"\nSaved to {RESULTS.relative_to(ROOT)}/: table3.csv, confusion_matrix.csv/.png/.pdf, model.joblib")
    print(f"Total time {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
