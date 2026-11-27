"""
Figures for the report (not needed for Table 3).

    cd code
    python make_figures.py      (run pipeline.py first for the confusion matrix)

report/figs/examples.pdf/.png  Figure 1: waveform (top) and spectrum (bottom), one clip per class,
                               same time scale and same frequency scale for all columns
report/figs/confusion.pdf/.png Figure 2: copied from results/ (made by pipeline.py)
report/figs/features.pdf/.png  box plots of the most useful features per class
report/figs/clusters.pdf/.png  all frames in 2-D (PCA of the 25 scaled features), coloured by class
report/figs/classifiers.png    confusion matrices of all four classifiers (same folds as pipeline.py)
report/figs/tree.png           the first levels of the decision tree (how it decides)
report/figs/settings.png       accuracy vs. k (k-NN) and vs. max_depth (tree)
Also prints why the worst clip is misclassified (report section 2d).
"""
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audio_processing import SR, LOW_HZ, HIGH_HZ, load_clip, bandpass, make_frames
from features import (CLASSES, FEATURE_NAMES, WINDOW, FREQS, IN_BAND, build_table, clip_features)

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "report" / "figs"
EXAMPLE = {c: ROOT / "data" / c / f"{c}_001.wav" for c in CLASSES}
COLORS = {"stirring": "C0", "pouring": "C1", "chopping": "C2"}


def mean_spectrum_db(path):
    """Average power spectrum (dB) over the 1 s frames of the band-passed clip."""
    frames = make_frames(bandpass(load_clip(path)))
    p = (np.abs(np.fft.rfft(frames * WINDOW, axis=1)) ** 2).mean(axis=0)
    return 10 * np.log10(p / p[IN_BAND].max() + 1e-12)


def examples_figure():
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), sharex="row", sharey="row")
    for j, c in enumerate(CLASSES):
        y = load_clip(EXAMPLE[c])
        t = np.arange(len(y)) / SR
        axes[0, j].plot(t, y, lw=0.4, color=COLORS[c])
        axes[0, j].set_title(f"{c} ({EXAMPLE[c].name})", fontsize=9)
        axes[0, j].set_xlim(0, 5)
        axes[0, j].set_ylim(-1, 1)
        axes[1, j].semilogx(FREQS[1:], mean_spectrum_db(EXAMPLE[c])[1:], lw=0.5, color=COLORS[c])
        axes[1, j].set_xlim(LOW_HZ, HIGH_HZ)
        axes[1, j].set_ylim(-80, 3)
        axes[1, j].set_xlabel("frequency (Hz)")
        axes[0, j].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("amplitude")
    axes[1, 0].set_ylabel("power (dB re. max)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"examples.{ext}", dpi=200)
    plt.close(fig)


def features_figure(X, y):
    names = ["zcr", "centroid", "peak_freq", "band_4000_8000", "band_80_250", "mfcc5"]
    fig, axes = plt.subplots(1, len(names), figsize=(11, 2.6))
    for ax, n in zip(axes, names):
        j = FEATURE_NAMES.index(n)
        bp = ax.boxplot([X[y == c, j] for c in CLASSES], patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(COLORS[c])
        ax.set_xticks([1, 2, 3], ["stir", "pour", "chop"], fontsize=8)
        ax.set_title(n, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"features.{ext}", dpi=200)
    plt.close(fig)


def clusters_figure(X, y, groups, mark="chopping_017"):
    """Squeeze the 25 standardised features into 2 dimensions with PCA (only for looking, not for training)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    pca = PCA(n_components=2)
    Z = pca.fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c in CLASSES:
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=0.55, color=COLORS[c], label=f"{c} ({m.sum()} frames)")
    m = groups == mark
    ax.scatter(Z[m, 0], Z[m, 1], s=60, facecolors="none", edgecolors="k", linewidths=1.2, label=f"{mark} (worst clip)")
    v = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC 1 ({v[0]:.0f} % of variance)")
    ax.set_ylabel(f"PC 2 ({v[1]:.0f} % of variance)")
    ax.set_title("Every 1 s frame as a dot (PCA of the 25 features)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"clusters.{ext}", dpi=200)
    plt.close(fig)


def classifier_figures(X, y, groups):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from evaluation import make_models, cross_validate_model, SEED
    from tune import knn, tree

    # 1. confusion matrix of every classifier
    models = make_models()
    fig, axes = plt.subplots(1, len(models), figsize=(16, 4.2))
    for ax, (name, model) in zip(axes, models.items()):
        r = cross_validate_model(model, X, y, groups)
        cm = confusion_matrix(y, r["pred"], labels=CLASSES)
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(f"{name}\naccuracy {r['acc'].mean():.1f} %, macro F1 {r['f1'].mean():.1f} %", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "classifiers.png", dpi=150)
    plt.close(fig)

    # 2. the decision tree's first questions (trained on all frames, depth 3 so it stays readable)
    t = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X, y)
    fig = plt.figure(figsize=(16, 6))
    plot_tree(t, feature_names=FEATURE_NAMES, class_names=list(t.classes_), filled=True,
              impurity=False, proportion=False, fontsize=9)
    fig.savefig(FIGS / "tree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. effect of the main setting
    ks = [1, 3, 5, 7, 9, 15, 25]
    depths = [1, 2, 3, 4, 5, 6, 8, 10, None]
    acc_k = [cross_validate_model(knn(k), X, y, groups)["acc"] for k in ks]
    acc_d = [cross_validate_model(tree(d), X, y, groups)["acc"] for d in depths]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].errorbar(ks, [a.mean() for a in acc_k], yerr=[a.std() for a in acc_k], fmt="o-", capsize=3)
    axes[0].set_xlabel("k (number of neighbours)"); axes[0].set_ylabel("accuracy %"); axes[0].set_title("k-NN")
    lab = [str(d) for d in depths]
    axes[1].errorbar(lab, [a.mean() for a in acc_d], yerr=[a.std() for a in acc_d], fmt="o-", capsize=3, color="C2")
    axes[1].set_xlabel("max_depth (None = unlimited)"); axes[1].set_title("decision tree")
    for ax in axes:
        ax.set_ylim(60, 100); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "settings.png", dpi=150)
    plt.close(fig)


def explain_clip(X, y, clip):
    """Compare one clip's median features with the class medians (evidence for report 2d)."""
    f = np.median(clip_features(ROOT / "data" / clip.split("_")[0] / f"{clip}.wav"), axis=0)
    print(f"\n{clip}: median features vs. class medians")
    print(f"  {'feature':16s} {'clip':>9s}" + "".join(f"{c:>10s}" for c in CLASSES))
    for n in ["rms", "zcr", "centroid", "flatness", "peak_freq", "band_80_250", "band_250_500", "band_2000_4000"]:
        j = FEATURE_NAMES.index(n)
        print(f"  {n:16s} {f[j]:9.4g}" + "".join(f"{np.median(X[y == c, j]):10.4g}" for c in CLASSES))


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    examples_figure()
    X, y, groups = build_table()
    features_figure(X, y)
    clusters_figure(X, y, groups)
    classifier_figures(X, y, groups)
    for ext in ("pdf", "png"):
        src = ROOT / "results" / f"confusion_matrix.{ext}"
        if src.exists():
            shutil.copy(src, FIGS / f"confusion.{ext}")
    explain_clip(X, y, "chopping_017")
    print(f"\nsaved figures to {FIGS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
