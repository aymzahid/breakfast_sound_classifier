"""
Hyperparameter search (not needed to reproduce Table 3, see Moodle forum 23 Nov).

Tries several settings per classifier on the SAME grouped folds as pipeline.py, plus the
feature-set comparison (own 12 features vs. own + MFCC), and logs everything to
results/tuning.csv. The chosen settings are hard-coded in evaluation.make_models().

    cd code
    python tune.py
"""
from pathlib import Path
import csv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import build_table, FEATURE_NAMES, BASE_NAMES
from evaluation import SEED, cross_validate_model

OUT = Path(__file__).resolve().parent.parent / "results" / "tuning.csv"


def knn(k, scaled=True):
    m = KNeighborsClassifier(n_neighbors=k)
    return make_pipeline(StandardScaler(), m) if scaled else m


def tree(d):
    return DecisionTreeClassifier(max_depth=d, random_state=SEED)


def mlp(hidden, alpha):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha,
                                                         max_iter=2000, random_state=SEED))


def main():
    X, y, groups = build_table()
    own = [FEATURE_NAMES.index(n) for n in BASE_NAMES]
    runs = []
    runs += [("k-NN", f"k={k}", "all", knn(k)) for k in (1, 3, 5, 7, 9, 15, 25)]
    runs += [("k-NN", "k=5, NOT scaled", "all", knn(5, scaled=False))]
    runs += [("Decision tree", f"max_depth={d}", "all", tree(d)) for d in (1, 2, 3, 4, 5, 6, 8, 10, None)]
    runs += [("MLP", f"hidden={h}, alpha={a}", "all", mlp(h, a))
             for h, a in [((16,), 1e-3), ((32,), 1e-3), ((64,), 1e-3), ((32,), 0.1), ((64, 32), 1e-3), ((64, 32), 0.1)]]
    runs += [("k-NN", "k=5", "own 12 (no MFCC)", knn(5)),
             ("Decision tree", "max_depth=5", "own 12 (no MFCC)", tree(5)),
             ("MLP", "hidden=(32,), alpha=0.1", "own 12 (no MFCC)", mlp((32,), 0.1))]

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classifier", "setting", "features", "acc_mean", "acc_std", "f1_mean", "f1_std"])
        for clf, setting, feats, model in runs:
            Xs = X if feats == "all" else X[:, own]
            r = cross_validate_model(model, Xs, y, groups)
            row = [clf, setting, feats, f"{r['acc'].mean():.1f}", f"{r['acc'].std():.1f}",
                   f"{r['f1'].mean():.1f}", f"{r['f1'].std():.1f}"]
            w.writerow(row)
            print(f"{clf:14s} {setting:26s} {feats:17s} acc {row[3]:>5s} ± {row[4]:>4s}   F1 {row[5]:>5s} ± {row[6]:>4s}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
