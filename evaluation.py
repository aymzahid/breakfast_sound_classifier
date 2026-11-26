"""
Pipeline step 6 (classify, test, report): classifiers, grouped cross-validation, metrics.

All classifiers see the same features and the same folds:
    5-fold StratifiedGroupKFold, grouped by clip, shuffled with SEED.
Grouping by clip means all frames of one recording are either in the training part or in the
test part of a fold, never in both (no leakage between frames of the same clip).
"""
import time

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
N_FOLDS = 5


def make_cv():
    return StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def make_models():
    """
    The classifiers of Table 3, with the final settings (chosen with tune.py).
    k-NN and MLP get a StandardScaler inside the pipeline, so in every fold the scaler
    is fitted on the training part only.
    """
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "k-NN (k=5, scaled)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision tree (max_depth=5)": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "MLP (32 hidden, alpha=0.1, scaled)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=2000, random_state=SEED)),
    }


def cross_validate_model(model, X, y, groups, cv=None):
    """
    Run grouped CV for one model.

    Returns a dict with per-fold accuracy and macro F1 (in %), mean fit/predict times (s),
    and the out-of-fold prediction for every frame (each frame predicted once, by a model
    that never saw its clip). Summing over folds = confusion matrix of these predictions.
    """
    cv = cv or make_cv()
    acc, f1, fit_t, pred_t = [], [], [], []
    oof = np.empty(len(y), dtype=object)
    for train, test in cv.split(X, y, groups):
        t0 = time.perf_counter()
        model.fit(X[train], y[train])
        t1 = time.perf_counter()
        p = model.predict(X[test])
        t2 = time.perf_counter()
        oof[test] = p
        acc.append(100 * accuracy_score(y[test], p))
        f1.append(100 * f1_score(y[test], p, average="macro"))
        fit_t.append(t1 - t0)
        pred_t.append(t2 - t1)
    return dict(acc=np.array(acc), f1=np.array(f1), fit_s=np.mean(fit_t), predict_s=np.mean(pred_t),
                pred=oof.astype(str))


def clip_votes(pred, y, groups):
    """Majority vote of the frame predictions per clip. Returns (true labels, voted labels) per clip."""
    true, voted = [], []
    for g in np.unique(groups):
        m = groups == g
        labels, counts = np.unique(pred[m], return_counts=True)
        true.append(y[m][0])
        voted.append(labels[np.argmax(counts)])
    return np.array(true), np.array(voted)


def confusion(y_true, y_pred, classes):
    return confusion_matrix(y_true, y_pred, labels=classes)


def format_confusion(cm, classes):
    """Confusion matrix as text: rows = actual, columns = predicted."""
    w = max(len(c) for c in classes) + 2
    lines = [" " * (w + 8) + "predicted", "actual".ljust(w) + "".join(c.rjust(w) for c in classes)]
    for c, row in zip(classes, cm):
        lines.append(c.ljust(w) + "".join(str(v).rjust(w) for v in row))
    return "\n".join(lines)
