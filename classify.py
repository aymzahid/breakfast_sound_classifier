"""
Classify new clips (live demo).

    cd code
    python classify.py path/to/clip.wav [more clips ...]

Each clip goes through the same chain as the training data (48 kHz mono, band-pass,
1 s frames, features), every frame is classified, and the clip gets the label most of its
frames voted for. Any length and sample rate works (.wav, .mp3, .m4a).
Uses results/model.joblib from pipeline.py (trains it first if it does not exist yet).
"""
from pathlib import Path
import sys

import numpy as np
import joblib

from features import clip_features

MODEL = Path(__file__).resolve().parent.parent / "results" / "model.joblib"


def load_model():
    if not MODEL.exists():
        print("No trained model yet, running pipeline.py first ...")
        import pipeline
        pipeline.main()
    return joblib.load(MODEL)


def classify(path, saved):
    frames = clip_features(path)
    votes = saved["model"].predict(frames)
    labels, counts = np.unique(votes, return_counts=True)
    order = np.argsort(counts)[::-1]
    return labels[order[0]], {labels[i]: int(counts[i]) for i in order}, len(votes)


def main(paths):
    if not paths:
        print(__doc__)
        sys.exit(1)
    saved = load_model()
    for p in paths:
        label, counts, n = classify(p, saved)
        detail = ", ".join(f"{k} {v}/{n}" for k, v in counts.items())
        print(f"{Path(p).name}: {label.upper()}   (frame votes: {detail})")


if __name__ == "__main__":
    main(sys.argv[1:])
