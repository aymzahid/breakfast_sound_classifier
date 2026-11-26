"""
Pipeline steps 3 (FFT) and 4 (features): one row of numbers per 1 s frame.

Own code: RMS, zero-crossing rate, spectral centroid, rolloff, flatness, peak frequency,
6 band-energy shares (12 features).
Library:  13 MFCC means from librosa.feature.mfcc (allowed, Moodle forum 18 Nov).

Run directly to build the feature table and save it for inspection:
    python features.py        -> ../results/features.csv
"""
from pathlib import Path
import csv

import numpy as np
import librosa

from audio_processing import SR, FRAME_LEN, LOW_HZ, HIGH_HZ, clip_to_frames, hann

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLASSES = ["stirring", "pouring", "chopping"]     # folder names in data/ = labels

# ---- FFT setup (step 3) ----
WINDOW = hann(FRAME_LEN)
FREQS = np.fft.rfftfreq(FRAME_LEN, d=1 / SR)         # Hz of every FFT bin (1 Hz apart for 1 s frames)
IN_BAND = (FREQS >= LOW_HZ) & (FREQS <= HIGH_HZ)     # only the bins the band-pass kept
BAND_FREQS = FREQS[IN_BAND]

BAND_EDGES = [80, 250, 500, 1000, 2000, 4000, 8000]  # Hz, for the band-energy shares
N_MFCC = 13

BASE_NAMES = (["rms", "zcr", "centroid", "rolloff", "flatness", "peak_freq"]
              + [f"band_{lo}_{hi}" for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])])
MFCC_NAMES = [f"mfcc{i}" for i in range(1, N_MFCC + 1)]
FEATURE_NAMES = BASE_NAMES + MFCC_NAMES


def power_spectrum(frame):
    """Hann window -> one rfft -> power per bin, restricted to 80 Hz - 8 kHz."""
    return (np.abs(np.fft.rfft(frame * WINDOW)) ** 2)[IN_BAND]


# ---- features on the raw frame ----
def rms(frame):
    """Loudness: root of the mean squared sample."""
    return np.sqrt(np.mean(frame ** 2))


def zcr(frame):
    """Fraction of neighbouring samples where the sign flips. Noisy/high sounds -> high."""
    s = np.sign(frame)
    return np.mean(s[:-1] != s[1:])


# ---- features on the power spectrum p (frequencies f) ----
def centroid(p, f):
    """Energy-weighted mean frequency ('brightness'), in Hz."""
    return np.sum(f * p) / (np.sum(p) + 1e-12)


def rolloff(p, f, share=0.85):
    """Frequency below which `share` of the energy lies, in Hz."""
    cum = np.cumsum(p)
    return f[min(np.searchsorted(cum, share * cum[-1]), len(f) - 1)]


def flatness(p):
    """Geometric mean / arithmetic mean: near 0 = tonal (peaks), higher = noise-like."""
    p = p + 1e-12
    return np.exp(np.mean(np.log(p))) / np.mean(p)


def peak_freq(p, f):
    """Frequency with the most energy, in Hz (e.g. the note a cup rings at)."""
    return f[np.argmax(p)]


def band_shares(p, f):
    """Share of the total energy in each band of BAND_EDGES (sums to 1, loudness-independent)."""
    total = np.sum(p) + 1e-12
    return [np.sum(p[(f >= lo) & (f < hi)]) / total for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]


def mfcc_means(frame):
    """librosa MFCC on the frame (it uses its own short sub-frames); mean over them -> 13 numbers."""
    return librosa.feature.mfcc(y=frame.astype(np.float32), sr=SR, n_mfcc=N_MFCC).mean(axis=1)


def frame_features(frame):
    """All features of one (band-passed, un-windowed) frame, in FEATURE_NAMES order."""
    p = power_spectrum(frame)
    f = BAND_FREQS
    base = [rms(frame), zcr(frame), centroid(p, f), rolloff(p, f), flatness(p), peak_freq(p, f)]
    return np.concatenate([base, band_shares(p, f), mfcc_means(frame)])


def clip_features(path):
    """Feature matrix of one clip: shape (n_frames, n_features)."""
    return np.array([frame_features(fr) for fr in clip_to_frames(path)])


def build_table(data_dir=DATA_DIR):
    """
    Features for every clip in data/<label>/*.wav.

    Returns X (frames x features), y (label per frame), groups (clip name per frame).
    `groups` keeps all frames of one clip together in cross-validation.
    """
    X, y, groups = [], [], []
    for label in CLASSES:
        for path in sorted((Path(data_dir) / label).glob("*.wav")):
            feats = clip_features(path)
            X.append(feats)
            y += [label] * len(feats)
            groups += [path.stem] * len(feats)
    return np.vstack(X), np.array(y), np.array(groups)


def save_table(X, y, groups, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "label"] + FEATURE_NAMES)
        for g, lab, row in zip(groups, y, X):
            w.writerow([g, lab] + [f"{v:.6g}" for v in row])


if __name__ == "__main__":
    X, y, groups = build_table()
    print(f"{X.shape[0]} frames x {X.shape[1]} features from {len(set(groups))} clips")
    for label in CLASSES:
        m = y == label
        print(f"  {label:9s} {m.sum():4d} frames, {len(set(groups[m])):3d} clips")
    print("\nmedian per class (own features):")
    print(f"  {'':16s}" + "".join(f"{c:>12s}" for c in CLASSES))
    for j, name in enumerate(BASE_NAMES):
        print(f"  {name:16s}" + "".join(f"{np.median(X[y == c, j]):12.4g}" for c in CLASSES))
    out = ROOT / "results" / "features.csv"
    save_table(X, y, groups, out)
    print(f"\nsaved {out.relative_to(ROOT)}")
