"""
Step 0 of the pipeline: turn the raw iPhone recordings into the clips the classifier uses.

    raw/<Class>/*.m4a            original Voice Memos recordings (not changed)
    raw/test-sample/<cls>/*.m4a  clips held back for a demo dry run

becomes

    data/<label>/<label>_NNN.wav       mono, 48 kHz, 16-bit, trimmed to at most 5 s
    data/test/<label>/<label>_test_NN.wav   held-out clips, converted but NOT trimmed
    data/clips.csv                      which new file came from which recording

Trimming (forum: a script is fine):
  1. drop the first and last 0.25 s (tap on the record button)
  2. if more than 5 s is left, keep the 5 s window with the highest RMS energy
     (searched in 0.1 s steps), so the window sits where the sound actually is.

Run from inside code/:   python prepare_data.py
"""
from pathlib import Path
import csv
import re
import shutil

import numpy as np
import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
DATA = ROOT / "data"

SR = 48_000          # the iPhone's native rate, so nothing is resampled
EDGE_S = 0.25        # cut from both ends (record-button taps)
MAX_S = 5.0          # assignment: 3-5 s per clip
STEP_S = 0.1         # step when searching for the loudest 5 s window

CLASSES = {          # raw folder name -> label (label = folder name in data/)
    "Stirring": "stirring",
    "Pouring": "pouring",
    "Chopping": "chopping",
}
TEST_DIRS = {"stir": "stirring", "pour": "pouring", "chop": "chopping"}


def recording_number(path):
    """'New Recording 6 copy.m4a' -> (6, 'New Recording 6 copy') for stable sorting."""
    m = re.search(r"(\d+)", path.stem)
    return (int(m.group(1)) if m else 0, path.stem)


def trim(y, sr):
    """Drop the edges, then keep the loudest MAX_S-second window. Returns (clip, start_s)."""
    edge = int(EDGE_S * sr)
    y = y[edge:len(y) - edge]
    n = int(MAX_S * sr)
    if len(y) <= n:
        return y, EDGE_S
    step = int(STEP_S * sr)
    starts = np.arange(0, len(y) - n + 1, step)
    energy = np.cumsum(np.concatenate([[0.0], y.astype(np.float64) ** 2]))
    window_energy = energy[starts + n] - energy[starts]   # sum of squares in each window
    best = starts[np.argmax(window_energy)]
    return y[best:best + n], EDGE_S + best / sr


def main():
    if not RAW.exists():                       # raw recordings are not part of the submission
        print(f"{RAW} not found: the original recordings are not included, data/ is already prepared. Nothing done.")
        return
    for sub in ("m4a", "wav", "test", *CLASSES.values()):
        if (DATA / sub).exists():
            shutil.rmtree(DATA / sub)          # rebuild from raw/ every time
    rows = []

    for raw_name, label in CLASSES.items():
        files = sorted((RAW / raw_name).glob("*.m4a"), key=recording_number)
        (DATA / label).mkdir(parents=True)
        for i, src in enumerate(files, start=1):
            name = f"{label}_{i:03d}"
            y, _ = librosa.load(src, sr=SR, mono=True)
            clip, start = trim(y, SR)
            sf.write(DATA / label / f"{name}.wav", clip, SR, subtype="PCM_16")
            rows.append(dict(file=f"{label}/{name}.wav", label=label, split="train",
                             original=str(src.relative_to(ROOT)),
                             original_s=round(len(y) / SR, 2),
                             kept_from_s=round(start, 2), length_s=round(len(clip) / SR, 2)))

    for test_dir, label in TEST_DIRS.items():
        files = sorted((RAW / "test-sample" / test_dir).glob("*.m4a"), key=recording_number)
        (DATA / "test" / label).mkdir(parents=True)
        for i, src in enumerate(files, start=1):
            name = f"{label}_test_{i:02d}"
            y, _ = librosa.load(src, sr=SR, mono=True)
            sf.write(DATA / "test" / label / f"{name}.wav", y, SR, subtype="PCM_16")
            rows.append(dict(file=f"test/{label}/{name}.wav", label=label, split="test",
                             original=str(src.relative_to(ROOT)),
                             original_s=round(len(y) / SR, 2),
                             kept_from_s=0.0, length_s=round(len(y) / SR, 2)))

    with open(DATA / "clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for split in ("train", "test"):
        print(split)
        for label in CLASSES.values():
            lens = [r["length_s"] for r in rows if r["split"] == split and r["label"] == label]
            print(f"  {label:9s} {len(lens):3d} clips, length {min(lens):.2f}-{max(lens):.2f} s")


if __name__ == "__main__":
    main()
