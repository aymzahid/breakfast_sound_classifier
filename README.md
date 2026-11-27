# Breakfast Sound Classifier

Recognises three kitchen activities from short audio clips: **stirring**, **pouring** and **chopping**.
Home Assignment 1 of *Pervasive Computing: Design and Development* (JKU Linz, 2026W).

Every step from the filter to the features is written by hand with NumPy/SciPy. scikit-learn is used only for scaling, the classifiers, cross-validation and the metrics.

## Results

5-fold cross-validation, grouped by clip (`StratifiedGroupKFold`, seed 42), 851 one-second frames from 98 clips, 25 features per frame:

| Classifier | Accuracy % | Macro F1 % |
|---|---|---|
| Majority baseline | 39.8 ± 2.3 | 19.0 ± 0.8 |
| **k-NN (k=5, scaled)** ← final | **97.3 ± 1.8** | **96.8 ± 2.1** |
| Decision tree (max_depth=5) | 93.1 ± 2.7 | 92.3 ± 2.8 |
| MLP (32 hidden, alpha=0.1, scaled) | 97.3 ± 1.7 | 96.8 ± 1.9 |

- A majority vote over each clip's frames classifies **97 of 98 clips** correctly.
- All **14 held-out test clips** are classified correctly.

## How it works

```
recording (.wav / .m4a / .mp3, any sample rate)
  → load as mono, 48 kHz
  → band-pass filter 80 Hz – 8 kHz (Butterworth, order 4, zero-phase)
  → 1 s frames, 50 % overlap
  → Hann window → FFT (np.fft.rfft)
  → 25 features per frame
  → classifier → one label per frame → majority vote per clip
```

**Features (25):**
- 12 of our own: RMS, zero-crossing rate, spectral centroid, roll-off (85 %), flatness, peak frequency, and the energy share in 6 bands (80–250, 250–500, 500–1k, 1k–2k, 2k–4k, 4k–8k Hz)
- 13 MFCC means (`librosa.feature.mfcc`)

**Fair evaluation:**
- All frames of one clip are always on the same side of a split (no leakage between neighbouring frames).
- The scaler is fitted inside each training fold only.

## Project layout

```
code/
  pipeline.py          main entry: features → cross-validation of all classifiers → table, confusion matrix, final model
  classify.py          classify new clips (frame votes + final label)
  audio_processing.py  loading, band-pass filter, framing, Hann window
  features.py          FFT and the 25 features, builds the feature table
  evaluation.py        grouped cross-validation, the classifiers, confusion matrix
  prepare_data.py      converts and trims the original recordings into data/ (needs raw/, not included)
  tune.py              settings search (k, tree depth, MLP size) and a with/without-MFCC comparison
  make_figures.py      figures used in the report
  requirements.txt
data/
  stirring/ pouring/ chopping/   98 clips, mono 48 kHz 16-bit WAV, trimmed to ≤ 5 s
  test/<label>/                  14 held-out clips, never used for training or cross-validation
  clips.csv                      which clip came from which recording
report/
  report.pdf
```

## Run it

Python 3.11:

```bash
pip install -r code/requirements.txt
cd code

python pipeline.py                                 # reproduces the results table + confusion matrix, saves results/
python classify.py ../data/test/*/*.wav            # classify clips (any .wav / .mp3 / .m4a)
python tune.py                                     # optional: settings search
python make_figures.py                             # optional: report figures
```

Example output of `classify.py`:

```
chopping_test_01.wav : classified as CHOPPING
  frame votes (12 frames of 1 s):
    stirring    0/12  (  0.0 %)
    pouring     0/12  (  0.0 %)
    chopping   12/12  (100.0 %)
```

## Data

- Recorded with iPhone Voice Memos (mainly an iPhone 15 Pro Max; a few stirring clips on an iPhone 16 Pro Max and an iPad mini 5).
- Three rooms (a kitchen with fridge and vent, a room with a fan, a room with the TV on), recorded at different times and distances.
- Stirring and pouring used a mug, a cup and a steel cup; pouring was from a bottle and a jug.
- Chopping used plastic and wooden boards with carrot, cucumber, potato and onion.

Author: Zahid Khan
