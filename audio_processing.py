"""
Pipeline steps 2 (filter) and frame cutting: everything that happens to the raw signal
before the FFT.

    load (48 kHz mono) -> band-pass 80 Hz - 8 kHz -> 1 s frames, 50 % overlap

The Hann window is applied right before the FFT, in features.py, because RMS and
zero-crossing rate are measured on the frame without a window.
"""
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

# ---- settings (all reported in the report, section 2a) ----
SR = 48_000              # every clip is loaded at this rate, whatever rate the file has
LOW_HZ = 80              # below: mains hum (50 Hz), handling noise, traffic
HIGH_HZ = 8_000          # above: mostly microphone hiss
FILTER_ORDER = 4         # Butterworth order: how steep the filter edges are
FRAME_S = 1.0            # frame length in seconds
OVERLAP = 0.5            # 50 % overlap between neighbouring frames

FRAME_LEN = int(FRAME_S * SR)               # 48 000 samples
HOP = int(FRAME_LEN * (1 - OVERLAP))        # 24 000 samples

# Filter design only depends on the settings, so do it once.
# output="sos" = numerically stable form; sosfiltfilt runs it forward and backward (no time shift).
_SOS = butter(FILTER_ORDER, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=SR, output="sos")


def load_clip(path):
    """Read an audio file as mono, resampled to SR if needed. Returns a 1-D float array."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def bandpass(y):
    """Apply the 80 Hz - 8 kHz Butterworth band-pass to a whole clip (before framing)."""
    return sosfiltfilt(_SOS, y)


def make_frames(y, frame_len=FRAME_LEN, hop=HOP):
    """
    Cut a signal into overlapping frames. Returns shape (n_frames, frame_len).

    n_frames = 1 + (len(y) - frame_len) // hop. Samples left over at the end are dropped.
    A clip shorter than one frame is padded with zeros to one full frame,
    so even a very short demo clip still gives one row.
    """
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    n_frames = 1 + (len(y) - frame_len) // hop
    return np.array([y[i * hop : i * hop + frame_len] for i in range(n_frames)])


def hann(n):
    """Hann window w[k] = 0.5 - 0.5 cos(2 pi k / (n-1)): 0 at both ends, 1 in the middle."""
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))


def clip_to_frames(path):
    """load -> band-pass -> frames. The full pre-FFT chain for one file."""
    return make_frames(bandpass(load_clip(path)))
