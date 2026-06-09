"""Signal decoding: turn an EEG trial into one of the 4 commands.

The default decoder is deliberately dependency-light (NumPy only): it computes
mu (8-12 Hz) and beta (13-30 Hz) band power per motor-cortex channel via an FFT,
z-scores the features, and classifies by nearest class centroid.

This is transparent and fast, not state-of-the-art. For higher accuracy install
the ``[sklearn]`` extra and use :class:`CSPDecoder`, which runs MNE's CSP + LDA.
"""

from __future__ import annotations

import numpy as np

from .commands import Command

DEFAULT_BANDS = ((8.0, 12.0), (13.0, 30.0))  # mu, beta


def band_power(trial: np.ndarray, sfreq: float, bands=DEFAULT_BANDS) -> np.ndarray:
    """Log band power per channel per band for a single trial.

    Parameters
    ----------
    trial : ndarray, shape (n_channels, n_times)
    sfreq : float, sampling rate in Hz
    bands : iterable of (low_hz, high_hz)

    Returns
    -------
    ndarray, shape (n_bands * n_channels,)
    """
    trial = np.asarray(trial, dtype=float)
    n = trial.shape[-1]
    x = trial - trial.mean(axis=-1, keepdims=True)
    x = x * np.hanning(n)
    psd = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    feats = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        bp = psd[:, mask].mean(axis=-1) if mask.any() else psd.mean(axis=-1)
        feats.append(np.log(bp + 1e-12))
    return np.concatenate(feats)


class BandPowerDecoder:
    """NumPy-only nearest-centroid decoder over mu/beta band power."""

    def __init__(self, ch_indices=None, sfreq: float = 160.0, bands=DEFAULT_BANDS):
        self.ch_indices = ch_indices
        self.sfreq = float(sfreq)
        self.bands = bands
        self.classes_ = None
        self.centroids_ = None
        self.mean_ = None
        self.std_ = None

    def _pick(self, trial: np.ndarray) -> np.ndarray:
        return trial if self.ch_indices is None else trial[self.ch_indices]

    def _features(self, X) -> np.ndarray:
        return np.array([band_power(self._pick(t), self.sfreq, self.bands) for t in X])

    def fit(self, X, y) -> "BandPowerDecoder":
        """Fit on trials ``X`` (n_trials, n_ch, n_times) and labels ``y``."""
        F = self._features(X)
        self.mean_ = F.mean(axis=0)
        self.std_ = F.std(axis=0) + 1e-9
        Fz = (F - self.mean_) / self.std_
        y = [Command(v) for v in y]
        self.classes_ = sorted({v.value for v in y})
        self.centroids_ = {
            c: Fz[[i for i, yy in enumerate(y) if yy.value == c]].mean(axis=0)
            for c in self.classes_
        }
        return self

    def predict_one(self, trial: np.ndarray):
        """Classify a single trial.

        Returns ``(command, confidence, scores)`` where ``scores`` maps each
        class to a softmax confidence in [0, 1].
        """
        if self.centroids_ is None:
            raise RuntimeError("Decoder is not fitted. Call fit() first.")
        f = band_power(self._pick(trial), self.sfreq, self.bands)
        fz = (f - self.mean_) / self.std_
        dists = np.array([np.linalg.norm(fz - self.centroids_[c]) for c in self.classes_])
        logits = -dists
        ex = np.exp(logits - logits.max())
        probs = ex / ex.sum()
        scores = {c: float(p) for c, p in zip(self.classes_, probs)}
        best = self.classes_[int(np.argmax(probs))]
        return Command(best), float(probs.max()), scores

    def predict(self, X):
        return [self.predict_one(t)[0] for t in X]

    def score(self, X, y) -> float:
        y = [Command(v) for v in y]
        preds = self.predict(X)
        return float(np.mean([p == t for p, t in zip(preds, y)]))


class CSPDecoder:
    """Optional CSP + LDA decoder (requires the ``[sklearn]`` extra + MNE).

    Substantially more accurate than :class:`BandPowerDecoder` but adds
    scikit-learn as a dependency. Same interface (fit / predict / score).
    """

    def __init__(self, n_components: int = 6):
        self.n_components = n_components
        self._pipe = None
        self._le = None

    def fit(self, X, y):
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder

        self._le = LabelEncoder()
        yi = self._le.fit_transform([Command(v).value for v in y])
        self._pipe = Pipeline(
            [
                ("csp", CSP(n_components=self.n_components, reg=None, log=True)),
                ("lda", LinearDiscriminantAnalysis()),
            ]
        )
        self._pipe.fit(np.asarray(X, dtype=float), yi)
        return self

    def predict(self, X):
        yi = self._pipe.predict(np.asarray(X, dtype=float))
        return [Command(c) for c in self._le.inverse_transform(yi)]

    def predict_one(self, trial):
        cmd = self.predict(trial[None])[0]
        return cmd, 1.0, {cmd.value: 1.0}

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y = [Command(v) for v in y]
        return float(np.mean([p == t for p, t in zip(preds, y)]))
