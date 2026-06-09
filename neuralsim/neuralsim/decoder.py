"""Signal decoding: turn an EEG trial into one of the 4 commands.

BandPowerDecoder uses log band power across delta/theta/alpha/beta/gamma bands
and a scikit-learn RandomForestClassifier with 5-fold cross-validation at fit
time.  CSPDecoder is kept unchanged.
"""

from __future__ import annotations

import numpy as np

from .commands import Command

# Five canonical EEG bands
DEFAULT_BANDS = (
    (0.5, 4.0),   # delta
    (4.0, 8.0),   # theta
    (8.0, 13.0),  # alpha
    (13.0, 30.0), # beta
    (30.0, 50.0), # gamma
)


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
    """Random Forest decoder over 5-band (delta/theta/alpha/beta/gamma) power features."""

    def __init__(self, ch_indices=None, sfreq: float = 160.0, bands=DEFAULT_BANDS):
        self.ch_indices = ch_indices
        self.sfreq = float(sfreq)
        self.bands = bands
        self.classes_ = None
        self._clf = None
        self._class_map = None

    def _pick(self, trial: np.ndarray) -> np.ndarray:
        return trial if self.ch_indices is None else trial[self.ch_indices]

    def _features(self, X) -> np.ndarray:
        return np.array([band_power(self._pick(t), self.sfreq, self.bands) for t in X])

    def fit(self, X, y) -> "BandPowerDecoder":
        """Fit on trials X (n_trials, n_ch, n_times) and labels y."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        F = self._features(X)
        y_vals = np.array([Command(v).value for v in y])
        self.classes_ = sorted(set(y_vals.tolist()))
        self._class_map = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([self._class_map[v] for v in y_vals])

        self._clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=0,
            n_jobs=-1,
        )
        cv_scores = cross_val_score(self._clf, F, yi, cv=5, scoring="accuracy")
        print(f"  [RF] 5-fold CV accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
        self._clf.fit(F, yi)
        return self

    def predict_one(self, trial: np.ndarray):
        """Classify a single trial.

        Returns (command, confidence, scores) where scores maps each
        class value to a probability in [0, 1].
        """
        if self._clf is None:
            raise RuntimeError("Decoder is not fitted. Call fit() first.")
        f = band_power(self._pick(trial), self.sfreq, self.bands)
        proba = self._clf.predict_proba(f[None])[0]
        best_idx = int(np.argmax(proba))
        best_class = self.classes_[best_idx]
        scores = {c: float(p) for c, p in zip(self.classes_, proba)}
        return Command(best_class), float(proba[best_idx]), scores

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
