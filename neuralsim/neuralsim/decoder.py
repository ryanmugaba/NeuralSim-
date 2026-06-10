"""Signal decoding: turn an EEG trial into one of the 4 commands.

BandPowerDecoder uses SMOTE oversampling to fix class imbalance, then trains
EEGNet (Lawhern et al. 2018) — a compact CNN built for EEG classification.
CSPDecoder is kept unchanged.
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


try:
    import torch
    import torch.nn as nn

    class _EEGNet(nn.Module):
        """EEGNet: temporal conv → depthwise spatial conv → separable conv → classifier."""

        def __init__(self, n_channels: int, n_times: int, n_classes: int,
                     F1: int = 8, D: int = 2, dropout: float = 0.5):
            super().__init__()
            F2 = F1 * D
            kern_t = min(64, n_times // 2)

            self.block1 = nn.Sequential(
                # Temporal convolution
                nn.Conv2d(1, F1, (1, kern_t), padding="same", bias=False),
                nn.BatchNorm2d(F1),
                # Depthwise spatial convolution across all channels
                nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
            )
            self.block2 = nn.Sequential(
                # Depthwise temporal
                nn.Conv2d(F2, F2, (1, 16), padding="same", groups=F2, bias=False),
                # Pointwise
                nn.Conv2d(F2, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                flat = self.block2(self.block1(dummy)).flatten(1).shape[1]
            self.classifier = nn.Linear(flat, n_classes)

        def forward(self, x):
            return self.classifier(self.block2(self.block1(x)).flatten(1))

except ImportError:
    _EEGNet = None  # type: ignore


class BandPowerDecoder:
    """SMOTE + EEGNet decoder for motor imagery EEG classification."""

    def __init__(self, ch_indices=None, sfreq: float = 160.0, bands=DEFAULT_BANDS):
        self.ch_indices = ch_indices
        self.sfreq = float(sfreq)
        self.bands = bands
        self.classes_ = None
        self._clf = None
        self._class_map = None
        self._norm = None
        self._device = None

    def _pick(self, trial: np.ndarray) -> np.ndarray:
        return trial if self.ch_indices is None else trial[self.ch_indices]

    def _features(self, X) -> np.ndarray:
        return np.array([band_power(self._pick(t), self.sfreq, self.bands) for t in X])

    def fit(self, X, y) -> "BandPowerDecoder":
        """Fit on trials X (n_trials, n_ch, n_times) and labels y."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from imblearn.over_sampling import SMOTE

        y_vals = np.array([Command(v).value for v in y])
        self.classes_ = sorted(set(y_vals.tolist()))
        self._class_map = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([self._class_map[v] for v in y_vals])

        X_arr = np.array([self._pick(t) for t in X], dtype=float)
        n_trials, n_ch, n_times = X_arr.shape

        # Step 1: SMOTE — oversample minority classes on flattened trials
        X_flat = X_arr.reshape(n_trials, -1)
        X_res, y_res = SMOTE(random_state=0).fit_resample(X_flat, yi)
        X_res = X_res.reshape(-1, n_ch, n_times)
        print(f"  [SMOTE] {n_trials} → {len(y_res)} trials (balanced)")

        # Normalize per-dataset
        mean, std = float(X_res.mean()), float(X_res.std()) + 1e-8
        self._norm = (mean, std)
        X_norm = (X_res - mean) / std

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 80/20 held-out split for val accuracy reporting
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y_res))
        cut = int(0.8 * len(idx))
        tr_idx, va_idx = idx[:cut], idx[cut:]

        X_t = torch.tensor(X_norm[:, None], dtype=torch.float32)
        y_t = torch.tensor(y_res, dtype=torch.long)

        tr_loader = DataLoader(
            TensorDataset(X_t[tr_idx], y_t[tr_idx]), batch_size=32, shuffle=True
        )
        X_va = X_t[va_idx].to(self._device)
        y_va = y_t[va_idx].to(self._device)

        # Step 2: EEGNet
        self._clf = _EEGNet(n_ch, n_times, len(self.classes_)).to(self._device)
        optimizer = torch.optim.Adam(self._clf.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self._clf.train()
        for _ in range(200):
            for xb, yb in tr_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                criterion(self._clf(xb), yb).backward()
                optimizer.step()

        self._clf.eval()
        with torch.no_grad():
            val_acc = (self._clf(X_va).argmax(1) == y_va).float().mean().item()
        print(f"  [EEGNet] val accuracy: {val_acc:.1%} (SMOTE-balanced 20% hold-out)")
        return self

    def predict_one(self, trial: np.ndarray):
        """Classify a single trial.

        Returns (command, confidence, scores) where scores maps each
        class value to a probability in [0, 1].
        """
        import torch
        import torch.nn.functional as F

        if self._clf is None:
            raise RuntimeError("Decoder is not fitted. Call fit() first.")
        t = self._pick(trial)
        mean, std = self._norm
        t_norm = (t - mean) / std
        x = torch.tensor(t_norm[None, None], dtype=torch.float32).to(self._device)
        self._clf.eval()
        with torch.no_grad():
            proba = F.softmax(self._clf(x), dim=1)[0].cpu().numpy()
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
