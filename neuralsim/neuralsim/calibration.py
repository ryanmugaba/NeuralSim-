"""Personal calibration: adapt a frozen EEGNet backbone to a specific user.

A tiny 2-layer adapter (flat→32→ELU→4) is trained on ~80 personal trials in
under 10 seconds.  Only the adapter is saved as the user's brain profile
(~31 KB); the full EEGNet backbone stays in the main decoder.
"""

from __future__ import annotations

import numpy as np

from .commands import Command


# Backbone wrapper lives at module level so it can be serialised cleanly.
try:
    import torch
    import torch.nn as nn

    class _Backbone(nn.Module):
        """EEGNet block1 + block2 with no classifier head."""

        def __init__(self, eegnet: nn.Module) -> None:
            super().__init__()
            self.block1 = eegnet.block1
            self.block2 = eegnet.block2

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.block2(self.block1(x)).flatten(1)

except ImportError:
    _Backbone = None  # type: ignore


# ---------------------------------------------------------------------------
# Module-level helpers (mirrors BandPowerDecoder internals, no coupling)

def _pick(trial: np.ndarray, ch_indices) -> np.ndarray:
    return trial if ch_indices is None else trial[ch_indices]


def _norm_trial(t: np.ndarray) -> np.ndarray:
    """Per-trial z-score across all channels × times (matches decoder at inference)."""
    return (t - t.mean()) / (t.std() + 1e-8)


def _norm_trials(X: np.ndarray) -> np.ndarray:
    """Batch version of _norm_trial for an (n, ch, t) array."""
    m = X.mean(axis=(1, 2), keepdims=True)
    s = X.std(axis=(1, 2), keepdims=True) + 1e-8
    return (X - m) / s


# ---------------------------------------------------------------------------

class PersonalCalibration:
    """Fine-tunable adapter on top of a frozen EEGNet backbone.

    The EEGNet backbone (block1 + block2) is frozen immediately on
    construction and never updated.  Only a tiny 2-layer adapter
    (flat → 32 → ELU → 4) learns the user's personal EEG characteristics.

    Parameters
    ----------
    decoder : BandPowerDecoder
        A fitted BandPowerDecoder.  Its EEGNet weights are shared (and
        frozen) inside this object; do not re-train the decoder afterwards.

    Examples
    --------
    >>> cal = PersonalCalibration(decoder)
    >>> cal.calibrate(my_trials, my_labels)
    >>> cmd, conf, _ = cal.predict(new_trial)
    >>> cal.save_profile("ryan.profile")
    >>> cal2 = PersonalCalibration.load_profile("ryan.profile", decoder)
    """

    def __init__(self, decoder) -> None:
        import torch
        import torch.nn as nn

        if decoder._clf is None:
            raise RuntimeError("decoder must be fitted before creating PersonalCalibration")

        self._device: "torch.device" = decoder._device or torch.device("cpu")
        self._ch_indices = decoder.ch_indices
        self._classes: list = list(decoder.classes_)
        self._class_map: dict = dict(decoder._class_map)
        self._flat: int = decoder._clf.classifier.in_features

        # Frozen backbone — shares block1/block2 parameter tensors with the
        # original _EEGNet.  requires_grad is set False in-place on those tensors.
        self._backbone = _Backbone(decoder._clf).to(self._device)
        for p in self._backbone.parameters():
            p.requires_grad_(False)
        self._backbone.eval()

        # Trainable adapter: flat → 32 → ELU → n_classes
        self._adapter = nn.Sequential(
            nn.Linear(self._flat, 32),
            nn.ELU(),
            nn.Linear(32, len(self._classes)),
        ).to(self._device)

        self._fitted = False

    # ------------------------------------------------------------------
    def calibrate(self, trials, labels) -> "PersonalCalibration":
        """Fine-tune the adapter on personal calibration trials.

        The backbone is never touched.  All 80 trials are forward-passed once
        to extract frozen features, then only the adapter is optimised for
        50 epochs — typically completes in under 10 seconds on CPU.

        Parameters
        ----------
        trials : sequence of ndarray, shape (n_ch, n_times)
            Calibration EEG trials.  ~80 trials (20 per class) recommended.
        labels : sequence of Command or str
            Ground-truth command for each trial.

        Returns
        -------
        self
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        y_vals = np.array([Command(v).value for v in labels])
        yi = np.array([self._class_map[v] for v in y_vals])

        X = _norm_trials(
            np.array([_pick(t, self._ch_indices) for t in trials], dtype=float))

        # Pre-extract backbone features once — frozen, so no need per epoch.
        with torch.no_grad():
            X_t = torch.tensor(X[:, None], dtype=torch.float32).to(self._device)
            feats = self._backbone(X_t)          # (n_trials, flat)
        feat_labels = torch.tensor(yi, dtype=torch.long).to(self._device)

        # Class-weighted loss handles any remaining imbalance in the cal set.
        n_cls = len(self._classes)
        counts = np.bincount(yi, minlength=n_cls).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        cw = torch.tensor(
            len(yi) / (n_cls * counts), dtype=torch.float32).to(self._device)

        loader = DataLoader(
            TensorDataset(feats, feat_labels),
            batch_size=16, shuffle=True)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = torch.optim.Adam(self._adapter.parameters(), lr=5e-3)

        self._adapter.train()
        for _ in range(50):
            for fb, yb in loader:
                optimizer.zero_grad()
                criterion(self._adapter(fb), yb).backward()
                optimizer.step()
        self._adapter.eval()

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, trial: np.ndarray):
        """Classify one trial with frozen backbone + trained adapter.

        Parameters
        ----------
        trial : ndarray, shape (n_ch, n_times)
            Raw (unselected, unnormalised) EEG trial.

        Returns
        -------
        (Command, confidence: float, scores: dict)
            Same structure as ``BandPowerDecoder.predict_one``.
        """
        import torch
        import torch.nn.functional as F

        if not self._fitted:
            raise RuntimeError("Call calibrate() before predict()")

        t = _pick(trial, self._ch_indices)
        x = torch.tensor(
            _norm_trial(t)[None, None], dtype=torch.float32).to(self._device)

        with torch.no_grad():
            feat = self._backbone(x)
            proba = F.softmax(self._adapter(feat), dim=1)[0].cpu().numpy()

        best_idx = int(np.argmax(proba))
        best_class = self._classes[best_idx]
        scores = {c: float(p) for c, p in zip(self._classes, proba)}
        return Command(best_class), float(proba[best_idx]), scores

    # ------------------------------------------------------------------
    def save_profile(self, path: str) -> None:
        """Save adapter weights + class metadata to *path*.

        Only the adapter is written — the EEGNet backbone is not included.
        File size is typically ~31 KB (2 linear layers, float32).

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"ryan.profile"``).
        """
        import torch

        torch.save({
            "adapter": self._adapter.state_dict(),
            "classes": self._classes,
            "class_map": self._class_map,
            "flat": self._flat,
        }, path)

    @classmethod
    def load_profile(cls, path: str, decoder) -> "PersonalCalibration":
        """Load a saved profile and attach it to *decoder*'s backbone.

        Parameters
        ----------
        path : str
            A file previously written by :meth:`save_profile`.
        decoder : BandPowerDecoder
            A fitted decoder whose backbone architecture matches the profile.

        Returns
        -------
        PersonalCalibration
            Ready to call :meth:`predict` immediately.
        """
        import torch

        cal = cls(decoder)
        data = torch.load(path, weights_only=True)
        if data["flat"] != cal._flat:
            raise ValueError(
                f"Profile flat={data['flat']} does not match decoder flat={cal._flat}. "
                "Profile was saved from a different model.")
        cal._adapter.load_state_dict(data["adapter"])
        cal._adapter.eval()
        cal._classes = data["classes"]
        cal._class_map = data["class_map"]
        cal._fitted = True
        return cal
