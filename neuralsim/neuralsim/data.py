"""Load the PhysioNet EEG Motor Movement/Imagery dataset (EEGBCI) via MNE.

Reference: Schalk et al., "BCI2000: A General-Purpose Brain-Computer Interface
System", and the EEGBCI dataset distributed through MNE
(``mne.datasets.eegbci``). The first call downloads the data to MNE's data
directory; subsequent calls reuse the cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .commands import (
    ALL_COMMANDS,
    LR_MAP,
    LR_RUNS,
    SELECT_MAP,
    SELECT_RUNS,
    Command,
)

# Motor-cortex channels we prefer for decoding (intersected with availability).
MOTOR_CHANNELS = [
    "C3", "C1", "Cz", "C2", "C4",
    "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
    "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",
]


@dataclass
class BCIDataset:
    """A bundle of trials ready for decoding/streaming."""

    signals: np.ndarray              # (n_trials, n_channels, n_times)
    commands: List[Command]          # length n_trials
    sfreq: float
    ch_names: List[str] = field(default_factory=list)

    @property
    def motor_indices(self):
        idx = [i for i, n in enumerate(self.ch_names) if n in MOTOR_CHANNELS]
        return idx or list(range(len(self.ch_names)))

    def __len__(self) -> int:
        return len(self.commands)

    def summary(self) -> str:
        from collections import Counter
        counts = Counter(c.value for c in self.commands)
        dist = ", ".join(f"{k}={counts.get(k, 0)}" for k in [c.value for c in ALL_COMMANDS])
        return (f"{len(self)} trials | {self.signals.shape[1]} ch @ {self.sfreq:.0f} Hz "
                f"| {self.signals.shape[2]} samples/trial | {dist}")


def _epoch_runs(subject, runs, label_map, tmin, tmax, fmin, fmax, verbose):
    import mne
    from mne.channels import make_standard_montage
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    fnames = eegbci.load_data(subject, list(runs), verbose=verbose)
    raw = concatenate_raws([read_raw_edf(f, preload=True, verbose=verbose) for f in fnames])
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"), on_missing="ignore")
    raw.filter(fmin, fmax, fir_design="firwin", verbose=verbose)

    events, event_id = mne.events_from_annotations(raw, verbose=verbose)
    # event_id maps description ("T0"/"T1"/"T2") -> integer code.
    keep = {desc: code for desc, code in event_id.items() if desc in label_map}
    if not keep:
        return None, None, raw.info["sfreq"], raw.ch_names

    epochs = mne.Epochs(raw, events, keep, tmin, tmax, baseline=None,
                        picks="eeg", preload=True, verbose=verbose)
    code_to_cmd = {event_id[desc]: label_map[desc] for desc in keep}
    data = epochs.get_data(copy=True)
    cmds = [code_to_cmd[c] for c in epochs.events[:, -1]]
    return data, cmds, epochs.info["sfreq"], epochs.ch_names


def load_eegbci(subject: int = 1, tmin: float = 0.5, tmax: float = 3.5,
                fmin: float = 7.0, fmax: float = 30.0, verbose: str | bool = "ERROR") -> BCIDataset:
    """Download (if needed) and assemble a 4-command dataset for one subject.

    Combines left/right imagery runs (LEFT/RIGHT/IDLE) with fists/feet imagery
    runs (SELECT). Requires network access on first run to fetch from PhysioNet.
    """
    lr_data, lr_cmds, sfreq, ch = _epoch_runs(
        subject, LR_RUNS, LR_MAP, tmin, tmax, fmin, fmax, verbose)
    sel_data, sel_cmds, _, _ = _epoch_runs(
        subject, SELECT_RUNS, SELECT_MAP, tmin, tmax, fmin, fmax, verbose)

    parts_d, parts_c = [], []
    if lr_data is not None:
        parts_d.append(lr_data)
        parts_c.extend(lr_cmds)
    if sel_data is not None:
        # align time length to the LR epochs if channel/sample counts match
        if parts_d and sel_data.shape[1:] == parts_d[0].shape[1:]:
            parts_d.append(sel_data)
            parts_c.extend(sel_cmds)
        elif not parts_d:
            parts_d.append(sel_data)
            parts_c.extend(sel_cmds)

    signals = np.concatenate(parts_d, axis=0)
    return BCIDataset(signals=signals, commands=parts_c, sfreq=sfreq, ch_names=ch)


def make_synthetic(n_per_class: int = 30, n_channels: int = 9, sfreq: float = 160.0,
                   seconds: float = 3.0, seed: int = 0) -> BCIDataset:
    """A no-download stand-in with class-separable mu/beta structure.

    Useful for testing the pipeline offline and for first-run demos. It is NOT
    real EEG — it injects event-related-desync-like power differences so the
    decoder demonstrably beats chance.
    """
    rng = np.random.default_rng(seed)
    n_times = int(seconds * sfreq)
    t = np.arange(n_times) / sfreq
    ch_names = ["C3", "C1", "Cz", "C2", "C4", "Cp3", "Cpz", "Cp4", "Fcz"][:n_channels]

    # Per-command mu-band amplitude gain per channel (lateralised ERD/ERS).
    gains = {
        Command.LEFT:   np.array([0.4, 0.5, 0.8, 1.2, 1.3, 0.5, 0.8, 1.2, 0.9][:n_channels]),
        Command.RIGHT:  np.array([1.3, 1.2, 0.8, 0.5, 0.4, 1.2, 0.8, 0.5, 0.9][:n_channels]),
        Command.SELECT: np.array([0.5, 0.6, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.6][:n_channels]),
        Command.IDLE:   np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][:n_channels]),
    }
    signals, commands = [], []
    for cmd, g in gains.items():
        for _ in range(n_per_class):
            mu = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
            beta = 0.5 * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi))
            base = mu + beta
            noise = rng.normal(0, 0.6, size=(n_channels, n_times))
            trial = g[:, None] * base[None, :] + noise
            signals.append(trial)
            commands.append(cmd)
    order = rng.permutation(len(commands))
    signals = np.array(signals)[order]
    commands = [commands[i] for i in order]
    return BCIDataset(signals=signals, commands=commands, sfreq=sfreq, ch_names=ch_names)
