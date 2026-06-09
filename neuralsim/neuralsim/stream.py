"""Replay recorded trials as if they were arriving live from a headset."""

from __future__ import annotations

import time

import numpy as np


class BCIStream:
    """Yield trials one at a time, paced to mimic real-time acquisition.

    Each iteration sleeps for ``trial_seconds / speed`` so that ``speed=1``
    replays at the original recording speed and ``speed=8`` is 8x faster.

    Yields
    ------
    (index, signal, true_command)
        ``signal`` is an ``(n_channels, n_times)`` array; ``true_command`` is
        the ground-truth :class:`~neuralsim.commands.Command`.
    """

    def __init__(self, signals, commands, sfreq: float, speed: float = 8.0,
                 shuffle: bool = True, seed: int = 0):
        self.signals = list(signals)
        self.commands = list(commands)
        self.sfreq = float(sfreq)
        self.speed = max(float(speed), 1e-6)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self):
        order = np.arange(len(self.signals))
        if self.shuffle:
            self.rng.shuffle(order)
        for idx in order:
            sig = np.asarray(self.signals[idx], dtype=float)
            time.sleep((sig.shape[-1] / self.sfreq) / self.speed)
            yield int(idx), sig, self.commands[idx]
