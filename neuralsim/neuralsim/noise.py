"""Simulated consumer EEG headset noise injection."""

from __future__ import annotations

import numpy as np

_FRONTAL = {"Fp1", "Fp2", "F3", "F4", "AF3", "AF4", "AF7", "AF8"}
_TEMPORAL = {"T7", "T8", "TP9", "TP10", "TP7", "TP8"}


class NoiseInjector:
    """Inject realistic consumer-grade EEG noise into single trials.

    Parameters
    ----------
    drift : bool   Slow baseline wandering (low-freq sines + random walk).
    blinks : bool  Eye-blink EOG artifacts on frontal channels.
    emg : bool     Jaw-clench EMG bursts on temporal channels.
    severity : float  0.0 = clean, 1.0 = horrific consumer-grade EEG.
    """

    def __init__(self, drift: bool = True, blinks: bool = True,
                 emg: bool = True, severity: float = 0.5) -> None:
        self.drift = drift
        self.blinks = blinks
        self.emg = emg
        self.severity = float(severity)
        self._rng = np.random.default_rng()   # random seed every run
        self._stats: dict[str, int] = {"drift": 0, "blinks": 0, "emg": 0}

    # ------------------------------------------------------------------
    def inject(self, trial: np.ndarray,
               ch_names: list[str] | None = None,
               sfreq: float = 160.0) -> np.ndarray:
        """Return a noisy copy of *trial* (shape: n_channels × n_times).

        Parameters
        ----------
        trial     : np.ndarray  shape (n_channels, n_times)
        ch_names  : list of str, optional  channel labels for spatial targeting
        sfreq     : float  sampling frequency in Hz
        """
        trial = trial.astype(float, copy=True)
        n_ch, n_t = trial.shape
        t = np.arange(n_t) / sfreq
        sig_std = float(np.std(trial)) + 1e-9

        frontal_idx = _channel_indices(ch_names, _FRONTAL, n_ch)
        temporal_idx = _channel_indices(ch_names, _TEMPORAL, n_ch)

        if self.drift:
            trial = self._add_drift(trial, t, sig_std, n_ch)
            self._stats["drift"] += 1

        if self.blinks:
            trial, n_blinks = self._add_blinks(trial, t, sig_std, sfreq, frontal_idx)
            self._stats["blinks"] += n_blinks

        if self.emg:
            trial, n_bursts = self._add_emg(trial, t, sig_std, sfreq, temporal_idx)
            self._stats["emg"] += n_bursts

        return trial

    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Print what noise sources are configured and how many were applied."""
        parts = []
        if self.drift:
            parts.append(f"baseline drift (trials={self._stats['drift']})")
        if self.blinks:
            parts.append(f"eye blinks (total={self._stats['blinks']})")
        if self.emg:
            parts.append(f"jaw-clench EMG (bursts={self._stats['emg']})")
        enabled = ", ".join(parts) if parts else "none"
        print(f"  NoiseInjector | severity={self.severity:.1f} | {enabled}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_drift(self, trial: np.ndarray, t: np.ndarray,
                   sig_std: float, n_ch: int) -> np.ndarray:
        amp = self.severity * sig_std * 1.5
        for ch in range(n_ch):
            # 1–3 low-frequency sine components (0.05–0.5 Hz)
            n_freqs = int(self._rng.integers(1, 4))
            drift = np.zeros(len(t))
            for _ in range(n_freqs):
                freq = self._rng.uniform(0.05, 0.5)
                phase = self._rng.uniform(0, 2 * np.pi)
                drift += np.sin(2 * np.pi * freq * t + phase)
            drift *= amp * self._rng.uniform(0.5, 1.5) / n_freqs

            # Slow random walk, smoothed to keep it low-frequency
            step_std = amp * 0.08
            walk = np.cumsum(self._rng.normal(0, step_std, size=len(t)))
            kernel = max(1, int(len(t) // 6))
            walk = np.convolve(walk, np.ones(kernel) / kernel, mode="same")

            trial[ch] += drift + walk * self._rng.uniform(0.3, 1.0)
        return trial

    def _add_blinks(self, trial: np.ndarray, t: np.ndarray, sig_std: float,
                    sfreq: float, frontal_idx: list[int],
                    ) -> tuple[np.ndarray, int]:
        duration = t[-1]
        # Higher severity → more frequent blinks
        mean_interval = self._rng.uniform(4.0, 6.0) / max(self.severity, 0.05)
        blink_amp = sig_std * self._rng.uniform(10, 100) * self.severity

        current_t = float(self._rng.exponential(mean_interval * 0.5))
        n_blinks = 0
        while current_t < duration:
            start = int(current_t * sfreq)
            if start >= trial.shape[1]:
                break

            blink_dur = self._rng.uniform(0.2, 0.4)   # 200–400 ms
            blink_samp = min(int(blink_dur * sfreq), trial.shape[1] - start)
            if blink_samp < 2:
                current_t += float(self._rng.exponential(mean_interval))
                continue

            shape = _blink_shape(blink_samp)
            amp = blink_amp * self._rng.uniform(0.7, 1.3)
            for ch in frontal_idx:
                trial[ch, start:start + blink_samp] += amp * self._rng.uniform(0.8, 1.2) * shape

            current_t += float(self._rng.exponential(mean_interval))
            n_blinks += 1
        return trial, n_blinks

    def _add_emg(self, trial: np.ndarray, t: np.ndarray, sig_std: float,
                 sfreq: float, temporal_idx: list[int],
                 ) -> tuple[np.ndarray, int]:
        duration = t[-1]
        mean_interval = self._rng.uniform(3.0, 8.0) / max(self.severity, 0.05)

        current_t = float(self._rng.exponential(mean_interval * 0.5))
        n_bursts = 0
        while current_t < duration:
            start = int(current_t * sfreq)
            if start >= trial.shape[1]:
                break

            burst_dur = self._rng.uniform(0.2, 1.5)
            burst_samp = min(int(burst_dur * sfreq), trial.shape[1] - start)
            if burst_samp < 2:
                current_t += float(self._rng.exponential(mean_interval))
                continue

            burst_t = np.arange(burst_samp) / sfreq
            envelope = np.sin(np.pi * burst_t / burst_dur) ** 2

            # Broadband 20–200 Hz approximated as harmonic sum
            emg = np.zeros(burst_samp)
            for _ in range(int(self._rng.integers(5, 15))):
                freq = self._rng.uniform(20, 200)
                phase = self._rng.uniform(0, 2 * np.pi)
                emg += np.sin(2 * np.pi * freq * burst_t + phase)
            emg *= envelope

            amp = sig_std * self._rng.uniform(2, 20) * self.severity
            for ch in temporal_idx:
                trial[ch, start:start + burst_samp] += amp * self._rng.uniform(0.6, 1.4) * emg

            current_t += float(self._rng.exponential(mean_interval))
            n_bursts += 1
        return trial, n_bursts


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _channel_indices(ch_names: list[str] | None,
                     target: set[str], n_ch: int) -> list[int]:
    if ch_names:
        idx = [i for i, n in enumerate(ch_names) if n in target]
        if idx:
            return idx
    return list(range(n_ch))


def _blink_shape(n_samples: int) -> np.ndarray:
    """Sharp rise (first 20%) then exponential-ish decay."""
    shape = np.empty(n_samples)
    rise_end = max(1, int(0.2 * n_samples))
    shape[:rise_end] = np.linspace(0, 1, rise_end)
    decay = n_samples - rise_end
    if decay > 0:
        tau = n_samples * 0.25
        shape[rise_end:] = np.exp(-np.arange(decay) / tau)
    return shape
