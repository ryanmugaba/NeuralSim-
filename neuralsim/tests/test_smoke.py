"""Offline smoke tests (no download) using the synthetic dataset."""

import numpy as np

from neuralsim import BandPowerDecoder, BCIStream, Command, make_synthetic
from neuralsim.decoder import band_power


def test_synthetic_shape():
    ds = make_synthetic(n_per_class=10, seed=0)
    assert ds.signals.shape[0] == len(ds.commands) == 40
    assert set(c for c in ds.commands) == set(Command)


def test_band_power_length():
    ds = make_synthetic(n_per_class=2, seed=0)
    f = band_power(ds.signals[0], ds.sfreq)
    assert f.shape == (2 * ds.signals.shape[1],)  # mu + beta per channel


def test_decoder_beats_chance():
    ds = make_synthetic(n_per_class=30, seed=0)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ds))
    tr, te = idx[:80], idx[80:]
    dec = BandPowerDecoder(ch_indices=ds.motor_indices, sfreq=ds.sfreq)
    dec.fit(ds.signals[tr], [ds.commands[i] for i in tr])
    acc = dec.score(ds.signals[te], [ds.commands[i] for i in te])
    assert acc > 0.25  # better than 4-class chance


def test_stream_yields_all():
    ds = make_synthetic(n_per_class=5, seed=0)
    stream = BCIStream(ds.signals, ds.commands, ds.sfreq, speed=1e9, seed=0)
    seen = list(stream)
    assert len(seen) == len(ds)
    idx, sig, cmd = seen[0]
    assert isinstance(cmd, Command) and sig.ndim == 2
