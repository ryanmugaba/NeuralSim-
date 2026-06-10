# NeuralSim

[![PyPI](https://img.shields.io/pypi/v/neuralsim)](https://pypi.org/project/neuralsim/)
[![GitHub stars](https://img.shields.io/github/stars/ryanmugaba/NeuralSim-)](https://github.com/ryanmugaba/NeuralSim-)

Replay PhysioNet EEG motor-imagery recordings through a live neural decoder in your terminal — then calibrate it to a specific person in under 10 seconds.

Built for developers and researchers who want to experiment with brain-computer interfaces without needing hardware.

---

## Install

```bash
pip install neuralsim
```

Python 3.9+. First run downloads ~50 MB of EEG recordings from PhysioNet.

---

## Basic usage

```python
from neuralsim import load_eegbci, BandPowerDecoder, BCIStream

ds = load_eegbci(subject=1)
decoder = BandPowerDecoder(ch_indices=ds.motor_indices, sfreq=ds.sfreq)
decoder.fit(ds.signals[:77], ds.commands[:77])

for _, signal, true_cmd in BCIStream(ds.signals[77:], ds.commands[77:], ds.sfreq, speed=8):
    pred, confidence, _ = decoder.predict_one(signal)
    print(f"{true_cmd.value:6} -> {pred.value:6}  ({confidence:.0%})")
```

No hardware? Use synthetic data:

```python
from neuralsim import make_synthetic
ds = make_synthetic()
```

---

## Personal calibration

Adapt the decoder to a specific user's EEG patterns with 80 labeled trials.
Only a tiny adapter (Linear → ELU → Linear) is trained; the EEGNet backbone is frozen.

```python
from neuralsim import load_eegbci, BandPowerDecoder
from neuralsim.calibration import PersonalCalibration

ds = load_eegbci(subject=1)
decoder = BandPowerDecoder(ch_indices=ds.motor_indices, sfreq=ds.sfreq)
decoder.fit(ds.signals[:77], ds.commands[:77])

# Accuracy with the base decoder on personal trials
before = sum(decoder.predict_one(t)[0] == c for t, c in zip(my_trials, my_labels))
print(f"Before: {before / len(my_labels):.0%}")   # e.g. 33.8%

# Calibrate — 80 labeled trials, <10 seconds on CPU
cal = PersonalCalibration(decoder)
cal.calibrate(my_trials, my_labels)

after = sum(cal.predict(t)[0] == c for t, c in zip(my_trials, my_labels))
print(f"After:  {after / len(my_labels):.0%}")    # e.g. 100.0%

# Save a 33 KB profile; reload it next session
cal.save_profile("ryan.profile")
cal = PersonalCalibration.load_profile("ryan.profile", decoder)
cmd, confidence, _ = cal.predict(new_trial)
```

---

## Terminal demo

```bash
neuralsim-demo --subject 1 --speed 8
neuralsim-demo --synthetic                       # no download
neuralsim-demo --subject 1 --calibrate           # run the calibration demo
neuralsim-demo --subject 1 --csp                 # CSP+LDA (needs scikit-learn)
```

```
  111 trials | 64 ch @ 160 Hz | 481 samples/trial | LEFT=23, RIGHT=22, SELECT=21, IDLE=45

  [EEGNet] best epoch 97/100 | train accuracy: 93.5%

  t=  1 | true SELECT -> pred [OK] SELECT OK  conf #####-------  39% | acc 100%
  t=  2 | true IDLE   -> pred ... IDLE   OK  conf ########----  70% | acc 100%
  ...
  Done. Final accuracy: 47.1% on 34 trials (chance = 25%).

NeuralSim :: Personal calibration simulation
  Accuracy BEFORE calibration : 33.8%  (base EEGNet, personal data)
  Calibrating adapter (50 epochs, adapter weights only)...
  Accuracy AFTER  calibration : 100.0%  (frozen backbone + adapter, same trials)
  Personal profile saved to ryan.profile  (33 KB)
```

---

## Accuracy

Subject 1, 4-class motor imagery, chance = 25%.

| Version | Decoder | Accuracy | Notes |
|---------|---------|----------|-------|
| v0.1.0 | Random Forest | 35.3% | 5 motor channels, band-power features |
| v0.1.1 | Random Forest | 35.3% | published to PyPI |
| v0.2.0 | EEGNet (CNN) | **47.1%** | 15 channels, weighted loss, noise augmentation, LR scheduler |
| v0.3.0 | EEGNet + PersonalCalibration | 33.8% → **100%** | frozen backbone + 2-layer adapter, 80 personal trials |

v0.3.0 accuracy is measured on the same 80 trials used for calibration (training set). The base accuracy drops on personal data due to per-channel amplitude differences between individuals; the adapter corrects for this.

`CSPDecoder` (optional, scikit-learn) scores 50% on the same split.

---

## How it works

```
PhysioNet EDF
  → MNE load + 7–30 Hz filter + epoch
  → BCIStream (paced replay at N× real-time)
  → BandPowerDecoder / PersonalCalibration / CSPDecoder
  → LEFT / RIGHT / SELECT / IDLE
```

**BandPowerDecoder** trains EEGNet on 15 motor-cortex channels (C3/Cz/C4 ± CP/FC neighbours). Class-weighted `CrossEntropyLoss` handles the IDLE imbalance. Gaussian noise augmentation per batch. `ReduceLROnPlateau` scheduler. Fixed seed for reproducibility.

**PersonalCalibration** freezes the EEGNet backbone and trains a tiny adapter (`flat → 32 → ELU → 4`) on ~80 personal trials. Backbone features are pre-extracted once; only the adapter optimises. Profile file contains adapter weights only (~33 KB).

**CSPDecoder** wraps MNE's `CSP` + scikit-learn `LDA`. Install with `pip install neuralsim[sklearn]`.

---

## Commands

| Command | Runs | Annotation | Imagery |
|---------|------|------------|---------|
| `LEFT` | 4, 8, 12 | T1 | Left fist |
| `RIGHT` | 4, 8, 12 | T2 | Right fist |
| `SELECT` | 6, 10, 14 | T1 | Both fists |
| `IDLE` | 4, 8, 12 | T0 | Rest |

---

## Package layout

```
neuralsim/
├── commands.py      Command enum + PhysioNet label map
├── data.py          PhysioNet loader (MNE) + synthetic generator
├── decoder.py       EEGNet decoder + optional CSP decoder
├── calibration.py   PersonalCalibration — frozen backbone + trainable adapter
├── stream.py        real-time trial replay
└── demo.py          terminal demo + CLI entry point
```

Run tests: `pip install neuralsim[dev] && pytest`

---

## Roadmap

- **OpenBCI hardware** — stream live from Cyton/Ganglion boards instead of replaying recordings
- **Edge deployment** — export to ONNX/TFLite for inference on Raspberry Pi and microcontrollers
- **Wheelchair and assistive tech** — higher-stakes 4-command profiles, latency tuning, fail-safe IDLE detection

---

## License

MIT. PhysioNet EEGBCI dataset: [physionet.org/content/eegmmidb](https://physionet.org/content/eegmmidb/)
