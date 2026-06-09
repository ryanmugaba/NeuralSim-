# NeuralSim

Replay real **PhysioNet EEG motor-imagery** recordings as a **simulated brain-computer interface (BCI)** and watch them get decoded into 4 commands in your terminal, in real time.

```
LEFT   <--      RIGHT  -->      SELECT  [OK]      IDLE  ...
```

NeuralSim is a small, readable teaching/prototyping library. It loads recorded brain signals with MNE-Python, streams them as if they were arriving live from a headset, and classifies each window with a transparent NumPy decoder.

---

## What it actually does

1. **Loads** the PhysioNet EEG Motor Movement/Imagery dataset via `mne.datasets.eegbci` (downloads on first use, then caches).
2. **Replays** recorded trials through a real-time streamer (`BCIStream`) that paces output to mimic live acquisition.
3. **Decodes** each trial into one of four commands: `LEFT`, `RIGHT`, `SELECT`, `IDLE`.
4. **Demo**: a terminal app shows each incoming signal window, the true label, the decoded command, a confidence bar, and a running accuracy.

### Honest caveats (read these)

- This is a **simulator and learning tool**, not a production BCI. It replays recorded data; it does not read a live brain.
- The default decoder is a deliberately simple **NumPy band-power nearest-centroid** classifier. On real EEG it lands roughly in the **50-70%** range for 4 classes (chance is 25%). That is normal for a lightweight motor-imagery decoder and far below clinical systems.
- For meaningfully better accuracy, install the `sklearn` extra and switch to `CSPDecoder` (Common Spatial Patterns + LDA), the standard baseline for this dataset.
- The `SELECT` command is mapped from "imagine both fists" trials (runs 6/10/14). It is the weakest-separated class.

---

## Install

```bash
pip install -e .                 # core: MNE + NumPy
pip install -e ".[sklearn]"      # + scikit-learn for the CSP+LDA decoder
```

Python 3.9+.

---

## Quickstart

Run the live demo. First run downloads ~50 MB from PhysioNet:

```bash
neuralsim-demo --subject 1 --speed 8
```

No network or just want to see it run instantly? Use the synthetic stand-in:

```bash
neuralsim-demo --synthetic --speed 8
```

Use the stronger decoder (needs the `sklearn` extra):

```bash
neuralsim-demo --subject 1 --csp
```

CLI flags: `--subject` (1-109), `--synthetic`, `--speed` (1 = real time), `--train-frac`, `--csp`, `--seed`.

### Sample output

```
NeuralSim :: PhysioNet EEG motor-imagery -> simulated BCI

  120 trials | 64 ch @ 160 Hz | 480 samples/trial | LEFT=30, RIGHT=30, SELECT=30, IDLE=30

  Decoder: NumPy band-power nearest-centroid | trained on 84 trials, streaming 36 live
  ----------------------------------------------------------------
  t=  1 | signal in [64ch x 480] | true LEFT   -> pred <-- LEFT   OK  conf #########--- 74% | acc 100%
  t=  2 | signal in [64ch x 480] | true RIGHT  -> pred ... IDLE   XX  conf ######------ 52% | acc  50%
  ...
```

---

## Use it as a library

```python
from neuralsim import load_eegbci, make_synthetic, BandPowerDecoder, BCIStream

# Real data (downloads on first call). Or: ds = make_synthetic()
ds = load_eegbci(subject=1)
print(ds.summary())

# Train a decoder
dec = BandPowerDecoder(ch_indices=ds.motor_indices, sfreq=ds.sfreq)
dec.fit(ds.signals, ds.commands)

# Stream trials in "real time" and classify them
for idx, signal, true_cmd in BCIStream(ds.signals, ds.commands, ds.sfreq, speed=8):
    pred, confidence, scores = dec.predict_one(signal)
    print(true_cmd, "->", pred, f"({confidence:.0%})")
```

---

## Command mapping

PhysioNet annotations (`T0/T1/T2`) mean different things per run. NeuralSim maps them like this:

| Command  | Source runs        | Annotation | Meaning                       |
|----------|--------------------|------------|-------------------------------|
| `LEFT`   | 4, 8, 12           | T1         | Imagine left fist             |
| `RIGHT`  | 4, 8, 12           | T2         | Imagine right fist            |
| `SELECT` | 6, 10, 14          | T1         | Imagine both fists            |
| `IDLE`   | 4, 8, 12           | T0         | Rest                          |

---

## How it works

```
PhysioNet EDF  ->  MNE load + standardize + 7-30 Hz filter  ->  epoch into trials
       |                                                              |
       |                                                       relabel to commands
       v                                                              v
   BCIStream (paced replay)  ----per trial---->  BandPowerDecoder / CSPDecoder
                                                          |
                                                  LEFT / RIGHT / SELECT / IDLE
```

- **Features (default):** log power in the mu (8-12 Hz) and beta (13-30 Hz) bands over motor-cortex channels (C3/Cz/C4 and neighbours), via FFT. Pure NumPy.
- **Classifier (default):** z-score features, then nearest class centroid; softmax over negative distance gives a confidence.
- **Classifier (optional):** `CSPDecoder` = MNE `CSP` + scikit-learn `LDA`.

---

## Package layout

```
neuralsim/
├── neuralsim/
│   ├── __init__.py      public API
│   ├── commands.py      Command enum + label mapping
│   ├── data.py          PhysioNet loader (MNE) + synthetic generator
│   ├── decoder.py       band-power decoder (NumPy) + CSP decoder (optional)
│   ├── stream.py        real-time replay
│   └── demo.py          terminal demo + CLI
├── tests/test_smoke.py
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

Run tests with `pip install -e ".[dev]" && pytest`.

---

## Dependencies

- [MNE-Python](https://mne.tools) (EEG loading and processing)
- NumPy
- scikit-learn (optional, only for `CSPDecoder`)

## License

MIT. The PhysioNet EEGBCI dataset has its own terms; see the
[PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/).
