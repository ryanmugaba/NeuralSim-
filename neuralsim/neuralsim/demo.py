"""Real-time terminal demo: read trials, classify them, print live results."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .commands import ALL_COMMANDS, Command
from .data import BCIDataset, load_eegbci, make_synthetic
from .decoder import BandPowerDecoder, CSPDecoder
from .stream import BCIStream

_ARROW = {Command.LEFT: "<--", Command.RIGHT: "-->", Command.SELECT: "[OK]", Command.IDLE: "..."}


def _bar(value: float, width: int = 12) -> str:
    filled = int(round(value * width))
    return "#" * filled + "-" * (width - filled)


def _split(dataset: BCIDataset, train_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    n = len(dataset)
    idx = rng.permutation(n)
    cut = int(train_frac * n)
    tr, te = idx[:cut], idx[cut:]
    return (dataset.signals[tr], [dataset.commands[i] for i in tr],
            dataset.signals[te], [dataset.commands[i] for i in te])


def run_demo(dataset: BCIDataset, speed: float = 8.0, train_frac: float = 0.7,
             use_csp: bool = False, seed: int = 0) -> float:
    print("NeuralSim :: PhysioNet EEG motor-imagery -> simulated BCI\n")
    print("  " + dataset.summary() + "\n")

    Xtr, ytr, Xte, yte = _split(dataset, train_frac, seed)
    decoder = CSPDecoder() if use_csp else BandPowerDecoder(
        ch_indices=dataset.motor_indices, sfreq=dataset.sfreq)
    decoder.fit(Xtr, ytr)
    print(f"  Decoder: {'CSP+LDA' if use_csp else 'NumPy band-power nearest-centroid'} "
          f"| trained on {len(ytr)} trials, streaming {len(yte)} live\n")
    print("  " + "-" * 64)

    stream = BCIStream(Xte, yte, sfreq=dataset.sfreq, speed=speed, seed=seed)
    correct = total = 0
    for _, sig, true_cmd in stream:
        pred, conf, _ = decoder.predict_one(sig)
        total += 1
        correct += int(pred == true_cmd)
        ok = "OK " if pred == true_cmd else "XX "
        line = (f"  t={total:>3} | signal in [{sig.shape[0]}ch x {sig.shape[1]}] "
                f"| true {true_cmd.value:<6} -> pred {_ARROW[pred]} {pred.value:<6} "
                f"{ok} conf {_bar(conf)} {conf:4.0%} | acc {correct/total:4.0%}")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    print("  " + "-" * 64)
    acc = correct / total if total else 0.0
    print(f"\n  Done. Final accuracy: {acc:.1%} on {total} trials "
          f"(chance = {1/len(ALL_COMMANDS):.0%}).\n")
    return acc


def _make_personal_trials(dataset: BCIDataset, decoder: BandPowerDecoder,
                          n_per_class: int = 20, seed: int = 42):
    """Return (trials, labels) simulating personal EEG via per-channel amplitude scaling.

    Each motor channel gets a fixed multiplicative scale simulating electrode
    placement variability.  Scaling changes inter-channel amplitude ratios —
    exactly what EEGNet's spatial conv uses — so the base decoder accuracy drops.
    The scale is consistent across all personal trials (same "person"), so the
    adapter can learn to correct for it.
    """
    rng = np.random.default_rng(seed)
    motor_idx = dataset.motor_indices

    # Personal amplitude scale per motor channel — the user's "EEG signature".
    # log-normal: always positive, different per channel, typical range 0.1–5x.
    channel_scale = rng.lognormal(0.0, 1.0, size=len(motor_idx))

    by_class: dict = {cmd: [] for cmd in ALL_COMMANDS}
    for i, cmd in enumerate(dataset.commands):
        by_class[cmd].append(i)

    trials, labels = [], []
    for cmd in ALL_COMMANDS:
        idx_pool = by_class[cmd]
        chosen = rng.choice(idx_pool,
                            size=min(n_per_class, len(idx_pool)),
                            replace=len(idx_pool) < n_per_class)
        while len(chosen) < n_per_class:
            chosen = np.concatenate([chosen,
                rng.choice(idx_pool, n_per_class - len(chosen), replace=True)])

        for i in chosen[:n_per_class]:
            sig = dataset.signals[i].astype(float).copy()
            sig[motor_idx] *= channel_scale[:, None]   # personal amplitude per channel
            trials.append(sig)
            labels.append(cmd)

    order = rng.permutation(len(labels))
    return [trials[i] for i in order], [labels[i] for i in order]


def run_calibration_demo(dataset: BCIDataset, decoder: BandPowerDecoder,
                         profile_path: str = "ryan.profile", seed: int = 0) -> None:
    """Show accuracy before and after personal calibration on simulated personal data."""
    from .calibration import PersonalCalibration

    print("\nNeuralSim :: Personal calibration simulation")
    print("  " + "-" * 64)
    print("  Generating 80 personal trials (20 per class) with")
    print("  simulated per-channel amplitude scaling...\n")

    personal_trials, personal_labels = _make_personal_trials(
        dataset, decoder, n_per_class=20, seed=seed + 12)

    # Accuracy BEFORE calibration (base decoder on personal data)
    correct_before = sum(
        decoder.predict_one(sig)[0] == cmd
        for sig, cmd in zip(personal_trials, personal_labels))
    acc_before = correct_before / len(personal_labels)
    print(f"  Accuracy BEFORE calibration : {acc_before:.1%}  (base EEGNet, personal data)")

    # Calibrate
    print("  Calibrating adapter (50 epochs, adapter weights only)...")
    cal = PersonalCalibration(decoder)
    cal.calibrate(personal_trials, personal_labels)

    # Accuracy AFTER calibration (adapter on same personal data)
    correct_after = sum(
        cal.predict(sig)[0] == cmd
        for sig, cmd in zip(personal_trials, personal_labels))
    acc_after = correct_after / len(personal_labels)
    print(f"  Accuracy AFTER  calibration : {acc_after:.1%}  (frozen backbone + adapter, same trials)")

    # Save profile
    cal.save_profile(profile_path)
    size_kb = os.path.getsize(profile_path) / 1024
    print(f"\n  Personal profile saved to {profile_path}  ({size_kb:.0f} KB)")
    print("  " + "-" * 64 + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="neuralsim-demo",
                                description="Replay EEG motor imagery as a live BCI in the terminal.")
    p.add_argument("--subject", type=int, default=1, help="PhysioNet subject id (1-109).")
    p.add_argument("--synthetic", action="store_true",
                   help="Skip the download; use a synthetic stand-in dataset.")
    p.add_argument("--speed", type=float, default=8.0, help="Replay speed multiplier (1 = real time).")
    p.add_argument("--train-frac", type=float, default=0.7, help="Fraction of trials used to train.")
    p.add_argument("--csp", action="store_true", help="Use CSP+LDA decoder (needs the [sklearn] extra).")
    p.add_argument("--calibrate", action="store_true",
                   help="After the main demo, run a personal calibration simulation.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.synthetic:
        ds = make_synthetic(seed=args.seed)
    else:
        try:
            print("Loading PhysioNet EEGBCI (first run downloads ~50 MB)...")
            ds = load_eegbci(subject=args.subject)
        except Exception as exc:  # network/download issues -> graceful fallback
            print(f"  Could not load real data ({exc.__class__.__name__}: {exc}).")
            print("  Falling back to --synthetic so you can still see it run.\n")
            ds = make_synthetic(seed=args.seed)

    run_demo(ds, speed=args.speed, train_frac=args.train_frac, use_csp=args.csp, seed=args.seed)

    if args.calibrate:
        if args.csp:
            print("  --calibrate is not supported with --csp (needs EEGNet backbone).\n")
        else:
            # Fit a fresh EEGNet decoder to use as the calibration backbone.
            Xtr, ytr, _Xte, _yte = _split(ds, args.train_frac, args.seed)
            cal_decoder = BandPowerDecoder(ch_indices=ds.motor_indices, sfreq=ds.sfreq)
            cal_decoder.fit(Xtr, ytr)
            run_calibration_demo(ds, cal_decoder, seed=args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
