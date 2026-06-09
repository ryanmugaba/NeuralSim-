"""Real-time terminal demo: read trials, classify them, print live results."""

from __future__ import annotations

import argparse
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="neuralsim-demo",
                                description="Replay EEG motor imagery as a live BCI in the terminal.")
    p.add_argument("--subject", type=int, default=1, help="PhysioNet subject id (1-109).")
    p.add_argument("--synthetic", action="store_true",
                   help="Skip the download; use a synthetic stand-in dataset.")
    p.add_argument("--speed", type=float, default=8.0, help="Replay speed multiplier (1 = real time).")
    p.add_argument("--train-frac", type=float, default=0.7, help="Fraction of trials used to train.")
    p.add_argument("--csp", action="store_true", help="Use CSP+LDA decoder (needs the [sklearn] extra).")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
