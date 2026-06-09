"""The 4 BCI commands NeuralSim emits and how PhysioNet labels map onto them."""

from enum import Enum


class Command(str, Enum):
    """A decoded brain-computer-interface command.

    Subclasses ``str`` so values compare/print as plain strings
    (e.g. ``Command.LEFT == "LEFT"`` is ``True``).
    """

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    SELECT = "SELECT"
    IDLE = "IDLE"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


# PhysioNet EEGBCI annotations only carry T0/T1/T2 *within a run*, and their
# meaning depends on which run produced them. We therefore map per run-group:
#
#   Left/Right imagery runs (4, 8, 12):  T1 -> LEFT,  T2 -> RIGHT, T0 -> IDLE
#   Fists/Feet  imagery runs (6, 10, 14): T1 -> SELECT (both fists), T0 -> IDLE
#
# This is the cleanest way to extract 4 distinct, physiologically grounded
# classes from this dataset without inventing fake signals.

LR_RUNS = (4, 8, 12)
SELECT_RUNS = (6, 10, 14)

LR_MAP = {"T0": Command.IDLE, "T1": Command.LEFT, "T2": Command.RIGHT}
SELECT_MAP = {"T1": Command.SELECT}  # T0 here is redundant IDLE; T2 (feet) dropped

ALL_COMMANDS = [Command.LEFT, Command.RIGHT, Command.SELECT, Command.IDLE]
