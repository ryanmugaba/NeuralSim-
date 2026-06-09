"""NeuralSim: replay PhysioNet EEG motor imagery as a simulated 4-command BCI."""

from .commands import ALL_COMMANDS, Command
from .data import BCIDataset, load_eegbci, make_synthetic
from .decoder import BandPowerDecoder, CSPDecoder, band_power
from .demo import run_demo
from .stream import BCIStream

__version__ = "0.1.0"

__all__ = [
    "Command",
    "ALL_COMMANDS",
    "BCIDataset",
    "load_eegbci",
    "make_synthetic",
    "BandPowerDecoder",
    "CSPDecoder",
    "band_power",
    "BCIStream",
    "run_demo",
    "__version__",
]
