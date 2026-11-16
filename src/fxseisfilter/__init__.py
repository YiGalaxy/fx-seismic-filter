"""F-X seismic random-noise filtering package."""
from ._algorithm import FXConfig, fx_filter, metrics
from .api import process_file, save_result
from .formats import read_seismic
from .processing import process

__version__ = "0.1.0"
__all__ = [
    "FXConfig", "fx_filter", "metrics", "process", "process_file",
    "read_seismic", "save_result",
]
