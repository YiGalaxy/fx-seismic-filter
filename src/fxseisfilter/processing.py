"""High-level F-X processing functions."""
from __future__ import annotations

import numpy as np

from ._algorithm import FXConfig, bandpass, fx_filter


def process(data: np.ndarray, dt: float, *, fb: float = 5.0, fe: float = 100.0,
            lw: int = 128, lx: int = 32, lop: int = 4, ac: float = .05,
            overlap: float = .5, bp_low: float = 5.0, bp_high: float = 100.0,
            use_bandpass: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the bandpass baseline and F-X filter on ``data[time, trace]``."""
    samples = np.asarray(data, dtype=np.float64)
    if samples.ndim != 2 or not np.isfinite(samples).all():
        raise ValueError("data must be a finite 2-D [time, trace] matrix")
    config = FXConfig(dt=dt, FB=fb, FE=fe, LW=lw, LX=lx, LOP=lop, AC=ac,
                      overlap=overlap)
    baseline = (bandpass(samples, dt, bp_low, bp_high) if use_bandpass
                else samples.copy())
    filtered = fx_filter(baseline, config)
    parameters = {
        "algorithm": "literature_bidir_lsq",
        "fx_config": config.__dict__,
        "bandpass": {
            "enabled": use_bandpass,
            "low_hz": bp_low,
            "high_hz": bp_high,
            "type": "4th-order zero-phase Butterworth",
        },
    }
    return baseline, filtered, parameters
