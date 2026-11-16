"""File-oriented API and result bundle serialization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .formats import read_seismic
from .processing import process


def save_result(path: str | Path, *, raw: np.ndarray, bandpass: np.ndarray,
                fx: np.ndarray, dt: float, source_metadata: dict[str, Any] | None = None,
                parameters: dict[str, Any] | None = None) -> Path:
    """Save input, baseline, filtered data, removed component and metadata."""
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".npz":
        destination = destination.with_suffix(".npz")
    destination = destination.resolve()
    original = np.asarray(raw)
    baseline = np.asarray(bandpass)
    filtered = np.asarray(fx)
    if original.shape != baseline.shape or original.shape != filtered.shape:
        raise ValueError("raw, bandpass and fx must have identical shapes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, raw=original, bandpass=baseline, fx=filtered,
        removed=baseline-filtered, dt=np.float64(dt),
        source_metadata_json=np.array(json.dumps(source_metadata or {}, ensure_ascii=False)),
        parameters_json=np.array(json.dumps(parameters or {}, ensure_ascii=False)),
    )
    return destination


def process_file(input_path: str | Path, output_path: str | Path | None = None, *,
                 dt: float | None = None, key: str | None = None,
                 transpose: bool = False, skiprows: int = 0,
                 fb: float = 5.0, fe: float = 100.0,
                 lw: int = 128, lx: int = 32, lop: int = 4, ac: float = .05,
                 overlap: float = .5, bp_low: float = 5.0, bp_high: float = 100.0,
                 use_bandpass: bool = True) -> dict[str, Any]:
    """Read, process, and save a seismic file without modifying the input."""
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = (Path(output_path).expanduser() if output_path is not None
                   else source.with_name(f"{source.stem}_fx_result.npz"))
    if destination.suffix.lower() != ".npz":
        destination = destination.with_suffix(".npz")
    if source == destination.resolve():
        raise ValueError("output path must not overwrite the input file")
    data, sampling_interval, source_metadata = read_seismic(
        source, dt=dt, key=key, transpose=transpose, skiprows=skiprows)
    baseline, filtered, parameters = process(
        data, sampling_interval, fb=fb, fe=fe, lw=lw, lx=lx, lop=lop, ac=ac,
        overlap=overlap, bp_low=bp_low, bp_high=bp_high,
        use_bandpass=use_bandpass)
    saved_to = save_result(destination, raw=data, bandpass=baseline, fx=filtered,
                           dt=sampling_interval, source_metadata=source_metadata,
                           parameters=parameters)
    return {
        "output_path": saved_to, "raw": data, "bandpass": baseline,
        "fx": filtered, "removed": baseline-filtered, "dt": sampling_interval,
        "source_metadata": source_metadata, "parameters": parameters,
    }
