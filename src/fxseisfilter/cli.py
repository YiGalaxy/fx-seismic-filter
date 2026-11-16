"""Command-line interface for processing a seismic file."""
from __future__ import annotations

import argparse
import json

from .api import process_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="supported seismic input file")
    parser.add_argument("output", nargs="?", help="result NPZ path")
    parser.add_argument("--dt", type=float, help="sampling interval in seconds")
    parser.add_argument("--key", help="matrix key in NPZ or MAT files")
    parser.add_argument("--transpose", action="store_true", help="input is [trace, time]")
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--fb", type=float, default=5.0)
    parser.add_argument("--fe", type=float, default=100.0)
    parser.add_argument("--lw", type=int, default=128)
    parser.add_argument("--lx", type=int, default=32)
    parser.add_argument("--lop", type=int, default=4)
    parser.add_argument("--ac", type=float, default=.05)
    parser.add_argument("--overlap", type=float, default=.5)
    parser.add_argument("--bp-low", type=float, default=5.0)
    parser.add_argument("--bp-high", type=float, default=100.0)
    parser.add_argument("--no-bandpass", action="store_true")
    args = parser.parse_args()
    result = process_file(
        args.input, args.output, dt=args.dt, key=args.key,
        transpose=args.transpose, skiprows=args.skiprows,
        fb=args.fb, fe=args.fe, lw=args.lw, lx=args.lx, lop=args.lop,
        ac=args.ac, overlap=args.overlap, bp_low=args.bp_low,
        bp_high=args.bp_high, use_bandpass=not args.no_bandpass,
    )
    print(json.dumps({
        "output": str(result["output_path"]),
        "shape_time_trace": list(result["fx"].shape),
        "sampling_interval_seconds": result["dt"],
        "parameters": result["parameters"],
    }, ensure_ascii=False, indent=2))
