"""Read, filter, save, and inspect one seismic data file."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fxseisfilter import process_file


def main() -> None:
    # argparse lets a user reuse this script with different surveys and outputs.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input seismic file")
    parser.add_argument("output", type=Path, help="output NPZ result bundle")
    # Matrix files often have no header from which the sample interval can be read.
    parser.add_argument("--dt", type=float, help="sampling interval in seconds, if absent from input")
    parser.add_argument("--key", help="matrix key for NPZ or MAT files")
    parser.add_argument("--transpose", action="store_true", help="input matrix is [trace, time]")
    args = parser.parse_args()

    # This call reads the source, runs the bandpass baseline and F-X filter,
    # and writes a separate result file so the original data stays untouched.
    result = process_file(
        args.input,
        args.output,
        dt=args.dt,
        key=args.key,
        transpose=args.transpose,
        fb=5.0,
        fe=100.0,
        lw=128,
        lx=32,
        lop=4,
        ac=0.05,
    )

    # These values confirm which data volume and sampling scale were processed.
    print(f"Input shape [time, trace]: {result['raw'].shape}")
    print(f"Sampling interval: {result['dt']} s")
    print(f"Output file: {result['output_path']}")

    # Reopen the saved bundle to verify its main arrays can be read back.
    # allow_pickle=False avoids loading executable Python objects from an archive.
    with np.load(result["output_path"], allow_pickle=False) as bundle:
        for name in ("raw", "bandpass", "fx", "removed"):
            print(f"{name}: shape={bundle[name].shape}, finite={np.isfinite(bundle[name]).all()}")


if __name__ == "__main__":
    main()
