"""Regenerate the small deterministic CSV fixtures used by the tests."""
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    sample_count = 128
    trace_count = 32
    dt = 0.002
    time = np.arange(sample_count)[:, None] * dt
    traces = np.arange(trace_count)[None, :]
    clean = np.zeros((sample_count, trace_count), dtype=np.float64)

    for center, frequency, amplitude, slope in (
        (0.045, 32.0, 1.0, 0.00008),
        (0.115, 24.0, 0.65, -0.00012),
        (0.190, 36.0, 0.45, 0.00005),
    ):
        arrival = center + slope * (traces - (trace_count - 1) / 2)
        phase = np.pi * frequency * (time - arrival)
        clean += amplitude * (1 - 2 * phase**2) * np.exp(-phase**2)

    rng = np.random.default_rng(20261116)
    noisy = clean + rng.normal(0.0, 0.08, clean.shape)
    np.savetxt(DATA_DIR / "synthetic_clean.csv", clean, delimiter=",", fmt="%.10e")
    np.savetxt(DATA_DIR / "synthetic_noisy.csv", noisy, delimiter=",", fmt="%.10e")


if __name__ == "__main__":
    main()
