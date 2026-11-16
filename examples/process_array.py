"""Run the F-X filter on an in-memory section."""
import numpy as np

from fxseisfilter import process


def main() -> None:
    rng = np.random.default_rng(7)
    section = rng.standard_normal((256, 64))
    baseline, filtered, parameters = process(
        section,
        dt=0.002,
        fb=5.0,
        fe=100.0,
        lw=128,
        lx=32,
        lop=4,
        ac=0.05,
    )
    np.savez_compressed(
        "example_result.npz",
        raw=section,
        bandpass=baseline,
        fx=filtered,
    )
    print("Filtered matrix:", filtered.shape)
    print("Algorithm settings:", parameters)


if __name__ == "__main__":
    main()
