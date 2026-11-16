# fx-seismic-filter

Python package for F-X domain prediction filtering of seismic sections. It reads a seismic section into a NumPy array, optionally applies a zero-phase bandpass baseline, runs local bidirectional least-squares F-X prediction, and can save the inputs and outputs in an NPZ result bundle.

The distribution name is `fx-seismic-filter`; the Python import name is `fxseisfilter`.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install "fx-seismic-filter[formats]"
```

The `formats` extra installs SEG-Y, SU, and SEG-2 readers. The core installation supports CST, NPY, NPZ, MAT, CSV, and TXT. Install from a local checkout while developing:

```bash
python -m pip install -e ".[dev,formats]"
```

The package has not yet been published to PyPI. Until then, install from a local checkout or a built wheel.

## Quick Start

```python
from fxseisfilter import process_file

result = process_file(
    "input.sgy",
    output_path="output_fx.npz",
    fb=5.0,
    fe=100.0,
    lw=128,
    lx=32,
    lop=4,
    ac=0.05,
)

fx_data = result["fx"]
print(fx_data.shape)       # (time samples, traces)
print(result["dt"])        # sampling interval in seconds
print(result["output_path"])
```

`process_file` does not modify the input file. If `output_path` is omitted, a sibling file named `<input>_fx_result.npz` is created. The return dictionary contains `raw`, `bandpass`, `fx`, `removed`, `dt`, `source_metadata`, `parameters`, and `output_path`.

To process an array already in memory:

```python
import numpy as np
from fxseisfilter import process

data = np.load("section.npy")  # shape: [time samples, traces]
baseline, fx_data, parameters = process(data, dt=0.002, lw=128, lx=32)
```

To read without processing:

```python
from fxseisfilter import read_seismic

data, dt, metadata = read_seismic("section.sgy")
```

The CLI is also installed as `fx-seismic-filter`:

```bash
fx-seismic-filter input.sgy output_fx.npz --fb 5 --fe 100 --lw 128 --lx 32 --lop 4 --ac 0.05
```

## Input Formats

All inputs are returned as finite, real, two-dimensional `float64` arrays in `[time sample, trace]` order. Files with an explicit sample interval use that metadata unless `dt` is passed to override it.

| Format | Extensions | Sample interval |
|---|---|---|
| CST | `.cst` | Read from the supported CST trace header |
| NumPy array | `.npy` | Pass `dt` |
| NumPy archive | `.npz` | Read `dt` or `dt_ms` when present; otherwise pass `dt` |
| SEG-Y | `.sgy`, `.segy` | Read from binary header (`segyio` extra) |
| SU | `.su` | Read from trace headers (`obspy` extra) |
| SEG-2 | `.sg2`, `.seg2` | Read from trace headers (`obspy` extra) |
| MATLAB | `.mat` | Pass `dt`; select a matrix with `key` if needed |
| Text matrix | `.csv`, `.txt` | Pass `dt`; `skiprows` skips leading header rows |

The CST reader currently supports the workspace-compatible record layout with IEEE float32 samples (format code 8). It does not claim support for every vendor-specific CST variant. SEG-2 and SEG-Y vendor extensions also vary; test representative files from the intended source before production use. Ragged text matrices and MATLAB v7.3/HDF5 are not supported by the current reader.

For arrays stored as `[trace, time]`, pass `transpose=True`. Sampling interval `dt` is always in seconds.

## Parameters

| Parameter | Meaning | Default |
|---|---|---:|
| `fb`, `fe` | F-X prediction frequency band in Hz | 5, 100 |
| `lw` | Local time-window length in samples | 128 |
| `lx` | Local trace-window width | 32 |
| `lop` | Prediction order on each side | 4 |
| `ac` | Relative diagonal loading for the complex least-squares solve | 0.05 |
| `overlap` | Window overlap ratio | 0.5 |
| `bp_low`, `bp_high` | Baseline Butterworth bandpass limits in Hz | 5, 100 |
| `use_bandpass` | Apply the baseline before F-X filtering | `True` |

The F-X frequency band must fit below the Nyquist frequency (`0.5 / dt`). Each effective trace window must contain more than `2 * lop` traces. `ac` is a dimensionless relative regularization parameter; it is not a vendor software parameter with the same name.

## Development

```bash
python -m pip install -e ".[dev,formats]"
pytest
python -m build
```

Package layout:

```text
src/fxseisfilter/   importable package
tests/              API and algorithm tests
examples/           runnable examples
```

## License

No license has been selected yet. Choose and add a license before publishing this repository as open source; a public GitHub repository without a license does not grant general reuse rights.
