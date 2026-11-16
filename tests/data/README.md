# Test Data

`synthetic_clean.csv` and `synthetic_noisy.csv` are deterministic synthetic sections in `[time sample, trace]` order. Each file contains 128 samples and 32 traces. The sampling interval is 0.002 seconds and is supplied by the tests because CSV has no seismic header.

The clean section contains three laterally coherent Ricker wavelets with simple moveout. The noisy section adds zero-mean Gaussian noise generated with NumPy seed `20261116` and standard deviation `0.08`. These files are software test fixtures, not field data or evidence of field performance.

Regenerate both files from this directory with:

```bash
python generate_fixture.py
```
