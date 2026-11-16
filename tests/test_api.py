import numpy as np
import pytest
from pathlib import Path

from fxseisfilter import FXConfig, fx_filter, metrics, process, process_file, read_seismic


FIXTURE_DIR = Path(__file__).resolve().parent / "data"


def test_identity_reconstruction():
    rng = np.random.default_rng(12)
    data = rng.standard_normal((64, 24))
    config = FXConfig(dt=.002, FB=5, FE=100, LW=32, LX=16, LOP=2)
    reconstructed = fx_filter(data, config, identity=True)
    np.testing.assert_allclose(reconstructed, data, rtol=1e-12, atol=1e-12)


def test_process_returns_finite_same_shape_arrays():
    rng = np.random.default_rng(22)
    data = rng.standard_normal((96, 40))
    baseline, filtered, _ = process(
        data, .002, fb=5, fe=100, lw=64, lx=24, lop=4,
        use_bandpass=False,
    )
    assert baseline.shape == filtered.shape == data.shape
    assert np.isfinite(baseline).all()
    assert np.isfinite(filtered).all()


def test_file_api_reads_processes_and_saves_npz(tmp_path):
    rng = np.random.default_rng(33)
    data = rng.standard_normal((96, 40))
    source = tmp_path / "input.npz"
    output = tmp_path / "result.npz"
    np.savez_compressed(source, data=data, dt=.002)
    source_bytes = source.read_bytes()

    result = process_file(
        source, output, key="data", lw=64, lx=24, lop=4,
        use_bandpass=False,
    )

    assert result["output_path"] == output
    assert output.is_file()
    assert source.read_bytes() == source_bytes
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["raw"], data)
        np.testing.assert_allclose(archive["fx"], result["fx"])
        assert float(archive["dt"]) == .002


def test_read_text_requires_explicit_dt(tmp_path):
    data = np.arange(64 * 8, dtype=float).reshape(64, 8)
    path = tmp_path / "section.csv"
    np.savetxt(path, data, delimiter=",")
    with pytest.raises(ValueError, match="采样间隔未知"):
        read_seismic(path)
    loaded, dt, _ = read_seismic(path, dt=.004)
    np.testing.assert_array_equal(loaded, data)
    assert dt == .004


def test_file_api_refuses_to_overwrite_input(tmp_path):
    data = np.ones((16, 8))
    path = tmp_path / "section.npz"
    np.savez_compressed(path, data=data, dt=.002)
    with pytest.raises(ValueError, match="must not overwrite"):
        process_file(path, path, key="data")


def test_checked_in_synthetic_fixture_complete_workflow(tmp_path):
    source = FIXTURE_DIR / "synthetic_noisy.csv"
    clean, dt, _ = read_seismic(FIXTURE_DIR / "synthetic_clean.csv", dt=.002)
    result = process_file(
        source, tmp_path / "synthetic_result.npz", dt=dt,
        lw=64, lx=24, lop=4,
    )

    assert result["raw"].shape == clean.shape == (128, 32)
    assert result["dt"] == dt
    assert np.isfinite(result["fx"]).all()
    scores = metrics(clean, result["fx"])
    assert set(scores) == {"snr_db", "rmse", "relative_error", "correlation"}
    assert all(np.isfinite(value) for value in scores.values())
