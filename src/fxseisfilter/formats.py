"""Import common seismic formats as a time-sample by trace NumPy matrix."""
from __future__ import annotations

from pathlib import Path
import struct

import numpy as np


def _dt(value: float | None, stored: float | None) -> float:
    result = value if value is not None else stored
    if result is None or not np.isfinite(result) or result <= 0:
        raise ValueError("采样间隔未知。请使用 --dt 指定秒，例如 --dt 0.002。")
    return float(result)


def _numeric_matrix(data: np.ndarray, transpose: bool) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 2 or data.dtype.kind not in "fiu" or not np.isfinite(data).all():
        raise ValueError("输入必须是有限值的二维实数矩阵。")
    if transpose:
        data = data.T
    if data.shape[0] < 8 or data.shape[1] < 1:
        raise ValueError("统一后的时间轴至少需要 8 个样点。请核对矩阵方向。")
    return np.asarray(data, dtype=np.float64)


def _read_cst(path: Path) -> tuple[np.ndarray, float, dict]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(512)
    if len(header) < 256:
        raise ValueError("CST 文件小于一个道头。")
    layouts = []
    for endian in ("<", ">"):
        samples = struct.unpack_from(endian + "i", header, 0)[0] + 1
        interval_us = struct.unpack_from(endian + "i", header, 32)[0]
        format_code = (struct.unpack_from(endian + "i", header, 40)[0] >> 20) & 0xFFF
        if 32 <= samples <= 10000 and 100 <= interval_us <= 10000 and format_code in (1, 2, 3, 4, 7, 8):
            bytes_per_sample = 4 if format_code in (3, 4, 7, 8) else 2
            record_bytes = ((256 + samples * bytes_per_sample + 511) // 512) * 512
            if size % record_bytes == 0:
                layouts.append((endian, samples, interval_us, format_code, record_bytes, size // record_bytes))
    if len(layouts) != 1:
        raise ValueError("不能唯一识别此 CST 道头布局。")
    endian, samples, interval_us, format_code, record_bytes, traces = layouts[0]
    if format_code != 8:
        raise ValueError(f"CST 格式码为 {format_code}，当前仅支持 IEEE float32（格式码 8）。")
    records = np.memmap(path, mode="r", dtype="u1", shape=(traces, record_bytes))
    headers = np.ndarray((traces, 64), dtype=endian + "i4", buffer=records, strides=(record_bytes, 4))
    trace_data = np.ndarray((traces, samples), dtype=endian + "f4", buffer=records,
                            offset=256, strides=(record_bytes, 4))
    data = np.array(trace_data.T, dtype=np.float64, copy=True)
    valid_samples = np.clip(headers[:, 0].astype(np.int64) + 1, 0, samples)
    data[np.arange(samples)[:, None] >= valid_samples[None, :]] = 0
    metadata = {
        "source_file": str(path), "source_format": "cst", "trace_count": traces,
        "sample_count": samples, "array_key": None, "dt_from_file": True,
        "sampling_interval_seconds": interval_us * 1e-6,
        "cdp_first": int(headers[0, 3]), "cdp_last": int(headers[-1, 3]),
    }
    del records
    return data, interval_us * 1e-6, metadata


def read_seismic(path: Path, *, dt: float | None = None, key: str | None = None,
                 transpose: bool = False, skiprows: int = 0) -> tuple[np.ndarray, float, dict]:
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    stored_dt = None
    metadata: dict = {"source_file": str(path), "source_format": suffix.lstrip(".")}

    if suffix == ".cst":
        data, stored_dt, metadata = _read_cst(path)
    elif suffix == ".npy":
        data = np.load(path, allow_pickle=False)
        metadata.update(array_key="data", dt_from_file=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files
                      if archive[name].ndim == 2 and archive[name].dtype.kind in "fiu"}
            if not arrays:
                raise ValueError("NPZ 文件中没有二维实数矩阵。")
            if key is None:
                preferred = ("raw", "noisy", "input", "data", "clean", "bandpass", "fx")
                key = next((candidate for candidate in preferred if candidate in arrays), None)
                if key is None and len(arrays) == 1:
                    key = next(iter(arrays))
                if key is None:
                    raise ValueError(f"NPZ 中有多个二维矩阵，请指定 key：{', '.join(arrays)}")
            if key not in arrays:
                raise ValueError(f"NPZ 中找不到数组 {key!r}；可用数组：{', '.join(arrays)}")
            data = arrays[key].copy()
            if "dt" in archive and np.asarray(archive["dt"]).size == 1:
                stored_dt = float(np.asarray(archive["dt"]).item())
            elif "dt_ms" in archive and np.asarray(archive["dt_ms"]).size == 1:
                stored_dt = float(np.asarray(archive["dt_ms"]).item()) * .001
            metadata.update(array_key=key, dt_from_file=stored_dt is not None)
    elif suffix in (".sgy", ".segy"):
        try:
            import segyio
        except ImportError as exc:
            raise RuntimeError("读取 SEG-Y 需要安装 segyio：python -m pip install segyio") from exc
        with segyio.open(str(path), "r", strict=False, ignore_geometry=True) as stream:
            traces = np.asarray(stream.trace.raw[:])
            interval_us = float(stream.bin[segyio.BinField.Interval])
            stored_dt = interval_us * 1e-6 if interval_us > 0 else None
            data = traces.T
            metadata.update(trace_count=int(traces.shape[0]), sample_count=int(traces.shape[1]),
                            dt_from_file=stored_dt is not None, sample_interval_us=interval_us)
    elif suffix in (".su", ".sg2", ".seg2"):
        try:
            from obspy import read
        except ImportError as exc:
            raise RuntimeError("读取 SU/SEG-2 需要安装 ObsPy：python -m pip install obspy") from exc
        stream = read(str(path), format="SU" if suffix == ".su" else "SEG2")
        if not stream or len({len(trace.data) for trace in stream}) != 1:
            raise ValueError("SU/SEG-2 文件为空，或各道样点数不同；请先统一道长。")
        intervals = {round(float(trace.stats.delta), 12) for trace in stream}
        if len(intervals) != 1:
            raise ValueError("SU/SEG-2 道间采样间隔不一致，不能直接组成规则矩阵。")
        stored_dt = intervals.pop()
        data = np.stack([trace.data for trace in stream], axis=1)
        metadata.update(trace_count=len(stream), sample_count=int(data.shape[0]), dt_from_file=True)
    elif suffix in (".csv", ".txt"):
        delimiter = "," if suffix == ".csv" else None
        data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows, ndmin=2)
        metadata.update(skiprows=skiprows, dt_from_file=False)
    elif suffix == ".mat":
        from scipy.io import loadmat
        arrays = {name: value for name, value in loadmat(path).items()
                  if not name.startswith("__") and isinstance(value, np.ndarray)
                  and value.ndim == 2 and value.size > 1 and value.dtype.kind in "fiu"}
        if not arrays:
            raise ValueError("MAT 文件中没有二维实数矩阵。")
        if key is None:
            if len(arrays) != 1:
                raise ValueError(f"MAT 中有多个二维矩阵，请用 --key 指定：{', '.join(arrays)}")
            key = next(iter(arrays))
        if key not in arrays:
            raise ValueError(f"MAT 中找不到矩阵 {key!r}；可用矩阵：{', '.join(arrays)}")
        data = arrays[key]
        metadata.update(array_key=key, dt_from_file=False)
    else:
        raise ValueError(f"暂不支持 {suffix or '(无扩展名)'}。请导出为 SEG-Y、SU、SEG-2、CST、NPY、NPZ、MAT 或数值 CSV/TXT。")

    data = _numeric_matrix(data, transpose)
    sampling_interval = _dt(dt, stored_dt)
    metadata.update(shape_time_trace=list(data.shape), sampling_interval_seconds=sampling_interval)
    return data, sampling_interval, metadata
