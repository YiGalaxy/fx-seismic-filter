# fx-seismic-filter

[中文](#中文) | [English](#english)

## 中文

`fx-seismic-filter` 是用于地震剖面 F-X 域预测滤波的 Python 包。它将地震资料读取为 NumPy 数组，可选执行零相位带通基线处理，再运行局部双向最小二乘 F-X 预测滤波，并可将输入、处理结果和参数保存为 NPZ 文件。

发行包名称为 `fx-seismic-filter`，Python 导入名称为 `fxseisfilter`。

### 安装

需要 Python 3.10 或更高版本。安装核心包及 SEG-Y、SU、SEG-2 格式支持：

```bash
python -m pip install "fx-seismic-filter[formats]"
```

`formats` 可选依赖包含 SEG-Y、SU 和 SEG-2 读取器。仅安装核心包可处理 CST、NPY、NPZ、MAT、CSV 和 TXT。开发时从本地仓库安装：

```bash
python -m pip install -e ".[dev,formats]"
```

本包目前尚未发布到 PyPI；可从本地源码目录或构建的 wheel 安装。

### 快速开始

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
print(fx_data.shape)       # (时间样点数, 道数)
print(result["dt"])        # 采样间隔，单位秒
print(result["output_path"])
```

`process_file` 不会改写输入文件。省略 `output_path` 时，会在输入文件旁生成 `<输入文件名>_fx_result.npz`。返回字典包含 `raw`、`bandpass`、`fx`、`removed`、`dt`、`source_metadata`、`parameters` 和 `output_path`。

处理内存中的数组：

```python
import numpy as np
from fxseisfilter import process

data = np.load("section.npy")  # 形状：[时间样点, 道]
baseline, fx_data, parameters = process(data, dt=0.002, lw=128, lx=32)
```

只读取数据、不执行滤波：

```python
from fxseisfilter import read_seismic

data, dt, metadata = read_seismic("section.sgy")
```

安装后也可以使用命令行入口 `fx-seismic-filter`：

```bash
fx-seismic-filter input.sgy output_fx.npz --fb 5 --fe 100 --lw 128 --lx 32 --lop 4 --ac 0.05
```

#### 一次完整使用

下面的脚本从指定文件开始，完成读入、滤波、保存和回读检查。每段注释都说明了这一步做什么、为什么做，以及得到什么。

```bash
python examples/complete_workflow.py input.sgy results/output_fx.npz
python examples/complete_workflow.py input.csv results/output_fx.npz --dt 0.002
python examples/complete_workflow.py tests/data/synthetic_noisy.csv results/test_output.npz --dt 0.002
```

第一条命令处理文件头包含采样间隔的 SEG-Y；第二条处理没有采样信息的 CSV，因此通过 `--dt 0.002` 指定 2 ms 采样间隔。第三条使用仓库自带的合成测试数据，输入为 128 个样点、32 道；它不需要外部地震文件。SEG-Y、SU、SEG-2 读取需要安装 `formats` 可选依赖。

```python
"""读取、滤波、保存并检查一份地震数据。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from fxseisfilter import process_file


def main() -> None:
    # 用命令行参数指定输入和输出，使同一脚本可以重复用于不同测线。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="输入地震数据文件")
    parser.add_argument("output", type=Path, help="输出 NPZ 结果包")
    # 矩阵类文件通常没有地震道头，需由使用者给出采样间隔。
    parser.add_argument("--dt", type=float, help="采样间隔，单位秒")
    parser.add_argument("--key", help="NPZ/MAT 文件中的矩阵名")
    parser.add_argument("--transpose", action="store_true",
                        help="输入矩阵方向是 [道, 时间样点] 时转置")
    args = parser.parse_args()

    # 一次调用完成读文件、单道带通基线、F-X 预测滤波和结果保存。
    # 输出写到另一个文件，避免覆盖原始资料；滤波参数在这里集中设置。
    result = process_file(
        args.input,
        args.output,
        dt=args.dt,
        key=args.key,
        transpose=args.transpose,
        fb=5.0,       # F-X 预测频带下限，Hz
        fe=100.0,     # F-X 预测频带上限，Hz
        lw=128,       # 每个局部时间窗的样点数
        lx=32,        # 每个局部空间窗包含的道数
        lop=4,        # 单侧预测阶数
        ac=0.05,      # 复数最小二乘的相对对角加载量
    )

    # 打印处理对象的尺度和保存位置，便于确认采样间隔与输出路径。
    print(f"输入矩阵 [时间样点, 道]：{result['raw'].shape}")
    print(f"采样间隔：{result['dt']} s")
    print(f"结果文件：{result['output_path']}")

    # 再打开刚写出的 NPZ，验证数据确实落盘且各结果数组可正常读取。
    # allow_pickle=False 拒绝从压缩包加载 Python 对象，限制反序列化风险。
    with np.load(result["output_path"], allow_pickle=False) as bundle:
        for name in ("raw", "bandpass", "fx", "removed"):
            array = bundle[name]
            print(f"{name}: shape={array.shape}, finite={np.isfinite(array).all()}")


if __name__ == "__main__":
    # 只有直接运行本文件时才启动流程；作为模块导入时不会自动处理数据。
    main()
```

运行后会得到 `output_fx.npz`，其中 `raw` 是读入矩阵，`bandpass` 是带通基线，`fx` 是 F-X 输出，`removed` 是 `bandpass - fx` 的差值，`dt` 是采样间隔。`removed` 表示算法输出与带通基线的差异，不等同于已知真实噪声。输入文件保持不变。

### Python 方法用法

包从 `fxseisfilter` 顶层公开以下方法和类型：

| 名称 | 用途 | 返回值 |
|---|---|---|
| `read_seismic(path, ...)` | 读取支持的地震文件 | `(data, dt, metadata)` |
| `process(data, dt, ...)` | 处理内存中的 `[时间样点, 道]` 数组 | `(baseline, fx, parameters)` |
| `process_file(input_path, output_path=None, ...)` | 读取、处理并写出 NPZ 结果 | 包含数组、参数和输出路径的字典 |
| `save_result(path, ...)` | 保存已经得到的输入、基线和滤波结果 | 输出文件的 `Path` |
| `FXConfig`、`fx_filter(data, config)` | 直接配置和调用 F-X 算法核心 | `FXConfig` 对象、滤波数组 |
| `metrics(clean, output)` | 用已知干净参考计算质量指标 | 含 SNR、RMSE、相对误差和相关系数的字典 |

#### 读取数据

`read_seismic` 的 `dt` 单位为秒；文件没有采样信息时必须传入。`key` 用于选择 NPZ/MAT 中的二维矩阵，`transpose=True` 用于将 `[道, 时间样点]` 转为标准方向，`skiprows` 用于跳过 CSV/TXT 的表头行。

```python
from fxseisfilter import read_seismic

data, dt, metadata = read_seismic("section.mat", key="seismic", dt=0.002,
                                  transpose=True)
```

#### 处理内存数组

`process` 参数与 `process_file` 中的滤波参数相同。它不写文件，适合把算法接入自己的 Python 流程。返回的 `baseline` 是带通基线；`use_bandpass=False` 时它是输入数据的副本，F-X 滤波仍会执行。

```python
from fxseisfilter import process

baseline, fx_data, parameters = process(
    data, dt=0.002, fb=5, fe=100, lw=128, lx=32, lop=4,
    ac=0.05, use_bandpass=False,
)
```

#### 配置底层算法

需要直接使用实现核心时，创建 `FXConfig` 再传给 `fx_filter`。此接口只执行 F-X 滤波，不自动做带通；带通可单独通过 `process` 配置。`identity=True` 仅用于测试分窗重构，不是正常滤波选项。

```python
from fxseisfilter import FXConfig, fx_filter

config = FXConfig(dt=0.002, FB=5, FE=100, LW=128, LX=32, LOP=4, AC=0.05)
fx_data = fx_filter(data, config)
```

#### 保存与计算指标

`save_result` 适用于已经通过 `process` 得到数组、但需要自行控制输出流程的情况。`metrics` 需要形状相同的干净参考数组和处理结果，只适用于有已知真值的合成或标定数据；实测资料没有真值时不要据此报告 SNR 或误差指标。

```python
from fxseisfilter import metrics, save_result

path = save_result("result.npz", raw=data, bandpass=baseline, fx=fx_data,
                   dt=dt, source_metadata=metadata, parameters=parameters)
scores = metrics(clean_reference, fx_data)
print(path, scores["snr_db"], scores["rmse"])
```

### 输入格式

读取结果统一为有限值的二维 `float64` 数组，排列为 `[时间样点, 道]`。文件包含采样间隔时默认读取文件信息；传入 `dt` 可覆盖文件信息。

| 格式 | 扩展名 | 采样间隔来源 |
|---|---|---|
| CST | `.cst` | 支持的 CST 道头 |
| NumPy 数组 | `.npy` | 需要传入 `dt` |
| NumPy 压缩包 | `.npz` | 优先读取 `dt` 或 `dt_ms`，否则传入 `dt` |
| SEG-Y | `.sgy`、`.segy` | 二进制文件头（需安装 `segyio`） |
| SU | `.su` | 道头（需安装 `obspy`） |
| SEG-2 | `.sg2`、`.seg2` | 道头（需安装 `obspy`） |
| MATLAB | `.mat` | 需要传入 `dt`；多矩阵时用 `key` 选择 |
| 文本矩阵 | `.csv`、`.txt` | 需要传入 `dt`；`skiprows` 可跳过开头的表头行 |

当前 CST 读取器支持工作区兼容的记录布局和 IEEE float32 样点（格式码 8），不代表支持所有厂商的 CST 变体。SEG-2、SEG-Y 也存在厂商扩展，正式处理前应使用目标来源的代表性文件验证。当前不支持不规则长文本记录和 MATLAB v7.3/HDF5 文件。

若数组按 `[道, 时间样点]` 存储，传入 `transpose=True`。`dt` 始终以秒为单位。

### 滤波参数

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `fb`、`fe` | F-X 预测频带上下限，单位 Hz | 5、100 |
| `lw` | 局部时间窗长度，单位样点 | 128 |
| `lx` | 局部空间窗宽度，单位道 | 32 |
| `lop` | 单侧预测阶数 | 4 |
| `ac` | 复数最小二乘求解的相对对角加载量 | 0.05 |
| `overlap` | 时间窗和空间窗重叠率 | 0.5 |
| `bp_low`、`bp_high` | 基线 Butterworth 带通频率，单位 Hz | 5、100 |
| `use_bandpass` | 是否在 F-X 滤波前执行带通基线处理 | `True` |

F-X 频带必须低于奈奎斯特频率（`0.5 / dt`）。有效空间窗的道数必须大于 `2 * lop`。`ac` 是无量纲的相对正则化参数，并非某一商业软件中同名参数的实现。

### 开发与测试

```bash
python -m pip install -e ".[dev,formats]"
pytest
python -m build
```

项目结构：

```text
src/fxseisfilter/   Python 包源码
tests/              API 与算法测试
examples/           可运行示例
```

### 测试数据

仓库提供固定种子生成的 `tests/data/synthetic_clean.csv` 和 `tests/data/synthetic_noisy.csv`。它们用于测试数据读取、滤波、保存/回读和指标计算；生成参数及重新生成方法见 [tests/data/README.md](tests/data/README.md)。这些合成数据不是实测资料，也不能证明对实测资料的处理效果。

### 许可证

本项目采用 MIT 许可证，许可条款见 [LICENSE](LICENSE)。

---

## English

`fx-seismic-filter` is a Python package for F-X domain prediction filtering of seismic sections. It reads seismic data into NumPy arrays, optionally applies a zero-phase bandpass baseline, runs local bidirectional least-squares F-X prediction, and can save inputs, outputs, and parameters in an NPZ bundle.

The distribution name is `fx-seismic-filter`; the Python import name is `fxseisfilter`.

### Installation

Python 3.10 or newer is required. Install the package with SEG-Y, SU, and SEG-2 support:

```bash
python -m pip install "fx-seismic-filter[formats]"
```

The `formats` extra installs the SEG-Y, SU, and SEG-2 readers. The core installation supports CST, NPY, NPZ, MAT, CSV, and TXT. For local development, install from the repository:

```bash
python -m pip install -e ".[dev,formats]"
```

The package has not been published to PyPI yet. Install it from a local source checkout or a built wheel in the meantime.

### Quick Start

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

`process_file` does not modify the input file. If `output_path` is omitted, a sibling file named `<input>_fx_result.npz` is created. The returned dictionary contains `raw`, `bandpass`, `fx`, `removed`, `dt`, `source_metadata`, `parameters`, and `output_path`.

Process an array already in memory:

```python
import numpy as np
from fxseisfilter import process

data = np.load("section.npy")  # shape: [time samples, traces]
baseline, fx_data, parameters = process(data, dt=0.002, lw=128, lx=32)
```

Read data without filtering:

```python
from fxseisfilter import read_seismic

data, dt, metadata = read_seismic("section.sgy")
```

The installed command-line entry point is `fx-seismic-filter`:

```bash
fx-seismic-filter input.sgy output_fx.npz --fb 5 --fe 100 --lw 128 --lx 32 --lop 4 --ac 0.05
```

#### Complete Example

The script below performs a complete read, filter, save, and read-back check. Inline comments explain what each step does, why it is used, and what it produces.

```bash
python examples/complete_workflow.py input.sgy results/output_fx.npz
python examples/complete_workflow.py input.csv results/output_fx.npz --dt 0.002
python examples/complete_workflow.py tests/data/synthetic_noisy.csv results/test_output.npz --dt 0.002
```

The first command reads the sample interval from the SEG-Y header. The second supplies `--dt 0.002` because a CSV matrix has no seismic trace header. The third command uses the repository's 128-sample by 32-trace synthetic fixture and does not need an external seismic file. SEG-Y, SU, and SEG-2 reading requires the `formats` extra.

```python
"""Read, filter, save, and inspect one seismic data file."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from fxseisfilter import process_file

def main() -> None:
    # Command-line paths let this same script process different surveys.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input seismic file")
    parser.add_argument("output", type=Path, help="output NPZ result bundle")
    # Matrix files usually lack trace headers, so their sample interval is explicit.
    parser.add_argument("--dt", type=float, help="sampling interval in seconds")
    parser.add_argument("--key", help="matrix name in NPZ/MAT files")
    parser.add_argument("--transpose", action="store_true",
                        help="transpose input stored as [trace, time sample]")
    args = parser.parse_args()

    # One call reads the file, applies the bandpass baseline and F-X filter,
    # then saves a separate output file so the original data is preserved.
    result = process_file(
        args.input, args.output, dt=args.dt, key=args.key,
        transpose=args.transpose,
        fb=5.0,       # lower F-X prediction frequency in Hz
        fe=100.0,     # upper F-X prediction frequency in Hz
        lw=128,       # samples in each local time window
        lx=32,        # traces in each local space window
        lop=4,        # prediction order on each side
        ac=0.05,      # relative diagonal loading for complex least squares
    )

    # Report dimensions, sampling scale, and destination to confirm the run.
    print(f"Input shape [time, trace]: {result['raw'].shape}")
    print(f"Sampling interval: {result['dt']} s")
    print(f"Result file: {result['output_path']}")

    # Reopen the NPZ to verify that the saved arrays are present and readable.
    # Disabling pickle prevents loading Python objects from the archive.
    with np.load(result["output_path"], allow_pickle=False) as bundle:
        for name in ("raw", "bandpass", "fx", "removed"):
            array = bundle[name]
            print(f"{name}: shape={array.shape}, finite={np.isfinite(array).all()}")

if __name__ == "__main__":
    # Run the workflow only when this file is executed as a script.
    main()
```

The output `output_fx.npz` contains `raw` (input), `bandpass` (baseline), `fx` (filtered output), `removed` (`bandpass - fx`), and `dt`. `removed` is the difference between the algorithm output and the bandpass baseline; it is not guaranteed to equal the true noise. The input file remains unchanged.

### Python API Methods

The package exports these functions and types from `fxseisfilter`:

| Name | Purpose | Returns |
|---|---|---|
| `read_seismic(path, ...)` | Read a supported seismic file | `(data, dt, metadata)` |
| `process(data, dt, ...)` | Process an in-memory `[time sample, trace]` array | `(baseline, fx, parameters)` |
| `process_file(input_path, output_path=None, ...)` | Read, process, and save an NPZ bundle | Dictionary with arrays, parameters, and output path |
| `save_result(path, ...)` | Save input, baseline, and filtered arrays already in memory | Output `Path` |
| `FXConfig`, `fx_filter(data, config)` | Configure and call the low-level F-X algorithm | Config object and filtered array |
| `metrics(clean, output)` | Calculate metrics against a known clean reference | Dictionary with SNR, RMSE, relative error, and correlation |

#### Read Data

`dt` is measured in seconds and is required when a file has no sample-interval metadata. Use `key` to select a matrix in an NPZ/MAT file, `transpose=True` for `[trace, time sample]` input, and `skiprows` to skip CSV/TXT header rows.

```python
from fxseisfilter import read_seismic

data, dt, metadata = read_seismic("section.mat", key="seismic", dt=0.002,
                                  transpose=True)
```

#### Process an In-Memory Array

`process` accepts the same filter parameters as `process_file` but does not write a file. Use it to integrate filtering into another Python workflow. `baseline` is the bandpass baseline; with `use_bandpass=False`, it is a copy of the input while F-X filtering still runs.

```python
from fxseisfilter import process

baseline, fx_data, parameters = process(
    data, dt=0.002, fb=5, fe=100, lw=128, lx=32, lop=4,
    ac=0.05, use_bandpass=False,
)
```

#### Configure the Low-Level Algorithm

Create an `FXConfig` and pass it to `fx_filter` to call the algorithm directly. This interface applies F-X filtering only; it does not apply a bandpass. `identity=True` is intended only for testing window reconstruction, not for normal filtering.

```python
from fxseisfilter import FXConfig, fx_filter

config = FXConfig(dt=0.002, FB=5, FE=100, LW=128, LX=32, LOP=4, AC=0.05)
fx_data = fx_filter(data, config)
```

#### Save Results and Calculate Metrics

Use `save_result` when arrays have already been produced by `process` and the caller needs to control saving. `metrics` requires a clean reference array with the same shape as the output. Use it only for synthetic or calibrated data with known truth; do not report SNR or error metrics for field data without a true reference.

```python
from fxseisfilter import metrics, save_result

path = save_result("result.npz", raw=data, bandpass=baseline, fx=fx_data,
                   dt=dt, source_metadata=metadata, parameters=parameters)
scores = metrics(clean_reference, fx_data)
print(path, scores["snr_db"], scores["rmse"])
```

### Input Formats

All inputs are returned as finite, real, two-dimensional `float64` arrays in `[time sample, trace]` order. File metadata supplies the sample interval when available; passing `dt` overrides it.

| Format | Extensions | Sample interval |
|---|---|---|
| CST | `.cst` | Read from supported CST trace headers |
| NumPy array | `.npy` | Pass `dt` |
| NumPy archive | `.npz` | Read `dt` or `dt_ms` when present; otherwise pass `dt` |
| SEG-Y | `.sgy`, `.segy` | Binary header (`segyio` required) |
| SU | `.su` | Trace headers (`obspy` required) |
| SEG-2 | `.sg2`, `.seg2` | Trace headers (`obspy` required) |
| MATLAB | `.mat` | Pass `dt`; use `key` to select a matrix when needed |
| Text matrix | `.csv`, `.txt` | Pass `dt`; `skiprows` skips leading header rows |

The current CST reader supports the workspace-compatible record layout with IEEE float32 samples (format code 8); it does not claim support for every vendor-specific CST variant. SEG-2 and SEG-Y vendor extensions also vary, so validate representative files from the intended source before production use. Ragged text records and MATLAB v7.3/HDF5 files are not supported.

For arrays stored as `[trace, time sample]`, pass `transpose=True`. The `dt` argument is always expressed in seconds.

### Filter Parameters

| Parameter | Meaning | Default |
|---|---|---:|
| `fb`, `fe` | F-X prediction frequency limits in Hz | 5, 100 |
| `lw` | Local time-window length in samples | 128 |
| `lx` | Local spatial-window width in traces | 32 |
| `lop` | Prediction order on each side | 4 |
| `ac` | Relative diagonal loading for the complex least-squares solve | 0.05 |
| `overlap` | Time and space window overlap ratio | 0.5 |
| `bp_low`, `bp_high` | Baseline Butterworth bandpass limits in Hz | 5, 100 |
| `use_bandpass` | Apply the bandpass baseline before F-X filtering | `True` |

The F-X frequency band must stay below the Nyquist frequency (`0.5 / dt`). The effective spatial window must contain more than `2 * lop` traces. `ac` is a dimensionless relative regularization parameter; it is not an implementation of a vendor parameter with the same name.

### Development and Tests

```bash
python -m pip install -e ".[dev,formats]"
pytest
python -m build
```

Project layout:

```text
src/fxseisfilter/   Python package source
tests/              API and algorithm tests
examples/           runnable examples
```

### Test Data

The repository includes fixed-seed fixtures at `tests/data/synthetic_clean.csv` and `tests/data/synthetic_noisy.csv`. They exercise file reading, filtering, saving/reloading, and metric calculation. Generation details and the regeneration command are in [tests/data/README.md](tests/data/README.md). These synthetic fixtures are not field data and do not establish field performance.

### License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full terms.
