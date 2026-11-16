"""独立的局部双向 F-X 预测滤波实现。

所有二维数组均采用 ``[时间样点, 道号]`` 排列。程序先在时间方向做实数
FFT，再对指定频带内的复数频谱进行空间预测，最后用重叠加权窗合成回时域。

AC 是无量纲相对对角加载量，其定义不等同于商业 FXNAT 的同名参数。
每个局部窗的 [FB, FE] 频带以外复谱保持输入值，不施加自动增益控制。
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfiltfilt

@dataclass(frozen=True)
class FXConfig:
    """F-X 滤波参数；数值与论文实验记录保持一致。"""
    dt: float = .002  # 时间采样间隔，单位秒
    FB: float = 5.    # 预测频带下限，单位 Hz
    FE: float = 100.  # 预测频带上限，不能超过奈奎斯特频率
    LW: int = 128    # 每个时间窗的样点数
    LX: int = 32     # 每个空间窗的道数
    LOP: int = 4     # 预测阶数：每侧使用的邻道数量
    AC: float = .05  # 相对对角加载量，越大通常越抑制回归系数
    overlap: float = .5  # 时间和空间窗口的重叠比例

def starts(n, w, overlap=.5):
    """返回覆盖长度 ``n`` 的滑动窗起点，并强制补上末端窗口。"""
    w = min(n, w)
    s = list(range(0, n-w+1, max(1, round(w*(1-overlap)))))
    if s[-1] != n-w: s.append(n-w)
    return s

def weight(n):
    # 合成窗的端点保持为正，避免重叠加权时出现零覆盖；它不参与分析窗加权。
    return np.maximum(np.sin(np.pi*(np.arange(n)+.5)/n)**2, 1.e-5)

def one_direction(z, p, ac):
    """按频率批量求解复数岭回归，预测空间位置 ``p`` 之后的道。

    ``z`` 的形状为 ``[频率, 道]``。每个频率独立估计 p 个复数预测系数，
    共轭转置用于构造复数正规方程；AC 是相对于平均对角能量的无量纲加载量。
    """
    nf, nx = z.shape
    # 每一行由最近到最远排列 p 个前向输入样本。
    X = np.stack([z[:, p-k:nx-k] for k in range(1,p+1)], axis=-1)
    y = z[:, p:]
    # 复数回归必须使用共轭转置，普通转置会破坏相位信息。
    H = X.conj().swapaxes(-1,-2)
    G = H @ X
    scale = np.maximum(np.trace(G,axis1=-2,axis2=-1).real/p, 1.e-30)
    # 用每个频率的平均对角能量标定 AC，使参数不依赖绝对振幅单位。
    reg = (ac+1.e-10)*scale
    G = G + reg[:,None,None]*np.eye(p)[None,:,:]
    b = H @ y[...,None]
    coef = np.linalg.solve(G,b)[...,0]
    pred = np.einsum('fnp,fp->fn',X,coef)
    return pred

def predict_spectrum(z, p, ac):
    """同时执行正向和反向预测，并平均两者在边界处的可用结果。"""
    if z.shape[1] <= 2*p:
        raise ValueError('Each spatial window must have more than 2*LOP traces')
    out = np.zeros_like(z)
    count = np.zeros(z.shape[1])
    # 正向预测使用左侧 p 道；反向预测等价于翻转道序后再翻回。
    out[:,p:] += one_direction(z,p,ac); count[p:] += 1
    out[:,:-p] += one_direction(z[:,::-1],p,ac)[:,::-1]; count[:-p] += 1
    return out/count[None,:]

def fx_filter(data, cfg=FXConfig(), identity=False):
    """执行分窗、频带预测、逆变换和重叠加权重构。

    ``identity=True`` 只走分窗与重构流程，供边界和覆盖测试验证；正常实验
    设为 False。频带外频谱在每个局部窗内原样保留，输入数据不做 AGC。
    """
    d = np.asarray(data,dtype=np.float64)
    if d.ndim != 2 or not np.isfinite(d).all(): raise ValueError('finite 2-D data required')
    if not np.isfinite(cfg.dt) or cfg.dt <= 0: raise ValueError('dt must be finite and positive')
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v <= 0
           for v in (cfg.LW, cfg.LX, cfg.LOP)):
        raise ValueError('LW, LX and LOP must be positive integers')
    if not 0 <= cfg.FB <= cfg.FE <= .5/cfg.dt: raise ValueError('invalid frequency interval')
    if not np.isfinite(cfg.AC) or cfg.AC < 0 or not 0 <= cfg.overlap < 1: raise ValueError('invalid AC or overlap')
    nt,nx=d.shape; tw=min(cfg.LW,nt); xw=min(cfg.LX,nx)
    if tw < 8 or cfg.LOP < 1 or xw <= 2*cfg.LOP: raise ValueError('invalid window/order')
    # 二维合成窗分别由时间和空间窗相乘得到。
    wt=weight(tw); wx=weight(xw); w=wt[:,None]*wx[None,:]
    f=np.fft.rfftfreq(tw,cfg.dt); band=(f>=cfg.FB)&(f<=cfg.FE)
    out=np.zeros_like(d); norm=np.zeros_like(d)
    for ti in starts(nt,tw,cfg.overlap):
        for xi in starts(nx,xw,cfg.overlap):
            sl=np.s_[ti:ti+tw,xi:xi+xw]
            # 每个局部时间窗独立变换，保留复数空间相位。
            z=np.fft.rfft(d[sl],axis=0)
            if not identity and band.any():
                z[band]=predict_spectrum(z[band],cfg.LOP,cfg.AC)
            q=np.fft.irfft(z,n=tw,axis=0)
            out[sl]+=q*w; norm[sl]+=w
    if np.any(norm<=0): raise RuntimeError('uncovered samples')
    return out/norm

def bandpass(data, dt, lo=5., hi=100.):
    """对时间轴施加零相位 Butterworth 带通，作为单道基线。"""
    return sosfiltfilt(butter(4,[lo,hi],fs=1/dt,btype='bandpass',output='sos'),data,axis=0)

def metrics(clean, out):
    """计算有干净参考时可用的 SNR、RMSE、相对误差和相关系数。"""
    e=np.asarray(out)-clean
    return dict(snr_db=float(10*np.log10(np.sum(clean**2)/max(np.sum(e**2),1.e-30))),
                rmse=float(np.sqrt(np.mean(e**2))),
                relative_error=float(np.linalg.norm(e)/np.linalg.norm(clean)),
                correlation=float(np.corrcoef(np.ravel(clean),np.ravel(out))[0,1]))
