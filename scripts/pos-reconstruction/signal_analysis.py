"""
信号幅度与定时分析
====================
AC-LGAD 波形特征提取：基线、幅度、上升时间、CFD 定时、抖动估计。

用法:
  from scripts.pos_reconstruction.signal_analysis import analyze_waveform, plot_analysis

  result = analyze_waveform(y, t_ns, polarity='positive')
  plot_analysis(result, out_path='analysis.png')

分析流程:
  1. 前基线: 取波形前 20%，计算 mean 和 sigma
  2. 后基线: 取波形后 20%，计算 mean 和 sigma
  3. 基线扣除: 整个波形减去前基线 mean
  4. 幅度: 查找最大值 A
  5. 上升时间: 10%→90% A 的时间区间，计算斜率
  6. CFD 定时: 40%/50%/60% 恒定比例定时
  7. 抖动: σ_noise / (0.8·A / t_rise)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


# ======================================================================
# 数据结构
# ======================================================================

@dataclass
class SignalResult:
    """单个波形信号分析结果。"""

    # ── 输入参数 ──
    polarity: str = "positive"   # "positive" 或 "negative"
    n_samples: int = 0

    # ── 基线 ──
    pre_baseline_mean: float = 0.0
    pre_baseline_sigma: float = 0.0
    pre_baseline_start: int = 0
    pre_baseline_end: int = 0

    post_baseline_mean: float = 0.0
    post_baseline_sigma: float = 0.0
    post_baseline_start: int = 0
    post_baseline_end: int = 0

    # ── 幅度 ──
    amplitude: float = 0.0           # 基线扣除后的最大幅度
    peak_index: int = 0              # 峰值位置 (样本索引)
    peak_time_ns: float = 0.0        # 峰值时间 (ns)

    # ── 上升时间 ──
    t10_idx: int = 0                 # 10% 幅度位置
    t10_ns: float = 0.0
    t90_idx: int = 0                 # 90% 幅度位置
    t90_ns: float = 0.0
    rise_time_ns: float = 0.0        # t90 - t10
    rise_slope: float = 0.0          # 0.8·A / t_rise (ADC/ns)

    # ── CFD 定时 ──
    cfd_40_ns: float = 0.0           # 40% 恒定比例定时
    cfd_50_ns: float = 0.0           # 50%
    cfd_60_ns: float = 0.0           # 60%

    # ── 抖动 ──
    jitter_ns: float = 0.0           # σ_noise / (0.8·A / t_rise)

    # ── 辅助 ──
    waveform_baseline_subtracted: Optional[np.ndarray] = None

    def __repr__(self):
        return (f"SignalResult(A={self.amplitude:.1f}, "
                f"t_rise={self.rise_time_ns:.3f}ns, "
                f"CFD50={self.cfd_50_ns:.3f}ns, "
                f"jitter={self.jitter_ns*1e3:.2f}ps, "
                f"σ_pre={self.pre_baseline_sigma:.2f})")


# ======================================================================
# 核心分析函数
# ======================================================================

def analyze_waveform(
    y: np.ndarray,
    t_ns: np.ndarray,
    polarity: str = "auto",
    baseline_fraction: float = 0.20,
    pre_baseline_fraction: Optional[float] = None,
    post_baseline_fraction: Optional[float] = None,
    trim_tail: int = 30,
) -> SignalResult:
    """
    Full signal analysis on a single waveform.

    Parameters
    ----------
    y, t_ns : ndarray
        Waveform and time axis (ns), same length.
    polarity : str
        "auto", "positive", or "negative".
    baseline_fraction : float
        Symmetric pre/post baseline fraction, default 0.20.
    pre_baseline_fraction : float, optional
        Pre-baseline fraction (overrides baseline_fraction).
    post_baseline_fraction : float, optional
        Post-baseline fraction (overrides baseline_fraction).
    trim_tail : int
        Samples to trim from tail (default 30 for V1742 DRS4 artifact).
    """
    y = np.asarray(y, dtype=np.float64)
    t_ns = np.asarray(t_ns, dtype=np.float64)

    if trim_tail > 0:
        y = y[:-trim_tail]
        t_ns = t_ns[:-trim_tail]

    n = len(y)
    result = SignalResult(polarity=polarity, n_samples=n)

    if n < 10:
        return result

    pre_frac = pre_baseline_fraction if pre_baseline_fraction is not None else baseline_fraction
    post_frac = post_baseline_fraction if post_baseline_fraction is not None else baseline_fraction

    # ── Step 1: 前基线 ──
    pre_end = max(int(n * pre_frac), 5)
    result.pre_baseline_start = 0
    result.pre_baseline_end = pre_end
    pre_region = y[:pre_end]
    result.pre_baseline_mean = float(np.mean(pre_region))
    result.pre_baseline_sigma = float(np.std(pre_region))

    # ── Step 2: 后基线 ──
    post_start = min(int(n * (1 - post_frac)), n - 5)
    result.post_baseline_start = post_start
    result.post_baseline_end = n
    post_region = y[post_start:]
    result.post_baseline_mean = float(np.mean(post_region))
    result.post_baseline_sigma = float(np.std(post_region))

    # ── Step 3: 基线扣除 ──
    y_sub = y - result.pre_baseline_mean
    result.waveform_baseline_subtracted = y_sub

    # ── Step 4: 幅度 & 极性检测 ──
    # 同时检查正/负最大偏离
    pos_max = float(np.max(y_sub))
    neg_max = float(np.abs(np.min(y_sub)))
    pos_idx = int(np.argmax(y_sub))
    neg_idx = int(np.argmin(y_sub))

    if polarity == "auto":
        # 自动判断: 负幅值 > 正幅值 → 负信号
        is_negative = (neg_max > pos_max)
    elif polarity == "negative":
        is_negative = True
    else:
        is_negative = False

    if is_negative:
        result.polarity = "negative"
        result.amplitude = neg_max
        result.peak_index = neg_idx
        y_sub = -y_sub   # 内部转为正脉冲处理
        result.waveform_baseline_subtracted = -result.waveform_baseline_subtracted
    else:
        result.polarity = "positive"
        result.amplitude = pos_max
        result.peak_index = pos_idx

    result.peak_time_ns = float(t_ns[result.peak_index])

    if result.amplitude <= 0:
        return result

    A = result.amplitude
    pk = result.peak_index

    # ── Step 5: 上升时间 (10%→90%) ──
    th10 = 0.10 * A
    th90 = 0.90 * A

    # 从峰值向前搜索 10% 和 90% 穿越点
    t10_idx = _find_crossing(y_sub, th10, pk, direction="backward")
    t90_idx = _find_crossing(y_sub, th90, pk, direction="backward")

    if t10_idx is not None and t90_idx is not None and t10_idx < t90_idx:
        result.t10_idx = t10_idx
        result.t90_idx = t90_idx
        result.t10_ns = float(t_ns[t10_idx])
        result.t90_ns = float(t_ns[t90_idx])
        result.rise_time_ns = result.t90_ns - result.t10_ns
        if result.rise_time_ns > 0:
            result.rise_slope = 0.8 * A / result.rise_time_ns
    else:
        # 回退: 使用峰值附近的粗略估计
        # 从峰值往回找达到稳定基线的区域
        search_start = max(0, pk - n // 2)
        rising_region = y_sub[search_start:pk + 1]
        if len(rising_region) > 3:
            idx10_rel = int(np.argmin(np.abs(rising_region - th10)))
            idx90_rel = int(np.argmin(np.abs(rising_region - th90)))
            # 确保 10% 在 90% 之前
            t10_temp = search_start + min(idx10_rel, idx90_rel)
            t90_temp = search_start + max(idx10_rel, idx90_rel)
            result.t10_idx = min(t10_temp, t90_temp)
            result.t90_idx = max(t10_temp, t90_temp)
            result.t10_ns = float(t_ns[result.t10_idx])
            result.t90_ns = float(t_ns[result.t90_idx])
            result.rise_time_ns = result.t90_ns - result.t10_ns
            if result.rise_time_ns > 0:
                result.rise_slope = 0.8 * A / result.rise_time_ns

    # ── Step 6: CFD 定时 (40%/50%/60%) ──
    for frac, attr in [(0.40, 'cfd_40_ns'), (0.50, 'cfd_50_ns'), (0.60, 'cfd_60_ns')]:
        threshold = frac * A
        cross_idx = _find_crossing(y_sub, threshold, pk, direction="backward")
        if cross_idx is not None:
            # 线性插值以提高精度
            t_cfd = _interpolate_crossing(t_ns, y_sub, cross_idx, threshold)
            setattr(result, attr, t_cfd)

    # ── Step 7: 抖动 ──
    if result.rise_slope > 0:
        dV_dt = result.rise_slope  # ADC/ns
        result.jitter_ns = result.pre_baseline_sigma / dV_dt
    else:
        result.jitter_ns = 0.0

    return result

# ======================================================================
# 信号有效性判断
# ======================================================================

def has_signal(result: SignalResult, n_sigma: float = 5.0) -> bool:
    """Return True if amplitude >= n_sigma * pre_baseline_sigma."""
    if result.pre_baseline_sigma <= 0:
        return result.amplitude > 0
    return result.amplitude >= n_sigma * result.pre_baseline_sigma

# ======================================================================
# 辅助函数
# ======================================================================

def _find_crossing(
    y: np.ndarray,
    threshold: float,
    start_idx: int,
    direction: str = "backward",
) -> Optional[int]:
    """
    从 start_idx 向前或向后搜索 threshold 穿越点。

    Returns
    -------
    int or None
        穿越前的最后一个样本索引 (低于 threshold 的那一侧)。
    """
    n = len(y)
    if direction == "backward":
        indices = range(start_idx, -1, -1)
    else:
        indices = range(start_idx, n)

    prev = y[start_idx]
    for i in indices:
        cur = y[i]
        # 检测穿越: 前一个在阈值上方，当前在下方 (或反之)
        if (prev >= threshold and cur < threshold) or (prev <= threshold and cur > threshold):
            # 返回阈值下方的那一侧 (上升沿时返回 i)
            if direction == "backward":
                return i if cur < prev else i + 1
            else:
                return i - 1 if cur < prev else i
        prev = cur

    return None


def _interpolate_crossing(
    t_ns: np.ndarray,
    y: np.ndarray,
    idx: int,
    threshold: float,
) -> float:
    """线性插值计算精确穿越时间。"""
    if idx < 0 or idx >= len(y) - 1:
        return float(t_ns[max(0, min(idx, len(t_ns) - 1))])

    y0, y1 = y[idx], y[idx + 1]
    t0, t1 = t_ns[idx], t_ns[idx + 1]

    if abs(y1 - y0) < 1e-12:
        return float(t0)

    frac = (threshold - y0) / (y1 - y0)
    return float(t0 + frac * (t1 - t0))


# ======================================================================
# 绘图
# ======================================================================

def plot_analysis(
    result: SignalResult,
    y_raw: np.ndarray,
    t_ns: np.ndarray,
    out_path: Optional[str] = None,
    title: str = "Signal Analysis",
    show: bool = False,
    trim_tail: int = 30,
    ax: Optional[object] = None,
    compact: bool = False,
    y_label: str = "ADC",
):
    """
    绘制信号分析结果。

    标注内容:
      - 前/后基线区间 (绿色/橙色半透明)
      - 10%-90% 上升时间区间 (红色半透明)
      - 峰值幅度 (红线)
      - 50% CFD 定时点 (蓝点)

    Parameters
    ----------
    result : SignalResult
        analyze_waveform 返回的结果。
    y_raw : np.ndarray
        原始波形 (未减基线)。
    t_ns : np.ndarray
        时间轴 (原始完整长度)。
    out_path : str, optional
        保存路径。
    title : str
        图标题。
    show : bool
        是否显示。
    trim_tail : int
        与 analyze_waveform 相同的尾部舍弃点数。
    ax : matplotlib.axes.Axes, optional
        If provided, draw on this axes (subplot mode); otherwise create new figure.
    compact : bool
        Compact mode: smaller fonts, no legend box, for multi-subplot layouts.
    """
    import matplotlib
    if not show and ax is None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    own_fig = (ax is None)
    if own_fig:
        fig, ax = plt.subplots(figsize=(14, 5))

    y = np.asarray(y_raw, dtype=np.float64)
    t_ns = np.asarray(t_ns, dtype=np.float64)

    # 尾部裁剪以匹配分析结果
    if trim_tail > 0:
        y = y[:-trim_tail]
        t_ns = t_ns[:-trim_tail]

    n = len(y)
    A = result.amplitude
    is_signal = has_signal(result)

    # ── 基线扣除后波形 ──
    lw_sub = 1.2 if compact else 1.0
    if result.waveform_baseline_subtracted is not None:
        ax.plot(t_ns, result.waveform_baseline_subtracted, 'steelblue', lw=lw_sub,
                label='Ped-subtracted')

    # ── 前基线区间 (绿色) ──
    ax.axvspan(t_ns[result.pre_baseline_start], t_ns[result.pre_baseline_end - 1],
               alpha=0.10, color='green', label=f'Pre-ped  μ={result.pre_baseline_mean:.1f} σ={result.pre_baseline_sigma:.2f}')

    # ── 后基线区间 (橙色) ──
    ax.axvspan(t_ns[result.post_baseline_start], t_ns[result.post_baseline_end - 1],
               alpha=0.10, color='orange', label=f'Post-ped μ={result.post_baseline_mean:.1f} σ={result.post_baseline_sigma:.2f}')

    if not is_signal:
        # ── 噪声: 仅显示基线信息 ──
        fs = 6.5 if compact else 8
        info_lines = [
            f"NOISE (A={A:.1f} < 5σ={5*result.pre_baseline_sigma:.1f})",
            f"σ_pre = {result.pre_baseline_sigma:.2f}",
            f"σ_post = {result.post_baseline_sigma:.2f}",
        ]
    else:
        # ── 有信号: 显示完整分析 ──
        # ── 上升时间区间 (红色) ──
        if result.t10_idx > 0 and result.t90_idx > 0:
            ax.axvspan(t_ns[result.t10_idx], t_ns[result.t90_idx],
                       alpha=0.15, color='red', label=f'Rise 10%→90% ({result.rise_time_ns:.3f} ns)')

        # ── 峰值幅度 ──
        lw_amp = 1.2 if compact else 0.8
        if A > 0:
            ax.axhline(A, color='red', ls='--', lw=lw_amp, alpha=0.6)
            ax.annotate(f'A={A:.1f}', xy=(t_ns[result.peak_index], A),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='red')

        # ── CFD 50% 定时点 ──
        lw_cfd = 1.2 if compact else 0.8
        if result.cfd_50_ns > 0:
            ax.axvline(result.cfd_50_ns, color='blue', ls=':', lw=lw_cfd, alpha=0.7)
            ax.plot(result.cfd_50_ns, 0.5 * A, 'bo', ms=7, alpha=0.9,
                    label=f'CFD 50% = {result.cfd_50_ns:.3f} ns')

        # ── 汇总信息 ──
        fs = 6.5 if compact else 8
        info_lines = [
            f"A = {A:.1f} ADC",
            f"t_rise = {result.rise_time_ns:.3f} ns",
            f"slope = {result.rise_slope:.1f} ADC/ns",
            f"CFD40 = {result.cfd_40_ns:.3f} ns",
            f"CFD50 = {result.cfd_50_ns:.3f} ns",
            f"CFD60 = {result.cfd_60_ns:.3f} ns",
            f"σ_pre = {result.pre_baseline_sigma:.2f} ADC",
            f"jitter = {result.jitter_ns*1e3:.2f} ps",
        ]

    ax.text(0.02, 0.98, '\n'.join(info_lines),
            transform=ax.transAxes, va='top', fontsize=fs, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=9 if compact else 11)
    if not compact:
        ax.legend(fontsize=7, loc='upper right', ncol=1, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    if own_fig:
        plt.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {out_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)


# ======================================================================
# 批量分析
# ======================================================================

def analyze_channel(
    waveforms: Dict[int, np.ndarray],
    t_ns: np.ndarray,
    channels: Optional[list] = None,
    polarity: str = "auto",
    trim_tail: int = 30,
) -> Dict[int, SignalResult]:
    """
    Batch analyze waveforms for multiple Digitizer channels in one event.

    Parameters
    ----------
    waveforms : dict
        {channel_key: waveform_array}.
    t_ns : np.ndarray
        Time axis.
    channels : list, optional
        Channels to analyze, default all.
    polarity : str
        Signal polarity.
    trim_tail : int
        Samples to trim from tail.

    Returns
    -------
    dict: {channel_key: SignalResult}
    """
    results = {}
    keys = channels if channels else sorted(waveforms.keys())
    for ch in keys:
        if ch in waveforms:
            results[ch] = analyze_waveform(waveforms[ch], t_ns, polarity=polarity, trim_tail=trim_tail)
    return results


# ======================================================================
# CLI 测试 — 查找负信号通道
# ======================================================================

if __name__ == "__main__":
    import sys, os, gc
    from collections import defaultdict
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from src.data.raw_data_recorder import load_binary_events

    DIGI_BIN = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "BT_20260610_094205", "run_0018", "digitizer", "V1742_events.bin")
    PLOTS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "plots", "signal_analysis")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    NS_PER_SAMPLE = 0.2
    N_SCAN = 1000
    N_SIGMA = 5.0

    print(f"Loading digitizer data ...")
    digi = load_binary_events(DIGI_BIN)
    print(f"  Total events: {len(digi)}")

    # 所有存在的通道
    all_channels = sorted(digi[0]['waveforms'].keys())
    print(f"  Channels: {all_channels}")

    # ── 逐通道扫描，统计极性 ──
    # neg_channels[ch] = count of negative-signal events (A ≥ 5σ)
    neg_counts: defaultdict = defaultdict(int)
    pos_counts: defaultdict = defaultdict(int)
    noise_counts: defaultdict = defaultdict(int)

    # 找每个通道第一个负信号事件 (用于绘图)
    first_neg_event: dict = {}

    print(f"\nScanning first {N_SCAN} events (polarity=auto, A≥{N_SIGMA}σ)...")

    for evt_i in range(min(N_SCAN, len(digi))):
        evt = digi[evt_i]
        for ch in all_channels:
            if ch not in evt['waveforms']:
                continue
            y = evt['waveforms'][ch]
            t = np.arange(len(y)) * NS_PER_SAMPLE
            r = analyze_waveform(y, t, polarity='auto')

            if has_signal(r, n_sigma=N_SIGMA):
                if r.polarity == 'negative':
                    neg_counts[ch] += 1
                    if ch not in first_neg_event:
                        first_neg_event[ch] = (evt_i, evt['event_number'], y, t, r)
                else:
                    pos_counts[ch] += 1
            else:
                noise_counts[ch] += 1

    # ── 报告 ──
    print(f"\n{'='*70}")
    print(f"  Digitizer 信号极性扫描结果 (前 {N_SCAN} events, A ≥ {N_SIGMA}σ)")
    print(f"{'='*70}")
    print(f"  {'Ch':<5} {'Negative':>8} {'Positive':>8} {'Noise':>8} {'Polarity':>10}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    neg_channels = []
    for ch in all_channels:
        neg_n = neg_counts[ch]
        pos_n = pos_counts[ch]
        noi_n = noise_counts[ch]
        total_sig = neg_n + pos_n
        if total_sig > 0:
            dominant = "NEGATIVE ⬇" if neg_n > pos_n else "positive ⬆"
        else:
            dominant = "no signal"
        print(f"  ch{ch:<3} {neg_n:>8} {pos_n:>8} {noi_n:>8} {dominant:>10}")
        if neg_n > 0 and (neg_n > pos_n or (neg_n > 0 and pos_n == 0)):
            neg_channels.append(ch)

    print(f"\n  负信号为主的通道: {neg_channels if neg_channels else '无'}")

    # ── 绘图: 负信号通道的首个负信号 ──
    if neg_channels:
        print(f"\n  绘制各负信号通道的首个负信号波形...")
        for ch in neg_channels:
            evt_i, evt_num, y, t, r = first_neg_event[ch]
            print(f"    ch{ch}: event #{evt_num} (idx={evt_i}), A={r.amplitude:.1f}")
            plot_analysis(r, y, t, trim_tail=30,
                          out_path=os.path.join(PLOTS_DIR, f"negative_ch{ch}_evt{evt_i}.png"),
                          title=f"Digitizer ch{ch} (NEG) — Event #{evt_num} (idx={evt_i}, run_0018)")

    print(f"\n✅ 完成，图保存到 {PLOTS_DIR}")
