"""
批量提取信号特征 → 二进制文件
==============================
遍历匹配事件，对每个通道执行信号分析，将所有结果保存为 .npz，
供位置重建直接加载。

输出文件结构:
  meta:         run_name, n_events, channels_digi, channels_osc, ...
  digi/{field}: (n_events, 8) float32 array, NaN=no data
  osc/{field}:  (n_events, 6) float32 array, NaN=no data
  fpga_id:      (n_events,) int32
  fpga_tick:    (n_events,) int64

用法:
  python scripts/pos-reconstruction/extract_signals.py --src <data_dir> [--n 100] [--out signals.npz]
"""

import sys, os, argparse
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
_POS_RECO = str(REPO / 'scripts' / 'pos-reconstruction')
if _POS_RECO not in sys.path:
    sys.path.insert(0, _POS_RECO)

import importlib.util

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_le = _load_module("load_events", os.path.join(_POS_RECO, "load_events.py"))
EventLoader = _le.EventLoader

_sa = _load_module("signal_analysis", os.path.join(_POS_RECO, "signal_analysis.py"))
analyze_waveform = _sa.analyze_waveform
has_signal = _sa.has_signal

# ======================================================================
# 配置
# ======================================================================
DIGI_CHS = [1, 2, 3, 4, 5, 6, 7, 15]   # 8 channels
OSC_CHS  = ['C3', 'C4', 'C5', 'C6', 'C7', 'C8']  # 6 channels
N_DIGI = len(DIGI_CHS)
N_OSC  = len(OSC_CHS)
NS_PER_SAMPLE = 0.2
TRIM_TAIL = 30
OSC_PRE_FRAC = 0.40
OSC_POST_FRAC = 0.30

# 要提取的字段 (来自 SignalResult)
FIELDS = [
    'amplitude', 'polarity_code',   # 0=positive, 1=negative
    'peak_time_ns', 'peak_index',
    't10_ns', 't90_ns', 'rise_time_ns', 'rise_slope',
    'cfd_40_ns', 'cfd_50_ns', 'cfd_60_ns',
    'pre_baseline_mean', 'pre_baseline_sigma',
    'post_baseline_mean', 'post_baseline_sigma',
    'jitter_ns', 'has_signal_int',
]


def result_to_dict(r) -> dict:
    """将 SignalResult 转为可序列化的 dict。"""
    return {
        'amplitude': r.amplitude,
        'polarity_code': 1 if r.polarity == 'negative' else 0,
        'peak_time_ns': r.peak_time_ns,
        'peak_index': r.peak_index,
        't10_ns': r.t10_ns,
        't90_ns': r.t90_ns,
        'rise_time_ns': r.rise_time_ns,
        'rise_slope': r.rise_slope,
        'cfd_40_ns': r.cfd_40_ns,
        'cfd_50_ns': r.cfd_50_ns,
        'cfd_60_ns': r.cfd_60_ns,
        'pre_baseline_mean': r.pre_baseline_mean,
        'pre_baseline_sigma': r.pre_baseline_sigma,
        'post_baseline_mean': r.post_baseline_mean,
        'post_baseline_sigma': r.post_baseline_sigma,
        'jitter_ns': r.jitter_ns,
        'has_signal_int': 1 if has_signal(r) else 0,
    }


# ======================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--n', type=int, default=None, help='max events (default: all)')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    loader = EventLoader(args.src)
    n_total = loader.n_matched - args.start
    if args.n is not None:
        n_total = min(args.n, n_total)
    print(f"Extracting {n_total} events (idx {args.start}..{args.start+n_total-1})")

    run_name = Path(args.src).parent.name + '_' + Path(args.src).name
    out_path = args.out or os.path.join(args.src, 'temp', 'signal_features.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── 预分配数组 ──
    digi_data = {f: np.full((n_total, N_DIGI), np.nan, dtype=np.float32) for f in FIELDS}
    osc_data  = {f: np.full((n_total, N_OSC), np.nan, dtype=np.float32) for f in FIELDS}
    fpga_ids = np.zeros(n_total, dtype=np.int32)
    fpga_ticks = np.zeros(n_total, dtype=np.int64)

    import gc
    t_digi = np.arange(1024) * NS_PER_SAMPLE

    for i in range(args.start, args.start + n_total):
        evt = loader.get_event(i)
        if evt is None:
            continue

        # 每 200 个事件释放一次 scope cache，防止 OOM
        if (i - args.start) % 200 == 0 and (i - args.start) > 0:
            loader._scope_cache.clear()
            loader._scope_trace_cache.clear()
            gc.collect()

        row = i - args.start
        fpga_ids[row] = evt.fpga_trigger_id
        fpga_ticks[row] = evt.fpga_tick

        # ── Digitizer ──
        if evt.has_digi and evt.digi_waveforms:
            for j, ch in enumerate(DIGI_CHS):
                if ch not in evt.digi_waveforms:
                    continue
                y = evt.digi_waveforms[ch]
                r = analyze_waveform(y, t_digi, polarity='auto', trim_tail=TRIM_TAIL)
                d = result_to_dict(r)
                for f in FIELDS:
                    digi_data[f][row, j] = d[f]

        # ── 示波器 ──
        if evt.has_scope and evt.scope_t_ns is not None and evt.scope_waveforms:
            t_osc = evt.scope_t_ns
            for j, ch in enumerate(OSC_CHS):
                if ch not in evt.scope_waveforms:
                    continue
                y = evt.scope_waveforms[ch]
                r = analyze_waveform(y, t_osc, polarity='auto',
                                     pre_baseline_fraction=OSC_PRE_FRAC,
                                     post_baseline_fraction=OSC_POST_FRAC,
                                     trim_tail=0)
                d = result_to_dict(r)
                for f in FIELDS:
                    osc_data[f][row, j] = d[f]

        if (row + 1) % 50 == 0:
            print(f"  ... {row+1}/{n_total}")

    # ── 保存 ──
    save_dict = {
        'meta_run': run_name,
        'meta_n_events': n_total,
        'meta_channels_digi': np.array(DIGI_CHS, dtype=np.int32),
        'meta_channels_osc': np.array(OSC_CHS, dtype='U2'),
        'fpga_id': fpga_ids,
        'fpga_tick': fpga_ticks,
        'digi_index': np.arange(N_DIGI, dtype=np.int32),
        'osc_index': np.arange(N_OSC, dtype=np.int32),
    }
    for f in FIELDS:
        save_dict[f'digi_{f}'] = digi_data[f]
        save_dict[f'osc_{f}'] = osc_data[f]

    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved: {out_path}")
    print(f"  Shape: digi=({n_total},{N_DIGI}), osc=({n_total},{N_OSC})")
    print(f"  Size: {os.path.getsize(out_path) / 1024:.0f} KB")
