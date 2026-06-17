"""
批量绘制 Digitizer + 示波器波形分析图
======================================
每个事件输出两张图 (digi + osc)，复用 plot_analysis 子图模式。

用法:
  python scripts/pos-reconstruction/plot_analyzed_events.py --src <data_dir> --n 100
"""

import sys, os, argparse, gc
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
plot_analysis = _sa.plot_analysis

_si = _load_module("sensor_index", os.path.join(_POS_RECO, "sensor_index.py"))
Instrument = _si.Instrument
ChannelKey = _si.ChannelKey
lookup_by_channel_key = _si.lookup_by_channel_key

DIGI_CHS = [1, 2, 3, 4, 5, 6, 7, 15]
OSC_CHS  = ['C3', 'C4', 'C5', 'C6', 'C7', 'C8']
NS_PER_SAMPLE = 0.2
TRIM_TAIL = 30
OSC_PRE_FRAC = 0.40
OSC_POST_FRAC = 0.30


def _label(ch_key, is_osc=False):
    inst = Instrument.OSC if is_osc else Instrument.DIGI
    r = lookup_by_channel_key(ChannelKey(inst, ch_key))
    if r: return f"{ch_key} ({r[0].alias} e{r[1].electrode})"
    return str(ch_key)


def plot_one_event(evt, idx, out_dir):
    evt_num = evt.fpga_trigger_id
    t_digi = np.arange(1024) * NS_PER_SAMPLE

    # ═══ Digitizer ═══
    fig_d, axes_d = plt.subplots(4, 2, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.50, wspace=0.25, top=0.92, bottom=0.06, left=0.07, right=0.97)
    axes_d = axes_d.flatten()

    for i, ch in enumerate(DIGI_CHS):
        ax = axes_d[i]
        if evt.has_digi and evt.digi_waveforms and ch in evt.digi_waveforms:
            y = evt.digi_waveforms[ch]
            r = analyze_waveform(y, t_digi, polarity='auto', trim_tail=TRIM_TAIL)
            plot_analysis(r, y, t_digi, ax=ax, compact=True, trim_tail=TRIM_TAIL, y_label='ADC',
                          title=f"ch{ch} {_label(ch)} [{r.polarity[0]}] A={r.amplitude:.0f} t_r={r.rise_time_ns:.2f}ns CFD50={r.cfd_50_ns:.2f}ns")
        else:
            ax.set_title(f"ch{ch} — no data"); ax.axis('off')

    fig_d.suptitle(f'Digitizer — Evt #{evt_num}  idx={idx}', fontsize=12, fontweight='bold')
    fig_d.savefig(os.path.join(out_dir, f"evt_{idx:04d}_digi.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig_d); gc.collect()

    # ═══ 示波器 ═══
    if not (evt.has_scope and evt.scope_t_ns is not None and evt.scope_waveforms):
        return

    fig_o, axes_o = plt.subplots(3, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.50, wspace=0.25, top=0.92, bottom=0.06, left=0.07, right=0.97)
    axes_o = axes_o.flatten()
    t_osc = evt.scope_t_ns

    for i, ch in enumerate(OSC_CHS):
        ax = axes_o[i]
        if ch in evt.scope_waveforms:
            y = evt.scope_waveforms[ch]
            r = analyze_waveform(y, t_osc, polarity='auto', pre_baseline_fraction=OSC_PRE_FRAC, post_baseline_fraction=OSC_POST_FRAC, trim_tail=0)
            plot_analysis(r, y, t_osc, ax=ax, compact=True, trim_tail=0, y_label='mV',
                          title=f"{ch} {_label(ch, True)} [{r.polarity[0]}] A={r.amplitude*1000:.1f}mV t_r={r.rise_time_ns:.2f}ns CFD50={r.cfd_50_ns:.2f}ns")
        else:
            ax.set_title(f"{ch} — no data"); ax.axis('off')

    fig_o.suptitle(f'Oscilloscope — Evt #{evt_num}  idx={idx}', fontsize=12, fontweight='bold')
    fig_o.savefig(os.path.join(out_dir, f"evt_{idx:04d}_osc.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig_o); gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--n', type=int, default=100, help='number of events')
    parser.add_argument('--start', type=int, default=0, help='start from event index')
    parser.add_argument('--single', type=int, default=None, help='plot only event at this CSV row index (overrides --n/--start)')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    out_dir = args.out or os.path.join(REPO, 'plots', 'analyzed_events',
                                       Path(args.src).parent.name + '_' + Path(args.src).name)
    os.makedirs(out_dir, exist_ok=True)

    loader = EventLoader(args.src)

    if args.single is not None:
        evt = loader.get_event(args.single)
        if evt is None:
            print(f"Event index {args.single} out of range")
            sys.exit(1)
        plot_one_event(evt, args.single, out_dir)
        print(f"Done: 2 figures")
    else:
        start = args.start
        n_total = min(args.n, loader.n_matched - start)
        print(f"Processing {n_total} events (idx {start}..{start+n_total-1}) -> {out_dir}")
        for i in range(start, start + n_total):
            evt = loader.get_event(i)
            if evt is None:
                continue
            if (i - start) % 200 == 0 and (i - start) > 0:
                loader._scope_cache.clear()
                loader._scope_trace_cache.clear(); gc.collect()
            plot_one_event(evt, i, out_dir)
            if (i - start + 1) % 10 == 0:
                print(f"  ... {i - start + 1}/{n_total}")
        print(f"Done: {n_total * 2} figures")
