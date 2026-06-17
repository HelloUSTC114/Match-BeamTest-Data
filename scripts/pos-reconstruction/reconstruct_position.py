"""
击中位置重建 — 幅度重心法
==========================
从 signal_features.npz 读取信号幅度，按传感器重建击中位置。

原理:  pos = sum(A_i * p_i) / sum(A_i)
  A_i = 通道幅度， p_i = electrode * pitch (电极编号 × strip间距)
  排除: 负信号 (串扰)、噪声 (A < 5σ)

Digitizer ADC→电压转换: V1742 12-bit, Vpp=1V → A_V = A_ADC / 4096 × 1.0 V

用法:
  python scripts/pos-reconstruction/reconstruct_position.py --src data/BT.../run_0018
  → 输出 <src>/temp/reco_positions.npz
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
def _load(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_si = _load("sensor_index", os.path.join(_POS_RECO, "sensor_index.py"))
SENSORS = _si.SENSORS
Instrument = _si.Instrument
ChannelKey = _si.ChannelKey
lookup_by_channel_key = _si.lookup_by_channel_key

# ======================================================================
# V1742 ADC 参数 (from UM4279 rev12)
# ======================================================================
ADC_BITS = 12
ADC_MAX  = 2 ** ADC_BITS       # 4096
VPP      = 1.0                  # 1 Vpp
ADC_TO_V = VPP / ADC_MAX        # ~0.244 mV/ADC

# ======================================================================
def ch_to_sensor_electrode(ch, is_osc):
    """Return (SensorInfo, electrode_number) or (None, None)."""
    inst = Instrument.OSC if is_osc else Instrument.DIGI
    r = lookup_by_channel_key(ChannelKey(inst, ch))
    if r is None:
        return None, None
    return r[0], r[1].electrode


def centroid(amplitudes, positions):
    """Amplitude-weighted centroid. Returns (pos, sum_A)."""
    if len(amplitudes) == 0:
        return np.nan, 0.0
    total = np.sum(amplitudes)
    if total <= 0:
        return np.nan, 0.0
    return float(np.sum(amplitudes * positions) / total), float(total)


def reconstruct_sensor(sensor_name, sn_info, d, digi_chs, osc_chs):
    """
    对一个 sensor 的所有事件执行位置重建。

    Returns
    -------
    pos_mm : (n_events,) float32, NaN if insufficient signal channels
    sum_amp_v : (n_events,) float32, total amplitude (V)
    n_channels_used : (n_events,) int, number of valid channels
    """
    n_events = d['meta_n_events']
    pitch_um = sn_info.pitch_um  # μm
    pos_mm = np.full(n_events, np.nan, dtype=np.float32)
    sum_amp = np.full(n_events, np.nan, dtype=np.float32)
    n_used = np.zeros(n_events, dtype=np.int32)

    # Collect channel indices and electrode positions for this sensor
    channel_data = []  # [(amp_array, pol_array, sig_array, electrode_num, is_osc)]

    # Digitizer
    for j, ch in enumerate(digi_chs):
        sn, elec = ch_to_sensor_electrode(ch, False)
        if sn is None or sn.name != sensor_name:
            continue
        amp_adc = d['digi_amplitude'][:, j]          # ADC
        pol     = d['digi_polarity_code'][:, j]
        sig     = d['digi_has_signal_int'][:, j]
        channel_data.append((amp_adc * ADC_TO_V, pol, sig, elec))

    # Oscilloscope
    for j, ch in enumerate(osc_chs):
        sn, elec = ch_to_sensor_electrode(ch, True)
        if sn is None or sn.name != sensor_name:
            continue
        amp_v  = d['osc_amplitude'][:, j]            # already in V
        pol    = d['osc_polarity_code'][:, j]
        sig    = d['osc_has_signal_int'][:, j]
        channel_data.append((amp_v, pol, sig, elec))

    n_ch = len(channel_data)
    if n_ch == 0:
        return pos_mm, sum_amp, n_used

    # Per-event reconstruction
    for evt in range(n_events):
        amps = []
        positions = []
        for amp_arr, pol_arr, sig_arr, elec in channel_data:
            a = amp_arr[evt]
            p = pol_arr[evt]
            s = sig_arr[evt]
            if np.isnan(a) or np.isnan(p):
                continue
            # 排除噪声和负信号(串扰)
            if s < 1 or p == 1:
                continue
            amps.append(a)
            positions.append(elec * pitch_um)  # μm

        if len(amps) >= 1:
            if len(amps) == 1:
                pos = positions[0]  # 单通道: 直接用电极位置
                total = amps[0]
            else:
                pos, total = centroid(np.array(amps), np.array(positions))
            pos_mm[evt] = pos / 1000.0 if not np.isnan(pos) else np.nan  # μm → mm
            sum_amp[evt] = total
        n_used[evt] = len(amps)

    return pos_mm, sum_amp, n_used


# ======================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--npz', default=None)
    args = parser.parse_args()

    npz_path = args.npz or os.path.join(args.src, 'temp', 'signal_features.npz')
    d = np.load(npz_path)
    digi_chs = d['meta_channels_digi']
    osc_chs  = d['meta_channels_osc']
    n_events = d['meta_n_events']

    run_label = Path(args.src).parent.name + '_' + Path(args.src).name
    out_path = os.path.join(args.src, 'temp', 'reco_positions.npz')

    print(f"Reconstructing positions from {npz_path}")
    print(f"  {n_events} events")
    print(f"  ADC: {ADC_BITS}-bit, {VPP} Vpp → {ADC_TO_V*1000:.3f} mV/ADC\n")

    save_dict = {
        'meta_run': d['meta_run'],
        'meta_n_events': n_events,
        'fpga_id': d['fpga_id'],
        'fpga_tick': d['fpga_tick'],
    }

    for sn_name, sn_info in SENSORS.items():
        pos, sum_a, n_used = reconstruct_sensor(sn_name, sn_info, d, digi_chs, osc_chs)
        valid = ~np.isnan(pos)
        n_valid = valid.sum()
        print(f"  {sn_name} ({sn_info.alias}, {sn_info.draw}): "
              f"{n_valid}/{n_events} = {n_valid/n_events*100:.1f}%  "
              f"channels={len([1 for j,ch in enumerate(digi_chs) if ch_to_sensor_electrode(ch,False)[0] and ch_to_sensor_electrode(ch,False)[0].name==sn_name]) + len([1 for j,ch in enumerate(osc_chs) if ch_to_sensor_electrode(ch,True)[0] and ch_to_sensor_electrode(ch,True)[0].name==sn_name])}  "
              f"n_used_mean={n_used[valid].mean():.1f}" if valid.sum() > 0 else f"  {sn_name}: 0 events")
        save_dict[f'{sn_name}_pos_mm'] = pos
        save_dict[f'{sn_name}_sum_amp_v'] = sum_a
        save_dict[f'{sn_name}_n_channels'] = n_used

    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved: {out_path}")
