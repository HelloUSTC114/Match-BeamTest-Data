"""
通道信号统计
============
从 signal_features.npz 读取分析结果，统计:
  1. 每个通道有信号占比
  2. 每个 sensor 至少一个通道"正常"(正极性+有信号)的事件占比

用法:
  python scripts/pos-reconstruction/channel_stats.py --src data/BT.../run_0018
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
parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True)
parser.add_argument('--npz', default=None, help='path to signal_features.npz (default: <src>/temp/signal_features.npz)')
args = parser.parse_args()

npz_path = args.npz or os.path.join(args.src, 'temp', 'signal_features.npz')
if not os.path.exists(npz_path):
    print(f"ERROR: {npz_path} not found. Run extract_signals.py first.")
    sys.exit(1)

d = np.load(npz_path)
n_events = d['meta_n_events']
digi_amp = d['digi_amplitude']
digi_sig = d['digi_has_signal_int']
digi_pol = d['digi_polarity_code']
digi_chs = d['meta_channels_digi']
osc_amp  = d['osc_amplitude']
osc_sig  = d['osc_has_signal_int']
osc_pol  = d['osc_polarity_code']
osc_chs  = d['meta_channels_osc']

print(f"Run: {d['meta_run']}, {n_events} events\n")

# ── Helper: map channel to sensor ──
def ch_to_sensor(ch, is_osc):
    inst = Instrument.OSC if is_osc else Instrument.DIGI
    r = lookup_by_channel_key(ChannelKey(inst, ch))
    return r[0] if r else None  # SensorInfo

# ═══════════════════════════════════════════════════════════
# 1. Per-channel signal fraction
# ═══════════════════════════════════════════════════════════
print("=" * 80)
print("  1. Per-channel statistics")
print("=" * 80)

print(f"\n  {'Channel':<12} {'Instrument':<7} {'Sensor':<10} {'Elec':<5} {'Polarity':<10} {'Signal%':>8} {'Mean A':>9} {'Median A':>9} {'Sigma_pre':>9}")
print(f"  {'-'*12} {'-'*7} {'-'*10} {'-'*5} {'-'*10} {'-'*8} {'-'*9} {'-'*9} {'-'*9}")

all_channels = []
# Digitizer
for j, ch in enumerate(digi_chs):
    sn = ch_to_sensor(ch, False)
    if sn is None: continue
    ec = sn.get_channel(ch) if hasattr(sn, 'get_channel') else None
    elec = ec.electrode if ec else '?'
    arr_a = digi_amp[:, j]
    arr_s = digi_sig[:, j]
    arr_p = digi_pol[:, j]
    valid = ~np.isnan(arr_a)
    if valid.sum() == 0: continue
    sig_frac = arr_s[valid].mean() * 100
    pos_frac = (arr_p[valid] == 0).mean() * 100
    neg_frac = (arr_p[valid] == 1).mean() * 100
    pol_str = f"{pos_frac:.0f}%P/{neg_frac:.0f}%N"
    mean_a = arr_a[valid].mean()
    med_a  = np.median(arr_a[valid])
    sigma  = d['digi_pre_baseline_sigma'][:, j][valid].mean()
    print(f"  {'ch'+str(ch):<12} {'Digi':<7} {sn.name:<10} {str(elec):<5} {pol_str:<10} {sig_frac:7.1f}% {mean_a:9.1f} {med_a:9.1f} {sigma:9.2f}")
    all_channels.append((sn.name, f"ch{ch}", 'Digi', j, arr_s[valid], arr_p[valid]))

# Oscilloscope
for j, ch in enumerate(osc_chs):
    sn = ch_to_sensor(ch, True)
    if sn is None: continue
    ec = sn.get_channel(ch) if hasattr(sn, 'get_channel') else None
    elec = ec.electrode if ec else '?'
    arr_a = osc_amp[:, j]
    arr_s = osc_sig[:, j]
    arr_p = osc_pol[:, j]
    valid = ~np.isnan(arr_a)
    if valid.sum() == 0: continue
    sig_frac = arr_s[valid].mean() * 100
    pos_frac = (arr_p[valid] == 0).mean() * 100
    neg_frac = (arr_p[valid] == 1).mean() * 100
    pol_str = f"{pos_frac:.0f}%P/{neg_frac:.0f}%N"
    mean_a = arr_a[valid].mean()
    med_a  = np.median(arr_a[valid])
    sigma  = d['osc_pre_baseline_sigma'][:, j][valid].mean()
    print(f"  {'OSC '+str(ch):<12} {'OSC':<7} {sn.name:<10} {str(elec):<5} {pol_str:<10} {sig_frac:7.1f}% {mean_a*1000:9.2f} {med_a*1000:9.2f} {sigma*1000:9.3f}")
    all_channels.append((sn.name, f"OSC{ch}", 'OSC', j, arr_s[valid], arr_p[valid]))

# ═══════════════════════════════════════════════════════════
# 2. Per-sensor "normal" fraction
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"  2. Per-sensor: at least 1 \"normal\" channel (positive + has signal)")
print(f"{'='*80}\n")

# "normal" per channel: polarity_code==0 AND has_signal_int==1
# Build per-sensor per-event mask
for sn_name, sensor in SENSORS.items():
    # Collect all channels for this sensor
    evt_normal = np.zeros(n_events, dtype=bool)  # at least one normal
    n_total_ch = 0

    # Digi
    for j, ch in enumerate(digi_chs):
        if ch_to_sensor(ch, False) is None: continue
        if ch_to_sensor(ch, False).name != sn_name: continue
        n_total_ch += 1
        # normal = positive + signal
        normal_mask = ~np.isnan(digi_pol[:, j]) & (digi_pol[:, j] == 0) & (digi_sig[:, j] == 1)
        evt_normal |= normal_mask

    # OSC
    for j, ch in enumerate(osc_chs):
        if ch_to_sensor(ch, True) is None: continue
        if ch_to_sensor(ch, True).name != sn_name: continue
        n_total_ch += 1
        normal_mask = ~np.isnan(osc_pol[:, j]) & (osc_pol[:, j] == 0) & (osc_sig[:, j] == 1)
        evt_normal |= normal_mask

    frac = evt_normal.mean() * 100
    n_normal = evt_normal.sum()
    print(f"  {sensor.name} ({sensor.alias}, {sensor.draw}): "
          f"{n_normal}/{n_events} = {frac:.1f}%  "
          f"(channels={n_total_ch}, role={sensor.role})")

print()
