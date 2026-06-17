"""
NPZ → ROOT TTree 转换
=====================
将 signal_features.npz + reco_positions.npz 合并写入 ROOT TTree，
附带 CSV 匹配信息副本，保证可通过 fpga_id 回溯原始波形。

用法:
  python scripts/pos-reconstruction/npz_to_root.py --src data/BT.../run_0018
"""

import sys, os, argparse, csv
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
# Sensor 编码
# ======================================================================
SENSOR_CODE = {"W6P18-6": 0, "W3P35-6": 1, "W3P3-8": 2}
SN_BY_CODE = {v: k for k, v in SENSOR_CODE.items()}

# ======================================================================
def ch_to_sensor_electrode(ch, is_osc):
    inst = Instrument.OSC if is_osc else Instrument.DIGI
    r = lookup_by_channel_key(ChannelKey(inst, ch))
    if r is None:
        return None, None, None
    sn, ec = r
    return sn.name, ec.electrode, SENSOR_CODE.get(sn.name, -1)


def load_and_merge(src_dir):
    """Load npz + match CSV, return flat data dict."""
    sig_path = os.path.join(src_dir, 'temp', 'signal_features.npz')
    reco_path = os.path.join(src_dir, 'temp', 'reco_positions.npz')
    csv_path = os.path.join(src_dir, 'temp', 'full_event_table.csv')

    sig = np.load(sig_path)
    reco = np.load(reco_path)
    n = int(sig['meta_n_events'])

    digi_chs = sig['meta_channels_digi']   # [1,2,3,4,5,6,7,15]
    osc_chs  = sig['meta_channels_osc']    # ['C3','C4','C5','C6','C7','C8']

    ADC_TO_V = 1.0 / 4096

    data = {}
    data['fpga_id'] = sig['fpga_id'].astype(np.int32)
    data['fpga_tick'] = sig['fpga_tick'].astype(np.int64)

    # ── Digitizer columns ──
    for j, ch in enumerate(digi_chs):
        sn, elec, scode = ch_to_sensor_electrode(ch, False)
        amp_v = sig['digi_amplitude'][:, j] * ADC_TO_V
        pol   = sig['digi_polarity_code'][:, j]
        sg    = sig['digi_has_signal_int'][:, j]
        cfd   = sig['digi_cfd_50_ns'][:, j]
        rise  = sig['digi_rise_time_ns'][:, j]
        jit   = sig['digi_jitter_ns'][:, j]

        pol_s = np.nan_to_num(pol, nan=-1.0)
        sg_s  = np.nan_to_num(sg, nan=-1.0)
        data[f'digi_A_ch{ch}']     = amp_v.astype(np.float32)
        data[f'digi_pol_ch{ch}']   = pol_s.astype(np.int32)
        data[f'digi_sig_ch{ch}']   = sg_s.astype(np.int32)
        data[f'digi_cfd50_ch{ch}'] = cfd.astype(np.float32)
        data[f'digi_rise_ch{ch}']  = rise.astype(np.float32)
        data[f'digi_jitter_ch{ch}'] = jit.astype(np.float32)
        data[f'digi_elec_ch{ch}']  = np.full(n, elec if elec is not None else -1, dtype=np.int32)
        data[f'digi_sensor_ch{ch}'] = np.full(n, scode, dtype=np.int32)

    # ── Oscilloscope columns ──
    for j, ch in enumerate(osc_chs):
        sn, elec, scode = ch_to_sensor_electrode(ch, True)
        amp_v = sig['osc_amplitude'][:, j]           # already V
        pol   = sig['osc_polarity_code'][:, j]
        sg    = sig['osc_has_signal_int'][:, j]
        cfd   = sig['osc_cfd_50_ns'][:, j]
        rise  = sig['osc_rise_time_ns'][:, j]
        jit   = sig['osc_jitter_ns'][:, j]

        pol_s = np.nan_to_num(pol, nan=-1.0)
        sg_s  = np.nan_to_num(sg, nan=-1.0)
        cs = ch.replace('C', '')
        data[f'osc_A_C{cs}']      = amp_v.astype(np.float32)
        data[f'osc_pol_C{cs}']    = pol_s.astype(np.int32)
        data[f'osc_sig_C{cs}']    = sg_s.astype(np.int32)
        data[f'osc_cfd50_C{cs}']  = cfd.astype(np.float32)
        data[f'osc_rise_C{cs}']   = rise.astype(np.float32)
        data[f'osc_jitter_C{cs}'] = jit.astype(np.float32)
        data[f'osc_elec_C{cs}']   = np.full(n, elec if elec is not None else -1, dtype=np.int32)
        data[f'osc_sensor_C{cs}'] = np.full(n, scode, dtype=np.int32)

    # ── 重建位置 ──
    for sn_name in SENSOR_CODE:
        alias = SENSORS[sn_name].alias
        data[f'pos_{alias}_mm'] = reco[f'{sn_name}_pos_mm'].astype(np.float32)
        data[f'nch_{alias}']    = reco[f'{sn_name}_n_channels'].astype(np.int32)

    data['_meta_n_events'] = n
    data['_meta_sensor_codes'] = SENSOR_CODE
    return data, csv_path


# ======================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    src_dir = args.src
    data, csv_src = load_and_merge(src_dir)
    n = data.pop('_meta_n_events')
    sensor_codes = data.pop('_meta_sensor_codes')

    run_label = Path(src_dir).parent.name + '_' + Path(src_dir).name
    out_root = args.out or os.path.join(src_dir, 'temp', f'analysis_{run_label}.root')
    os.makedirs(os.path.dirname(out_root), exist_ok=True)

    # ── 写 ROOT TTree ──
    import uproot
    print(f"Writing ROOT TTree: {out_root}")
    print(f"  {n} events, {len(data)} branches")

    with uproot.recreate(out_root) as f:
        f.mktree("events", data, title=f"Analysis {run_label}")

    fsize = os.path.getsize(out_root)
    print(f"  Saved: {out_root} ({fsize/1024:.0f} KB)")

    # ── 复制 CSV 匹配表 ──
    out_csv = out_root.replace('.root', '_match.csv')
    if os.path.exists(csv_src):
        import shutil
        shutil.copy2(csv_src, out_csv)
        print(f"  CSV: {out_csv}")

    # ── 存 sensor 编码表 ──
    code_json = out_root.replace('.root', '_sensor_codes.json')
    import json
    with open(code_json, 'w') as f:
        json.dump(sensor_codes, f, indent=2)
    print(f"  Sensor codes: {code_json}")

    print(f"\n  Sensor codes: {sensor_codes}")
    print(f"  Example branches: fpga_id, digi_A_ch3, osc_A_C4, pos_BT4_mm, nch_BT4, ...")
    print(f"  In ROOT: events->Draw(\"pos_BT4_mm:(pos_M4_mm+pos_BT2_mm)/2\", \"nch_BT4>=2\")")
