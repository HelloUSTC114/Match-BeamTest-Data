#!/usr/bin/env python
"""npz → flat binary for C++ ROOT. No external deps beyond numpy."""

import sys, os, argparse, struct
import numpy as np

DIGI_CH = [1, 2, 3, 4, 5, 6, 7, 15]
OSC_CH  = ['C3', 'C4', 'C5', 'C6', 'C7', 'C8']

# Hardcoded channel→(electrode, sensor_code), matches sensor_index.py
_CH_MAP = {
    # Digi
    (False, 1):  (1, 2),   # ch1  = W3P3-8  e1
    (False, 2):  (4, 1),   # ch2  = W3P35-6 e4
    (False, 3):  (7, 1),   # ch3  = W3P35-6 e7
    (False, 4):  (6, 1),   # ch4  = W3P35-6 e6
    (False, 5):  (5, 1),   # ch5  = W3P35-6 e5
    (False, 6):  (3, 1),   # ch6  = W3P35-6 e3
    (False, 7):  (6, 0),   # ch7  = W6P18-6 e6
    (False, 15): (2, 2),   # ch15 = W3P3-8  e2
    # OSC
    (True, 'C3'): (4, 0),  # C3 = W6P18-6 e4
    (True, 'C4'): (5, 0),  # C4 = W6P18-6 e5
    (True, 'C5'): (3, 0),  # C5 = W6P18-6 e3
    (True, 'C6'): (3, 2),  # C6 = W3P3-8  e3
    (True, 'C7'): (4, 2),  # C7 = W3P3-8  e4
    (True, 'C8'): (5, 2),  # C8 = W3P3-8  e5
}

ADC_TO_V = 1.0 / 4096

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    args = parser.parse_args()

    src = args.src
    sig = np.load(os.path.join(src, 'temp', 'signal_features.npz'))
    reco = np.load(os.path.join(src, 'temp', 'reco_positions.npz'))
    n = int(sig['meta_n_events'])

    out_path = os.path.join(src, 'temp', 'export.bin')
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<i', n))
        for evt in range(n):
            f.write(struct.pack('<iq', int(sig['fpga_id'][evt]), int(sig['fpga_tick'][evt])))

            for j, ch in enumerate(DIGI_CH):
                elec, scode = _CH_MAP.get((False, ch), (-1, -1))
                a = sig['digi_amplitude'][evt, j]
                a_v = a * ADC_TO_V if not np.isnan(a) else float('nan')
                f.write(struct.pack('<f', float(a_v)))
                for field in ['digi_polarity_code', 'digi_has_signal_int',
                              'digi_cfd_50_ns', 'digi_rise_time_ns', 'digi_jitter_ns']:
                    v = sig[field][evt, j]
                    f.write(struct.pack('<f', float(v) if not np.isnan(v) else float('nan')))
                f.write(struct.pack('<ii', elec, scode))

            for j, ch in enumerate(OSC_CH):
                elec, scode = _CH_MAP.get((True, ch), (-1, -1))
                for field in ['osc_amplitude', 'osc_polarity_code', 'osc_has_signal_int',
                              'osc_cfd_50_ns', 'osc_rise_time_ns', 'osc_jitter_ns']:
                    v = sig[field][evt, j]
                    f.write(struct.pack('<f', float(v) if not np.isnan(v) else float('nan')))
                f.write(struct.pack('<ii', elec, scode))

            for sn in ['W6P18-6', 'W3P35-6', 'W3P3-8']:
                v = reco[f'{sn}_pos_mm'][evt]
                f.write(struct.pack('<f', float(v) if not np.isnan(v) else float('nan')))
            for sn in ['W6P18-6', 'W3P35-6', 'W3P3-8']:
                f.write(struct.pack('<i', int(reco[f'{sn}_n_channels'][evt])))

    sz = os.path.getsize(out_path)
    print(f"Exported {n} events, {sz/1024:.0f} KB -> {out_path}")

if __name__ == '__main__':
    main()
