"""
关联图: 示踪器预测位置 vs DUT 测量位置
========================================
(x1 + x3)/2  vs  x2  二维直方图 (ROOT TH2F)

x1 = M4 (Draw6)   x2 = BT4 (Draw4 DUT)   x3 = BT2 (Draw1)

用法:
  python scripts/pos-reconstruction/correlation_plot.py --src data/BT.../run_0018
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
_si = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("sensor_index", os.path.join(_POS_RECO, "sensor_index.py")))
sys.modules['sensor_index'] = _si
importlib.util.spec_from_file_location("sensor_index", os.path.join(_POS_RECO, "sensor_index.py")).loader.exec_module(_si)
SENSORS = _si.SENSORS

parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True)
parser.add_argument('--out', default=None)
args = parser.parse_args()

npz_path = os.path.join(args.src, 'temp', 'reco_positions.npz')
d = np.load(npz_path)

# ── 三个传感器的位置 ──
x1 = d['W6P18-6_pos_mm']    # M4  (Draw6, upstream tracker)
x2 = d['W3P35-6_pos_mm']    # BT4 (Draw4, DUT)
x3 = d['W3P3-8_pos_mm']     # BT2 (Draw1, downstream tracker)
n2  = d['W3P35-6_n_channels']  # DUT 使用的通道数

# 三系统同时有效 + DUT ≥ 2 通道
valid = ~np.isnan(x1) & ~np.isnan(x2) & ~np.isnan(x3) & (n2 >= 2)
n_valid = valid.sum()
print(f"Events: {len(x1)} total, {n_valid} triple + DUT>=2ch ({n_valid/len(x1)*100:.1f}%)")

x_pred = (x1[valid] + x3[valid]) / 2.0
x_meas = x2[valid]

# ── 直方图参数 ──
run_label = Path(args.src).parent.name + '_' + Path(args.src).name
x_min, x_max = float(min(x_pred.min(), x_meas.min())), float(max(x_pred.max(), x_meas.max()))
margin = (x_max - x_min) * 0.05
xbins = 80

print(f"  x range: [{x_min:.2f}, {x_max:.2f}] mm")
print(f"  RMS(x_pred - x_meas) = {np.std(x_pred - x_meas):.4f} mm")

# ── ROOT 文件 (uproot) ──
import uproot

counts, x_edges, y_edges = np.histogram2d(x_pred, x_meas, bins=xbins,
                                           range=[[0.7, 1.0], [0.8, 1.1]])

out_path = args.out or os.path.join(args.src, 'temp', f'correlation_{run_label}.root')
with uproot.recreate(out_path) as f:
    f["h_corr"] = (counts, x_edges, y_edges)
print(f"Saved: {out_path}")

# ── PNG 预览 (matplotlib) ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 7))
counts = counts.astype(float)
counts[counts == 0] = np.nan

im = ax.pcolormesh(x_edges, y_edges, counts.T, cmap='viridis', shading='flat', vmin=1)
ax.plot([0.7, 1.0], [0.8, 1.1], 'r--', lw=1.5, label='y = x')
ax.set_xlim(0.7, 1.0)
ax.set_ylim(0.8, 1.1)
ax.set_xlabel('(x_M4 + x_BT2) / 2  [mm]')
ax.set_ylabel('x_BT4 (DUT)  [mm]')
ax.set_title(f'{run_label}  (DUT >= 2ch)\nTriple: {n_valid} events, '
             f'RMS(x_pred - x_meas) = {np.std(x_pred - x_meas):.3f} mm')
ax.set_aspect('auto')
cbar = plt.colorbar(im, ax=ax, label='Counts')
ax.legend()

png_path = out_path.replace('.root', '.png')
fig.savefig(png_path, dpi=130, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {png_path}")
