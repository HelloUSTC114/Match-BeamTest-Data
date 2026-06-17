"""
绘制 Digitizer ch1-7 & ch15 前 50 个事件的波形
================================================
每个 event 独立一张图，8 通道绘制在同一坐标轴上。
输出到 plots/digi_waveforms/ 文件夹。

通道映射:
  ch1  = Draw1 BT2, electrode 1
  ch2  = Draw4 BT4, electrode 4
  ch3  = Draw4 BT4, electrode 7
  ch4  = Draw4 BT4, electrode 6
  ch5  = Draw4 BT4, electrode 5
  ch6  = Draw4 BT4, electrode 3
  ch7  = Draw6 M4,  electrode 6
  ch15 = Draw1 BT2, electrode 2
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── 路径设置 ───
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from src.data.raw_data_recorder import load_binary_events

# ─── 配置 (支持命令行参数) ───
if len(sys.argv) > 1:
    RUN_DIR = sys.argv[1]
else:
    RUN_DIR = os.path.join(WORKSPACE, 'data', 'BT_20260610_094205', 'run_0018')

DIGI_BIN = os.path.join(RUN_DIR, 'digitizer', 'V1742_events.bin')
if not os.path.exists(DIGI_BIN):
    print(f"ERROR: {DIGI_BIN} not found!")
    sys.exit(1)

# 用 run 名作为子目录名
run_name = os.path.basename(RUN_DIR.rstrip('/\\'))
session_name = os.path.basename(os.path.dirname(RUN_DIR.rstrip('/\\')))
PLOTS_DIR = os.path.join(WORKSPACE, 'plots', 'digi_waveforms', f'{session_name}_{run_name}')
os.makedirs(PLOTS_DIR, exist_ok=True)

# 通道索引 & 标签 & 颜色 (按传感器分组着色)
# 注意: key 就是 Digitizer 物理通道编号 (1-based for signal ch)
CH_INFO = [
    (1,  'ch1',  'Draw1 BT2, elec1',  '#e74c3c'),   # red
    (2,  'ch2',  'Draw4 BT4, elec4',  '#2ecc71'),   # green
    (3,  'ch3',  'Draw4 BT4, elec7',  '#27ae60'),   # dark green
    (4,  'ch4',  'Draw4 BT4, elec6',  '#1abc9c'),   # teal
    (5,  'ch5',  'Draw4 BT4, elec5',  '#16a085'),   # dark teal
    (6,  'ch6',  'Draw4 BT4, elec3',  '#2ecc71'),   # green
    (7,  'ch7',  'Draw6 M4,  elec6',  '#3498db'),   # blue
    (15, 'ch15', 'Draw1 BT2, elec2',  '#e67e22'),   # orange
]

N_EVENTS = 50
SAMPLING_PS = 200
NS_PER_SAMPLE = SAMPLING_PS / 1000.0  # 0.2 ns/sample
RECORD_LENGTH = 1024

# ─── 加载数据 ───
print(f"Loading digitizer data: {DIGI_BIN}")
digi = load_binary_events(DIGI_BIN)
print(f"  Total events: {len(digi)}")

t_ns = np.arange(RECORD_LENGTH) * NS_PER_SAMPLE

# ═══════════════════════════════════════════════════════════
# 每个 event 独立一张图, 8 通道同一坐标轴
# ═══════════════════════════════════════════════════════════
print(f"\nPlotting {N_EVENTS} individual event figures...")

for evt_i in range(min(N_EVENTS, len(digi))):
    evt = digi[evt_i]
    evt_num = evt['event_number']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for ch_idx, ch_name, ch_label, color in CH_INFO:
        if ch_idx in evt['waveforms']:
            wf = evt['waveforms'][ch_idx]
            ax.plot(t_ns, wf, color=color, lw=0.8, alpha=0.85, label=f'{ch_name} ({ch_label})')
    
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('ADC')
    ax.set_title(f'Digi Waveforms — Event #{evt_num} ({session_name}/{run_name}, idx={evt_i})', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f'evt_{evt_i:04d}_digi_ch1-7_ch15.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    
    if (evt_i + 1) % 10 == 0:
        print(f"  ... {evt_i + 1}/{N_EVENTS} saved")

print(f"\n✅ All {min(N_EVENTS, len(digi))} figures saved to {PLOTS_DIR}")
