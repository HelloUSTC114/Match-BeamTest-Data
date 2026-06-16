"""
Step 4: 示波器 ↔ FPGA 匹配 (Sequence 模式展开 + Spill 对齐)

方案:
  1. 展开示波器 Sequence 文件, 计算每个 segment 的绝对时间戳
  2. 分析示波器数据的 spill 结构
  3. 分析 FPGA 数据的 spill 结构
  4. 通过 spill 对齐两者, 再逐事件匹配
"""
import sys, os, numpy as np, re, glob, warnings, importlib.util
from pathlib import Path
warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parent.parent.parent
SCR = str(REPO / 'scripts')
sys.path.insert(0, str(REPO))
if SCR not in sys.path: sys.path.insert(0, SCR)

import lecroyparser

# 用 importlib 加载 parse_trc_binary
spec = importlib.util.spec_from_file_location("awm", os.path.join(SCR, "analyze_lecroy_wfm.py"))
awm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awm)
parse_trc_binary = awm.parse_trc_binary

# ── 配置: 修改 RUN 指向你的数据目录 ──
RUN = str(REPO.parent / 'Analyze' / 'data' / 'BT_20260530_010301_run_0001')
SCOPE_DIR = os.path.join(RUN, 'lecroy_wfm')
FPGA_BIN = os.path.join(RUN, 'fpga', 'fpga_events.bin')
FPGA_CLOCK_HZ = 200e6

# ═══ 1. 加载 FPGA 数据 ═══
from src.data.fpga_parser import load_fpga_events
fpga = load_fpga_events(FPGA_BIN)
f_tick = np.array([e.t_fpga for e in fpga])
f_id   = np.array([e.trigger_id for e in fpga])
n_fpga = len(fpga)
f_time_s = f_tick.astype(float) / FPGA_CLOCK_HZ

print(f"FPGA: {n_fpga} events")
print(f"  time range: {f_time_s[0]:.2f} ~ {f_time_s[-1]:.2f}s")
print(f"  span: {f_time_s[-1]-f_time_s[0]:.2f}s")

# FPGA spills (50ms threshold)
f_dt = np.diff(f_time_s) * 1000  # ms
f_edges = np.where(f_dt > 50)[0]
f_sp = np.zeros(n_fpga, dtype=int)
for e in f_edges: f_sp[e+1:] += 1
n_f_sp = f_sp.max() + 1
print(f"  FPGA spills: {n_f_sp}")

# ═══ 2. 展开示波器 Sequence TRC 文件 ═══
from datetime import datetime

def get_file_idx(path):
    return int(re.search(r'--Trace--(\d+)\.trc', os.path.basename(path)).group(1))

# 选择一个通道 (C4)
ch_files = sorted(glob.glob(os.path.join(SCOPE_DIR, 'C4--Trace--*.trc')))
print(f"\n示波器 (C4): {len(ch_files)} 个文件")

scope_segments = []  # [{time_abs, time_rel, file_idx, seg_idx}]
for fp in ch_files:
    fi = get_file_idx(fp)
    try:
        s = lecroyparser.ScopeData(path=fp)
        # 文件第一个segment的绝对UTC时间
        t0_str = s.triggerTime
        try:
            t0_dt = datetime.strptime(t0_str, "%Y-%m-%d %H:%M:%S.%f")
        except:
            t0_dt = datetime.strptime(t0_str, "%Y-%m-%d %H:%M:%S")
        t0_unix = t0_dt.timestamp()
        
        # 读取trigger_time_offset
        info = parse_trc_binary(fp)
        ns = info['n_segments']
        offsets = info['trig_time_offsets']  # seconds relative to first trigger
        
        for si in range(ns):
            scope_segments.append({
                'file_idx': fi,
                'seg_idx': si,
                't_abs': t0_unix + offsets[si],  # UTC (seconds since epoch)
                'offset_s': offsets[si],
            })
    except Exception as e:
        print(f"  ❌ {os.path.basename(fp)}: {e}")

n_scope = len(scope_segments)
if n_scope == 0:
    print("❌ 无示波器数据"); exit()

t_abs_all = np.array([s['t_abs'] for s in scope_segments])
print(f"  展开后: {n_scope} segments")
print(f"  时间范围 (UTC): {datetime.fromtimestamp(t_abs_all[0])} ~ {datetime.fromtimestamp(t_abs_all[-1])}")

# 相对时间 (文件内第一个segment为0)
t_rel = t_abs_all - t_abs_all[0]
print(f"  相对时间: {t_rel[0]:.3f} ~ {t_rel[-1]:.3f}s")
print(f"  跨度: {t_rel[-1]-t_rel[0]:.3f}s")

# ═══ 3. 示波器 spills ═══
s_dt = np.diff(t_rel) * 1000  # ms
s_edges = np.where(s_dt > 50)[0]
s_sp = np.zeros(n_scope, dtype=int)
for e in s_edges: s_sp[e+1:] += 1
n_s_sp = s_sp.max() + 1
print(f"\n示波器 spills: {n_s_sp}")
for sp in range(min(n_s_sp, 30)):
    n = np.sum(s_sp == sp)
    if n > 10:
        mask = s_sp == sp
        t0 = t_rel[mask][0]
        print(f"  Spill{sp:2d}: {n:5d} segs, t_start={t0:.2f}s")

# ═══ 4. 对比 FPGA 和示波器时间范围 ═══
print(f"\n{'='*60}")
print(f"时间范围对比")
print(f"{'='*60}")
print(f"  FPGA:        {f_time_s[0]:.2f} ~ {f_time_s[-1]:.2f}s (跨度 {f_time_s[-1]-f_time_s[0]:.2f}s)")
print(f"  示波器(C4):  {t_rel[0]:.2f} ~ {t_rel[-1]:.2f}s (跨度 {t_rel[-1]-t_rel[0]:.2f}s)")

# ═══ 5. 对齐: 找示波器 ↔ FPGA 的时间偏移 ═══
# 示波器相对时间 (以第一个segment为0)
t_scope_s = t_abs_all - t_abs_all[0]

# FPGA 相对时间 (以第一个OSCI事件为0)
t_fpga_s = f_time_s - f_time_s[0]

# 找示波器 ↔ FPGA 的时间偏移 (Δt 直方图)
all_dt = []
for i in range(min(5000, n_scope)):
    ts = t_scope_s[i]
    ni = int(np.searchsorted(t_fpga_s, ts))
    for fi in range(max(0, ni-50), min(n_fpga, ni+50)):
        all_dt.append(ts - t_fpga_s[fi])

all_dt = np.array(all_dt)
h, be = np.histogram(all_dt, bins=500, range=(-500, 500))
bc = (be[:-1] + be[1:]) / 2
if np.max(h) > 10:
    peak = bc[np.argmax(h)]
    print(f"\n  Scope-FPGA Δt 峰值: {peak:.6f}s ({peak*1000:.3f}ms)")
    near = all_dt[np.abs(all_dt - peak) < 0.01]
    if len(near) > 10:
        sigma = np.std(near)
        print(f"  σ: {sigma*1000:.3f}ms")
else:
    print(f"\n  ⚠️ 未找到 clean peak. max count = {np.max(h)}")
    # 尝试更宽范围
    h2, be2 = np.histogram(all_dt, bins=2000, range=(-600, 600))
    bc2 = (be2[:-1] + be2[1:]) / 2
    if np.max(h2) > 10:
        peak = bc2[np.argmax(h2)]
        print(f"  (宽范围) Scope-FPGA Δt 峰值: {peak:.6f}s ({peak*1000:.3f}ms)")
    else:
        peak = 0.0

# ═══ 6. 保存结果 ═══
np.savez(os.path.join(RUN, 'temp', 'scope_fpga_aligned.npz'),
    scope_t_abs=t_abs_all, scope_t_rel=t_rel, scope_spill=s_sp,
    fpga_t_fpga=f_tick, fpga_t_s=f_time_s, fpga_spill=f_sp,
    fpga_id=f_id, scope_n=n_scope, fpga_n=n_fpga)
print(f"\n已保存: {RUN}/temp/scope_fpga_aligned.npz")
