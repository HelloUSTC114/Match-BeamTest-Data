"""
BT_20260610_094205 run_0005 逐步诊断
====================================
不跑完整 pipeline，逐步骤检查每一步的中间结果。
"""

import sys, os, numpy as np, glob, warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from src.data.raw_data_recorder import load_binary_events
from src.data.fpga_parser import load_fpga_events, FPGA_CLOCK_HZ

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

RUN = str(REPO / 'data' / 'BT_20260610_094205' / 'run_0005')
DIGI_NS = 8.5
FPGA_1HZ = 200_000_020

# ═══════════════════════════════════════════
print("=" * 60)
print("  run_0005 逐步诊断")
print("=" * 60)

# ─── Step 1: 加载数据 ───
print("\n--- Step 1: 加载 Digitizer ---")
digi = load_binary_events(os.path.join(RUN, 'digitizer', 'V1742_events.bin'))
n_digi = len(digi)
print(f"  Digitizer 事件数: {n_digi}")

d_num = np.array([e['event_number'] for e in digi], dtype=np.int64)
d_tt  = np.array([e['trigger_time_tag'] for e in digi], dtype=np.int64)

# 检查 TT 范围
print(f"  TT 范围: {d_tt.min()} ~ {d_tt.max()}")
print(f"  TT 跨度 (tick): {d_tt[-1] - d_tt[0]:,}")
print(f"  TT 跨度 (秒): {(d_tt[-1] - d_tt[0]) * DIGI_NS / 1e9:.1f}")

# 分离 1Hz
TR_CH = [32, 33, 34, 35]
N_HEAD, N_TAIL = 20, 50
td = np.full((n_digi, 4), np.nan)
has_tr = np.zeros(n_digi, dtype=bool)
for i, e in enumerate(digi):
    for j, ch in enumerate(TR_CH):
        if ch not in e['waveforms']:
            continue
        has_tr[i] = True
        d = e['waveforms'][ch]
        ns = len(d)
        h = min(N_HEAD, ns)
        tl = min(N_TAIL, ns)
        td[i, j] = abs(np.mean(d[:h]) - np.mean(d[-tl:]))

is_hz = np.full(n_digi, False)
is_hz[has_tr] = np.nanmax(td[has_tr], axis=1) < 5.0
is_hz[~has_tr] = True

n_hz = is_hz.sum()
n_tr = n_digi - n_hz
print(f"  1Hz: {n_hz}, TR: {n_tr}")

# 检查 1Hz 间隔
hz_idx = np.where(is_hz)[0]
if len(hz_idx) >= 2:
    hz_intervals = np.diff(d_tt[hz_idx])
    print(f"  1Hz 间隔: min={hz_intervals.min()}, max={hz_intervals.max()}, mean={hz_intervals.mean():.0f}")
    print(f"  预期 1s 间隔: {1e9/DIGI_NS:.0f} ticks")
    n_bad = np.sum(np.abs(hz_intervals - 1e9/DIGI_NS) > 1e9/DIGI_NS * 0.1)
    print(f"  偏离 >10% 的: {n_bad}/{len(hz_intervals)}")

# ─── Step 2: 时钟标定 ───
print("\n--- Step 2: 时钟标定 ---")
tt_1hz = d_tt[hz_idx]
NOM_1S = 1e9 / DIGI_NS

# 检查首脉冲是否完整（偏离 >20% 则跳过）
start_i = 0
if len(hz_idx) >= 3:
    first_dt = tt_1hz[1] - tt_1hz[0]
    if abs(first_dt - NOM_1S) / NOM_1S > 0.02:
        print(f"  [FIX] 跳过不完整首脉冲: dt={first_dt:.0f} (expected={NOM_1S:.0f}, ratio={first_dt/NOM_1S:.4f})")
        start_i = 1

theory_ft = np.zeros(len(hz_idx), dtype=np.float64)
cum = 0.0
for i in range(len(hz_idx)):
    if i < start_i:
        theory_ft[i] = -FPGA_1HZ  # 标记无效
        continue
    if i == start_i:
        n_s = 1
    else:
        n_s = max(1, round((tt_1hz[i] - tt_1hz[i - 1]) / NOM_1S))
    cum += n_s * FPGA_1HZ
    theory_ft[i] = cum

# 只用有效段算 f_pd
valid = theory_ft >= 0
vt = theory_ft[valid]; vtt = tt_1hz[valid]
dt_ft = np.diff(vt); dt_cyc = np.diff(vtt)
f_pd_raw = dt_ft.astype(float) / np.maximum(dt_cyc, 1)
f_pd = np.zeros(len(hz_idx))
f_pd[valid] = np.insert(f_pd_raw, 0, np.mean(f_pd_raw[:5]) if len(f_pd_raw) >= 5 else f_pd_raw[0])

print(f"  avg f_pd (valid): {np.mean(f_pd[f_pd > 0]):.6f}")
print(f"  f_pd 范围: {f_pd[f_pd > 0].min():.6f} ~ {f_pd.max():.6f}")
print(f"  f_pd std: {np.std(f_pd[f_pd > 0]):.6f}")

def conv(t):
    s = int(np.searchsorted(tt_1hz, t, side='right')) - 1
    s = max(0, min(len(tt_1hz) - 2, s))
    if theory_ft[s] < 0 and s + 1 < len(theory_ft):
        s = s + 1
    if theory_ft[s] < 0:
        s = np.argmax(theory_ft >= 0)
    return theory_ft[s] + (float(t) - float(tt_1hz[s])) * f_pd[s]

t_ft = np.array([conv(int(t)) for t in d_tt])
if start_i > 0:
    vh = hz_idx[start_i:]
    err = np.max(np.abs((t_ft[vh] - t_ft[vh[0]]) - (theory_ft[start_i:] - theory_ft[start_i])))
else:
    err = np.max(np.abs((t_ft[hz_idx] - t_ft[hz_idx[0]]) - (theory_ft - theory_ft[0])))
print(f"  1Hz max error: {err:.0f} ticks")

tr_idx = np.where(~is_hz)[0]
t_tr = t_ft[tr_idx]
print(f"  t_tr 范围: {t_tr.min():.0f} ~ {t_tr.max():.0f}")

# ─── Step 3: 加载 FPGA ───
print("\n--- Step 3: 加载 FPGA ---")
fpga = load_fpga_events(os.path.join(RUN, 'fpga', 'fpga_events.bin'))
n_fpga = len(fpga)
f_id = np.array([e.trigger_id for e in fpga], dtype=np.int64)
f_tick = np.array([e.t_fpga for e in fpga], dtype=np.int64)
print(f"  FPGA 事件数: {n_fpga}")
print(f"  f_tick 范围: {f_tick.min():,} ~ {f_tick.max():,}")
print(f"  f_tick 跨度: {(f_tick[-1] - f_tick[0]) / FPGA_CLOCK_HZ:.1f} 秒")

# 检查 digi 和 fpga 的时间覆盖
print(f"  Digi t_tr[0]: {t_tr[0]:.0f}, t_tr[-1]: {t_tr[-1]:.0f}")
print(f"  FPGA f_tick[0]: {f_tick[0]:,}, f_tick[-1]: {f_tick[-1]:,}")

# ─── Step 4: Δt 直方图分析 ───
print("\n--- Step 4: Delta-t 直方图 ---")
n_trig = len(t_tr)

# 先用完整范围看看
all_dt_full = []
for i in range(min(n_trig, 500)):  # 先抽样看
    ni = int(np.searchsorted(f_tick, t_tr[i]))
    lo = max(0, ni - 50)
    hi = min(n_fpga, ni + 50)
    for fi in range(lo, hi):
        all_dt_full.append(t_tr[i] - float(f_tick[fi]))
all_dt_full = np.array(all_dt_full)

# 完整直方图 (宽范围)
h_full, be_full = np.histogram(all_dt_full, bins=np.arange(-10000, 10001, 1))
bc_full = (be_full[:-1] + be_full[1:]) / 2
peak_idx = np.argmax(h_full)
print(f"  峰值位置: {bc_full[peak_idx]:.1f} ticks")
print(f"  峰值高度: {h_full[peak_idx]:,}")
print(f"  bin -5000..0 总计数: {np.sum(h_full[bc_full < 0]):,}")
print(f"  bin 0..5000 总计数: {np.sum(h_full[bc_full > 0]):,}")

# 检查峰是否在 +/-5000 窗口内
if abs(bc_full[peak_idx]) > 4000:
    print(f"  ⚠️ 峰值接近搜索窗口边缘 ({bc_full[peak_idx]:.0f})")
    
    # 扩大搜索范围
    h_wide, be_wide = np.histogram(all_dt_full, bins=np.arange(-50000, 50001, 10))
    bc_wide = (be_wide[:-1] + be_wide[1:]) / 2
    peak_wide = bc_wide[np.argmax(h_wide)]
    print(f"  扩大搜索 (50k): 峰值在 {peak_wide:.0f} ticks ({peak_wide*5/1e3:.1f} us)")

# ─── 检查 1Hz 事件的 TT 间隔是否正常 ───
print("\n--- 1Hz 脉冲间隔检查 ---")
for i in range(min(10, len(hz_idx) - 1)):
    interval = d_tt[hz_idx[i+1]] - d_tt[hz_idx[i]]
    expected = FPGA_1HZ / np.mean(f_pd)
    print(f"  1Hz[{i}]: DT={interval:.0f} ticks, expected={expected:.0f}, ratio={interval/expected:.4f}")

# ─── 检查 f_pd 是否稳定 ───
print("\n--- f_pd 稳定性 ---")
for i in range(min(10, len(f_pd_raw))):
    print(f"  f_pd[{i+start_i}]: {f_pd_raw[i]:.6f}")

# ─── 画图 ───
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1Hz 分类
ax = axes[0, 0]
valid_td = td[has_tr]
if len(valid_td) > 0:
    ax.hist(np.nanmax(valid_td, axis=1), bins=50, color='steelblue', alpha=0.8)
ax.axvline(5.0, color='r', ls='--')
ax.set_title(f'TR tail-baseline (1Hz={n_hz}, TR={n_tr})')
ax.set_xlabel('max |tail-head| (ADC)')

# 1Hz 标定
ax = axes[0, 1]
ax.plot(theory_ft / 1e6, tt_1hz / 1e6, 'o-', ms=2, lw=1)
ax.set_xlabel('Theory FPGA tick (M)')
ax.set_ylabel('Digi raw TT (M)')
ax.set_title('1Hz calibration')

# Δt 直方图 (完整范围)
ax = axes[1, 0]
ax.hist(all_dt_full * 5, bins=200, range=(-5000, 5000), color='steelblue', alpha=0.8)
ax.axvline(x=bc_full[peak_idx] * 5, color='r', lw=2, ls='--', 
           label=f'peak={bc_full[peak_idx]*5:.0f}ns')
ax.set_xlabel('dt (ns)')
ax.set_ylabel('Count')
ax.set_title(f'dt histogram ({len(all_dt_full):,} pairs, window=+/-100)')
ax.legend()

# f_pd 分布
ax = axes[1, 1]
ax.plot(f_pd, 'g.', ms=2)
ax.axhline(np.mean(f_pd), color='r', ls='--', label=f'mean={np.mean(f_pd):.4f}')
ax.set_xlabel('1Hz index')
ax.set_ylabel('f_pd (FPGA ticks / DRS4 cycle)')
ax.set_title('f_pd stability')
ax.legend()

plt.tight_layout()
out_path = os.path.join(RUN, 'temp', 'run0005_debug.png')
fig.savefig(out_path, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"\n  诊断图已保存: {out_path}")

# ═══════════════════════════════════════════
# ─── Step 5: 一对一 Digi ↔ FPGA 匹配 ───
print("\n--- Step 5: 一对一 Digi <-> FPGA 匹配 ---")

# 完整 Δt 直方图确定 offset
all_dt = np.array([t_tr[i] - float(f_tick[fi])
    for i in range(n_trig)
    for fi in range(max(0, int(np.searchsorted(f_tick, t_tr[i])) - 30),
                    min(n_fpga, int(np.searchsorted(f_tick, t_tr[i])) + 30))])

h, be = np.histogram(all_dt, bins=np.arange(-5000, 5001, 0.5))
bc = (be[:-1] + be[1:]) / 2
offset = bc[np.argmax(h)]
near = all_dt[np.abs(all_dt - offset) < 50]
sigma = np.std(near) if len(near) > 10 else 5.0
print(f"  Offset: {offset:.1f} ticks ({offset*5:.0f} ns)")
print(f"  Sigma: {sigma:.1f} ticks ({sigma*5:.1f} ns)")

mw = 3 * sigma; used = set(); matched = []
for i in range(n_trig):
    td_i = t_tr[i]; tgt = td_i - offset; ni = int(np.searchsorted(f_tick, tgt))
    for fi in range(max(0, ni - 15), min(n_fpga, ni + 15)):
        if fi in used: continue
        if abs(td_i - f_tick[fi] - offset) < mw:
            matched.append((int(d_num[tr_idx[i]]), int(f_id[fi]), float(td_i - f_tick[fi])))
            used.add(fi); break
n_m = len(matched)
print(f"  Digi->FPGA: {n_m}/{n_trig} ({n_m/max(n_trig,1)*100:.1f}%)")

# ─── Step 6: Scope ↔ FPGA ───
print("\n--- Step 6: Scope <-> FPGA ---")
import re, lecroyparser, importlib.util
from datetime import datetime
spec = importlib.util.spec_from_file_location("awm", os.path.join(str(REPO/'scripts'), "analyze_lecroy_wfm.py"))
awm = importlib.util.module_from_spec(spec); spec.loader.exec_module(awm)
parse_trc = awm.parse_trc_binary

ch_files = sorted(glob.glob(os.path.join(RUN, 'lecroy_wfm', 'C4--Trace--*.trc')))
print(f"  C4 trc 文件: {len(ch_files)}")

scope_data = []
for fp in ch_files:
    file_idx = int(re.search(r'--Trace--(\d+)\.trc', os.path.basename(fp)).group(1))
    s = lecroyparser.ScopeData(path=fp)
    try: file_utc = datetime.strptime(s.triggerTime, '%Y-%m-%d %H:%M:%S.%f').timestamp()
    except: file_utc = datetime.strptime(s.triggerTime, '%Y-%m-%d %H:%M:%S').timestamp()
    info = parse_trc(fp)
    for si, off in enumerate(info['trig_time_offsets']):
        scope_data.append({'file_idx': file_idx, 'seg_idx': si, 'utc': file_utc + float(off)})

scope_utc = np.array([s['utc'] for s in scope_data])
scope_rel = scope_utc - scope_utc[0]; n_scope = len(scope_data)
print(f"  Scope C4 段数: {n_scope}")

s_dt = np.diff(scope_rel) * 1000; s_ed = np.where(s_dt > 50)[0]
s_sp = np.zeros(n_scope, dtype=int)
for e in s_ed: s_sp[e + 1:] += 1

f_dt_sp = np.diff(f_tick).astype(float) / FPGA_CLOCK_HZ * 1000
f_ed = np.where(f_dt_sp > 50)[0]
f_sp = np.zeros(n_fpga, dtype=int)
for e in f_ed: f_sp[e + 1:] += 1

f_large = sorted([sp for sp in range(f_sp.max() + 1) if np.sum(f_sp == sp) > 10])
s_large = sorted([sp for sp in range(s_sp.max() + 1) if np.sum(s_sp == sp) > 10])
f_cnts = [np.sum(f_sp == sp) for sp in f_large]
s_cnts = [np.sum(s_sp == sp) for sp in s_large]

f_arr, s_arr = np.array(f_cnts, dtype=float), np.array(s_cnts, dtype=float)
best_off, best_rms = 0, float('inf')
for off in range(-10, 11):
    if off >= 0: fs, ss = f_arr[off:], s_arr[:len(f_arr)-off]
    else: fs, ss = f_arr[:len(f_arr)+off], s_arr[-off:]
    no = min(len(fs), len(ss))
    if no < 3: continue
    rms = np.sqrt(np.mean((fs[:no] - ss[:no])**2))
    if rms < best_rms: best_rms, best_off = rms, off
print(f"  Best spill offset: {best_off:+d}, RMS={best_rms:.1f}")

# 匹配 spill 内部事件
all_sp = []
for i in range(min(len(f_large) - max(0, best_off), len(s_large) - max(0, -best_off))):
    fsp = f_large[i + max(0, best_off)]; ssp = s_large[i + max(0, -best_off)]
    fn, sn = np.sum(f_sp == fsp), np.sum(s_sp == ssp)
    if fn != sn: continue
    f_gi, s_gi = np.where(f_sp == fsp)[0], np.where(s_sp == ssp)[0]
    nv = min(100, len(f_gi), len(s_gi))
    ftv = f_tick[f_gi[:nv]].astype(float) / FPGA_CLOCK_HZ; stv = scope_rel[s_gi[:nv]]
    rms_v = np.sqrt(np.mean(((ftv-ftv[0]) - (stv-stv[0]))**2)) * 1e6
    if rms_v < 10:
        for fi, si in zip(f_gi, s_gi):
            all_sp.append((int(fi), scope_data[si]['file_idx'], scope_data[si]['seg_idx']))
n_sp = len(all_sp)
print(f"  Scope-FPGA: {n_sp} pairs")

# ─── Step 7: 完整事件表 ───
print("\n--- Step 7: 完整事件表 ---")
fs_map = {p[0]: {'file_idx': p[1], 'seg_idx': p[2]} for p in all_sp}
fd_map = {}
for m in matched:
    fd_map[m[1]] = {'digi_evnum': m[0], 'dt_ns': m[2] * 5}

import csv
out_csv = os.path.join(RUN, 'temp', 'full_event_table.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['fpga_trigger_id','fpga_tick','has_scope','scope_file_idx',
                'scope_seg_idx','has_digi','digi_evnum','digi_tt_raw',
                'digi_is_1hz','dt_ns'])
    for fi in range(n_fpga):
        fid = f_id[fi]; s = fs_map.get(fi, None); d = fd_map.get(fid, None)
        w.writerow([fid, f_tick[fi],
                    1 if s else 0, s['file_idx'] if s else -1, s['seg_idx'] if s else -1,
                    1 if d else 0, d['digi_evnum'] if d else -1,
                    d_tt[d['digi_evnum']] if d and d['digi_evnum'] < n_digi else -1,
                    int(is_hz[d['digi_evnum']]) if d and d['digi_evnum'] < n_digi else -1,
                    d['dt_ns'] if d else np.nan])

n_s = len(fs_map); n_d = len(matched)
both = sum(1 for fi in range(n_fpga) if fi in fs_map and f_id[fi] in fd_map)
print(f"  FPGA:{n_fpga}, Scope:{n_s}, Digi:{n_d}, Both:{both}")
print(f"  Saved: {out_csv}")

print(f"\n{'=' * 60}")
print("  诊断完成")
print(f"{'=' * 60}")
