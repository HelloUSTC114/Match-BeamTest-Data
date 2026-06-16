"""
三系统 (Digitizer - FPGA - 示波器) 时间戳匹配分析
=================================================

示波器为 Sequence 模式，每个 TRC 文件包含多个 segment，
需要先将 segment 展开后再做匹配。

数据格式:
  Digitizer: V1742_events.bin, trigger_time_tag @ 8.5ns/cyc
  FPGA:       fpga_events.bin, 16B/packet, timestamp @ 200MHz (5ns/tick)
  示波器:     .trc 文件, Sequence 模式, 每文件 2800 segments
              每个 segment 有 trigger_time_offset (s)

匹配策略:
  Scope ↔ FPGA : global_seg_index = file_idx × n_segments + seg_idx
                 对应 FPGA OSCI trigger_id (从 1 开始)
  FPGA ↔ Digitizer : 校准时钟后, 通过 Δt 分布峰值做时间匹配
"""

import lecroyparser
import importlib.util
from src.data.fpga_parser import parse_fpga_packets, FPGA_CLOCK_HZ
from src.data.raw_data_recorder import load_binary_events
import matplotlib.pyplot as plt
from pathlib import Path
import gc
import os
import sys
import re
import glob
import csv
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# ─── 仓库根目录 ───
REPO = Path(__file__).resolve().parent.parent  # Match-BeamTest-Data
sys.path.insert(0, str(REPO))
# scripts 是非包目录，用 importlib 加载
spec = importlib.util.spec_from_file_location(
    "analyze_lecroy_wfm",
    str(REPO / "scripts" / "analyze_lecroy_wfm.py"))
alcw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alcw)
parse_trc_binary = alcw.parse_trc_binary

# ─── 配置: 修改 RUN_DIR 指向你的数据目录 ───
RUN_DIR = str(REPO.parent / 'Analyze' / 'data' / 'BT_20260530_010301_run_0001')
OUT_DIR = str(REPO / 'plots')
DIGI_NS = 8.5          # V1742 时钟周期 (ns)
SCOPE_CH = 'C4'          # 用于匹配的示波器通道
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=130, bbox_inches='tight')
    print(f"  ✅ {name}")
    plt.close(fig)
    gc.collect()


# ═══════════════════════════════════════════════════════════
# 1. 加载三系统数据
# ═══════════════════════════════════════════════════════════
print("=" * 66)
print("  Step 1: 加载数据")
print("=" * 66)

# ── 1a. Digitizer ──
digi = load_binary_events(os.path.join(
    RUN_DIR, 'digitizer', 'V1742_events.bin'))
n_digi = len(digi)
d_num = np.array([e['event_number'] for e in digi], dtype=np.int64)
d_tt = np.array([e['trigger_time_tag'] for e in digi], dtype=np.int64)
d_et = np.array([e.get('event_time_tag', 0) for e in digi], dtype=np.int64)
d_t_ns = d_tt.astype(np.float64) * DIGI_NS
d_t_s = d_t_ns * 1e-9
# 检查 event_number 连续性
d_missing = np.setdiff1d(np.arange(d_num[0], d_num[-1]+1), d_num)
print(f"  ✅ Digitizer: {n_digi} events, event# {d_num[0]}~{d_num[-1]}"
      f" (缺失 {len(d_missing)})")
print(f"     trigger_time_tag: {d_tt[0]} ~ {d_tt[-1]}")
print(f"     总时长: {d_t_s[-1]-d_t_s[0]:.3f} s")

# ── 1b. FPGA ──
with open(os.path.join(RUN_DIR, 'fpga', 'fpga_events.bin'), 'rb') as f:
    fpga_raw = f.read()
fpga_all = parse_fpga_packets(fpga_raw)
osci = [p for p in fpga_all if p.is_osci]
lgad = [p for p in fpga_all if not p.is_osci]
n_osci = len(osci)
o_id = np.array([p.event_id for p in osci], dtype=np.int64)
o_tick = np.array([p.timestamp for p in osci], dtype=np.int64)
o_t_s = o_tick.astype(np.float64) / FPGA_CLOCK_HZ
o_missing = np.setdiff1d(np.arange(o_id[0], o_id[-1]+1), o_id)
print(f"  ✅ FPGA: {len(fpga_all)} packets (OSCI={n_osci}, LGAD={len(lgad)})")
print(f"     trigger_id: {o_id[0]}~{o_id[-1]} (缺失 {len(o_missing)})")
print(f"     总时长: {o_t_s[-1]-o_t_s[0]:.3f} s")

# ── 1c. 示波器 (Sequence 模式 → 展开 segment) ──
scope_dir = os.path.join(RUN_DIR, 'lecroy_wfm')
ch_files = sorted(glob.glob(os.path.join(
    scope_dir, f'{SCOPE_CH}--Trace--*.trc')))
print(f"  ✅ 示波器 ({SCOPE_CH}): {len(ch_files)} 个文件")


def get_file_idx(fpath):
    m = re.search(r'--Trace--(\d+)\.trc', os.path.basename(fpath))
    return int(m.group(1))


# ── 展开所有 segment ──
scope_segments = []  # list of dicts
for fp in ch_files:
    fi = get_file_idx(fp)
    try:
        info = parse_trc_binary(fp)
        ns = info['n_segments']
        off = info['trig_time_offsets']  # ndarray of shape (ns,)
        for si in range(ns):
            scope_segments.append({
                'file_idx': fi,
                'seg_idx': si,
                'global_idx': fi * ns + si,  # 全局序号
                'offset_s': float(off[si]),
            })
    except Exception as e:
        print(f"    ⚠️  {os.path.basename(fp)}: {e}")

scope_segments.sort(key=lambda x: x['global_idx'])
n_scope = len(scope_segments)
print(f"    展开后共 {n_scope} 个 segments")

# ═══════════════════════════════════════════════════════════
# 2. 数据量对比
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*66}")
print(f"  Step 2: 数据量对比")
print(f"{'='*66}")
print(f"  {'系统':<18} {'数量':<12} {'时间跨度(s)':<15}")
print(f"  {'-'*45}")
print(f"  {'Digitizer':<18} {n_digi:<12} {d_t_s[-1]-d_t_s[0]:<15.3f}")
print(f"  {'FPGA OSCI':<18} {n_osci:<12} {o_t_s[-1]-o_t_s[0]:<15.3f}")
print(f"  {f'示波器({SCOPE_CH})':<18} {n_scope:<12} {'(seq展开)':<15}")

# ═══════════════════════════════════════════════════════════
# 3. 示波器 ↔ FPGA 按序号匹配
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*66}")
print(f"  Step 3: 示波器 ↔ FPGA 序号匹配")
print(f"{'='*66}")
print(f"  FPGA trigger_id 从 {o_id[0]} 开始 (1-based)")
print(f"  示波器 global_idx 从 {scope_segments[0]['global_idx']} 开始 (0-based)")
print(f"  对应关系: FPGA trigger_id = scope_global_idx + 1")

# 示波器第 i 个 segment ↔ FPGA 第 i 个 OSCI 事件
min_n = min(n_scope, n_osci)
n_scope_extra = n_scope - n_osci
n_osci_extra = n_osci - n_scope
print(f"  可匹配: {min_n} 对")
if n_scope_extra > 0:
    print(f"  ⚠️ 示波器比 FPGA 多 {n_scope_extra} 个 (示波器可能记录了额外触发)")
if n_osci_extra > 0:
    print(f"  ⚠️ FPGA 比示波器多 {n_osci_extra} 个 (FPGA可能收到了噪声触发)")

# ── 验证匹配质量: 示波器 offset vs FPGA 时间 ──
if min_n > 1:
    n_fit = min(1000, min_n)
    sc_off = np.array([scope_segments[i]['offset_s'] for i in range(n_fit)])
    fpga_t = o_tick[:n_fit].astype(float) / FPGA_CLOCK_HZ
    # 减去各自的 t0 看相对时间差
    sc_rel = sc_off - sc_off[0]
    fpga_rel = fpga_t - fpga_t[0]
    dt_rel = sc_rel - fpga_rel
    print(f"\n  示波器 offset vs FPGA 时间 (前{n_fit}个, 均归零化):")
    print(f"    Δt 均值: {dt_rel.mean()*1e6:.3f} μs")
    print(f"    Δt 标准差: {dt_rel.std()*1e6:.3f} μs")
    print(f"    总漂移: {(dt_rel[-1]-dt_rel[0])*1e6:.1f} μs")

    # 线性拟合: 检测时钟漂移
    if n_fit > 10:
        coeff = np.polyfit(np.arange(n_fit), dt_rel*1e9, 1)  # ns
        print(f"    漂移率: {coeff[0]:.6f} ns/事件 ({coeff[0]*1e3:.3f} ps/事件)")
        if abs(coeff[0]) > 0.1:
            print(f"    ⚠️ 存在时钟漂移，需修正")

# ═══════════════════════════════════════════════════════════
# 4. Digitizer 时钟校准 (1Hz GSYNC 脉冲)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*66}")
print(f"  Step 4: Digitizer 时钟校准")
print(f"{'='*66}")

d_dt = np.diff(d_tt)
hz_th = int(0.5e9 / DIGI_NS)  # 0.5秒阈值
# ═══════════════════════════════════════════════════════════
# 4. Digitizer 时钟校准 (1Hz GSYNC 脉冲)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*66}")
print(f"  Step 4: Digitizer 时钟校准")
print(f"{'='*66}")

# 加载已完成的1Hz分类结果 (由 step1 + step2 生成)
corr_data = np.load(os.path.join(RUN_DIR, 'temp', 'corrected_timetags.npz'))
is_hz = corr_data['is_1hz']
tr_idx = np.where(~is_hz)[0]
tr_evnums = corr_data['ev_numbers'][~is_hz]
t_tr_ft = corr_data['t_corrected_fticks'][~is_hz].astype(np.float64)

print(f"  从 calibration 结果加载:")
print(f"    1Hz事件: {is_hz.sum()}, TR事件: {len(tr_idx)}")

# ═══════════════════════════════════════════════════════════
# 5. FPGA ↔ Digitizer Δt 峰值查找 (只用 TR 事件)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*66}")
print(f"  Step 5: FPGA ↔ Digitizer Δt 校准 (仅TR事件)")
print(f"{'='*66}")

# 对每个 Digi TR 事件, 找最近的 FPGA 事件
all_dt = []
n_trig = len(t_tr_ft)
for i in range(n_trig):
    td = t_tr_ft[i]
    # 在 FPGA 时间轴中定位
    nearest = int(np.searchsorted(o_tick, td))
    left = max(0, nearest - 50)
    right = min(n_osci, nearest + 50)
    for fi in range(left, right):
        dt = td - o_tick[fi]
        if abs(dt) < 50000:  # 只保留合理范围
            all_dt.append(dt)

all_dt = np.array(all_dt)
print(f"  收集到 {len(all_dt)} 个 Δt 值")

# 直方图找峰
if len(all_dt) > 0:
    bins = np.arange(-5000, 5001, 0.5)
    h, be = np.histogram(all_dt, bins=bins)
    bc = (be[:-1] + be[1:]) / 2
    pi = np.argmax(h)
    peak = bc[pi]

    near = all_dt[np.abs(all_dt - peak) < 50]
    sigma_est = np.std(near) if len(near) > 10 else 1.0
    match_win = 5 * sigma_est
    print(f"  Δt = t_dig_corr - t_fpga 峰值: {peak*5:.1f} ns ({peak:.1f} ticks)")
    print(f"  σ: {sigma_est*5:.1f} ns")
    print(f"  匹配窗口: ±{match_win*5:.1f} ns")
else:
    print("  ❌ 无有效 Δt 值!")
    peak, match_win = 0.0, 50

# ═══════════════════════════════════════════════════════════
# 6. Digi ↔ FPGA 匹配
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*66}")
print(f"  Step 6: Digi ↔ FPGA 事件匹配")
print(f"{'='*66}")

# 为每个 Digi TR 事件找最近的 FPGA 事件
matched = []
used_fpga = set()
for i in range(len(t_tr_ft)):
    td = t_tr_ft[i]
    tgt = td - peak  # 预期的 FPGA 时间
    nearest = int(np.searchsorted(o_tick, tgt))
    for fi in range(max(0, nearest - 10), min(n_osci, nearest + 10)):
        if fi in used_fpga:
            continue
        dt = td - o_tick[fi]
        if abs(dt - peak) < match_win:
            matched.append({
                'digi_evnum': int(tr_evnums[i]),
                'fpga_idx': fi,
                'fpga_id': int(o_id[fi]),
                'dt_s': dt * 5e-9,
            })
            used_fpga.add(fi)
            break

n_matched = len(matched)
print(f"  匹配: {n_matched}/{n_trig} Digi ({n_matched/max(n_trig,1)*100:.1f}%)")
if matched:
    dt_arr = np.array([m['dt_s'] for m in matched])
    print(f"  Δt 均值: {np.mean(dt_arr)*1e9:.1f} ns")
    print(f"  Δt 标准差: {np.std(dt_arr)*1e9:.1f} ns")

# 直方图找峰
h = np.histogram(deltas, bins=300, range=(-0.3, 0.3))
bc = (h[1][:-1] + h[1][1:]) / 2
pi = np.argmax(h[0])
peak = bc[pi]
print(f"  Δt = t_FPGA - t_Digi 峰值: {peak*1e6:.2f} μs")

# 估算 σ: 从峰值附近 ±100 μs 的数据
near_mask = np.abs(deltas - peak) < 100e-6
near_vals = deltas[near_mask]
sigma_est = np.std(near_vals) if len(near_vals) > 10 else 5e-6
match_win = 5 * sigma_est
print(f"  σ: {sigma_est*1e6:.2f} μs")
print(f"  匹配窗口: ±{match_win*1e6:.2f} μs")

# ═══════════════════════════════════════════════════════════
# 6. 三系统联合匹配
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*66}")
print(f"  Step 6: 三系统联合匹配")
print(f"{'='*66}")

# 为 FPGA 第 i 个事件匹配 Digitizer 事件
# t_digi_expected = t_fpga - peak
d_t_all = d_tt.astype(float) * DIGI_NS * 1e-9 / correction

matched = []
for i in range(min_n):
    fpga_t = o_tick[i].astype(float) / FPGA_CLOCK_HZ
    exp_digi = fpga_t - peak
    best = np.argmin(np.abs(d_t_all - exp_digi))
    dt = d_t_all[best] - exp_digi
    if abs(dt) < match_win:
        matched.append({
            'scope_global_idx': scope_segments[i]['global_idx'],
            'scope_file_idx': scope_segments[i]['file_idx'],
            'scope_seg_idx': scope_segments[i]['seg_idx'],
            'scope_offset_s': scope_segments[i]['offset_s'],
            'fpga_idx': i,
            'fpga_id': int(o_id[i]),
            'fpga_t_s': fpga_t,
            'digi_idx': best,
            'digi_event': int(d_num[best]),
            'digi_t_s': d_t_all[best],
            'dt_s': dt,
        })

n_matched = len(matched)
print(f"  示波器-FPGA 按序号: {min_n} 对")
print(f"  三系统匹配: {n_matched} 对 ({n_matched/min_n*100:.1f}%)")
if matched:
    dts = np.array([m['dt_s'] for m in matched])
    print(f"  Δt 均值: {np.mean(dts)*1e6:.2f} μs")
    print(f"  Δt 标准差: {np.std(dts)*1e6:.2f} μs")

# ═══════════════════════════════════════════════════════════
# 7. 可视化
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*66}")
print(f"  Step 7: 可视化")
print(f"{'='*66}")

# ── 图1: Δt 分布 ──
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.hist(deltas*1e6, bins=200, range=(-300, 300), color='steelblue', alpha=0.8,
        label=f'取样 {n_samp}×{n_samp} 组合')
ax.axvline(x=peak*1e6, color='r', lw=2, ls='--', label=f'峰值 {peak*1e6:.1f} μs')
ax.axvline(x=(peak-match_win)*1e6, color='orange',
           ls=':', label=f'匹配窗口 ±{match_win*1e6:.0f} μs')
ax.axvline(x=(peak+match_win)*1e6, color='orange', ls=':')
ax.set_xlabel('Δt = t_FPGA - t_Digitizer (μs)')
ax.set_ylabel('Count')
ax.set_title('FPGA vs Digitizer Δt 分布 (所有组合)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, 'match_deltahist.png')

# ── 图2: Δt 放大 ──
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.hist(deltas*1e6, bins=200, range=(-50, 50), color='steelblue', alpha=0.8)
ax.axvline(x=peak*1e6, color='r', lw=2, ls='--')
ax.axvline(x=(peak-match_win)*1e6, color='orange', ls=':')
ax.axvline(x=(peak+match_win)*1e6, color='orange', ls=':')
ax.set_xlabel('Δt = t_FPGA - t_Digitizer (μs)')
ax.set_ylabel('Count')
ax.set_title('Δt 分布 (放大, ±50 μs)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, 'match_deltahist_zoom.png')

# ── 图3: 匹配验证 ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
ax = axes[0]
if matched:
    dts = np.array([m['dt_s'] for m in matched]) * 1e6
    idx = np.arange(len(matched))
    ax.plot(idx, dts, 'g.', markersize=0.8, alpha=0.4)
    ax.axhline(y=0, color='r', ls='--')
    ax.axhline(y=np.std(dts)*3, color='orange', ls=':',
               label=f'±3σ={np.std(dts)*3:.1f}μs')
    ax.axhline(y=-np.std(dts)*3, color='orange', ls=':')
    ax.set_xlabel('Match index')
    ax.set_ylabel('Δt (μs)')
    ax.set_title(f'匹配 Δt vs 序号 (n={n_matched}, σ={np.std(dts):.2f} μs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

ax = axes[1]
if matched:
    dts = np.array([m['dt_s'] for m in matched]) * 1e6
    ax.hist(dts, bins=80, color='green', alpha=0.7, edgecolor='k', lw=0.5)
    ax.axvline(x=0, color='r', ls='--')
    ax.set_xlabel('Δt (μs)')
    ax.set_ylabel('Count')
    ax.set_title(f'匹配 Δt 直方图 (μ={np.mean(dts):.2f}, σ={np.std(dts):.2f}) μs')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, 'match_validation.png')

# ── 图4: 数据量对比 ──
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
labels = ['Digitizer', 'FPGA OSCI', f'示波器({SCOPE_CH})', '三系统匹配']
counts = [n_digi, n_osci, n_scope, n_matched]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='k', lw=0.5)
for b, c in zip(bars, counts):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(counts)*0.01,
            str(c), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Events')
ax.set_title('三系统数据量对比')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
savefig(fig, 'match_summary.png')

# ── 图5: 示波器 offset vs FPGA 时间 ──
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
if min_n > 1:
    n_plt = min(2000, min_n)
    sc_off = np.array([scope_segments[i]['offset_s'] for i in range(n_plt)])
    fpga_t_prof = o_tick[:n_plt].astype(float) / FPGA_CLOCK_HZ
    sc_rel = sc_off - sc_off[0]
    fpga_rel = fpga_t_prof - fpga_t_prof[0]

    ax = axes[0]
    ax.plot(sc_rel*1e6, 'b-', lw=0.5, alpha=0.7, label='示波器 offset')
    ax.plot(fpga_rel*1e6, 'r-', lw=0.5, alpha=0.7, label='FPGA 时间')
    ax.set_ylabel('相对时间 (μs)')
    ax.set_title(f'示波器 trigger_offset vs FPGA 时间 (前{n_plt})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    dt_vs = (sc_rel - fpga_rel) * 1e9  # ns
    ax.plot(dt_vs, 'g-', lw=0.5, alpha=0.7)
    ax.axhline(y=0, color='r', ls='--')
    ax.set_xlabel('Event index')
    ax.set_ylabel('时间差 (ns)')
    ax.set_title(f'差: 示波器 - FPGA (前{n_plt}, σ={np.std(dt_vs):.2f} ns)')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, 'match_scope_vs_fpga.png')

# ═══════════════════════════════════════════════════════════
# 8. 输出结果
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*66}")
print(f"  Step 8: 结果输出")
print(f"{'='*66}")

csv_path = os.path.join(OUT_DIR, 'matched_events.csv')
if matched:
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'scope_global_idx', 'scope_file_idx', 'scope_seg_idx', 'scope_offset_s',
            'fpga_idx', 'fpga_id', 'fpga_t_s',
            'digi_idx', 'digi_event', 'digi_t_s', 'dt_s'])
        w.writeheader()
        w.writerows(matched)
    print(f"  ✅ CSV: {csv_path} ({n_matched} rows)")

# 摘要
print(f"\n{'='*66}")
print(f"  📊 匹配摘要")
print(f"{'='*66}")
print(f"  Digitizer:        {n_digi:>8}")
print(f"  FPGA OSCI:        {n_osci:>8}")
print(f"  示波器({SCOPE_CH}):        {n_scope:>8} (展开后)")
print(f"  {'─'*30}")
print(f"  三系统匹配:       {n_matched:>8} ({n_matched/min_n*100:>5.1f}%)")
if matched:
    dts_a = np.array([m['dt_s'] for m in matched])
    print(f"  匹配精度 (σ):     {np.std(dts_a)*1e6:.2f} μs")
print(f"{'='*66}")
