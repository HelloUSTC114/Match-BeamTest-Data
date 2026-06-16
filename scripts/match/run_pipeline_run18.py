"""
run_0018 三系统匹配流水线
=========================
Digitizer (V1742) ↔ FPGA ↔ 示波器 (LeCroy C3-C8)

Step 1: 分离 1Hz 校准脉冲 vs 物理触发
Step 2: DRS4 → FPGA 时钟域标定
Step 3: Digitizer ↔ FPGA 时间戳匹配（Δt 直方图）
Step 4: 示波器 Sequence 展开 + Scope ↔ FPGA 匹配
Step 5: 生成完整三系统事件表
"""

import sys, os, numpy as np, re, glob, warnings, csv, gc
from pathlib import Path
from datetime import datetime
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ─── 路径设置 ───
WORKSPACE = r'd:\Codes\Analyze'
PROJ = os.path.join(WORKSPACE, 'beamtest-control-master', 'beamtest-control-master')
SCRIPTS = os.path.join(WORKSPACE, 'scripts')
sys.path.insert(0, PROJ)
sys.path.insert(0, SCRIPTS)

from src.data.raw_data_recorder import load_binary_events
from src.data.fpga_parser import load_fpga_events, FPGA_CLOCK_HZ

import lecroyparser
import importlib.util
spec = importlib.util.spec_from_file_location("awm", os.path.join(SCRIPTS, "analyze_lecroy_wfm.py"))
awm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awm)
parse_trc = awm.parse_trc_binary

# ─── 运行配置 ───
RUN = os.path.join(WORKSPACE, 'data', 'BT_20260610_094205-run_0018')
TEMP = os.path.join(RUN, 'temp')
OUT = os.path.join(WORKSPACE, 'plots')
os.makedirs(TEMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

DIGI_NS = 8.5          # V1742 时钟周期 (ns)
FPGA_1HZ = 200_000_020 # 1秒对应 FPGA tick 数
SCOPE_CH = 'C4'        # 用于匹配的主示波器通道

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    print(f"  ✅ 已保存: {path}")
    plt.close(fig)
    gc.collect()

# ═══════════════════════════════════════════
print("=" * 60)
print("  run_0018 三系统匹配流水线")
print("=" * 60)

# ═══ Step 1: 加载 Digitizer 数据 + 分离 1Hz ═══
print("\n" + "=" * 60)
print("Step 1: 加载 Digitizer + 分离 1Hz vs 物理触发")
print("=" * 60)

digi_path = os.path.join(RUN, 'digitizer', 'V1742_events.bin')
digi = load_binary_events(digi_path)
n_digi = len(digi)
print(f"  Digitizer 事件数: {n_digi}")

d_num = np.array([e['event_number'] for e in digi], dtype=np.int64)
d_tt  = np.array([e['trigger_time_tag'] for e in digi], dtype=np.int64)
d_et  = np.array([e.get('event_time_tag', 0) for e in digi], dtype=np.int64)

# 分离 1Hz 脉冲 vs 物理触发：TR 通道 (32-35) 尾部-头部差异
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

# 无 TR 通道的事件也视为 1Hz（可能是纯模拟通道触发）
is_hz = np.full(n_digi, False)
is_hz[has_tr] = np.nanmax(td[has_tr], axis=1) < 5.0
is_hz[~has_tr] = True  # 无 TR 通道 → 视为 1Hz

n_hz = is_hz.sum()
n_tr = n_digi - n_hz
print(f"  1Hz 事件: {n_hz}, 物理触发: {n_tr}")

fig, ax = plt.subplots(figsize=(10, 5))
valid_td = td[has_tr]
ax.hist(np.nanmax(valid_td, axis=1), bins=100, color='steelblue', alpha=0.8)
ax.axvline(5.0, color='r', ls='--', label='threshold=5.0 ADC')
ax.set_xlabel('max |tail-head| (ADC)')
ax.set_ylabel('Count')
ax.set_title(f'TR tail-baseline diff (1Hz={n_hz}, TR={n_tr})')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); savefig(fig, 'run0018_step1_classify.png')

# ═══ Step 2: 时钟标定 (DRS4 → FPGA tick 域) ═══
print("\n" + "=" * 60)
print("Step 2: 时钟标定 (DRS4 → FPGA)")
print("=" * 60)

hz_idx = np.where(is_hz)[0]
if len(hz_idx) < 2:
    print("  ⚠️ 1Hz 事件不足，无法标定！尝试用所有事件做近似标定...")
    hz_idx = np.arange(n_digi)

tt_1hz = d_tt[hz_idx]
theory_ft = np.zeros(len(hz_idx), dtype=np.float64)
cum = 0.0
NOM_1S = 1e9 / DIGI_NS
for i in range(len(hz_idx)):
    n_s = 1 if i == 0 else max(1, round((tt_1hz[i] - tt_1hz[i - 1]) / NOM_1S))
    cum += n_s * FPGA_1HZ
    theory_ft[i] = cum

dt_ft = np.diff(theory_ft)
dt_cyc = np.diff(tt_1hz.astype(np.float64))
f_pd = dt_ft / np.maximum(dt_cyc, 1)
f_pd = np.insert(f_pd, 0, np.mean(f_pd[:5]) if len(f_pd) >= 5 else f_pd[0])

def conv(t):
    s = int(np.searchsorted(tt_1hz, t, side='right')) - 1
    s = max(0, min(len(tt_1hz) - 2, s))
    return theory_ft[s] + (float(t) - float(tt_1hz[s])) * f_pd[s]

t_ft = np.array([conv(int(t)) for t in d_tt])
err = np.max(np.abs((t_ft[hz_idx] - t_ft[hz_idx[0]]) - (theory_ft - theory_ft[0])))
print(f"  1Hz 标定最大误差: {err:.0f} FPGA ticks")
print(f"  平均 f_pd (FPGA ticks / DRS4 cycle): {np.mean(f_pd):.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theory_ft / 1e6, tt_1hz / 1e6, 'o-', color='steelblue', ms=3, lw=1)
ax.set_xlabel('Theory FPGA tick (M)')
ax.set_ylabel('Digi raw TT (M)')
ax.set_title('1Hz events: theory FPGA tick vs DRS4 raw TT')
ax.grid(True, alpha=0.3)
plt.tight_layout(); savefig(fig, 'run0018_step2_calibration.png')

tr_idx = np.where(~is_hz)[0]
np.savez_compressed(os.path.join(TEMP, 'corrected_timetags.npz'),
    tt_tags=d_tt, ev_numbers=d_num, is_1hz=is_hz, t_corrected_fticks=t_ft,
    idx_1hz=hz_idx, idx_trig=tr_idx, f_ticks_per_drs4=f_pd)
print(f"  标定数据已保存: {TEMP}/corrected_timetags.npz")

# ═══ Step 3: Digitizer ↔ FPGA 时间匹配 ═══
print("\n" + "=" * 60)
print("Step 3: Digitizer ↔ FPGA 时间戳匹配")
print("=" * 60)

fpga_path = os.path.join(RUN, 'fpga', 'fpga_events.bin')
fpga = load_fpga_events(fpga_path)
n_fpga = len(fpga)
f_id = np.array([e.trigger_id for e in fpga], dtype=np.int64)
f_tick = np.array([e.t_fpga for e in fpga], dtype=np.int64)
print(f"  FPGA 事件数: {n_fpga}")

t_tr = t_ft[tr_idx]
ev_tr = d_num[tr_idx]
n_trig = len(t_tr)

# Δt 分布计算（限定搜索窗口避免组合爆炸）
all_dt = []
for i in range(n_trig):
    ni = int(np.searchsorted(f_tick, t_tr[i]))
    lo = max(0, ni - 30)
    hi = min(n_fpga, ni + 30)
    for fi in range(lo, hi):
        all_dt.append(t_tr[i] - f_tick[fi])
all_dt = np.array(all_dt)

h, be = np.histogram(all_dt, bins=np.arange(-5000, 5001, 0.5))
bc = (be[:-1] + be[1:]) / 2
offset = bc[np.argmax(h)]
near = all_dt[np.abs(all_dt - offset) < 50]
sigma = np.std(near) if len(near) > 10 else 5.0
print(f"  Δt 峰值偏移: {offset:.1f} FPGA ticks ({offset*5:.1f} ns)")
print(f"  σ = {sigma:.1f} ticks ({sigma*5:.1f} ns)")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.hist(all_dt * 5, bins=200, range=(-500, 500), color='steelblue', alpha=0.8)
ax1.axvline(x=offset * 5, color='r', lw=2, ls='--', label=f'peak={offset*5:.0f}ns')
ax1.set_xlabel('Δt (ns)')
ax1.set_ylabel('Count')
ax1.set_title(f'Δt 分布 ({len(all_dt):,} 对)')
ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.hist(near * 5, bins=50, color='green', alpha=0.8)
ax2.axvline(x=offset * 5, color='r', lw=2, ls='--')
ax2.set_xlabel('Δt near peak (ns)')
ax2.set_ylabel('Count')
ax2.set_title(f'峰值放大 (n={len(near)}, sigma={sigma*5:.1f}ns)')
ax2.grid(True, alpha=0.3)
plt.tight_layout(); savefig(fig, 'run0018_step3_deltahist.png')

# 一对一匹配
mw = 3 * sigma
used = set()
matched = []
for i in range(n_trig):
    td_i = t_tr[i]
    tgt = td_i - offset
    ni = int(np.searchsorted(f_tick, tgt))
    for fi in range(max(0, ni - 15), min(n_fpga, ni + 15)):
        if fi in used:
            continue
        if abs(td_i - f_tick[fi] - offset) < mw:
            matched.append((int(ev_tr[i]), int(f_id[fi]), float(td_i - f_tick[fi])))
            used.add(fi)
            break
n_m = len(matched)
match_rate_digi = n_m / max(n_trig, 1) * 100
print(f"  Digi→FPGA 匹配: {n_m}/{n_trig} ({match_rate_digi:.1f}%)")

if n_m > 0:
    dts_a = np.array([m[2] for m in matched]) * 5  # ns
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dts_a, 'g.', ms=2, alpha=0.5)
    ax.axhline(offset * 5, color='r', ls='--', label=f'offset={offset*5:.0f}ns')
    ax.axhline(np.std(dts_a) * 3, color='orange', ls=':', label=f'3σ={np.std(dts_a)*3:.0f}ns')
    ax.set_xlabel('Match index')
    ax.set_ylabel('Δt (ns)')
    ax.set_title(f'Digi-FPGA 匹配残差 (n={n_m}, σ={np.std(dts_a):.1f}ns)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); savefig(fig, 'run0018_step3_match_residual.png')

arr = np.array(matched, dtype=[('digi_evnum', 'i4'), ('fpga_id', 'i4'), ('dt', 'f4')])
np.savez(os.path.join(TEMP, 'matched_pairs.npz'), pairs=arr, offset_fticks=offset, sigma_fticks=sigma)

# ═══ Step 4: 示波器 ↔ FPGA 匹配 ═══
print("\n" + "=" * 60)
print(f"Step 4: 示波器 {SCOPE_CH} ↔ FPGA 匹配")
print("=" * 60)

ch_pattern = os.path.join(RUN, 'lecroy_wfm', f'{SCOPE_CH}--Trace--*.trc')
ch_files = sorted(glob.glob(ch_pattern))
print(f"  示波器 {SCOPE_CH} 文件数: {len(ch_files)}")

scope_data = []
for fp in ch_files:
    file_idx = int(re.search(r'--Trace--(\d+)\.trc', os.path.basename(fp)).group(1))
    s = lecroyparser.ScopeData(path=fp)
    try:
        file_utc = datetime.strptime(s.triggerTime, '%Y-%m-%d %H:%M:%S.%f').timestamp()
    except:
        file_utc = datetime.strptime(s.triggerTime, '%Y-%m-%d %H:%M:%S').timestamp()
    info = parse_trc(fp)
    for si, off in enumerate(info['trig_time_offsets']):
        scope_data.append({
            'file_idx': file_idx,
            'seg_idx': si,
            'utc': file_utc + float(off)
        })

scope_utc = np.array([s['utc'] for s in scope_data])
scope_rel = scope_utc - scope_utc[0]
n_scope = len(scope_data)
print(f"  示波器 {SCOPE_CH} 总段数: {n_scope}")

# 检测 Spill 结构（>50ms 间隙）
s_dt = np.diff(scope_rel) * 1000
s_ed = np.where(s_dt > 50)[0]
s_sp = np.zeros(n_scope, dtype=int)
for e in s_ed:
    s_sp[e + 1:] += 1

f_dt = np.diff(f_tick).astype(float) / FPGA_CLOCK_HZ * 1000
f_ed = np.where(f_dt > 50)[0]
f_sp = np.zeros(n_fpga, dtype=int)
for e in f_ed:
    f_sp[e + 1:] += 1

n_spills_fpga = f_sp.max() + 1
n_spills_scope = s_sp.max() + 1
print(f"  FPGA Spill 数: {n_spills_fpga}")
print(f"  示波器 {SCOPE_CH} Spill 数: {n_spills_scope}")

# 过滤小 spill（事件数 > 10）
f_large = sorted([sp for sp in range(n_spills_fpga) if np.sum(f_sp == sp) > 10])
s_large = sorted([sp for sp in range(n_spills_scope) if np.sum(s_sp == sp) > 10])
f_cnts = [np.sum(f_sp == sp) for sp in f_large]
s_cnts = [np.sum(s_sp == sp) for sp in s_large]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.bar(range(len(f_cnts)), f_cnts, color='red', alpha=0.7, label=f'FPGA ({len(f_cnts)} spills)')
for i, c in enumerate(f_cnts):
    ax1.text(i, c + max(f_cnts)*0.02, str(c), ha='center', fontsize=7)
ax1.set_ylabel('Events/spill')
ax1.set_title('FPGA 各 Spill 事件数')
ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')
ax2.bar(range(len(s_cnts)), s_cnts, color='blue', alpha=0.7, label=f'Scope {SCOPE_CH} ({len(s_cnts)} spills)')
for i, c in enumerate(s_cnts):
    ax2.text(i, c + max(s_cnts)*0.02, str(c), ha='center', fontsize=7)
ax2.set_xlabel('Spill index')
ax2.set_ylabel('Events/spill')
ax2.set_title(f'示波器 {SCOPE_CH} 各 Spill 段数')
ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); savefig(fig, 'run0018_step4_spill_counts.png')

# RMS 扫描找最佳 spill 偏移
f_arr = np.array(f_cnts, dtype=float)
s_arr = np.array(s_cnts, dtype=float)
best_off, best_rms = 0, float('inf')
for off in range(-len(f_cnts) + 1, len(s_cnts)):
    if off >= 0:
        fs = f_arr[off:]
        ss = s_arr[:len(f_arr) - off]
    else:
        fs = f_arr[:len(f_arr) + off]
        ss = s_arr[-off:]
    no = min(len(fs), len(ss))
    if no < 3:
        continue
    rms = np.sqrt(np.mean((fs[:no] - ss[:no]) ** 2))
    if rms < best_rms:
        best_rms, best_off = rms, off

print(f"  最佳 Spill 偏移: {best_off:+d}, RMS={best_rms:.1f}")

off_range = max(10, len(f_cnts) // 2)
offs_scan = list(range(-off_range, off_range + 1))
rms_scan = []
for off in offs_scan:
    if off >= 0:
        fs = f_arr[off:]
        ss = s_arr[:len(f_arr) - off]
    else:
        fs = f_arr[:len(f_arr) + off]
        ss = s_arr[-off:]
    n_o = min(len(fs), len(ss))
    rms_scan.append(np.sqrt(np.mean((fs[:n_o] - ss[:n_o]) ** 2)) if n_o >= 3 else 1e9)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(offs_scan, rms_scan, color='steelblue', alpha=0.8)
ax.bar(best_off, best_rms, color='red', alpha=0.8, label=f'Best={best_off:+d}, RMS={best_rms:.0f}')
ax.set_xlabel('Spill offset')
ax.set_ylabel('RMS of spill counts')
ax.set_title(f'Spill Count RMS Scan (best offset={best_off:+d})')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); savefig(fig, 'run0018_step4_spill_rms.png')

# 按最佳偏移配对 spill → 内部事件配对
all_sp = []
for i in range(min(len(f_large) - max(0, best_off), len(s_large) - max(0, -best_off))):
    fsp = f_large[i + max(0, best_off)]
    ssp = s_large[i + max(0, -best_off)]
    fn = np.sum(f_sp == fsp)
    sn = np.sum(s_sp == ssp)
    if fn != sn:
        continue  # 不配对数量不一致的 spill
    f_gi = np.where(f_sp == fsp)[0]
    s_gi = np.where(s_sp == ssp)[0]
    nv = min(100, len(f_gi), len(s_gi))
    ftv = f_tick[f_gi[:nv]].astype(float) / FPGA_CLOCK_HZ
    stv = scope_rel[s_gi[:nv]]
    rms_v = np.sqrt(np.mean(((ftv - ftv[0]) - (stv - stv[0])) ** 2)) * 1e6
    status = 'OK' if rms_v < 10 else f'REJECT(RMS={rms_v:.1f}us)'
    if rms_v < 10:
        for fi, si in zip(f_gi, s_gi):
            all_sp.append((int(fi), scope_data[si]['file_idx'], scope_data[si]['seg_idx']))

n_sp = len(all_sp)
print(f"  Scope-FPGA 匹配: {n_sp} 对")

# ═══ Step 5: 完整事件表 ═══
print("\n" + "=" * 60)
print("Step 5: 生成完整三系统事件表")
print("=" * 60)

fs_map = {p[0]: {'file_idx': p[1], 'seg_idx': p[2]} for p in all_sp}
fd_map = {}
for m in matched:
    fd_map[m[1]] = {'digi_evnum': m[0], 'dt_ns': m[2] * 5}  # ticks→ns

out_csv = os.path.join(TEMP, 'full_event_table.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['fpga_trigger_id', 'fpga_tick', 'has_scope', 'scope_file_idx',
                'scope_seg_idx', 'has_digi', 'digi_evnum', 'digi_tt_raw',
                'digi_is_1hz', 'dt_ns'])
    for fi in range(n_fpga):
        fid = f_id[fi]
        s = fs_map.get(fi, None)
        d = fd_map.get(fid, None)
        w.writerow([
            fid, f_tick[fi],
            1 if s else 0, s['file_idx'] if s else -1, s['seg_idx'] if s else -1,
            1 if d else 0, d['digi_evnum'] if d else -1,
            d_tt[d['digi_evnum']] if d and d['digi_evnum'] < n_digi else -1,
            int(is_hz[d['digi_evnum']]) if d and d['digi_evnum'] < n_digi else -1,
            d['dt_ns'] if d else np.nan
        ])

n_s = len(fs_map)
n_d = len(matched)
both = sum(1 for fi in range(n_fpga) if fi in fs_map and f_id[fi] in fd_map)

# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("  匹配结果汇总")
print("=" * 60)
print(f"  Digitizer 总事件:    {n_digi:>8}")
print(f"    其中 1Hz 脉冲:     {n_hz:>8}")
print(f"    其中 物理触发:     {n_tr:>8}")
print(f"  FPGA 总事件:         {n_fpga:>8}")
print(f"  示波器 C4 总段数:    {n_scope:>8}")
print(f"")
print(f"  示波器 Spill 数:     {len(s_large):>8}")
print(f"  FPGA Spill 数:       {len(f_large):>8}")
print(f"  最佳 Spill 偏移:     {best_off:>+8}")
print(f"")
print(f"  Digi→FPGA 匹配:      {n_m:>8} / {n_trig} ({match_rate_digi:.1f}%)")
print(f"  Scope→FPGA 匹配:     {n_sp:>8}")
print(f"  三系统同时匹配:      {both:>8} ({both/max(n_fpga,1)*100:.1f}%)")
print(f"")
print(f"  事件表已保存: {out_csv}")
print(f"  图片已保存:   {OUT}/run0018_*.png")
print("=" * 60)
