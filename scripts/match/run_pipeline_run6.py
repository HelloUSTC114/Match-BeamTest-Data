"""
run_pipeline_run6.py — 三系统匹配流水线 (Run 6)
=================================================
基于 run_pipeline.py 改编，数据源改为 BT_20260530_192428_run_0006

Step 1: 分离1Hz/TR
Step 2: 时钟标定 (DRS4 → FPGA tick域)
Step 3: Digi ↔ FPGA 匹配 (DeltaT直方图+逐事件)
Step 4: 示波器 Sequence 展开 + Scope ↔ FPGA (spill count RMS)
Step 5: Scope ↔ FPGA Δt 直方图
Step 6: 已配对 spill 逐事件时间线
Step 7: 三系统综合 Dashboard
Step 8: 三系统完整事件表
"""

from src.data.fpga_parser import load_fpga_events, FPGA_CLOCK_HZ
from src.data.raw_data_recorder import load_binary_events
import matplotlib.pyplot as plt
import importlib.util
import lecroyparser
import sys
import os
import numpy as np
import re
import glob
import warnings
import csv
import gc
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# ── 仓库根目录 (Match-BeamTest-Data) ──
REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS = str(REPO / 'scripts')
sys.path.insert(0, str(REPO))

# 导入 analyze_lecroy_wfm 中的 parse_trc_binary
spec = importlib.util.spec_from_file_location(
    "awm", os.path.join(SCRIPTS, "analyze_lecroy_wfm.py"))
awm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awm)
parse_trc = awm.parse_trc_binary

# ═══ 配置: 修改 RUN 指向你的数据目录 ═══
RUN = str(REPO.parent / 'Analyze' / 'data' / 'BT_20260530_192428_run_0006')
TEMP = os.path.join(RUN, 'temp')
OUT = str(REPO / 'plots')
os.makedirs(TEMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

DIGI_NS = 8.5
FPGA_1HZ = 200_000_020  # 硬件实测值


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches='tight')
    plt.close(fig)
    gc.collect()


# ═══ Step 1: 分离1Hz ═══
print("=" * 60)
print("Step 1: 分离 1Hz vs TR")
print("=" * 60)
digi_bin = os.path.join(RUN, 'digitizer', 'V1742_events.bin')
if not os.path.exists(digi_bin):
    print(f"❌ 未找到 Digitizer 数据: {digi_bin}")
    sys.exit(1)

digi = load_binary_events(digi_bin)
n_digi = len(digi)
d_num = np.array([e['event_number'] for e in digi], dtype=np.int64)
d_tt = np.array([e['trigger_time_tag'] for e in digi], dtype=np.int64)
d_et = np.array([e.get('event_time_tag', 0) for e in digi], dtype=np.int64)

TR_CH, N_HEAD, N_TAIL = [32, 33, 34, 35], 20, 50
td = np.full((n_digi, 4), np.nan)
for i, e in enumerate(digi):
    for j, ch in enumerate(TR_CH):
        if ch not in e['waveforms']:
            continue
        d = e['waveforms'][ch]
        ns = len(d)
        h = min(N_HEAD, ns)
        tl = min(N_TAIL, ns)
        td[i, j] = abs(np.mean(d[:h]) - np.mean(d[-tl:]))
is_hz = np.nanmax(td, axis=1) < 5.0
n_hz, n_tr = is_hz.sum(), n_digi - is_hz.sum()
print(f"  Digitizer: {n_digi} events, 1Hz={n_hz}, TR={n_tr}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(np.nanmax(td, axis=1), bins=100, color='steelblue', alpha=0.8)
ax.axvline(5.0, color='r', ls='--', label='threshold=5.0')
ax.set_xlabel('max |tail-head| (ADC)')
ax.set_ylabel('Count')
ax.set_title(f'Run6 TR tail-baseline diff (1Hz={n_hz}, TR={n_tr})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, 'run6_step1_classify.png')

# ═══ Step 2: 时钟标定 ═══
print("\n" + "=" * 60)
print("Step 2: 时钟标定")
print("=" * 60)
hz_idx = np.where(is_hz)[0]
tt_1hz = d_tt[hz_idx]
theory_ft = np.zeros(len(hz_idx))
cum = 0.0
NOM_1S = 1e9 / DIGI_NS
for i in range(len(hz_idx)):
    n_s = 1 if i == 0 else max(1, round((tt_1hz[i] - tt_1hz[i - 1]) / NOM_1S))
    cum += n_s * FPGA_1HZ
    theory_ft[i] = cum
dt_ft = np.diff(theory_ft)
dt_cyc = np.diff(tt_1hz)
f_pd = dt_ft.astype(float) / dt_cyc
f_pd[0] = np.mean(f_pd[1:])


def conv(t):
    s = int(np.searchsorted(tt_1hz, t, side='right')) - 1
    s = max(0, min(len(tt_1hz) - 2, s))
    return theory_ft[s] + (t - tt_1hz[s]) * f_pd[s]


t_ft = np.array([conv(int(t)) for t in d_tt])
err = np.max(
    np.abs((t_ft[hz_idx] - t_ft[hz_idx[0]]) - (theory_ft - theory_ft[0])))
print(f"  1Hz max error: {err:.0f} ticks, avg f_pd={np.mean(f_pd):.6f}")

tr_idx = np.where(~is_hz)[0]
np.savez_compressed(os.path.join(TEMP, 'corrected_timetags.npz'),
                    tt_tags=d_tt, ev_numbers=d_num, is_1hz=is_hz, t_corrected_fticks=t_ft,
                    idx_1hz=hz_idx, idx_trig=tr_idx, f_ticks_per_drs4=f_pd)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theory_ft / 1e6, tt_1hz / 1e6, 'o-', color='steelblue', ms=4, lw=1)
ax.set_xlabel('Theory FPGA tick (M)')
ax.set_ylabel('Digi raw TT (M)')
ax.set_title('Run6 1Hz events: theory_ft vs TT_raw')
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, 'run6_step2_calibration.png')

# ═══ Step 3: Digi ↔ FPGA ═══
print("\n" + "=" * 60)
print("Step 3: Digi ↔ FPGA")
print("=" * 60)
fpga_bin = os.path.join(RUN, 'fpga', 'fpga_events.bin')
if not os.path.exists(fpga_bin):
    print(f"❌ 未找到 FPGA 数据: {fpga_bin}")
    sys.exit(1)

fpga = load_fpga_events(fpga_bin)
f_id = np.array([e.trigger_id for e in fpga], dtype=np.int64)
f_tick = np.array([e.t_fpga for e in fpga], dtype=np.int64)
n_fpga = len(f_id)

t_tr = t_ft[tr_idx]
ev_tr = d_num[tr_idx]
n_trig = len(t_tr)

all_dt = np.array([t_tr[i] - f_tick[fi]
                   for i in range(n_trig)
                   for fi in range(max(0, int(np.searchsorted(f_tick, t_tr[i])) - 30),
                                   min(n_fpga, int(np.searchsorted(f_tick, t_tr[i])) + 30))])

h, be = np.histogram(all_dt, bins=np.arange(-5000, 5001, 0.5))
bc = (be[:-1] + be[1:]) / 2
offset = bc[np.argmax(h)]
near = all_dt[np.abs(all_dt - offset) < 50]
sigma = np.std(near) if len(near) > 10 else 5.0
print(
    f"  Offset: {offset:.1f} ticks = {offset*5:.0f}ns, sigma={sigma:.1f} ticks = {sigma*5:.0f}ns")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.hist(all_dt * 5, bins=200, range=(-500, 500), color='steelblue', alpha=0.8)
ax1.axvline(x=offset * 5, color='r', lw=2, ls='--',
            label=f'peak={offset*5:.0f}ns')
ax1.set_xlabel('Dt (ns)')
ax1.set_ylabel('Count')
ax1.set_title(f'Run6 Dt distribution ({len(all_dt)} pairs)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.hist(near * 5, bins=50, color='green', alpha=0.8)
ax2.axvline(x=offset * 5, color='r', lw=2, ls='--')
ax2.set_xlabel('Dt near peak (ns)')
ax2.set_ylabel('Count')
ax2.set_title(f'Run6 Peak zoom (n={len(near)}, sigma={sigma*5:.1f}ns)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, 'run6_step3_deltahist.png')

mw = 3 * sigma
used = set()
matched = []
for i in range(n_trig):
    td = t_tr[i]
    tgt = td - offset
    ni = int(np.searchsorted(f_tick, tgt))
    for fi in range(max(0, ni - 15), min(n_fpga, ni + 15)):
        if fi in used:
            continue
        if abs(td - f_tick[fi] - offset) < mw:
            matched.append(
                (int(ev_tr[i]), int(f_id[fi]), float(td - f_tick[fi])))
            used.add(fi)
            break
n_m = len(matched)
print(f"  Digi-FPGA: {n_m}/{n_trig} ({n_m/max(n_trig,1)*100:.1f}%)")

if n_m > 0:
    dts_a = np.array([m[2] for m in matched]) * 5
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dts_a, 'g.', ms=2, alpha=0.5)
    ax.axhline(offset * 5, color='r', ls='--',
               label=f'offset={offset*5:.0f}ns')
    ax.axhline(np.std(dts_a) * 3, color='orange', ls=':',
               label=f'3sigma={np.std(dts_a)*3:.0f}ns')
    ax.set_xlabel('Match index')
    ax.set_ylabel('Dt (ns)')
    ax.set_title(f'Run6 Matched (n={n_m}, sigma={np.std(dts_a):.0f}ns)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig, 'run6_step3_match_dt.png')

arr = np.array(matched, dtype=[
               ('digi_evnum', 'i4'), ('fpga_id', 'i4'), ('dt', 'f4')])
np.savez(os.path.join(TEMP, 'matched_pairs.npz'), pairs=arr,
         offset_fticks=offset, sigma_fticks=sigma)

# ═══ Step 4: Scope ↔ FPGA ═══
print("\n" + "=" * 60)
print("Step 4: Scope ↔ FPGA")
print("=" * 60)

# Run6 有 C3-C8 通道，选择 C4 做对齐
ch_files = sorted(glob.glob(os.path.join(
    RUN, 'lecroy_wfm', 'C4--Trace--*.trc')))
print(f"  示波器 C4: {len(ch_files)} 个文件")

scope_segments = []  # [{file_idx, t_abs (unix), t_rel}]
for fp in ch_files:
    file_idx = int(re.search(r'--Trace--(\d+)\.trc',
                   os.path.basename(fp)).group(1))
    try:
        s = lecroyparser.ScopeData(path=fp)
        try:
            file_utc = datetime.strptime(
                s.triggerTime, '%Y-%m-%d %H:%M:%S.%f').timestamp()
        except:
            file_utc = datetime.strptime(
                s.triggerTime, '%Y-%m-%d %H:%M:%S').timestamp()
    except Exception as e:
        print(
            f"  lecroyparser failed for {os.path.basename(fp)}: {e}, trying binary parser...")
        info = parse_trc(fp)
        try:
            file_utc = datetime.strptime(
                info['trigger_time'], '%Y-%m-%d %H:%M:%S').timestamp()
        except:
            print(f"    Cannot get UTC time, skipping...")
            continue

    info = parse_trc(fp)
    if info['n_segments'] > 1:
        for si, off_s in enumerate(info['trig_time_offsets']):
            scope_segments.append({
                'file_idx': file_idx, 'seg_idx': si,
                't_abs': file_utc + float(off_s),
                'offset_s': float(off_s)
            })
    else:
        scope_segments.append({
            'file_idx': file_idx, 'seg_idx': 0,
            't_abs': file_utc,
            'offset_s': 0.0
        })

n_scope = len(scope_segments)
if n_scope == 0:
    print("❌ 无示波器数据，跳过 Step 4-5")
    sys.exit(1)

scope_t_abs = np.array([s['t_abs'] for s in scope_segments])
scope_rel = scope_t_abs - scope_t_abs[0]
print(f"  展开后: {n_scope} segments")
print(
    f"  UTC范围: {datetime.fromtimestamp(scope_t_abs[0])} ~ {datetime.fromtimestamp(scope_t_abs[-1])}")
print(f"  跨度: {scope_t_abs[-1]-scope_t_abs[0]:.1f}s")

# Scope spills (dt>50ms = boundary)
s_dt = np.diff(scope_rel) * 1000
s_ed = np.where(s_dt > 50)[0]
s_sp = np.zeros(n_scope, dtype=int)
for e in s_ed:
    s_sp[e + 1:] += 1
n_s_sp = s_sp.max() + 1

# FPGA spills
f_time_s = f_tick.astype(float) / FPGA_CLOCK_HZ
f_dt_ms = np.diff(f_time_s) * 1000
f_ed = np.where(f_dt_ms > 50)[0]
f_sp = np.zeros(n_fpga, dtype=int)
for e in f_ed:
    f_sp[e + 1:] += 1
n_f_sp = f_sp.max() + 1

print(f"  FPGA spills: {n_f_sp}, Scope spills: {n_s_sp}")

# 过滤大spill
f_large = sorted([sp for sp in range(f_sp.max() + 1)
                 if np.sum(f_sp == sp) > 10])
s_large = sorted([sp for sp in range(s_sp.max() + 1)
                 if np.sum(s_sp == sp) > 10])
f_cnts = [np.sum(f_sp == sp) for sp in f_large]
s_cnts = [np.sum(s_sp == sp) for sp in s_large]
print(f"  FPGA大spills: {len(f_cnts)}, Scope大spills: {len(s_cnts)}")

# Spill counts bar chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.bar(range(len(f_cnts)), f_cnts, color='red',
        alpha=0.7, label=f'FPGA ({len(f_cnts)} spills)')
for i, c in enumerate(f_cnts):
    ax1.text(i, c + 20, str(c), ha='center', fontsize=7)
ax1.set_ylabel('Events/spill')
ax1.set_title('Run6 FPGA spill counts')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax2.bar(range(len(s_cnts)), s_cnts, color='blue',
        alpha=0.7, label=f'Scope C4 ({len(s_cnts)} spills)')
for i, c in enumerate(s_cnts):
    ax2.text(i, c + 20, str(c), ha='center', fontsize=7)
ax2.set_xlabel('Spill index')
ax2.set_ylabel('Events/spill')
ax2.set_title('Run6 Scope C4 spill counts')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
savefig(fig, 'run6_step4_spill_counts.png')

# RMS scan 找 best spill offset
f_arr, s_arr = np.array(f_cnts, dtype=float), np.array(s_cnts, dtype=float)
best_off, best_rms = 0, float('inf')
for off in range(-10, 11):
    if off >= 0:
        fs, ss = f_arr[off:], s_arr[:len(f_arr)-off]
    else:
        fs, ss = f_arr[:len(f_arr)+off], s_arr[-off:]
    no = min(len(fs), len(ss))
    if no < 3:
        continue
    rms = np.sqrt(np.mean((fs[:no] - ss[:no])**2))
    if rms < best_rms:
        best_rms, best_off = rms, off
print(f"  Best spill offset: {best_off:+d}, RMS={best_rms:.1f}")

offs_scan = range(-10, 11)
rms_scan = []
for off in offs_scan:
    if off >= 0:
        fs, ss = f_arr[off:], s_arr[:len(f_arr)-off]
    else:
        fs, ss = f_arr[:len(f_arr)+off], s_arr[-off:]
    n_o = min(len(fs), len(ss))
    rms_scan.append(
        np.sqrt(np.mean((fs[:n_o]-ss[:n_o])**2)) if n_o >= 3 else 1e9)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(offs_scan, rms_scan, color='steelblue', alpha=0.8)
ax.bar(best_off, best_rms, color='red', alpha=0.8, label=f'Best={best_off:+d}')
ax.set_xlabel('Spill offset')
ax.set_ylabel('RMS of counts')
ax.set_title(f'Run6 Count RMS scan (best={best_off:+d}, RMS={best_rms:.0f})')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
savefig(fig, 'run6_step4_spill_rms.png')

# 按 best_off 配对 spill
all_sp = []
for i in range(min(len(f_large) - max(0, best_off), len(s_large) - max(0, -best_off))):
    fsp = f_large[i + max(0, best_off)]
    ssp = s_large[i + max(0, -best_off)]
    fn, sn = np.sum(f_sp == fsp), np.sum(s_sp == ssp)
    if fn != sn:
        print(f"  Skip {fsp}<->{ssp}: {fn}!={sn}")
        continue
    f_gi, s_gi = np.where(f_sp == fsp)[0], np.where(s_sp == ssp)[0]
    nv = min(100, len(f_gi), len(s_gi))
    ftv = f_tick[f_gi[:nv]].astype(float) / FPGA_CLOCK_HZ
    stv = scope_rel[s_gi[:nv]]
    rms_v = np.sqrt(np.mean(((ftv-ftv[0]) - (stv-stv[0]))**2)) * 1e6
    status = 'OK' if rms_v < 10 else f'REJECT(RMS={rms_v:.1f}us)'
    print(
        f"  FPGA spill {fsp:2d}({fn}) <-> Scope spill {ssp:2d}({sn}): {status}")
    if rms_v < 10:
        for fi, si in zip(f_gi, s_gi):
            all_sp.append(
                (int(fi), scope_segments[si]['file_idx'], scope_segments[si]['seg_idx']))
n_sp = len(all_sp)
print(f"  Scope-FPGA: {n_sp} pairs")

# ═══ Step 5: Scope ↔ FPGA Δt 直方图 ═══
print("\n" + "=" * 60)
print("Step 5: Scope ↔ FPGA Δt 直方图")
print("=" * 60)
# 对已匹配的 spill 内事件，计算 scope vs fpga 的时间差
sp_dt_all = []
for i in range(min(len(f_large) - max(0, best_off), len(s_large) - max(0, -best_off))):
    fsp = f_large[i + max(0, best_off)]
    ssp = s_large[i + max(0, -best_off)]
    fn, sn = np.sum(f_sp == fsp), np.sum(s_sp == ssp)
    if fn != sn:
        continue
    f_gi = np.where(f_sp == fsp)[0]
    s_gi = np.where(s_sp == ssp)[0]
    f_time = f_tick[f_gi].astype(float) / FPGA_CLOCK_HZ
    s_time = scope_rel[s_gi]
    # 对齐后计算逐事件 Δt
    f_time_adj = f_time - f_time[0]
    s_time_adj = s_time - s_time[0]
    for dt in (f_time_adj - s_time_adj) * 1e6:  # µs
        sp_dt_all.append(dt)

sp_dt_all = np.array(sp_dt_all)
print(f"  共 {len(sp_dt_all)} 对 scope-fpga Δt")
dt_peak = np.median(sp_dt_all)
dt_sigma = np.std(sp_dt_all)
print(f"  Median Δt: {dt_peak:.3f} µs, σ: {dt_sigma:.3f} µs")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(sp_dt_all, bins=200, color='purple', alpha=0.8)
ax1.axvline(dt_peak, color='r', ls='--', label=f'median={dt_peak:.3f}µs')
ax1.set_xlabel('Scope-FPGA Δt (µs)')
ax1.set_ylabel('Count')
ax1.set_title(f'Run6 Scope↔FPGA Δt ({len(sp_dt_all)} pairs, σ={dt_sigma:.2f}µs)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.hist(sp_dt_all, bins=200, range=(dt_peak-1, dt_peak+1), color='purple', alpha=0.8)
ax2.axvline(dt_peak, color='r', ls='--')
ax2.set_xlabel('Scope-FPGA Δt (µs)')
ax2.set_ylabel('Count')
ax2.set_title(f'Run6 Zoom (±1µs around median)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, 'run6_step5_scope_fpga_dt.png')

# ═══ Step 6: 已配对 spill 内逐事件 Timeline ═══
print("\n" + "=" * 60)
print("Step 6: 已配对 spill 逐事件时间线")
print("=" * 60)
# 选一个典型 spill 画出 scope vs fpga 的事件时间线
demo_sp = min(5, len(f_large) - max(0, best_off) - 1)
fdemo = f_large[demo_sp + max(0, best_off)]
sdemo = s_large[demo_sp + max(0, -best_off)]
f_gi_d = np.where(f_sp == fdemo)[0]
s_gi_d = np.where(s_sp == sdemo)[0]
fd_time = f_tick[f_gi_d].astype(float) / FPGA_CLOCK_HZ
sd_time = scope_rel[s_gi_d]
fd_time_adj = fd_time - fd_time[0]
sd_time_adj = sd_time - sd_time[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
ax1.plot(fd_time_adj*1e3, np.ones(len(fd_time_adj)), 'r.', ms=8, alpha=0.7, label='FPGA')
ax1.plot(sd_time_adj*1e3, np.zeros(len(sd_time_adj)), 'b.', ms=8, alpha=0.7, label='Scope C4')
ax1.set_xlabel('Time within spill (ms)')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Scope', 'FPGA'])
ax1.set_title(f'Run6 Spill#{demo_sp} FPGA({len(f_gi_d)}) vs Scope({len(s_gi_d)}) events')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, axis='x')

dn = min(len(fd_time_adj), len(sd_time_adj))
ax2.plot(np.arange(dn), (fd_time_adj[:dn]-sd_time_adj[:dn])*1e6, 'g.-', ms=4, lw=0.5)
ax2.axhline(0, color='gray', ls='--')
ax2.set_xlabel('Event index')
ax2.set_ylabel('FPGA - Scope Δt (µs)')
ax2.set_title(f'Run6 Spill#{demo_sp} per-event Δt')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, 'run6_step6_spill_timeline.png')

# ═══ Step 7: 三系统综合 Dashboard ═══
print("\n" + "=" * 60)
print("Step 7: 三系统综合 Dashboard")
print("=" * 60)
fs_map_temp = {}
for p in all_sp:
    fs_map_temp[p[0]] = True
fd_map_temp = {}
for m in matched:
    fd_map_temp[m[1]] = True

has_s = np.array([fi in fs_map_temp for fi in range(n_fpga)])
has_d = np.array([f_id[fi] in fd_map_temp for fi in range(n_fpga)])
f_time_s = f_tick.astype(float) / FPGA_CLOCK_HZ

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 7a: Venn/饼图
ax = axes[0, 0]
both = np.sum(has_s & has_d)
only_s = np.sum(has_s & ~has_d)
only_d = np.sum(~has_s & has_d)
neither = np.sum(~has_s & ~has_d)
labels = [f'Both\n({both})', f'Scope only\n({only_s})', f'Digi only\n({only_d})', f'FPGA only\n({neither})']
colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
patches, texts, autotexts = ax.pie([both, only_s, only_d, neither], labels=labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90)
ax.set_title(f'Run6 三系统匹配概览 (FPGA={n_fpga})')

# 7b: Spill 内三系统事件率对比
ax = axes[0, 1]
spill_ids = sorted(set(f_sp[f_sp >= 0]))
f_counts = [np.sum(f_sp == s) for s in spill_ids]
s_counts = [np.sum(s_sp == s) if s < n_s_sp else 0 for s in spill_ids]
ax.scatter(f_counts, s_counts, alpha=0.6, c='steelblue', edgecolors='navy')
mx = max(max(f_counts), max(s_counts)) * 1.1
ax.plot([0, mx], [0, mx], 'r--', lw=1, label='y=x')
ax.set_xlabel('FPGA events/spill')
ax.set_ylabel('Scope events/spill')
ax.set_title('Run6 Spill事件数对比')
ax.legend()
ax.grid(True, alpha=0.3)

# 7c: Scope-Digi 通过 FPGA 桥接的 Δt 分布
ax = axes[1, 0]
fs_map_full = {p[0]: p[1:] for p in all_sp}  # fi -> (file_idx, seg_idx)
fd_map_full = {}
for m in matched:
    fd_map_full[m[1]] = m[0]  # fpga_id -> digi_evnum
ts_from_fpga = f_tick.astype(float) / FPGA_CLOCK_HZ
# 找同时有 scope 和 digi 的事件
both_fi = np.where(has_s & has_d)[0]
if len(both_fi) > 0:
    b_dt_ns = np.array([fd_map.get(f_id[fi], {}).get('dt_ns', np.nan) if isinstance(fd_map.get(f_id[fi]), dict) 
                        else (fd_map[f_id[fi]][2]*5 if f_id[fi] in fd_map else np.nan)
                        for fi in both_fi])
    # 简化: 直接用 matched 中的 dt_ns
    b_dt_vals = []
    for m in matched:
        fid = m[1]
        fi_match = np.where(f_id == fid)[0]
        if len(fi_match) > 0 and fi_match[0] in fs_map_full:
            b_dt_vals.append(m[2] * 5)  # dt_ns
    b_dt_vals = np.array(b_dt_vals)
    ax.hist(b_dt_vals, bins=80, color='#e67e22', alpha=0.8)
    ax.axvline(np.median(b_dt_vals), color='r', ls='--', label=f'median={np.median(b_dt_vals):.0f}ns')
    ax.set_xlabel('Digi-FPGA Δt (ns)')
    ax.set_ylabel('Count')
    ax.set_title(f'Run6 三系统同时事件 Digi-FPGA Δt (n={len(b_dt_vals)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 7d: 匹配统计汇总表
ax = axes[1, 1]
ax.axis('off')
summary = [
    f"Run6 三系统匹配汇总",
    f"",
    f"FPGA 总事件:     {n_fpga}",
    f"Digitizer TR:    {n_tr} / 1Hz: {n_hz}",
    f"",
    f"Digi↔FPGA:       {n_m} ({n_m/max(n_trig,1)*100:.1f}%)",
    f"Scope↔FPGA:      {n_sp}",
    f"三系统同时:       {both}",
    f"",
    f"Digi-FPGA offset: {offset*5:.0f} ns",
    f"Digi-FPGA sigma:  {sigma*5:.0f} ns",
    f"Scope spill RMS:  {best_rms:.1f}",
]
for i, line in enumerate(summary):
    c = 'white' if i == 0 else 'black'
    fs = 16 if i == 0 else 12
    ax.text(0.1, 0.95 - i*0.065, line, transform=ax.transAxes,
            fontsize=fs, fontweight='bold' if i == 0 else 'normal',
            color='#2c3e50')
ax.set_title('')

plt.tight_layout()
savefig(fig, 'run6_step7_dashboard.png')

# ═══ Step 8: 完整事件表 ═══
print("\n" + "=" * 60)
print("Step 8: 完整事件表")
print("=" * 60)
fs_map = {p[0]: {'file_idx': p[1], 'seg_idx': p[2]} for p in all_sp}
fd_map = {}
for m in matched:
    fd_map[m[1]] = {'digi_evnum': m[0], 'dt_ns': m[2] * 5}

out_csv = os.path.join(TEMP, 'full_event_table.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['fpga_trigger_id', 'fpga_tick', 'has_scope', 'scope_file_idx', 'scope_seg_idx',
                'has_digi', 'digi_evnum', 'digi_tt_raw', 'digi_is_1hz', 'dt_ns'])
    for fi in range(n_fpga):
        fid = f_id[fi]
        s = fs_map.get(fi, None)
        d = fd_map.get(fid, None)
        w.writerow([fid, f_tick[fi],
                    1 if s else 0, s['file_idx'] if s else -
                    1, s['seg_idx'] if s else -1,
                    1 if d else 0, d['digi_evnum'] if d else -1,
                    d_tt[d['digi_evnum']] if d and d['digi_evnum'] < n_digi else -1,
                    int(is_hz[d['digi_evnum']]
                        ) if d and d['digi_evnum'] < n_digi else -1,
                    d['dt_ns'] if d else np.nan])

n_s = len(fs_map)
n_d = len(matched)
both = sum(1 for fi in range(n_fpga) if fi in fs_map and f_id[fi] in fd_map)
print(f"  FPGA:{n_fpga}, Scope:{n_s}, Digi:{n_d}, Both={both}")
print(f"  Saved: {out_csv}")

# 汇总
print(f"\n{'='*60}")
print(f"Run6 匹配完成!")
print(f"  1Hz事件: {n_hz}  |  TR事件: {n_tr}")
print(
    f"  Digi↔FPGA: {n_m}/{min(n_trig, n_fpga)} ({n_m/max(min(n_trig,n_fpga),1)*100:.1f}%)")
print(f"  Scope↔FPGA: {n_sp}")
print(f"  三系统同时: {both}")
print(f"  图表保存在: {OUT}/")
print(f"{'='*60}")
