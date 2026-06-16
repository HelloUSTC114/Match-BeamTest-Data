"""
Step 3: 基于spill的Digi↔FPGA匹配

流程:
  1. 用FPGA时间戳分离spill (Δt > 50ms 为边界)
  2. 把Digi事件按时间范围分配到各spill
  3. 每个spill内部逐事件匹配
  4. 输出匹配统计到终端, 保存pairs到temp/
"""
from src.data.fpga_parser import load_fpga_events, FPGA_CLOCK_HZ
import sys
import os
import numpy as np
from pathlib import Path

# ── 仓库根目录 ──
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# 重新导入确保使用正确的 src
from src.data.fpga_parser import load_fpga_events, FPGA_CLOCK_HZ


def match_by_spill(run_dir, dt_thresh_ms=50, match_window=50, verbose=True):
    run_dir = Path(run_dir)
    temp = run_dir / "temp"

    corr = np.load(str(temp / "corrected_timetags.npz"))
    t_ft = corr['t_corrected_fticks'].astype(float)
    is_hz = corr['is_1hz']
    tr_idx = np.where(~is_hz)[0]
    t_tr_ft = t_ft[tr_idx]
    n_trig = len(tr_idx)

    fpga = load_fpga_events(str(run_dir / "fpga" / "fpga_events.bin"))
    f_id = np.array([e.trigger_id for e in fpga])
    f_tick = np.array([e.t_fpga for e in fpga])
    n_fpga = len(f_id)

    # ── 1. 分离spill ──
    f_dt_ms = np.diff(f_tick).astype(float) / FPGA_CLOCK_HZ * 1000
    edges = np.where(f_dt_ms > dt_thresh_ms)[0]
    f_sp = np.zeros(n_fpga, dtype=int)
    for e in edges:
        f_sp[e+1:] += 1
    n_sp = f_sp.max() + 1

    # Digi分配到spill (按时间范围)
    d_sp = np.full(n_trig, -1, dtype=int)
    for sp in range(n_sp):
        m = f_sp == sp
        t0, t1 = f_tick[m][0], f_tick[m][-1]
        d_sp[(t_tr_ft >= t0) & (t_tr_ft <= t1)] = sp

    # ── 2. 从spill 0估offset ──
    sp0_f = np.where(f_sp == 0)[0]
    sp0_d = np.where(d_sp == 0)[0]
    all_dt = np.array([t_tr_ft[di] - f_tick[fi]
                       for di in sp0_d for fi in range(
        max(0, int(np.searchsorted(f_tick, t_tr_ft[di]))-50),
        min(n_fpga, int(np.searchsorted(f_tick, t_tr_ft[di]))+50))
        if abs(t_tr_ft[di] - f_tick[fi]) < 50000])

    h, be = np.histogram(all_dt, bins=np.arange(-5000, 5001, 0.5))
    bc = (be[:-1] + be[1:]) / 2
    offset = bc[np.argmax(h)]
    near = all_dt[np.abs(all_dt-offset) < 50]
    sigma = np.std(near) if len(near) > 10 else 1.0
    print(f"Offset: {offset:.1f} ticks, sigma={sigma:.1f} ticks")

    # ── 3. 逐spill匹配 ──
    total = 0
    pairs = []
    for sp in range(n_sp):
        mf = np.where(f_sp == sp)[0]
        md = np.where(d_sp == sp)[0]
        if len(mf) == 0 or len(md) == 0:
            continue
        used = set()
        spm = 0
        for di in md:
            td = t_tr_ft[di]
            tgt = td - offset
            ni = int(np.searchsorted(f_tick, tgt))
            for fi in range(max(0, ni-10), min(n_fpga, ni+10)):
                if fi in used:
                    continue
                if abs(td - f_tick[fi] - offset) < match_window:
                    pairs.append((int(tr_idx[di]), int(
                        f_id[fi]), float(td-f_tick[fi])))
                    used.add(fi)
                    spm += 1
                    break
        total += spm
        if verbose and sp < 30:
            print(
                f"  Spill{sp:2d}: FPGA={len(mf):5d} Digi={len(md):5d} match={spm:4d}({spm/max(len(md),1)*100:5.1f}%)")

    print(f"\n总匹配: {total}/{n_trig} Digi ({total/max(n_trig,1)*100:.1f}%)")
    print(f"       {total}/{n_fpga} FPGA  ({total/max(n_fpga,1)*100:.1f}%)")

    arr = np.array(pairs, dtype=[('digi_idx', 'i4'),
                   ('fpga_id', 'i4'), ('dt', 'f4')])
    np.savez(str(temp/"spill_matched_pairs.npz"), pairs=arr, offset_fticks=offset,
             f_spill=f_sp, d_spill=d_sp, f_id=f_id, f_tick=f_tick,
             digi_tr_idx=tr_idx, digi_tr_t_ft=t_tr_ft, n_matched=total, n_trig=n_trig)

    return arr, total, n_trig


if __name__ == '__main__':
    # 修改为你的数据路径
    RUN = REPO.parent / 'Analyze' / 'data' / 'BT_20260530_010301_run_0001'
    match_by_spill(str(RUN))
