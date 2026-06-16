"""
digi_fpga_matching.py — Digi↔FPGA 时间戳匹配（参照 match 文件夹的四步法）

Step 1: 分离 1Hz 与 TR 事件 (TR通道波形 tail-diff)
Step 2: 用 1Hz 事件将 Digi TT 映射到 FPGA tick 域
Step 3: Δt 峰值查找 + 逐事件匹配
Step 4: (暂略) 关联示波器

用法:
    python digi_fpga_matching.py <data_dir>
"""

import sys
import os
import numpy as np
from pathlib import Path

FPGA_TICKS_PER_1HZ = 200_000_020  # 硬件实测值，比标称+20ppm


def load_data(run_dir):
    """加载 Digitizer 和 FPGA 数据"""
    import sys
    import os
    from pathlib import Path
    _repo = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_repo))
    from src.data.raw_data_recorder import load_binary_events
    from src.data.fpga_parser import load_fpga_events, parse_fpga_packets, FPGA_CLOCK_HZ

    # Digi
    digi = load_binary_events(str(run_dir / "digitizer" / "V1742_events.bin"))
    n_digi = len(digi)
    d_num = np.array([e['event_number'] for e in digi], dtype=np.int64)
    d_tt = np.array([e['trigger_time_tag'] for e in digi], dtype=np.int64)
    d_et = np.array([e.get('event_time_tag', 0) for e in digi], dtype=np.int64)
    print(f"\nDigitizer: {n_digi} events, event# {d_num[0]}~{d_num[-1]}")

    # FPGA
    fpga_ev = load_fpga_events(str(run_dir / "fpga" / "fpga_events.bin"))
    n_fpga = len(fpga_ev)
    f_id = np.array([e.trigger_id for e in fpga_ev], dtype=np.int64)
    f_tick = np.array([e.t_fpga for e in fpga_ev], dtype=np.int64)
    print(f"FPGA OSCI: {n_fpga} events, trigger_id {f_id[0]}~{f_id[-1]}")

    return digi, d_num, d_tt, d_et, fpga_ev, f_id, f_tick


def step1_separate_1hz(digi, d_num, d_tt, run_dir):
    """Step 1: 用 TR 通道 tail-diff 分离 1Hz 和 TR 事件"""
    print("\n" + "="*60)
    print("  Step 1: 分离 1Hz 与 TR 事件")
    print("="*60)

    TR_CH = [32, 33, 34, 35]
    N_HEAD, N_TAIL = 20, 50
    n = len(digi)

    tail_diff = np.full((n, 4), np.nan)
    for i, e in enumerate(digi):
        wf = e['waveforms']
        for j, ch in enumerate(TR_CH):
            if ch not in wf:
                continue
            d = wf[ch]
            ns = len(d)
            h = min(N_HEAD, ns)
            tl = min(N_TAIL, ns)
            head_m = np.mean(d[:h])
            tail_m = np.mean(d[-tl:])
            tail_diff[i, j] = abs(tail_m - head_m)

    max_tail = np.nanmax(tail_diff, axis=1)
    TR_DIFF_THRESHOLD = 5.0
    is_hz = max_tail < TR_DIFF_THRESHOLD

    n_hz = is_hz.sum()
    n_tr = n - n_hz
    hz_idx = np.where(is_hz)[0]
    tr_idx = np.where(~is_hz)[0]
    print(f"  1Hz: {n_hz}, TR triggers: {n_tr}")
    print(f"  1Hz indices (first 20): {hz_idx[:20].tolist()}")

    # 理论 ft 标定
    N = FPGA_TICKS_PER_1HZ
    NOM_1S = 1e9 / 8.5  # 理论每秒 DRS4 cycles
    tt_1hz = d_tt[hz_idx]

    # 跳过高偏离的首脉冲（采集可能始于 1Hz 周期中间）
    start_i = 0
    if len(hz_idx) >= 3:
        first_dt = tt_1hz[1] - tt_1hz[0]
        if abs(first_dt - NOM_1S) / NOM_1S > 0.02:
            start_i = 1

    theory_ft = np.zeros(len(hz_idx), dtype=np.float64)
    cum = 0.0
    for i in range(len(hz_idx)):
        if i < start_i:
            theory_ft[i] = -N
            continue
        if i == start_i:
            n_s = 1
        else:
            n_s = max(1, round((tt_1hz[i] - tt_1hz[i-1]) / NOM_1S))
        cum += n_s * N
        theory_ft[i] = cum

    # 保存
    temp_dir = run_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(temp_dir / "tr_waveform_analysis.npz"),
                        is_1hz=is_hz, ev_numbers=d_num, tt_tags=d_tt,
                        idx_1hz=hz_idx, tt_1hz=tt_1hz, theory_ft=theory_ft,
                        max_tail_diff=max_tail)

    print(f"  理论ft标定完成, 已保存到 {temp_dir}/tr_waveform_analysis.npz")
    return temp_dir


def step2_calibrate(run_dir):
    """Step 2: 将 Digi TT 映射到 FPGA tick 域"""
    print("\n" + "="*60)
    print("  Step 2: DRS4 时钟标定 → FPGA tick 域")
    print("="*60)

    data = np.load(str(run_dir / "temp" / "tr_waveform_analysis.npz"))
    is_hz = data['is_1hz']
    tt_tags = data['tt_tags'].astype(np.int64)
    tt_1hz = data['tt_1hz'].astype(np.int64)
    theory_ft = data['theory_ft'].astype(np.float64)
    hz_idx = data['idx_1hz']

    if len(tt_1hz) < 2:
        raise ValueError(f"Need >= 2 1Hz events, found {len(tt_1hz)}")

    # 每段映射因子: f_pd = Δtheory_ft / ΔTT (仅用有效段)
    valid = theory_ft >= 0
    vt, vtt = theory_ft[valid], tt_1hz[valid]
    dt_ft = np.diff(vt)
    dt_cyc = np.diff(vtt)
    f_pd_raw = dt_ft.astype(float) / np.maximum(dt_cyc, 1)
    f_pd = np.zeros(len(hz_idx))
    f_pd[valid] = np.insert(f_pd_raw, 0, np.mean(f_pd_raw[:5]) if len(f_pd_raw) >= 5 else f_pd_raw[0])

    # 逐段线性映射
    def conv(t):
        s = int(np.searchsorted(tt_1hz, t, side='right') - 1)
        if s < 0:
            s = 0
        if s >= len(tt_1hz) - 1:
            s = len(tt_1hz) - 2
        if theory_ft[s] < 0 and s + 1 < len(theory_ft):
            s = s + 1
        if theory_ft[s] < 0:
            s = np.argmax(theory_ft >= 0)
        return theory_ft[s] + (t - tt_1hz[s]) * f_pd[s]

    t_ft = np.array([conv(int(t)) for t in tt_tags])

    # 验证 1Hz 映射精度
    t_1hz_ft = t_ft[hz_idx]
    if theory_ft[0] < 0:
        start = np.argmax(theory_ft >= 0)
        err = (t_1hz_ft[start:] - t_1hz_ft[start]) - (theory_ft[start:] - theory_ft[start])
    else:
        err = (t_1hz_ft - t_1hz_ft[0]) - (theory_ft - theory_ft[0])
    print(f"  1Hz max error: {np.max(np.abs(err)):.0f} ticks")
    print(f"  平均 f_pd: {np.mean(f_pd[f_pd > 0]):.6f} FPGA ticks / DRS4 cycle")

    # 保存
    tr_idx = np.where(~is_hz)[0]
    np.savez_compressed(str(run_dir / "temp" / "corrected_timetags.npz"),
                        tt_tags=tt_tags, ev_numbers=data['ev_numbers'],
                        is_1hz=is_hz, t_corrected_fticks=t_ft,
                        t_corrected_ns=t_ft * 5,  # FPGA-ns
                        idx_1hz=hz_idx, idx_trig=tr_idx,
                        f_ticks_per_drs4=f_pd)

    return t_ft, tr_idx, hz_idx


def step3_match(run_dir, window=50):
    """Step 3: FPGA ↔ Digitizer 事件匹配"""
    from src.data.fpga_parser import load_fpga_events

    print("\n" + "="*60)
    print("  Step 3: FPGA ↔ Digitizer 事件匹配")
    print("="*60)

    corr = np.load(str(run_dir / "temp" / "corrected_timetags.npz"))
    t_ft = corr['t_corrected_fticks'].astype(np.float64)
    is_hz = corr['is_1hz']
    ev = corr['ev_numbers'].astype(np.int64)

    idx_trig = np.where(~is_hz)[0]
    t_trig_ft = t_ft[idx_trig]
    n_trig = len(idx_trig)
    print(f"  TR triggers (可匹配): {n_trig}")

    # FPGA
    fpga_ev = load_fpga_events(str(run_dir / "fpga" / "fpga_events.bin"))
    f_id = np.array([e.trigger_id for e in fpga_ev], dtype=np.int64)
    f_tick = np.array([e.t_fpga for e in fpga_ev], dtype=np.int64)
    n_fpga = len(fpga_ev)
    print(f"  FPGA OSCI: {n_fpga}")

    # ── 3a. Δt 直方图找偏移 ──
    print(f"\n  扫描 Δt 直方图...")
    all_dt = []
    n_scan = min(2000, n_fpga)
    for fi in range(n_scan):
        ft = f_tick[fi]
        l = int(np.searchsorted(t_trig_ft, ft - 5000))
        r = int(np.searchsorted(t_trig_ft, ft + 5000))
        for di in range(l, r):
            all_dt.append(t_trig_ft[di] - ft)

    all_dt = np.array(all_dt)
    bins = np.arange(all_dt.min(), all_dt.max() + 0.5, 0.5)
    h, _ = np.histogram(all_dt, bins=bins)
    bc = (bins[:-1] + bins[1:]) / 2
    offset = float(bc[np.argmax(h)])

    # 估算 σ: 取 peak ±50 ticks 内的数据
    near = all_dt[np.abs(all_dt - offset) < 50]
    sigma_est = np.std(near) if len(near) > 10 else 1.0
    print(f"  Offset: {offset:.1f} ticks ({offset*5:.0f} ns)")
    print(f"  σ: {sigma_est:.1f} ticks ({sigma_est*5:.0f} ns)")

    # ── 3b. 逐事件匹配 ──
    print(f"\n  匹配中 (窗口 ±{window} ticks)...")
    fpga2d = {}  # fpga_idx → (digi_idx_in_trig, dt)
    digi2f = {}  # digi_idx_in_trig → fpga_idx

    for fi in range(n_fpga):
        tgt = f_tick[fi] + offset
        l = int(np.searchsorted(t_trig_ft, tgt - window))
        r = int(np.searchsorted(t_trig_ft, tgt + window))
        if l >= r:
            continue

        dl = t_trig_ft[l:r] - tgt
        bi = int(np.argmin(np.abs(dl)))
        bdi = l + bi
        bdt = float(dl[bi])
        bdt_abs = abs(bdt)

        if bdi in digi2f:
            # 冲突: 保留更近的那个
            pfi = digi2f[bdi]
            pdt = abs(t_trig_ft[bdi] - (f_tick[pfi] + offset))
            if bdt_abs < pdt:
                del fpga2d[pfi]
                fpga2d[fi] = (bdi, bdt)
                digi2f[bdi] = fi
        else:
            fpga2d[fi] = (bdi, bdt)
            digi2f[bdi] = fi

    n_m = len(fpga2d)
    print(f"  匹配: {n_m}/{n_fpga} FPGA ({n_m/n_fpga*100:.1f}%)")

    if n_m > 0:
        md_signed = np.array([d for _, d in fpga2d.values()])
        md_abs = np.abs(md_signed)
        print(f"  DT: med={np.median(md_signed):.1f}, "
              f"med_abs={np.median(md_abs):.1f}, "
              f"max_abs={md_abs.max():.1f} ticks")

    # 保存
    # 结果数组: 长度 = n_trig, fpga_matched_id[digi_trig_idx] = FPGA trigger_id
    fpga_mid = np.full(n_trig, -1, dtype=np.int64)
    fpga_mdt = np.full(n_trig, np.nan)
    for fi, (di, dt) in fpga2d.items():
        fpga_mid[di] = int(f_id[fi])
        fpga_mdt[di] = float(dt)

    np.savez_compressed(str(run_dir / "temp" / "fpga_digi_match.npz"),
                        fpga_id=f_id, fpga_tick=f_tick,
                        fpga_matched_id=fpga_mid, fpga_matched_dt=fpga_mdt,
                        global_offset_fticks=offset,
                        trig_ev_numbers=ev[idx_trig],
                        fpga_only_idx=np.array(
                            sorted(set(range(n_fpga)) - set(fpga2d.keys()))),
                        digi_only_trig_idx=np.array(sorted(set(range(n_trig)) - set(digi2f.keys()))))

    print(f"  已保存: {run_dir}/temp/fpga_digi_match.npz")
    return fpga2d, digi2f


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python digi_fpga_matching.py <data_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    # 加载
    digi, d_num, d_tt, d_et, fpga_ev, f_id, f_tick = load_data(run_dir)

    # Step 1
    temp_dir = step1_separate_1hz(digi, d_num, d_tt, run_dir)

    # Step 2
    t_ft, tr_idx, hz_idx = step2_calibrate(run_dir)

    # Step 3
    fpga2d, digi2f = step3_match(run_dir, window=50)
