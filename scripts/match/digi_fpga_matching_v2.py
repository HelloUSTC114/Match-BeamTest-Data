"""
改进的 Digi↔FPGA 匹配算法

方法:
  1. 以 Digi 事件为锚点，每个 Digi 找最近的 N 个 FPGA 事件
  2. 计算 Δt = t_dig - t_fpga，画直方图
  3. 直方图形的峰值 = FPGA与Digi信号的传输延迟
  4. 在延迟附近找精确的 1:1 匹配
"""

import sys
import os
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def load_and_calibrate(run_dir):
    """加载数据 + 1Hz分离 + 时钟标定，返回校正后的时间"""
    from src.data.raw_data_recorder import load_binary_events
    from src.data.fpga_parser import load_fpga_events

    # 加载原始数据
    digi = load_binary_events(str(run_dir / "digitizer" / "V1742_events.bin"))
    d_tt = np.array([e['trigger_time_tag'] for e in digi], dtype=np.int64)
    d_num = np.array([e['event_number'] for e in digi], dtype=np.int64)
    n_digi = len(digi)

    fpga_ev = load_fpga_events(str(run_dir / "fpga" / "fpga_events.bin"))
    f_tick = np.array([e.t_fpga for e in fpga_ev], dtype=np.int64)
    f_id = np.array([e.trigger_id for e in fpga_ev], dtype=np.int64)
    n_fpga = len(fpga_ev)

    print(f"Digitizer: {n_digi} events, event# {d_num[0]}~{d_num[-1]}")
    print(f"FPGA OSCI: {n_fpga} events, trigger_id {f_id[0]}~{f_id[-1]}")

    # ── Step 1: TR 通道分离 1Hz ──
    TR_CH = [32, 33, 34, 35]
    N_HEAD, N_TAIL = 20, 50
    tail_diff = np.full((n_digi, 4), np.nan)
    for i, e in enumerate(digi):
        wf = e['waveforms']
        for j, ch in enumerate(TR_CH):
            if ch not in wf:
                continue
            d = wf[ch]
            ns = len(d)
            h = min(N_HEAD, ns)
            tl = min(N_TAIL, ns)
            tail_diff[i, j] = abs(np.mean(d[:h]) - np.mean(d[-tl:]))
    is_hz = np.nanmax(tail_diff, axis=1) < 5.0
    n_hz, n_tr = is_hz.sum(), n_digi - is_hz.sum()
    print(f"  1Hz: {n_hz}, TR triggers: {n_tr}")

    # ── Step 2: 1Hz 理论 ft 标定 ──
    FPGA_TICKS_PER_1HZ = 200_000_020  # 硬件实测值, 比标称+20ppm
    NOM_1S = 1e9 / 8.5
    hz_idx = np.where(is_hz)[0]
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
            theory_ft[i] = -FPGA_TICKS_PER_1HZ
            continue
        if i == start_i:
            n_s = 1
        else:
            n_s = max(1, round((tt_1hz[i] - tt_1hz[i-1]) / NOM_1S))
        cum += n_s * FPGA_TICKS_PER_1HZ
        theory_ft[i] = cum

    # ── Step 3: 分段线性映射 DRS4 → FPGA tick ──
    valid = theory_ft >= 0
    vt, vtt = theory_ft[valid], tt_1hz[valid]
    dt_ft = np.diff(vt)
    dt_cyc = np.diff(vtt)
    f_pd_raw = dt_ft.astype(float) / np.maximum(dt_cyc, 1)
    f_pd = np.zeros(len(hz_idx))
    f_pd[valid] = np.insert(f_pd_raw, 0, np.mean(f_pd_raw[:5]) if len(f_pd_raw) >= 5 else f_pd_raw[0])

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

    t_ft = np.array([conv(int(t)) for t in d_tt])

    # 物理触发事件的时间 (FPGA tick 域)
    tr_idx = np.where(~is_hz)[0]
    t_tr_ft = t_ft[tr_idx]
    d_tr_num = d_num[tr_idx]

    print(f"  FPGA tick 域范围: {t_tr_ft[0]:.0f} ~ {t_tr_ft[-1]:.0f}")
    print(f"  FPGA tick 域跨度: {(t_tr_ft[-1]-t_tr_ft[0])/200e6:.1f}s")

    return t_tr_ft, d_tr_num, f_tick, f_id


def find_deltas_histogram(t_tr_ft, f_tick, n_closest=100):
    """
    以 Digi 事件为锚点:
      每个 Digi 找最近的 n_closest 个 FPGA 事件
      计算 Δt = t_dig - t_fpga
      画直方图 → 峰值 = 信号传输延迟
    """
    print(f"\n{'='*60}")
    print(f"  计算 Δt = t_dig - t_fpga 分布")
    print(f"  (每个 Digi 找最近 {n_closest} 个 FPGA)")
    print(f"{'='*60}")

    n_digi = len(t_tr_ft)
    all_dt = []

    for i in range(n_digi):
        td = t_tr_ft[i]
        # 找最近的 FPGA 事件索引
        nearest = np.searchsorted(f_tick, td)
        left = max(0, nearest - n_closest//2)
        right = min(len(f_tick), nearest + n_closest//2)
        for fi in range(left, right):
            dt = td - f_tick[fi]
            # 只保留合理范围内的 Δt (±100000 ticks = ±0.5ms)
            if abs(dt) < 100000:
                all_dt.append(dt)

    all_dt = np.array(all_dt)
    print(f"  Δt 在 ±100000 内的数量: {len(all_dt)}")
    if len(all_dt) == 0:
        print("  ❌ 没有 Δt 在 ±100000 内！检查时间校准")
        return 0.0, 1.0, np.array([]), np.array([]), np.array([])

    print(f"  Δt 范围: {all_dt.min():.0f} ~ {all_dt.max():.0f} ticks")

    # 直方图: ±5000 ticks (= ±25 μs) 足够
    bins = np.arange(-5000, 5001, 0.5)
    h, be = np.histogram(all_dt, bins=bins)
    bc = (be[:-1] + be[1:]) / 2

    peak_idx = np.argmax(h)
    peak = bc[peak_idx]
    peak_height = h[peak_idx]

    # 估算 σ: 取峰值 ±50 内的数据
    near = all_dt[np.abs(all_dt - peak) < 50]
    sigma_est = np.std(near) if len(near) > 10 else 1.0

    print(f"  组合数: {len(all_dt)}")
    print(f"  峰值: {peak:.1f} ticks = {peak*5:.0f} ns")
    print(f"  峰值计数: {peak_height}")
    print(f"  σ: {sigma_est:.1f} ticks = {sigma_est*5:.0f} ns")

    # 显示直方图关键部分
    print(f"\n  Δt 分布 (峰值附近):")
    for i in range(max(0, peak_idx-5), min(len(bc), peak_idx+6)):
        bar = '█' * min(h[i]//max(1, peak_height//40), 40)
        pct = h[i]/peak_height*100 if peak_height > 0 else 0
        print(f"    {bc[i]:+8.1f} ticks: {h[i]:5d} ({pct:5.1f}%) {bar}")

    return peak, sigma_est, all_dt, bc, h


def match_events(t_tr_ft, d_tr_num, f_tick, f_id, peak, window=50):
    """
    在 peak ± window 内精确匹配。
    双向匹配: 每个 Digi → 最近的 FPGA; 解决冲突 (多个 Digi 抢同一个 FPGA)。
    """
    print(f"\n{'='*60}")
    print(f"  精确匹配 (窗口 = {peak:.1f} ± {window} ticks)")
    print(f"{'='*60}")

    n_digi = len(t_tr_ft)
    n_fpga = len(f_tick)

    # 对每个 Digi 事件，在 peak ± window 内找最近的 FPGA
    digi2fpga = {}  # digi_idx → (fpga_idx, dt)
    fpga2digi = {}  # fpga_idx → digi_idx

    for di in range(n_digi):
        td = t_tr_ft[di]
        tgt = td - peak  # 预期的 FPGA 时间
        l = int(np.searchsorted(f_tick, tgt - window))
        r = int(np.searchsorted(f_tick, tgt + window))
        if l >= r:
            continue

        # 找最近的 FPGA
        dl = f_tick[l:r] - tgt
        bi = int(np.argmin(np.abs(dl)))
        fi = l + bi
        dt = td - f_tick[fi]  # 实际 Δt

        # 冲突处理: 如果两个 Digi 抢同一个 FPGA，保留 Δt 更接近 peak 的那个
        if fi in fpga2digi:
            old_di = fpga2digi[fi]
            old_dt = t_tr_ft[old_di] - f_tick[fi]
            if abs(dt - peak) < abs(old_dt - peak):
                del digi2fpga[old_di]
                digi2fpga[di] = (fi, dt)
                fpga2digi[fi] = di
        else:
            digi2fpga[di] = (fi, dt)
            fpga2digi[fi] = di

    n_m = len(digi2fpga)
    print(f"  Digi 匹配: {n_m}/{n_digi} ({n_m/n_digi*100:.1f}%)")
    print(f"  FPGA 匹配: {len(set(fi for fi,_ in digi2fpga.values()))}/{n_fpga}")

    if n_m > 0:
        dts = np.array([dt for _, dt in digi2fpga.values()])
        print(f"  Δt: 中位数={np.median(dts):.1f}, "
              f"中位|Δt|={np.median(np.abs(dts)):.1f}, "
              f"max|Δt|={np.max(np.abs(dts)):.1f} ticks")

    # 结果数组
    result_fpga_id = np.full(n_digi, -1, dtype=np.int64)
    result_dt = np.full(n_digi, np.nan)
    result_fpga_tick = np.full(n_digi, -1, dtype=np.int64)

    for di, (fi, dt) in digi2fpga.items():
        result_fpga_id[di] = int(f_id[fi])
        result_dt[di] = float(dt)
        result_fpga_tick[di] = int(f_tick[fi])

    return (digi2fpga, fpga2digi,
            result_fpga_id, result_dt, result_fpga_tick)


def plot_results(t_tr_ft, d_tr_num, f_tick, f_id,
                 all_dt, bc, h, peak, sigma_est,
                 digi2fpga, run_dir):
    """生成诊断图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plots_dir = run_dir / "temp" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 图1: Δt 分布
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(bc * 5, h, width=2.5, color='steelblue',
               alpha=0.8, label='Δt histogram')
        ax.axvline(peak * 5, color='r', lw=2, ls='--',
                   label=f'peak={peak*5:.0f}ns')
        ax.axvline((peak - sigma_est*3) * 5, color='orange', ls=':',
                   label=f'±3σ={sigma_est*3*5:.0f}ns')
        ax.axvline((peak + sigma_est*3) * 5, color='orange', ls=':')
        ax.set_xlabel('Δt = t_dig - t_fpga (ns)')
        ax.set_ylabel('Count')
        ax.set_title(
            f'Δt distribution (each Digi → {len(all_dt)//max(1,len(t_tr_ft))} nearest FPGA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(str(plots_dir / "match_deltahist.png"), dpi=150)
        plt.close()
        print(f"  图 -> {plots_dir}/match_deltahist.png")

        # 图2: 匹配 Δt vs 事件序号
        if digi2fpga:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            ax = axes[0]
            di_list = sorted(digi2fpga.keys())
            dt_vals = np.array([digi2fpga[di][1] for di in di_list]) * 5
            ax.plot(di_list, dt_vals, 'g.', ms=2, alpha=0.5)
            ax.axhline(peak*5, color='r', ls='--')
            ax.axhline(np.std(dt_vals)*3, color='orange', ls=':',
                       label=f'±3σ={np.std(dt_vals)*3:.1f}ns')
            ax.axhline(-np.std(dt_vals)*3, color='orange', ls=':')
            ax.set_xlabel('Digi event index (TR only)')
            ax.set_ylabel('Δt (ns)')
            ax.set_title(
                f'Matched Δt (n={len(di_list)}, σ={np.std(dt_vals):.1f}ns)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            ax.hist(dt_vals, bins=50, color='green',
                    alpha=0.7, edgecolor='k', lw=0.5)
            ax.axvline(peak*5, color='r', ls='--')
            ax.set_xlabel('Δt (ns)')
            ax.set_ylabel('Count')
            ax.set_title(
                f'Matched Δt histogram (μ={np.mean(dt_vals):.1f}, σ={np.std(dt_vals):.1f}) ns')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(str(plots_dir / "match_validation.png"), dpi=150)
            plt.close()
            print(f"  图 -> {plots_dir}/match_validation.png")

        print(f"  所有图已保存到 {plots_dir}/")

    except ImportError:
        print("  ⚠️ matplotlib 未安装，跳过绘图")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Digi↔FPGA 匹配')
    parser.add_argument('data_dir', help='run 目录')
    parser.add_argument('--n-closest', type=int, default=100,
                        help='每个 Digi 找最近 N 个 FPGA 来计算 Δt 分布')
    parser.add_argument('--window', type=int, default=50,
                        help='匹配窗口 (ticks)')
    args = parser.parse_args()

    run_dir = Path(args.data_dir)

    # 加载 + 标定
    t_tr_ft, d_tr_num, f_tick, f_id = load_and_calibrate(run_dir)

    # Δt 直方图找峰值
    peak, sigma_est, all_dt, bc, h = find_deltas_histogram(
        t_tr_ft, f_tick, n_closest=args.n_closest)

    # 精确匹配
    digi2fpga, fpga2digi, result_id, result_dt, result_tick = match_events(
        t_tr_ft, d_tr_num, f_tick, f_id, peak, window=args.window)

    # 绘图
    plot_results(t_tr_ft, d_tr_num, f_tick, f_id,
                 all_dt, bc, h, peak, sigma_est,
                 digi2fpga, run_dir)

    # 保存匹配结果
    out = run_dir / "temp" / "fpga_digi_match.npz"
    fpga_only = sorted(set(range(len(f_tick))) -
                       {fi for fi, _ in digi2fpga.values()})
    digi_only = sorted(set(range(len(t_tr_ft))) - set(digi2fpga.keys()))

    np.savez_compressed(str(out),
                        trig_ev_numbers=d_tr_num,
                        trig_t_ft=t_tr_ft,
                        fpga_id=f_id, fpga_tick=f_tick,
                        matched_fpga_id=result_id,
                        matched_fpga_tick=result_tick,
                        matched_dt=result_dt,
                        global_offset_fticks=peak,
                        matching_window=args.window,
                        fpga_only_idx=np.array(fpga_only, dtype=np.int64),
                        digi_only_idx=np.array(digi_only, dtype=np.int64))

    print(f"\n  已保存: {out}")
    print(
        f"  Digi总: {len(t_tr_ft)}, 匹配: {len(digi2fpga)} ({len(digi2fpga)/len(t_tr_ft)*100:.1f}%)")


if __name__ == '__main__':
    main()
