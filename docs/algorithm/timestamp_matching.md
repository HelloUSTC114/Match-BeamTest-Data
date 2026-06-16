# 三系统时间戳匹配 — 经验总结

## 信号路径

```
束流spill → 探测器 → 示波器触发 → AUX OUT → 甄别器 → 分两路:
  一路 → FPGA:        OSCI事件 (trigger_id++, t_fpga @200MHz, 5ns/tick)
  一路 → Digitizer:   TR0/TR1: 波形采集 (trigger_time_tag @8.5ns/cycle)

FPGA GSYNC(1Hz) ──→ Digitizer TRIGGER IN: 校准脉冲 (清零两边时间计数器)
```

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| FPGA 时钟 | 200 MHz | 5 ns/tick |
| FPGA_TICKS_PER_1HZ | **200,000,020** | 硬件实测值，比标称 200M 多 20 ticks (+100 ppm) |
| Digitizer 时钟 | V1742 | 8.5 ns/cycle (DRS4) |
| Digitizer 采样率 | 5 GHz | 200 ps/sample |
| 示波器采样率 | 10 GS/s | 100 ps/sample, 50Ω DC 耦合 |

## 数据量

| 系统 | 总数 | 分类 | 时间跨度 |
|------|------|------|---------|
| **Digitizer** | **5,006** events | 558 个 1Hz + **4,448** 个 TR 触发 | ~595 s |
| **FPGA OSCI** | **26,230** events | 全部为物理触发 (OSCI) | ~594 s |
| **示波器 (C4)** | **8** 个 TRC 文件 | 展开后 **22,400** segments | — |

## 匹配流程

### Step 1 — 分离 1Hz 校准事件

**方法**: 检查 TR 通道 (ch32-35) 波形的尾部-基线差异

```python
tail_diff = |mean(waveform[-50:]) - mean(waveform[:20])|
is_1hz = max_tail_diff < 5.0   # 阈值 5.0 ADC 单位
```

- **1Hz 事件** — TR 通道无脉冲信号 (来自 FPGA GSYNC, 绕过甄别器)
- **TR 触发** — TR 通道有脉冲信号 (来自示波器 AUX OUT → 甄别器)

同时计算每个 1Hz 事件的理论 FPGA tick (`theory_ft`):

```python
for i in range(len(hz_events)):
    n_s = 1 if i == 0 else round(ΔTT / NOM_1S)
    theory_ft[i] += n_s * FPGA_TICKS_PER_1HZ
    # FPGA_TICKS_PER_1HZ = 200,000,020 (硬件实测)
```

### Step 2 — 时钟标定 (关键)

将 Digitizer 的 `trigger_time_tag` (DRS4 时钟域, 8.5 ns/cycle) **分段线性映射** 到 FPGA tick 域 (5 ns/tick):

```
对每对相邻1Hz事件:
  f_pd = Δtheory_ft / ΔTT          # FPGA ticks per DRS4 cycle
       ≈ 200,000,020 / 117,186,474
       ≈ 1.70668

对任意事件 t:
  找到所在区间 s = searchsorted(tt_1hz, t) - 1
  t_ft = theory_ft[s] + (t - tt_1hz[s]) × f_pd[s]
```

> **经验**: `FPGA_TICKS_PER_1HZ` 必须用硬件实测值 (200,000,020 而非 200,000,000)。用错会引入系统性时钟漂移 (~20 ticks/s)。

### Step 3 — Δt 峰值定位

对所有 Digi TR 事件, 在 FPGA 时间轴中定位, 取附近 ±30 FPGA 事件:

```python
for each Digi TR event:
    ni = searchsorted(f_tick, t_dig)
    for fi in range(ni-30, ni+30):
        all_dt.append(t_dig - f_tick[fi])

hist(all_dt, bins=np.arange(-5000, 5001, 0.5))  # bins = 0.5 tick
peak = bin_center[argmax(counts)]                # Δt peak
```

精细直方图 (0.5 tick bins) 找到的峰值:

```
Δt 峰值 = -12.8 ticks (-64 ns)    ← 信号传输延迟
σ = 1.1 ticks (5.5 ns)           ← 精度极高
```

### Step 4 — 逐事件匹配

```python
for each Digi TR event td:
    expected_fpga_time = td - peak
    ni = searchsorted(f_tick, expected_fpga_time)
    for fi in range(ni-15, ni+15):
        if used(fi): skip
        if |td - f_tick[fi] - peak| < 3σ:
            matched! → 标记 used(fi)
```

结果:

| 指标 | 值 |
|------|-----|
| 匹配数 | **4,408 / 4,448 (99.1%)** |
| Δt σ | **1.1 ticks (5.5 ns)** |
| 未匹配 | 40 个 (0.9%) |

## 关键经验

### 1. FPGA 时钟频率必须用实测值

用 `200,000,000` (标称) vs `200,000,020` (实测):

| 参数 | 结果 |
|------|------|
| 200,000,000 | 每个 spill offset 漂移 ~-21 ticks/s, 匹配率 < 5% |
| **200,000,020** | offset 恒定 -12.8 ticks, **匹配率 99.1%** |

### 2. 1Hz/物理触发需用波形区分

不能只靠 Δt 阈值 (>0.5s) 区分, 要用 TR 通道波形:

- **过度简单**: `tt_gap > 0.5s → 1Hz` (误分类多)
- **正确方法**: `|tail_mean - head_mean| < 5 → 1Hz` (TR 通道无脉冲)

### 3. 直方图 bin 大小很重要

| bin 大小 | 效果 |
|----------|------|
| 200 bins in ±10000 ticks | 找不到精细 offset, 匹配率 0% |
| **0.5 tick × 20000 bins** | 找到 -12.8 tick 峰值, 匹配率 99.1% |

### 4. Spill 分离

基于 FPGA 时间戳的 spill 分离 (Δt > 50ms) 效果很好:

- FPGA: 29 spills, 每个 ~1000-1800 事件
- Digi: 分配到相同 spill, 每个 ~180-290 事件
- 逐 spill 匹配 vs 全局匹配: 用全局 offset + 精细直方图即可, 无需逐 spill

## 文件结构

```
scripts/
├── match/
│   ├── step1_separate_1hz.py      — 1Hz 分离
│   ├── step2_calibrate_clock.py   — 时钟标定
│   ├── step3_match_by_spill.py    — 基于spill的匹配
│   └── digi_fpga_matching.py      — Digi-FPGA匹配
├── analyze_lecroy_wfm.py          — TRC Sequence 模式分析
├── match_timestamps.py            — 三系统时间戳匹配
└── plot_all_trc.py                — 批量绘图

data/BT_20260530_010301_run_0001/
└── temp/
    ├── tr_waveform_analysis.npz       — Step 1 结果
    ├── corrected_timetags.npz         — Step 2 结果
    ├── spill_matched_pairs.npz        — Step 3 结果
    └── matched_pairs.npz              — 最终匹配对

debug/                                 — 诊断图
plots/                                 — 完整分析图
```
