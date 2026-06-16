# 示波器 ↔ FPGA Spill 匹配经验

## 匹配策略

### 1. 构建示波器时间戳

每个 TRC 文件包含 2800 个 Sequence 模式 segments。时间戳计算：
```python
# 文件的 UTC 触发时间 (triggerTime 字段, WAVEDESC 中)
file_utc = triggerTime → unix timestamp

# 每个 segment 的绝对 UTC 时间
seg_utc = file_utc + trigger_time_offset[seg_idx]  # trigger_time_offset 来自 TRIG_TIME_ARRAY
```

跨文件无需估算——直接用每个文件自身的 `triggerTime` + 内部 `trigger_time_offsets`。

### 2. Spill 内部事件匹配方法

**RMS 扫描法**：

1. 选择 FPGA 和 Scope 中待匹配的 spill，各取前 N 个事件
2. 对 index offset ∈ [-5, +5]：
   - offset ≥ 0：`Δt_i = t_scope[i+offset] - t_fpga[i]`
   - offset < 0：`Δt_i = t_scope[i] - t_fpga[i-offset]`
3. 计算 `RMS = sqrt(mean(Δt²))`
4. 最小 RMS 对应的 offset 即为最佳匹配

### 3. 匹配结果

| FPGA Spill | 事件数 | Scope Spill | seg数 | 最佳offset | RMS | 结论 |
|-----------|--------|------------|-------|-----------|-----|------|
| 1 | 1721 | 0 | 1721 | 0 | 0.08 μs | ✅ 完美匹配 |
| 3 | 1078 | 待确认 | | | | |

### 4. 关键经验

- **Spill 间隔太规律**（~56s大/~9.6s小交替），无法通过间隔匹配 spill
- **RMS 扫描法**简单有效：同一 spill 内事件按序号 1:1 对应
- N=100 个事件足够判定最佳 offset
- FPGA Spill 1 ↔ Scope Spill 0 时间零点差 < 80 ns（几乎完美对齐）
