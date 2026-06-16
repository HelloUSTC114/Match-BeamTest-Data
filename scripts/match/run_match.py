"""Step 1: 区分Digitizer中的1Hz校准事件和物理触发事件"""
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import numpy as np
from pathlib import Path

# ── 仓库根目录 ──
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from src.data.raw_data_recorder import load_binary_events

matplotlib.use('Agg')

# ── 配置: 修改 RUN 指向你的数据目录 ──
RUN = str(REPO.parent / 'Analyze' / 'data' / 'BT_20260610_094205-run_0018')
TMP = os.path.join(RUN, 'temp')
os.makedirs(TMP, exist_ok=True)

DIGI_NS = 8.5
TR_CH = [32, 33, 34, 35]
N_HEAD, N_TAIL = 20, 50

# 加载Digi
digi = load_binary_events(os.path.join(RUN, 'digitizer', 'V1742_events.bin'))
n = len(digi)
d_tt = np.array([e['trigger_time_tag'] for e in digi], dtype=np.int64)
d_ev = np.array([e['event_number'] for e in digi], dtype=np.int64)
d_et = np.array([e.get('event_time_tag', 0) for e in digi], dtype=np.int64)

print(f'Digitizer: {n} events, event# {d_ev[0]}~{d_ev[-1]}')

# ── 计算TR通道尾部-基线差异 ──
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

# ── 区分1Hz vs TR ──
# 1Hz GSYNC脉冲: TR通道无信号 → tail_diff ~ 0
# 物理触发: TR通道有信号 → tail_diff > 阈值
TR_THRESH = 5.0
is_hz = max_tail < TR_THRESH
n_hz = is_hz.sum()
n_tr = n - n_hz
print(f'1Hz事件: {n_hz}, 物理触发事件: {n_tr}')

# ── 检查分类质量 ──
hz_idx = np.where(is_hz)[0]
tr_idx = np.where(~is_hz)[0]
print(f'1Hz索引(前20): {hz_idx[:20].tolist()}')
print(f'物理触发索引(前20): {tr_idx[:20].tolist()}')

# 检查1Hz事件的间隔
d_dt_s = np.diff(d_tt).astype(float) * DIGI_NS * 1e-9
print(f'\n1Hz事件的Δt:')
for i, idx in enumerate(hz_idx[:20]):
    dt = d_dt_s[idx-1] if idx > 0 else -1
    dt_n = d_dt_s[idx] if idx < n-1 else -1
    print(
        f'  ev#{d_ev[idx]:5d}  ttag={d_tt[idx]:15d}  dt_before={dt:.3f}s  dt_after={dt_n:.3f}s')

# ── 保存 ──
np.savez(os.path.join(TMP, 'classified_events.npz'),
         ev_numbers=d_ev, tt_tags=d_tt,
         is_hz=is_hz, hz_idx=hz_idx, tr_idx=tr_idx,
         tail_diff=tail_diff, max_tail=max_tail,
         threshold=TR_THRESH)
print(f'\n已保存: {TMP}/classified_events.npz')

# ── 绘图 ──
plt_dir = os.path.join(TMP, 'plots')
os.makedirs(plt_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(max_tail, bins=100, color='steelblue', alpha=0.8, edgecolor='none')
ax.axvline(TR_THRESH, color='r', ls='--', label=f'threshold={TR_THRESH}')
ax.set_xlabel('max |tail-head| (ADC)')
ax.set_ylabel('Count')
ax.set_title(f'TR tail-baseline diff (1Hz={n_hz}, TR={n_tr})')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(plt_dir, 'step1_classification.png'), dpi=150)
plt.close()
print(f'图 -> {plt_dir}/step1_classification.png')
