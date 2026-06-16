import sys, os, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from src.data.fpga_parser import load_fpga_events, FPGA_CLOCK_HZ
from src.data.raw_data_recorder import load_binary_events
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── 配置 ──
RUN = str(REPO.parent / 'Analyze' / 'data' / 'BT_20260530_010301_run_0001')
OUT = str(REPO / 'plots')
os.makedirs(OUT, exist_ok=True)
DIGI_NS = 8.5
FPGA_1HZ = 200_000_020

# ═══ Step 1: 分离1Hz ═══
digi = load_binary_events(os.path.join(RUN, 'digitizer', 'V1742_events.bin'))
n = len(digi)
tt = np.array([e['trigger_time_tag'] for e in digi])
ev = np.array([e['event_number'] for e in digi])

TR_CH = [32, 33, 34, 35]
N_HEAD, N_TAIL = 20, 50
td = np.full((n, 4), np.nan)
for i, e in enumerate(digi):
    wf = e['waveforms']
    for j, ch in enumerate(TR_CH):
        if ch not in wf:
            continue
        d = wf[ch]
        ns = len(d)
        h = min(N_HEAD, ns)
        tl = min(N_TAIL, ns)
        td[i, j] = abs(np.mean(d[:h])-np.mean(d[-tl:]))
is_hz = np.nanmax(td, axis=1) < 5.0
hz_idx = np.where(is_hz)[0]
tt_1hz = tt[hz_idx]

# theory ft
theory_ft = np.zeros(len(hz_idx))
cum = 0.0
NOM_1S = 1e9/DIGI_NS
for i in range(len(hz_idx)):
    n_s = 1 if i == 0 else max(1, round((tt_1hz[i]-tt_1hz[i-1])/NOM_1S))
    cum += n_s*FPGA_1HZ
    theory_ft[i] = cum

# ═══ Step 2: 转换到FPGA tick域 ═══
dt_ft = np.diff(theory_ft)
dt_cyc = np.diff(tt_1hz)
f_pd = dt_ft.astype(float)/dt_cyc
f_pd[0] = np.mean(f_pd[1:])

t_ft = np.zeros(n)
for i in range(n):
    s = int(np.searchsorted(tt_1hz, tt[i], side='right'))-1
    s = max(0, min(len(tt_1hz)-2, s))
    t_ft[i] = theory_ft[s]+(tt[i]-tt_1hz[s])*f_pd[s]

# ═══ 取数据 ═══
tr_idx = np.where(~is_hz)[0]
dn_tr, dt_tr, tc_tr = ev[tr_idx], tt[tr_idx], t_ft[tr_idx]
dn_ri = np.arange(len(dn_tr))

fp = load_fpga_events(os.path.join(RUN, 'fpga', 'fpga_events.bin'))
fi = np.array([e.trigger_id for e in fp])
ft = np.array([e.t_fpga for e in fp])

# ═══ 图1: Digi原始 ═══
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(dn_ri, dt_tr.astype(float)*DIGI_NS*1e-9, 'b-', lw=0.6)
ax.set_xlabel('TR event index (re-indexed)')
ax.set_ylabel('Time (s)')
ax.set_title('Digi raw trigger_time_tag vs TR event index (1Hz removed)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_digi_raw_not1hz.png'), dpi=150)
plt.close()
print('fig1 done')

# ═══ 图2: Digi校正 ═══
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(dn_ri, tc_tr/FPGA_CLOCK_HZ, 'g-', lw=0.6)
ax.set_xlabel('TR event index (re-indexed)')
ax.set_ylabel('Time (s)')
ax.set_title('Digi corrected to FPGA domain vs TR event index (1Hz removed)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_digi_corrected_not1hz.png'), dpi=150)
plt.close()
print('fig2 done')

# ═══ 图3: FPGA ═══
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(fi, ft.astype(float)/FPGA_CLOCK_HZ, 'r-', lw=0.6)
ax.set_xlabel('trigger_id')
ax.set_ylabel('Time (s)')
ax.set_title('FPGA OSCI timestamp vs trigger_id')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_fpga_raw.png'), dpi=150)
plt.close()
print('fig3 done')

print(f'\\nAll saved to {OUT}/')
