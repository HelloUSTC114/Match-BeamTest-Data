"""绘制示波器和FPGA的spill起始时间对比图"""
import sys, os, numpy as np, matplotlib, matplotlib.pyplot as plt, re, glob, importlib.util, warnings
from pathlib import Path
warnings.filterwarnings('ignore'); matplotlib.use('Agg')

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from src.data.fpga_parser import load_fpga_events, FPGA_CLOCK_HZ
import lecroyparser
spec=importlib.util.spec_from_file_location('awm', str(REPO / 'scripts' / 'analyze_lecroy_wfm.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); parse_trc=m.parse_trc_binary

RUN=str(REPO.parent / 'Analyze' / 'data' / 'BT_20260530_010301_run_0001')
OUT=str(REPO / 'plots')
os.makedirs(OUT, exist_ok=True)

# ===== FPGA =====
fp=load_fpga_events(os.path.join(RUN,'fpga','fpga_events.bin'))
f_time=np.array([e.t_fpga for e in fp]).astype(float)/FPGA_CLOCK_HZ
f_dt=np.diff(f_time)*1000; f_edges=np.where(f_dt>50)[0]
f_sp=np.zeros(len(f_time),dtype=int)
for e in f_edges: f_sp[e+1:]+=1; n_fsp=f_sp.max()+1

f_sp_starts=[]; f_sp_ends=[]
for sp in range(n_fsp):
    m=f_sp==sp
    if m.sum()>5:
        f_sp_starts.append(f_time[m][0])
        f_sp_ends.append(f_time[m][-1])
print(f'FPGA: {len(f_sp_starts)} large spills')

# ===== 示波器 =====
ch_files=sorted(glob.glob(os.path.join(RUN,'lecroy_wfm','C4--Trace--*.trc')))
all_segs=[]
for fp_ in ch_files:
    fi=int(re.search(r'--Trace--(\d+)\.trc',os.path.basename(fp_)).group(1))
    info=parse_trc(fp_); ns=info['n_segments']; off=info['trig_time_offsets']
    for si in range(ns):
        all_segs.append({'fi':fi,'si':si,'off':float(off[si])})
all_segs.sort(key=lambda x:x['fi']*2800+x['si'])

# 构建绝对时间: 每个TRC文件内,第一个segment的offset=0,最后一个≈10秒(2800*20.2ns)
# 文件之间: 下一个文件的第一个segment = 上一个文件最后一个segment的时间 + 间隔
# 用文件采集的总时长来估算: 每个文件内2800 segments × 20.2ns/seg ≈ 56.56μs? 不对
# 实际上每个segment之间是连续的,触发间隔就是trigger_time_offset
# 所以文件内最后一个segment的offset ≈ 总采集时长
s_abs_time=np.zeros(len(all_segs))
prev_end=0
for fi in range(len(ch_files)):
    segs_in_f=[s for s in all_segs if s['fi']==fi]
    if not segs_in_f: continue
    # 第一个segment offset = 0
    for s in segs_in_f:
        s_abs_time[all_segs.index(s)] = prev_end + s['off']
    # 推算下一个文件的开始时间 = 这个文件结束时间 + 约56秒(束流周期)
    last_off=segs_in_f[-1]['off']
    prev_end = prev_end + last_off + 56  # 估算56秒gap

# 用实际数据: 文件间的gap按FPGA的时间来算
# 直接用seq index做spill检测: 文件间自然有gap
s_t=np.array([s['fi']*2800+s['si'] for s in all_segs]).astype(float)

# 示波器spills (在seq index轴上检测)
s_dt=np.diff(s_t); s_edges=np.where(s_dt>100)[0]
s_sp=np.zeros(len(s_t),dtype=int)
for e in s_edges: s_sp[e+1:]+=1; n_ssp=s_sp.max()+1

s_sp_starts=[]
for sp in range(n_ssp):
    m=s_sp==sp
    if m.sum()>5:
        s_sp_starts.append(float(np.where(m)[0][0]))
print(f'Scope C4: {len(s_sp_starts)} large spills')

# ===== 图: spill起始时间对比 =====
fig,ax=plt.subplots(figsize=(16,6))
ax.plot(range(len(f_sp_starts)), f_sp_starts, 'ro-', ms=6, lw=1.5, label=f'FPGA ({len(f_sp_starts)} spills)')
ax.plot(range(len(s_sp_starts)), s_sp_starts, 'bs-', ms=6, lw=1.5, label=f'Scope C4 ({len(s_sp_starts)} spills)')
ax.set_xlabel('Spill index'); ax.set_ylabel('Spill start time (s)')
ax.set_title('Spill start times: FPGA vs Scope C4')
ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig7_spill_start_times.png'),dpi=150); plt.close()
print(f'Saved fig7_spill_start_times.png')

# ===== 图: 对齐时间轴 =====
# 将示波器的spill索引乘以2(FPGA有~2倍spill)
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(16,8),sharex=True)
ax1.plot(f_sp_starts,'ro-',ms=5,lw=1,label='FPGA')
ax1.set_ylabel('Spill start (s)'); ax1.set_title('FPGA spill starts'); ax1.grid(True,alpha=0.3); ax1.legend()

# 示波器用实际segment序号
seg_indices=np.array([s['fi']*2800+s['si'] for s in all_segs])
ax2.plot(s_sp_starts,'bs-',ms=5,lw=1,label='Scope C4 (seg index)')
ax2.set_xlabel('Spill index'); ax2.set_ylabel('Seg index')
ax2.set_title(f'Scope C4 spill starts ({len(all_segs)} segs)'); ax2.grid(True,alpha=0.3); ax2.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig7_spill_separate.png'),dpi=150); plt.close()
print(f'Saved fig7_spill_separate.png')

print(f'\nAll: {OUT}/fig7_*.png')
