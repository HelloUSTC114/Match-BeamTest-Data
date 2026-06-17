"""
数据加载器 — 通过匹配表自动定位 Digitizer & 示波器波形
==========================================================
用于击中位置重建的前端数据接口。

输入:  raw data 目录路径 (含 digitizer/, lecroy_wfm/, temp/)
输出: 可迭代的事件对象，每个事件包含 Digitizer 波形 + OSC 波形

用法:
  python scripts/pos-reconstruction/load_events.py --src <raw_dir> --n <count> [--plot]

若无 temp/full_event_table.csv (未匹配):
  → 提示用户运行 run_pipeline.py 进行三系统匹配
"""

import sys, os, argparse, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import namedtuple

import numpy as np

# ── 仓库路径 ──
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.data.raw_data_recorder import load_binary_events

# 导入 parse_trc_binary (用于正确提取 Sequence 模式 segment)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "awm", str(REPO / "scripts" / "analyze_lecroy_wfm.py"))
_awm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_awm)
parse_trc_binary = _awm.parse_trc_binary

try:
    import lecroyparser
except ImportError:
    lecroyparser = None

warnings.filterwarnings('ignore')

# ======================================================================
# 通道映射 (来自 exp-logs/tracker-timing-resolution/README.md)
# ======================================================================

# Digitizer 物理通道 → 标签
DIGI_CH_INFO = {
    1:  ('ch1',  'Draw1 BT2, elec1', '#e74c3c'),
    2:  ('ch2',  'Draw4 BT4, elec4', '#2ecc71'),
    3:  ('ch3',  'Draw4 BT4, elec7', '#27ae60'),
    4:  ('ch4',  'Draw4 BT4, elec6', '#1abc9c'),
    5:  ('ch5',  'Draw4 BT4, elec5', '#16a085'),
    6:  ('ch6',  'Draw4 BT4, elec3', '#2ecc71'),
    7:  ('ch7',  'Draw6 M4,  elec6', '#3498db'),
    15: ('ch15', 'Draw1 BT2, elec2', '#e67e22'),
}

# 示波器通道 → 标签
OSC_CH_INFO = {
    'C3': ('C3', 'Draw6 M4,  elec4', '#3498db'),
    'C4': ('C4', 'Draw6 M4,  elec5', '#2980b9'),
    'C5': ('C5', 'Draw6 M4,  elec3', '#1f618d'),
    'C6': ('C6', 'Draw1 BT2, elec3', '#e74c3c'),
    'C7': ('C7', 'Draw1 BT2, elec4', '#c0392b'),
    'C8': ('C8', 'Draw1 BT2, elec5', '#e67e22'),
}

SAMPLING_PS = 200  # V1742: 5 GHz
NS_PER_SAMPLE = SAMPLING_PS / 1000.0
RECORD_LENGTH = 1024

# ======================================================================
# 数据结构
# ======================================================================

class MatchedEvent:
    """一个三系统匹配事件"""
    __slots__ = (
        'fpga_trigger_id', 'fpga_tick', 'dt_ns',
        'has_scope', 'has_digi',
        'scope_file_idx', 'scope_seg_idx',
        'digi_evnum', 'digi_waveforms',
        'scope_waveforms', 'scope_t_ns',
    )

    def __init__(self):
        self.fpga_trigger_id = -1
        self.fpga_tick = 0
        self.dt_ns = 0.0
        self.has_scope = False
        self.has_digi = False
        self.scope_file_idx = -1
        self.scope_seg_idx = -1
        self.digi_evnum = -1
        self.digi_waveforms: Dict[int, np.ndarray] = {}
        self.scope_waveforms: Dict[str, np.ndarray] = {}
        self.scope_t_ns: Optional[np.ndarray] = None

    def __repr__(self):
        parts = [f"Evt(fpga_id={self.fpga_trigger_id}"]
        if self.has_digi:
            parts.append(f"digi=#{self.digi_evnum}")
        if self.has_scope:
            parts.append(f"scope=CX@{self.scope_file_idx}:{self.scope_seg_idx}")
        parts.append(f"dt={self.dt_ns:.2f}ns")
        return ", ".join(parts) + ")"


# ======================================================================
# 核心加载逻辑
# ======================================================================

class EventLoader:
    """
    通过匹配表加载三系统数据。

    Parameters
    ----------
    src_dir : str
        原始数据目录，含 digitizer/, lecroy_wfm/, temp/。
    """
    def __init__(self, src_dir: str):
        self.src_dir = Path(src_dir)
        self.temp_dir = self.src_dir / 'temp'
        self._digi_file = self.src_dir / 'digitizer' / 'V1742_events.bin'
        self._fpga_file = self.src_dir / 'fpga' / 'fpga_events.bin'
        self._scope_dir = self.src_dir / 'lecroy_wfm'

        self._digi_data: List[dict] = []
        self._csv_rows: List[dict] = []
        # _scope_cache: (trace_num, seg_idx) → (t_ns, {ch: waveform})
        self._scope_cache: Dict[Tuple[int, int], Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}
        # _scope_trace_map: trace_num → {ch: filepath}
        self._scope_trace_map: Dict[int, Dict[str, str]] = {}
        # _scope_t_info: trace_num → (n_segments, samples_per_seg, horiz_interval_ns)
        self._scope_t_info: Dict[int, Tuple[int, int, float]] = {}
        # _scope_trace_cache: trace_num → {ch: full_voltages}  (预加载，避免重复 parse)
        self._scope_trace_cache: Dict[int, Dict[str, np.ndarray]] = {}

        self._validate()
        self._load_match_table()
        self._index_scope_files()

    # ── 验证 ──────────────────────────────────────────────

    def _validate(self):
        """验证输入目录结构，未匹配则提示。"""
        csv_path = self.temp_dir / 'full_event_table.csv'
        if not csv_path.exists():
            print(f"\n{'='*60}")
            print(f"  ⚠️  未找到匹配结果文件:")
            print(f"     {csv_path}")
            print(f"\n  该 run 尚未进行三系统匹配。")
            print(f"  请先运行匹配流水线:\n")
            print(f"    python scripts/match/run_pipeline.py \\")
            print(f"      --src \"{self.src_dir}\"")
            if self.src_dir.name.startswith('run_'):
                print(f"      --out \"data/BT_*/{self.src_dir.name}/temp\"")
            print(f"\n{'='*60}\n")
            sys.exit(1)

        if not self._digi_file.exists():
            print(f"⚠️  Digitizer 数据不存在: {self._digi_file}")
            sys.exit(1)

    # ── 加载匹配表 ────────────────────────────────────────

    def _load_match_table(self):
        """读取 temp/full_event_table.csv。"""
        import csv
        csv_path = self.temp_dir / 'full_event_table.csv'
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            self._csv_rows = list(reader)
        print(f"✅ 已加载匹配表: {len(self._csv_rows)} 行")

    # ── 索引示波器文件 ────────────────────────────────────

    def _index_scope_files(self):
        """
        建立 scope_file_idx → 所有通道文件路径的映射。

        scope_file_idx 是 trc 文件名中的数字后缀 (如 C3--Trace--00012.trc → 12)。
        同一个 file_idx 对应所有通道 (C3-C8) 的同序号文件，
        它们是在同一次触发下由不同通道同时保存的。

        存储结构:
          self._scope_trace_map[trace_num] = {'C3': path, 'C4': path, ...}
          self._scope_t_info[trace_num] = (n_segments, samples_per_seg, horiz_interval_ns)
        """
        if not self._scope_dir.exists():
            return

        import re
        self._scope_trace_map: Dict[int, Dict[str, str]] = {}
        self._scope_t_info: Dict[int, Tuple[int, int, float]] = {}

        trc_files = sorted(self._scope_dir.glob('*.trc'))
        for p in trc_files:
            basename = p.name
            # 文件名格式: C3--Trace--00012.trc
            m = re.match(r'(C\d+)--Trace--(\d+)\.trc', basename)
            if not m:
                continue
            ch = m.group(1)       # e.g. "C3"
            tnum = int(m.group(2))  # e.g. 12

            if tnum not in self._scope_trace_map:
                self._scope_trace_map[tnum] = {}
            self._scope_trace_map[tnum][ch] = str(p)

        # 为每个 trace_num 读取 sequence 元信息
        for tnum, ch_paths in self._scope_trace_map.items():
            # 取任意一个通道的文件读取元信息 (所有通道参数相同)
            sample_path = next(iter(ch_paths.values()))
            try:
                info = parse_trc_binary(sample_path)
                self._scope_t_info[tnum] = (
                    info['n_segments'],
                    info['samples_per_seg'],
                    info['horiz_interval'] * 1e9,  # 转为 ns
                )
            except Exception:
                pass

        n_traces = len(self._scope_trace_map)
        n_total = sum(len(v) for v in self._scope_trace_map.values())
        print(f"✅ 已索引示波器文件: {n_traces} 个 trace, 共 {n_total} 个文件")
        print(f"   通道: {sorted(next(iter(self._scope_trace_map.values())).keys())}")

    # ── 加载 Digitizer ────────────────────────────────────

    def _ensure_digi_loaded(self):
        if self._digi_data:
            return
        print(f"⏳ 加载 Digitizer 原始数据...")
        self._digi_data = load_binary_events(str(self._digi_file))
        print(f"✅ Digitizer: {len(self._digi_data)} 事件")

    # ── 加载示波器 segment ─────────────────────────────────

    def _ensure_trace_loaded(self, tnum: int):
        """预加载 trace 的所有通道完整波形（只解析一次）。"""
        if tnum in self._scope_trace_cache:
            return
        if tnum not in self._scope_trace_map or tnum not in self._scope_t_info:
            return

        ch_paths = self._scope_trace_map[tnum]
        waveforms_all: Dict[str, np.ndarray] = {}
        for ch, path in ch_paths.items():
            try:
                info = parse_trc_binary(path)
                waveforms_all[ch] = info['voltages']
            except Exception:
                pass
        if waveforms_all:
            self._scope_trace_cache[tnum] = waveforms_all

    def _load_scope_segment(self, file_idx: int, seg_idx: int) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        加载指定 trace_num 的第 seg_idx 个 segment 的【所有示波器通道】波形。

        LeCroy Sequence 模式: 每个 .trc 文件包含 n_segments 个波形,
        首尾拼接存储在文件末尾。需要按 samples_per_seg 切片提取。

        Args:
            file_idx: trc 文件名中的数字后缀 (trace number)
            seg_idx:  segment 索引 (0-based)

        Returns:
            (t_ns, waveforms_dict) 或 None
            waveforms_dict 格式: {'C3': array, 'C4': array, ...}
        """
        cache_key = (file_idx, seg_idx)
        if cache_key in self._scope_cache:
            return self._scope_cache[cache_key]

        if file_idx not in self._scope_trace_map or file_idx not in self._scope_t_info:
            return None

        n_segs, samples_per_seg, horiz_interval_ns = self._scope_t_info[file_idx]
        if seg_idx >= n_segs:
            return None

        seg_start = seg_idx * samples_per_seg
        seg_end = seg_start + samples_per_seg
        t_ns = np.arange(samples_per_seg) * horiz_interval_ns

        # 预加载 trace 全波形（只解析一次文件）
        self._ensure_trace_loaded(file_idx)
        if file_idx not in self._scope_trace_cache:
            return None

        waveforms: Dict[str, np.ndarray] = {}
        for ch, voltage_all in self._scope_trace_cache[file_idx].items():
            waveforms[ch] = voltage_all[seg_start:seg_end]

        if not waveforms:
            return None

        self._scope_cache[cache_key] = (t_ns, waveforms)
        return self._scope_cache[cache_key]

    # ── 公开 API ──────────────────────────────────────────

    def get_event(self, row_idx: int) -> Optional[MatchedEvent]:
        """获取第 row_idx 行的匹配事件。"""
        if row_idx >= len(self._csv_rows):
            return None
        row = self._csv_rows[row_idx]
        evt = MatchedEvent()

        evt.fpga_trigger_id = int(row['fpga_trigger_id'])
        evt.fpga_tick = int(row['fpga_tick'])
        evt.dt_ns = float(row['dt_ns'])
        evt.has_digi = int(row.get('has_digi', '0')) == 1

        if evt.has_digi:
            evt.digi_evnum = int(row['digi_evnum'])
            self._ensure_digi_loaded()
            if evt.digi_evnum < len(self._digi_data):
                evt.digi_waveforms = self._digi_data[evt.digi_evnum]['waveforms']
            else:
                evt.has_digi = False

        evt.has_scope = int(row.get('has_scope', '0')) == 1
        if evt.has_scope:
            evt.scope_file_idx = int(row['scope_file_idx'])
            evt.scope_seg_idx = int(row['scope_seg_idx'])
            result = self._load_scope_segment(evt.scope_file_idx, evt.scope_seg_idx)
            if result:
                evt.scope_t_ns, evt.scope_waveforms = result
            else:
                evt.has_scope = False

        return evt

    def iter_events(self, n: int = None, require_both: bool = False):
        """迭代匹配事件。"""
        total = min(n, len(self._csv_rows)) if n else len(self._csv_rows)
        for i in range(total):
            evt = self.get_event(i)
            if evt is None:
                continue
            if require_both and not (evt.has_digi and evt.has_scope):
                continue
            yield evt

    @property
    def n_matched(self) -> int:
        return len(self._csv_rows)

    @property
    def scope_trace_num_at(self, file_idx: int) -> Optional[str]:
        """返回 file_idx (trace number) 对应的示波器通道列表。"""
        ch_paths = self._scope_trace_map.get(file_idx, {})
        if ch_paths:
            return ', '.join(sorted(ch_paths.keys()))
        return None


# ======================================================================
# 绘图辅助
# ======================================================================

def plot_event(evt: MatchedEvent, loader: EventLoader,
               out_path: Optional[str] = None,
               show_scope: bool = True):
    """绘制单个匹配事件的 Digitizer + OSC 波形。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_plots = 1 + (1 if (show_scope and evt.has_scope) else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]

    t_ns_digi = np.arange(RECORD_LENGTH) * NS_PER_SAMPLE

    # ── Digitizer 波形 ──
    ax = axes[0]
    for ch_key, (ch_name, ch_label, color) in DIGI_CH_INFO.items():
        if ch_key in evt.digi_waveforms:
            ax.plot(t_ns_digi, evt.digi_waveforms[ch_key],
                    color=color, lw=0.8, alpha=0.85,
                    label=f'{ch_name} ({ch_label})')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('ADC')
    ax.set_title(f'Digitizer — Event #{evt.fpga_trigger_id}  (digi_evt=#{evt.digi_evnum})')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── 示波器波形 ──
    if show_scope and evt.has_scope and evt.scope_t_ns is not None:
        ax = axes[1]
        for ch, wf in evt.scope_waveforms.items():
            info = OSC_CH_INFO.get(ch, (ch, ch, 'gray'))
            ax.plot(evt.scope_t_ns, wf, color=info[2], lw=0.8, alpha=0.85,
                    label=f'{info[0]} ({info[1]})')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Oscilloscope — file_idx={evt.scope_file_idx}, seg={evt.scope_seg_idx}')
        ax.legend(fontsize=7, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='事件数据加载器 — 通过匹配表自动定位 Digi + OSC 波形')
    parser.add_argument('--src', required=True,
                        help='原始数据目录 (含 digitizer/, lecroy_wfm/, temp/)')
    parser.add_argument('--n', type=int, default=5,
                        help='加载的事件数 (默认: 5)')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='绘制波形并保存到 plots/pos-reco/')
    parser.add_argument('--require-both', action='store_true', default=False,
                        help='仅加载三系统均匹配的事件')
    args = parser.parse_args()

    loader = EventLoader(args.src)

    print(f"\n📊 匹配统计: {loader.n_matched} 个 FPGA 触发事件\n")

    if args.plot:
        out_dir = REPO / 'plots' / 'pos-reco'
        out_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    count = 0
    for evt in loader.iter_events(n=args.n, require_both=args.require_both):
        print(f"  {evt}")
        if args.plot:
            out_path = str(out_dir / f'evt_{count:04d}_fpga{evt.fpga_trigger_id}.png')
            plot_event(evt, loader, out_path=out_path)
            print(f"    → 已保存: {out_path}")
        count += 1

    print(f"\n✅ 共加载 {count} 个事件")


if __name__ == '__main__':
    main()
