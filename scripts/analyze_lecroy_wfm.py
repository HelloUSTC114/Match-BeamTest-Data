"""
分析 LeCroy WR8208HD 示波器 Sequence 模式 TRC 文件

功能:
  - 检测 TRC 文件是否为 Sequence 模式
  - 统计每个文件中的 segment 数量
  - 解析数据格式 (采样率、垂直增益、触发时间等)
  - 汇总所有通道和 run 的统计信息

依赖: lecroyparser, numpy, matplotlib
"""

import os
import sys
import struct
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 尝试导入 lecroyparser
try:
    import lecroyparser
except ImportError:
    print("⚠️  lecroyparser 未安装，请运行: pip install lecroyparser")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
#  TRC 文件原始二进制解析 (不依赖 lecroyparser 的完整解析)
# ============================================================================

def read_wavedesc_field(data, desc_offset, field_offset, fmt):
    """从 WAVEDESC 中读取指定偏移的字段"""
    pos = desc_offset + field_offset
    if fmt == 'I':  # uint32
        return struct.unpack('<I', data[pos:pos+4])[0]
    elif fmt == 'H':  # uint16
        return struct.unpack('<H', data[pos:pos+2])[0]
    elif fmt == 'h':  # int16
        return struct.unpack('<h', data[pos:pos+2])[0]
    elif fmt == 'f':  # float32
        return struct.unpack('<f', data[pos:pos+4])[0]
    elif fmt == 'd':  # float64
        return struct.unpack('<d', data[pos:pos+8])[0]
    elif fmt == 's16':  # 16-byte string
        s = data[pos:pos+16]
        return s.decode('ascii', errors='replace').strip('\x00').strip()
    elif fmt == 'I64':  # uint64
        return struct.unpack('<Q', data[pos:pos+8])[0]
    return None


def parse_trc_binary(filepath):
    """
    解析 TRC 文件的二进制结构。

    返回 dict，包含:
      - instrument: 仪器型号
      - record_type: 记录类型
      - wave_count: 总采样点数
      - n_segments: 段数 (Sequence 模式下 > 1)
      - samples_per_seg: 每段采样数
      - horiz_interval: 采样间隔 (s)
      - horiz_offset: 水平偏移 (s)
      - vert_gain: 垂直增益
      - vert_offset: 垂直偏移
      - trigger_time: 触发时间 (字符串)
      - trig_time_offsets: 各段触发时间偏移数组 (Sequence 模式)
      - sampling_rate: 采样率 (Hz)
      - time_per_seg: 每段时间 (s)
      - has_sequence: 是否为 Sequence 模式
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    # 定位 WAVEDESC
    wavedesc_pos = data.find(b'WAVEDESC')
    if wavedesc_pos < 0:
        raise ValueError(f"无法在文件中找到 WAVEDESC 标记: {filepath}")

    DESC = wavedesc_pos

    # --- 读取 WAVEDESC 字段 ---
    # 偏移参考: LECROY_2_3 模板 (标准 LeCroy 格式)
    template_name = read_wavedesc_field(data, DESC, 16, 's16')
    comm_order = read_wavedesc_field(data, DESC, 34, 'H')
    comm_type = read_wavedesc_field(data, DESC, 32, 'H')

    # WAVEDESC 布局字段
    desc_size = read_wavedesc_field(data, DESC, 36, 'I')  # WAVE_DESCRIPTOR
    user_text = read_wavedesc_field(data, DESC, 40, 'I')  # USER_TEXT
    trig_time_arr = read_wavedesc_field(data, DESC, 48, 'I')  # TRIG_TIME_ARRAY
    wave_array_1 = read_wavedesc_field(
        data, DESC, 60, 'I')  # WAVE_ARRAY_1 (bytes)
    wave_array_2 = read_wavedesc_field(data, DESC, 64, 'I')  # WAVE_ARRAY_2

    # 仪器信息
    instrument_name = read_wavedesc_field(data, DESC, 76, 's16')
    instrument_num = read_wavedesc_field(data, DESC, 92, 'I')

    # 波形参数
    wave_array_count = read_wavedesc_field(data, DESC, 116, 'I')  # 总采样数
    vert_gain = read_wavedesc_field(data, DESC, 156, 'f')
    vert_offset = read_wavedesc_field(data, DESC, 160, 'f')
    nom_bits = read_wavedesc_field(data, DESC, 172, 'H')

    horiz_interval = read_wavedesc_field(data, DESC, 176, 'f')
    horiz_offset = read_wavedesc_field(data, DESC, 180, 'd')

    # 触发时间 (偏移参考 lecroyparser: offset 296 = 0x128)
    trig_second = read_wavedesc_field(data, DESC, 296, 'd')
    trig_minute = data[DESC + 304] if DESC + 304 < len(data) else 0
    trig_hour = data[DESC + 305] if DESC + 305 < len(data) else 0
    trig_day = data[DESC + 306] if DESC + 306 < len(data) else 0
    trig_month = data[DESC + 307] if DESC + 307 < len(data) else 0
    trig_year = read_wavedesc_field(data, DESC, 308, 'H')

    try:
        trigger_time_str = f"{trig_year}-{trig_month:02d}-{trig_day:02d} {trig_hour:02d}:{trig_minute:02d}:{trig_second:.2f}"
    except:
        trigger_time_str = f"{trig_year}-??-?? ??:??:??"

    # 记录类型 (offset 316 = 0x13C)
    record_type_idx = read_wavedesc_field(data, DESC, 316, 'H')
    record_types = [
        "single_sweep", "interleaved", "histogram", "graph",
        "filter_coefficient", "complex", "extrema", "sequence_obsolete",
        "centered_RIS", "peak_detect"
    ]
    record_type = record_types[record_type_idx] if record_type_idx < len(
        record_types) else f"unknown({record_type_idx})"

    # 波形源 (offset 344 = 0x158, WR8208HD 8通道: 0=C1..7=C8)
    wave_source_idx = read_wavedesc_field(data, DESC, 344, 'H')
    # WR8208HD 是 8 通道示波器，索引 0-7 对应 C1-C8
    if wave_source_idx < 8:
        wave_source = f"Channel {wave_source_idx + 1}"
    else:
        wave_source = f"Ch{wave_source_idx}"

    # --- Sequence 模式检测 ---
    n_segments = 1
    samples_per_seg = wave_array_count
    trig_time_offsets = None

    # 判断依据: TRIG_TIME_ARRAY > 0 且波形数据被分段
    # 在 LeCroy 的 Sequence 模式中，TRIG_TIME_ARRAY 存储每个 segments 的触发时间
    # 每 16 字节 = 1 个 trigger 记录 (8 bytes double 偏移 + 8 bytes 其他/水平偏移)
    has_sequence = (trig_time_arr > 0)

    if has_sequence:
        # 触发时间数组起始
        trig_start = DESC + desc_size + user_text
        trig_raw = data[trig_start:trig_start + trig_time_arr]

        # 从触发时间数组中提取: 每 16 字节一组
        # 格式: [offset_double, horizOffset_copy, offset_double, horizOffset_copy, ...]
        trig_doubles = np.frombuffer(
            trig_raw[:trig_time_arr], dtype=np.float64)
        trig_time_offsets = trig_doubles[0::2]  # 偶数索引 = trigger offset
        n_segments = len(trig_time_offsets)

        if n_segments > 0:
            samples_per_seg = wave_array_count // n_segments

    # 采样率和每段时长
    sampling_rate = 1.0 / horiz_interval if horiz_interval > 0 else 0
    time_per_seg = samples_per_seg * horiz_interval

    # --- 读取波形数据 ---
    wave_start = DESC + desc_size + user_text + trig_time_arr
    raw_wave_bytes = data[wave_start:wave_start + wave_array_1]

    # 按 16-bit 整数读取
    all_samples = np.frombuffer(
        raw_wave_bytes, dtype=np.int16).astype(np.float64)

    # 转换为电压
    voltages = (all_samples - vert_offset) * vert_gain

    return {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'filesize': len(data),
        'template': template_name,
        'instrument': instrument_name,
        'instrument_sn': instrument_num,
        'wave_source': wave_source,
        'record_type': record_type,
        'wave_count': wave_array_count,
        'wave_bytes': wave_array_1,
        'horiz_interval': horiz_interval,
        'horiz_offset': horiz_offset,
        'vert_gain': vert_gain,
        'vert_offset': vert_offset,
        'nominal_bits': nom_bits,
        'sampling_rate': sampling_rate,
        'trigger_time': trigger_time_str,
        'has_sequence': has_sequence,
        'n_segments': n_segments,
        'samples_per_seg': samples_per_seg,
        'time_per_seg': time_per_seg,
        'trig_time_offsets': trig_time_offsets,
        'voltages': voltages,
        'all_samples': all_samples,
        'trig_time_arr_size': trig_time_arr,
    }


def parse_trc_lecroyparser(filepath, sparse=-1):
    """
    使用 lecroyparser 库解析 TRC 文件。
    返回基本元数据 (lecroyparser 不支持 Sequence 模式，只能做基本解析)
    """
    s = lecroyparser.ScopeData(path=filepath, sparse=sparse)
    return s


# ============================================================================
#  批量分析函数
# ============================================================================

def scan_trc_files(data_dir):
    """
    扫描目录下所有 .trc 文件，按通道和文件编号分组。

    返回:
      files_by_channel: { 'C3': ['...C3--Trace--00000.trc', ...], ... }
      all_files: 所有 TRC 文件的完整路径列表
    """
    trc_files = sorted(glob.glob(os.path.join(
        data_dir, '**', '*.trc'), recursive=True))

    files_by_channel = defaultdict(list)
    for fp in trc_files:
        basename = os.path.basename(fp)
        # 文件名格式: C3--Trace--00000.trc
        if '--Trace--' in basename:
            channel = basename.split('--')[0]  # C3, C4, ...
            files_by_channel[channel].append(fp)
        else:
            files_by_channel['OTHER'].append(fp)

    # 确保各通道内按编号排序
    for ch in files_by_channel:
        files_by_channel[ch].sort()

    return files_by_channel, trc_files


def analyze_all_files(data_dir, max_files=None):
    """
    分析指定目录下的所有 TRC 文件。

    参数:
      data_dir: 数据目录路径
      max_files: 最多分析的文件数 (None = 全部)

    返回:
      results: 解析结果列表
      summary: 汇总信息 dict
    """
    files_by_channel, all_files = scan_trc_files(data_dir)

    print(f"📂 扫描目录: {data_dir}")
    print(f"  找到 {len(all_files)} 个 .trc 文件")

    if not all_files:
        return [], {}

    # 分析文件
    results = []
    files_to_analyze = all_files[:max_files] if max_files else all_files

    for i, fp in enumerate(files_to_analyze):
        try:
            info = parse_trc_binary(fp)
            results.append(info)

            basename = os.path.basename(fp)
            seg_str = f"{info['n_segments']} segments" if info['has_sequence'] else "single sweep"
            print(f"  [{i+1}/{len(files_to_analyze)}] {basename}: {seg_str}, "
                  f"{info['wave_count']} pts, {info['sampling_rate']/1e9:.1f} GS/s", end='')

            if info['has_sequence']:
                print(f", {info['samples_per_seg']} pts/seg, "
                      f"{info['time_per_seg']*1e9:.2f} ns/seg")
            else:
                print()

        except Exception as e:
            print(
                f"  [{i+1}/{len(files_to_analyze)}] {os.path.basename(fp)}: ❌ {e}")

    # 汇总
    summary = generate_summary(results)
    return results, summary


def generate_summary(results):
    """生成所有文件的汇总统计"""
    summary = {}

    if not results:
        return summary

    # 基本信息
    r0 = results[0]
    summary['instrument'] = r0['instrument']
    summary['template'] = r0['template']
    summary['total_files'] = len(results)

    # 序列模式统计
    seq_files = [r for r in results if r['has_sequence']]
    non_seq = [r for r in results if not r['has_sequence']]
    summary['sequence_files'] = len(seq_files)
    summary['non_sequence_files'] = len(non_seq)

    if seq_files:
        summary['n_segments'] = seq_files[0]['n_segments']
        summary['samples_per_seg'] = seq_files[0]['samples_per_seg']
        summary['horiz_interval'] = seq_files[0]['horiz_interval']
        summary['sampling_rate'] = seq_files[0]['sampling_rate']
        summary['time_per_seg_ns'] = seq_files[0]['time_per_seg'] * 1e9

    # 按通道分组 (从文件名提取, 如 C3--Trace--00000.trc → C3)
    by_channel = defaultdict(list)
    for r in results:
        basename = os.path.basename(r['filepath'])
        ch = basename.split('--')[0] if '--' in basename else 'Unknown'
        by_channel[ch].append(r)
    summary['by_channel'] = {ch: len(files)
                             for ch, files in sorted(by_channel.items())}

    # 触发时间范围
    if seq_files and seq_files[0]['trig_time_offsets'] is not None:
        offsets = seq_files[0]['trig_time_offsets']
        summary['trig_time_range_s'] = (float(offsets[0]), float(offsets[-1]))
        summary['total_acquisition_s'] = float(offsets[-1] - offsets[0])

    # 文件大小信息
    sizes = [r['filesize'] for r in results]
    summary['file_size_min'] = min(sizes)
    summary['file_size_max'] = max(sizes)
    summary['file_size_mean'] = np.mean(sizes)

    return summary


# ============================================================================
#  可视化函数
# ============================================================================

def plot_segment_overview(info, seg_indices=None, save_path=None):
    """
    绘制某个 TRC 文件的段波形概览。

    参数:
      info: parse_trc_binary 返回的结果 dict
      seg_indices: 要绘制的段索引列表 (None = 前几个)
      save_path: 图片保存路径 (None = 显示)
    """
    if not HAS_MPL:
        print("⚠️  matplotlib 未安装，无法绘图")
        return

    n_seg = info['n_segments']
    sps = info['samples_per_seg']
    dt = info['horiz_interval']
    trig_offsets = info['trig_time_offsets']
    volts = info['voltages']

    if seg_indices is None:
        # 默认绘制前 4 段
        seg_indices = list(range(min(4, n_seg)))

    n_plot = len(seg_indices)
    fig, axes = plt.subplots(n_plot, 1, figsize=(
        12, 2.5 * n_plot), sharex=False)
    if n_plot == 1:
        axes = [axes]

    for ax, seg_idx in zip(axes, seg_indices):
        if seg_idx >= n_seg:
            continue
        start = seg_idx * sps
        end = start + sps
        seg_y = volts[start:end]
        seg_t = np.arange(sps) * dt * 1e9  # ns

        trig_t = trig_offsets[seg_idx] if trig_offsets is not None else 0

        ax.plot(seg_t, seg_y, 'b-', linewidth=0.8)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Trigger')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Segment {seg_idx} | Trigger offset: {trig_t:.6e} s')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # 统计
        stats = f"Vpp={seg_y.max()-seg_y.min():.3f}V  mean={seg_y.mean():.4f}V"
        ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 图片已保存: {save_path}")
    else:
        plt.show()


def plot_trigger_timeline(info, save_path=None):
    """
    绘制序列模式的触发时间线。
    """
    if not HAS_MPL:
        return

    if not info['has_sequence'] or info['trig_time_offsets'] is None:
        print("⚠️  不是 Sequence 模式，无法绘制触发时间线")
        return

    offsets = info['trig_time_offsets']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # 触发时间偏移
    ax1.plot(offsets * 1e3, 'b.', markersize=2)
    ax1.set_xlabel('Segment index')
    ax1.set_ylabel('Trigger time offset (ms)')
    ax1.set_title(f'Sequence Mode: {len(offsets)} segments')
    ax1.grid(True, alpha=0.3)

    # 触发间隔
    if len(offsets) > 1:
        intervals = np.diff(offsets)
        ax2.plot(intervals * 1e3, 'r.', markersize=2)
        ax2.set_xlabel('Segment index')
        ax2.set_ylabel('Inter-trigger interval (ms)')
        ax2.set_title(
            f'Trigger intervals: mean={np.mean(intervals)*1e3:.3f} ms')
        ax2.grid(True, alpha=0.3)

        print(f"  触发间隔统计:")
        print(f"    均值: {np.mean(intervals)*1e3:.3f} ms")
        print(f"    中位: {np.median(intervals)*1e3:.3f} ms")
        print(f"    标准差: {np.std(intervals)*1e3:.3f} ms")
        print(f"    最小: {np.min(intervals)*1e3:.3f} ms")
        print(f"    最大: {np.max(intervals)*1e3:.3f} ms")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 图片已保存: {save_path}")
    else:
        plt.show()


def plot_segment_waterfall(info, n_segments=20, save_path=None):
    """
    绘制多个段的瀑布图 (堆叠显示)。
    """
    if not HAS_MPL:
        return

    sps = info['samples_per_seg']
    volts = info['voltages']
    dt = info['horiz_interval']

    n_plot = min(n_segments, info['n_segments'])
    seg_t = np.arange(sps) * dt * 1e9  # ns

    fig, ax = plt.subplots(figsize=(12, 8))

    offset = 0
    for i in range(n_plot):
        start = i * sps
        end = start + sps
        seg_y = volts[start:end]
        ax.plot(seg_t, seg_y + offset, linewidth=0.6,
                label=f'Seg {i}' if i % 5 == 0 else '')
        offset += 0.5  # 垂直偏移

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Voltage (V, stacked)')
    ax.set_title(f'First {n_plot} segments (waterfall)')
    ax.grid(True, alpha=0.3)
    if n_plot <= 10:
        ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 图片已保存: {save_path}")
    else:
        plt.show()


# ============================================================================
#  主程序
# ============================================================================

def print_info(info):
    """打印单个文件的详细解析信息"""
    print(f"\n{'='*60}")
    print(f"📄 文件: {info['filename']}")
    print(f"{'='*60}")
    print(f"  仪器:        {info['instrument']} (SN: {info['instrument_sn']})")
    print(f"  模板:        {info['template']}")
    print(f"  通道:        {info['wave_source']}")
    print(f"  记录类型:    {info['record_type']}")
    print(f"  触发时间:    {info['trigger_time']}")
    print(f"  采样率:      {info['sampling_rate']/1e9:.2f} GS/s")
    print(f"  采样间隔:    {info['horiz_interval']*1e12:.2f} ps")
    print(f"  垂直增益:    {info['vert_gain']:.6e}")
    print(f"  垂直偏移:    {info['vert_offset']:.4f}")
    print(f"  标称位数:    {info['nominal_bits']}-bit")

    print(f"\n  📊 Sequence 模式: {'✅ 是' if info['has_sequence'] else '❌ 否'}")

    if info['has_sequence']:
        print(f"  段数 (segments):      {info['n_segments']}")
        print(f"  每段采样数:           {info['samples_per_seg']}")
        print(f"  每段时间:             {info['time_per_seg']*1e9:.3f} ns")
        print(f"  总采样数:             {info['wave_count']}")
        print(
            f"  总时长:               {info['wave_count'] * info['horiz_interval'] * 1e6:.2f} µs")

        if info['trig_time_offsets'] is not None:
            offsets = info['trig_time_offsets']
            print(f"\n  触发时间偏移:")
            print(f"    第一个: {offsets[0]:.6e} s")
            print(f"    最后一个: {offsets[-1]:.6e} s")
            if len(offsets) > 1:
                total_time = offsets[-1] - offsets[0]
                print(
                    f"    总采集时间: {total_time:.3f} s ({total_time/60:.2f} min)")
    else:
        print(f"  采样数:       {info['wave_count']}")
        print(
            f"  总时长:       {info['wave_count'] * info['horiz_interval'] * 1e6:.2f} µs")

    print(f"  文件大小:     {info['filesize']/1024:.1f} KB")


def print_summary(summary):
    """打印汇总信息"""
    if not summary:
        print("\n⚠️  没有数据可供汇总")
        return

    print(f"\n{'='*60}")
    print(f"📊 汇总统计")
    print(f"{'='*60}")
    print(f"  仪器: {summary.get('instrument', 'N/A')}")
    print(f"  文件总数: {summary['total_files']}")
    print(f"  Sequence 模式文件: {summary.get('sequence_files', 0)}")
    print(f"  非 Sequence 模式文件: {summary.get('non_sequence_files', 0)}")

    if 'sampling_rate' in summary:
        print(f"\n  采样率: {summary['sampling_rate']/1e9:.2f} GS/s")

    if 'n_segments' in summary:
        print(f"\n  📈 Sequence 模式参数:")
        print(f"    段数:         {summary['n_segments']}")
        print(f"    每段采样数:   {summary['samples_per_seg']}")
        print(f"    每段时间:     {summary['time_per_seg_ns']:.3f} ns")

    if 'by_channel' in summary:
        print(f"\n  通道分布:")
        for ch, cnt in sorted(summary['by_channel'].items()):
            print(f"    {ch}: {cnt} 个文件")

    if 'trig_time_range_s' in summary:
        t0, t1 = summary['trig_time_range_s']
        total = summary.get('total_acquisition_s', 0)
        print(f"\n  触发时间范围: {t0:.3f} ~ {t1:.3f} s")
        print(f"  总采集时间跨度: {total:.1f} s ({total/60:.1f} min)")

    if 'file_size_mean' in summary:
        print(f"\n  文件大小: {summary['file_size_min']/1024:.0f} ~ {summary['file_size_max']/1024:.0f} KB "
              f"(平均 {summary['file_size_mean']/1024:.0f} KB)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='分析 LeCroy TRC 示波器波形文件 (Sequence 模式)')
    parser.add_argument('data_dir', nargs='?',
                        default=r'D:\hgtd\alice3\Analyze\data',
                        help='数据目录路径 (包含 .trc 文件)')
    parser.add_argument('--file', '-f', help='分析单个 TRC 文件')
    parser.add_argument('--plot', '-p', action='store_true', help='绘制波形图')
    parser.add_argument('--segments', '-s', type=int, default=4, help='要绘制的段数')
    parser.add_argument('--waterfall', '-w', action='store_true', help='绘制瀑布图')
    parser.add_argument('--timeline', '-t',
                        action='store_true', help='绘制触发时间线')
    parser.add_argument('--max-files', '-m', type=int,
                        default=None, help='最多分析的文件数')
    parser.add_argument('--save', action='store_true', help='保存图片')

    args = parser.parse_args()

    if args.file:
        # 分析单个文件
        info = parse_trc_binary(args.file)
        print_info(info)

        if args.plot:
            plot_segment_overview(info, seg_indices=list(range(min(args.segments, info['n_segments']))),
                                  save_path=f"segments_{Path(args.file).stem}.png" if args.save else None)

        if args.waterfall:
            plot_segment_waterfall(info, n_segments=args.segments,
                                   save_path=f"waterfall_{Path(args.file).stem}.png" if args.save else None)

        if args.timeline and info['has_sequence']:
            plot_trigger_timeline(info,
                                  save_path=f"timeline_{Path(args.file).stem}.png" if args.save else None)

    else:
        # 批量分析
        results, summary = analyze_all_files(
            args.data_dir, max_files=args.max_files)

        if results:
            print_summary(summary)

            # 显示第一个文件的详细信息
            print(f"\n{'='*60}")
            print(f"📄 第一个文件详情 (示例)")
            print_info(results[0])

            if args.plot:
                plot_segment_overview(results[0],
                                      seg_indices=list(range(min(args.segments, results[0]['n_segments']))))

            if args.waterfall:
                plot_segment_waterfall(results[0], n_segments=args.segments)

            if args.timeline and results[0]['has_sequence']:
                plot_trigger_timeline(results[0])


if __name__ == '__main__':
    main()
