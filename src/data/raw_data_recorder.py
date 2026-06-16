"""
Raw Data Recorder — Binary Recording with WaveDump-Compatible Format
=====================================================================

Online recording phase:
  Digitizer data saved as a single binary file with WaveDump-compatible
  event headers. Extremely fast — no analysis, just raw fwrite.

  Format v4 (single file: {Model}_events.bin, e.g. V1742_events.bin):
    FileHeader(56B) + [EventHeader(56B) + OptionalGroupData(0/16B) + ChannelData] × N

  FPGA:  (待实现 — FPGA 通信协议未确定)
  Scope: (由 AutoSave CSV 文件处理, 不经过此 recorder)

Offline matching phase (in event_merger.py):
  Uses the same binary format for loading and matching events.

Format history:
  v1: Initial format — per-channel files like WaveDump.
  v2: Single file format — merged all channels into one file.
  v3: ch_mask upgraded from uint32 to uint64 (supports ch32-35).
  v4: Added event_time_tag (uint64), per-group TriggerTimeTags,
      and flags field for future extensibility.
  v4-legacy: Buggy v4 with event_time_tag packed as uint32 (52 B headers).
  v5: Fixed event_time_tag to uint64 (56 B headers).
"""

import logging
import struct
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Binary Format Constants
# ======================================================================

# Magic number: "CAEN" + "D742" → 0xCAE1D742
BIN_MAGIC = 0xCAE1D742
BIN_VERSION = 5  # v5: fixed event_time_tag from uint32 to uint64

# File header: 56 bytes (unchanged from v3)
#   uint32 magic          — magic number 0xCAE1D742
#   uint32 version        — format version
#   uint32 run_number     — run number
#   uint32 n_channels     — number of enabled channels
#   uint32 record_length  — samples per waveform
#   uint32 n_bits         — ADC bits (12 for V1742)
#   uint32 sampling_ps    — picoseconds per sample (200 for 5GHz)
#   uint32 family_code    — board family (6=XX742)
#   uint32 reserved[4]    — future use
_FILE_HEADER_FMT = "=IIIIIIIIIIIII"  # 13 × uint32 = 52 bytes
_FILE_HEADER_SIZE = struct.calcsize(_FILE_HEADER_FMT)  # = 52
_FILE_HEADER_FMT_PAD = "=IIIIIIIIIIIIII"  # 14 × uint32 = 56 bytes
_FILE_HEADER_SIZE_PAD = struct.calcsize(_FILE_HEADER_FMT_PAD)  # = 56

# ---- v3 Event header: 40 bytes ----
#   uint32 total_size     — event header + all channel data (bytes)
#   uint32 board_id       — board ID
#   uint32 pattern        — trigger pattern
#   uint64 ch_mask        — bitmask of channels with data (up to ch63)
#   uint32 event_counter  — EventCounter = TriggerID
#   uint64 trigger_time_tag  — extended 64-bit Group Trigger Time Tag (8.5 ns/cyc)
#   uint32 dc_offset      — DC offset (DAC value)
#   uint32 start_index_cell — DRS4 start index cell
_EVENT_HEADER_FMT_V3 = "=IIIQIQII"  # = 40 bytes
_EVENT_HEADER_SIZE_V3 = struct.calcsize(_EVENT_HEADER_FMT_V3)  # = 40

# ---- v4 Event header (legacy, buggy): 52 bytes ----
# event_time_tag was incorrectly packed as uint32 instead of uint64.
# Only the v4 writer had this bug; v5+ uses the correct 56-byte format.
#   uint32 total_size, board_id, pattern, uint64 ch_mask,
#   uint32 event_counter, uint64 trigger_time_tag,
#   uint32 event_time_tag (BUG: should be uint64!),
#   uint32 dc_offset, start_index_cell, flags, reserved
_EVENT_HEADER_FMT_V4_LEGACY = "=IIIQIQIIIII"  # 11 fields = 52 bytes
_EVENT_HEADER_SIZE_V4_LEGACY = struct.calcsize(_EVENT_HEADER_FMT_V4_LEGACY)  # = 52

# ---- v5+ Event header: 56 bytes ----
# Extends v3 with: event_time_tag(uint64), flags(uint32), reserved(uint32)
#
#   uint32 total_size       — fixed header + optional group section + channel data (bytes)
#   uint32 board_id         — board ID
#   uint32 pattern          — trigger pattern
#   uint64 ch_mask          — bitmask of channels with data (up to ch63)
#   uint32 event_counter    — EventCounter = TriggerID
#   uint64 trigger_time_tag — Group Trigger Time Tag (物理时间, extended 64-bit)
#   uint64 event_time_tag   — Event Time Tag (读取时间戳, extended 64-bit)
#   uint32 dc_offset        — DC offset (DAC value)
#   uint32 start_index_cell — DRS4 start index cell
#   uint32 flags            — bit flags (see EVENT_FLAG_*)
#   uint32 reserved         — future use (pad to 56 bytes)
_EVENT_HEADER_FMT_V5 = "=IIIQIQQIIII"  # 11 fields = 56 bytes
_EVENT_HEADER_SIZE_V5 = struct.calcsize(_EVENT_HEADER_FMT_V5)  # = 56

# Alias: current version uses v5 format
_EVENT_HEADER_FMT = _EVENT_HEADER_FMT_V5
_EVENT_HEADER_SIZE = _EVENT_HEADER_SIZE_V5

# Event header flags (for v4+)
EVENT_FLAG_HAS_GROUP_TTAGS = 0x01  # bit0: optional group TriggerTimeTags section present (16 bytes)

# Optional per-group TriggerTimeTags section (only when flags & HAS_GROUP_TTAGS)
# Stored between the fixed event header and channel data.
#   uint32 gr0_trigger_time_tag  — raw 30-bit Group0 TriggerTimeTag
#   uint32 gr1_trigger_time_tag  — raw 30-bit Group1 TriggerTimeTag
#   uint32 gr2_trigger_time_tag  — raw 30-bit Group2 TriggerTimeTag
#   uint32 gr3_trigger_time_tag  — raw 30-bit Group3 TriggerTimeTag
_GROUP_TTAGS_FMT = "=IIII"  # 4 × uint32 = 16 bytes
_GROUP_TTAGS_SIZE = struct.calcsize(_GROUP_TTAGS_FMT)  # = 16



# Per-channel data within an event:
#   uint32 channel_id
#   uint32 n_samples
#   float32[n_samples] waveform_data
_CH_DATA_HEADER_FMT = "=II"  # 8 bytes
_CH_DATA_HEADER_SIZE = struct.calcsize(_CH_DATA_HEADER_FMT)  # = 8


# ======================================================================
# WaveDump Binary Writer — Single File Format
# ======================================================================

class WaveDumpBinaryWriter:
    """
    Writes digitizer data to a single binary file with WaveDump-compatible format.

    File structure (v4):
      [FileHeader: 56B]
      [Event0: EventHeader(56B) + (OptionalGroupData: 16B) + ChData × N]
      [Event1: EventHeader(56B) + (OptionalGroupData: 16B) + ChData × N]
      ...

    This format is:
    - Extremely fast to write (raw fwrite, no analysis, no compression)
    - Self-describing (file header has all metadata)
    - 64-bit extended trigger_time_tag AND event_time_tag (handles counter overflow)
    - Per-group TriggerTimeTags for X742 (all 4 groups)
    - Single file (not per-channel files like original WaveDump)

    Usage:
        writer = WaveDumpBinaryWriter("./data_raw")
        writer.open_run(run_number=1, n_channels=32, record_length=1024)
        writer.write_event(event_dict)
        writer.close_run()
    """

    def __init__(self, output_dir: str = "./data_raw"):
        self._output_dir = Path(output_dir)
        self._run_dir: Optional[Path] = None
        self._file: Optional[BinaryIO] = None
        self._event_count = 0
        self._n_channels = 0
        self._record_length = 0

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def open_run(self,
                 run_number: int,
                 n_channels: int = 32,
                 record_length: int = 1024,
                 n_bits: int = 12,
                 sampling_ps: int = 200,
                 family_code: int = 6,
                 model_name: str = "V1742") -> None:
        """
        Open a new binary file for a run.

        Args:
            run_number: Run number.
            n_channels: Number of enabled digitizer channels.
            record_length: Samples per waveform.
            n_bits: ADC resolution (12 for V1742).
            sampling_ps: Sampling period in picoseconds (200 for 5GHz).
            family_code: Board family code (6=XX742).
            model_name: Digitizer model name, used in filename (e.g. "V1742").
        """
        self._run_dir = self._output_dir
        self._run_dir.mkdir(parents=True, exist_ok=True)

        filepath = self._run_dir / f"{model_name}_events.bin"
        self._file = open(filepath, "wb")
        self._event_count = 0
        self._n_channels = n_channels
        self._record_length = record_length

        # Write file header (v4)
        header = struct.pack(
            _FILE_HEADER_FMT_PAD,
            BIN_MAGIC,
            BIN_VERSION,  # = 4
            run_number,
            n_channels,
            record_length,
            n_bits,
            sampling_ps,
            family_code,
            0, 0, 0, 0,  # reserved[4]
            0, 0,        # padding to 56 bytes
        )
        self._file.write(header)
        self._file.flush()

        logger.info(
            f"Binary file opened: {filepath} "
            f"({n_channels}ch, {record_length}samples, {sampling_ps}ps)"
        )

    def close_run(self) -> None:
        """Close the binary file."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

        if self._run_dir:
            logger.info(
                f"Binary file closed: {self._event_count} events written"
            )

    def write_event(self, event_data: dict) -> None:
        """
        Write one digitizer event to the binary file (v4 format).

        Args:
            event_data: Dict from read_all_events() with keys:
                'event_number', 'trigger_time_tag', 'event_time_tag',
                'gr_trigger_time_tags' (Dict[int,int], optional),
                'pattern', 'board_id', 'start_index_cell',
                'channel_mask', 'waveforms' (dict of {ch: ndarray})
        """
        if self._file is None:
            return

        waveforms = event_data.get('waveforms', {})
        if not waveforms:
            return

        # Serialize channel data
        ch_mask = 0
        ch_data = b""
        for ch_idx in sorted(waveforms.keys()):
            arr = np.asarray(waveforms[ch_idx], dtype=np.float32)
            n_samp = len(arr)
            ch_mask |= (1 << ch_idx)
            ch_data += struct.pack(_CH_DATA_HEADER_FMT, ch_idx, n_samp)
            ch_data += arr.tobytes()

        # ---- Build v4 event header ----
        flags = 0
        optional_section = b""

        # Optional: per-group TriggerTimeTags (X742 only)
        gr_ttags = event_data.get('gr_trigger_time_tags', {})
        if gr_ttags:
            flags |= EVENT_FLAG_HAS_GROUP_TTAGS
            optional_section += struct.pack(
                _GROUP_TTAGS_FMT,
                gr_ttags.get(0, 0) & 0x3FFFFFFF,  # raw 30-bit
                gr_ttags.get(1, 0) & 0x3FFFFFFF,
                gr_ttags.get(2, 0) & 0x3FFFFFFF,
                gr_ttags.get(3, 0) & 0x3FFFFFFF,
            )

        total_size = _EVENT_HEADER_SIZE_V5 + len(optional_section) + len(ch_data)

        ev_header = struct.pack(
            _EVENT_HEADER_FMT_V5,
            total_size,
            event_data.get('board_id', 0),
            event_data.get('pattern', 0),
            ch_mask & 0xFFFFFFFFFFFFFFFF,                    # uint64
            event_data.get('event_number', self._event_count),
            event_data.get('trigger_time_tag', 0),            # uint64: physical time
            event_data.get('event_time_tag', 0),              # uint64: ★ NEW readout timestamp
            0,                                                # dc_offset (not saved in readout)
            event_data.get('start_index_cell', 0),
            flags,                                            # ★ NEW flags
            0,                                                # reserved
        )

        # Write to file (no flush — let OS buffer for speed)
        self._file.write(ev_header + optional_section + ch_data)
        self._event_count += 1

    def flush(self) -> None:
        """Flush file buffer to disk."""
        if self._file is not None:
            self._file.flush()

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def run_dir(self) -> Optional[Path]:
        return self._run_dir


# ======================================================================
# Binary File Reader (for offline loading)
# ======================================================================

def load_binary_events(filepath: str) -> List[dict]:
    """
    Load all events from a WaveDump-compatible binary file.

    Supports v3 and v4 formats.  v4 adds event_time_tag and optional
    per-group TriggerTimeTags.

    Args:
        filepath: Path to *_events.bin file (e.g. V1742_events.bin).

    Returns:
        List of event dicts (same format as read_all_events).
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"Binary file not found: {path}")
        return []

    events = []
    with open(path, "rb") as f:
        # Read file header
        header_data = f.read(_FILE_HEADER_SIZE_PAD)
        if len(header_data) < _FILE_HEADER_SIZE_PAD:
            logger.error("File too small for header")
            return events

        magic, version, run_num, n_ch, rec_len, n_bits, sps, fc, *_ = \
            struct.unpack(_FILE_HEADER_FMT_PAD, header_data)

        if magic != BIN_MAGIC:
            logger.error(f"Invalid magic number: 0x{magic:08X}")
            return events

        SUPPORTED_MAX = max(BIN_VERSION, 3)  # supports v3 and v4
        if version > SUPPORTED_MAX:
            logger.error(
                f"Binary format version {version} > supported {SUPPORTED_MAX}. "
                f"Please update the software."
            )
            return events

        logger.info(
            f"Loading binary (v{version}): run={run_num}, {n_ch}ch, "
            f"{rec_len}samples, {sps}ps"
        )

        # ---- Determine event header size for this version ----
        if version <= 3:
            ev_hdr_fmt = _EVENT_HEADER_FMT_V3
            ev_hdr_size = _EVENT_HEADER_SIZE_V3
        elif version == 4:  # v4-legacy (buggy 52-byte headers)
            ev_hdr_fmt = _EVENT_HEADER_FMT_V4_LEGACY
            ev_hdr_size = _EVENT_HEADER_SIZE_V4_LEGACY
        else:  # v5+
            ev_hdr_fmt = _EVENT_HEADER_FMT_V5
            ev_hdr_size = _EVENT_HEADER_SIZE_V5

        # Read events
        while True:
            ev_header = f.read(ev_hdr_size)
            if len(ev_header) < ev_hdr_size:
                break

            if version <= 3:
                (total_size, board_id, pattern, ch_mask,
                 ev_counter, tt_tag, dc_off, start_cell) = \
                    struct.unpack(ev_hdr_fmt, ev_header)
                event_time_tag = 0
                flags = 0
            else:
                (total_size, board_id, pattern, ch_mask,
                 ev_counter, tt_tag, event_time_tag,
                 dc_off, start_cell, flags, _reserved) = \
                    struct.unpack(ev_hdr_fmt, ev_header)

            # v4: data after fixed header = optional group section + channel data
            data_size = total_size - ev_hdr_size
            if data_size <= 0:
                continue

            ch_data = f.read(data_size)
            if len(ch_data) < data_size:
                break

            # ---- Parse optional per-group TriggerTimeTags (v4+) ----
            gr_ttags: Dict[int, int] = {}
            ch_offset = 0
            if version >= 4 and (flags & EVENT_FLAG_HAS_GROUP_TTAGS):
                if ch_offset + _GROUP_TTAGS_SIZE <= len(ch_data):
                    g0, g1, g2, g3 = struct.unpack(
                        _GROUP_TTAGS_FMT,
                        ch_data[ch_offset:ch_offset + _GROUP_TTAGS_SIZE])
                    ch_offset += _GROUP_TTAGS_SIZE
                    if g0:  gr_ttags[0] = g0
                    if g1:  gr_ttags[1] = g1
                    if g2:  gr_ttags[2] = g2
                    if g3:  gr_ttags[3] = g3

            # ---- Parse channel data ----
            waveforms = {}
            while ch_offset + _CH_DATA_HEADER_SIZE <= len(ch_data):
                ch_idx = struct.unpack("=I", ch_data[ch_offset:ch_offset+4])[0]
                n_samp = struct.unpack("=I", ch_data[ch_offset+4:ch_offset+8])[0]
                ch_offset += _CH_DATA_HEADER_SIZE
                samp_bytes = n_samp * 4
                if ch_offset + samp_bytes > len(ch_data):
                    break
                arr = np.frombuffer(
                    ch_data[ch_offset:ch_offset+samp_bytes],
                    dtype=np.float32
                ).copy()
                waveforms[ch_idx] = arr
                ch_offset += samp_bytes

            event = {
                'event_number': ev_counter,
                'trigger_time_tag': tt_tag,
                'event_time_tag': event_time_tag,
                'pattern': pattern,
                'channel_mask': ch_mask,
                'board_id': board_id,
                'start_index_cell': start_cell,
                'waveforms': waveforms,
            }
            if gr_ttags:
                event['gr_trigger_time_tags'] = gr_ttags

            events.append(event)

    logger.info(f"Loaded {len(events)} events from {filepath}")
    return events
