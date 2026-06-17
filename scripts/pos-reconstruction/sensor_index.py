"""
传感器-电极-信号-读出通道索引
================================
中心化的映射表，供击中位置重建等分析脚本查询。

三传感器 (name=wafer ID, alias=板子编号, draw=抽屉位置):
  W6P18-6 (M4)  — Draw6, upstream tracker,   pitch=150μm
  W3P35-6 (BT4) — Draw4, DUT,                pitch=150μm
  W3P3-8  (BT2) — Draw1, downstream tracker, pitch=200μm

用法:
  from scripts.pos_reconstruction.sensor_index import (
      SENSORS, get_sensor, lookup_by_instrument, lookup_electrode,
      ChannelKey, Instrument,
  )

  s = get_sensor("W3P35-6")   # by wafer ID
  s = get_sensor("BT4")       # by board
  s = get_sensor("Draw4")     # by drawer position
  for ch in s.channels:
      print(ch.electrode, ch.signal, ch.instrument, ch.channel_id)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ======================================================================
# 枚举类型
# ======================================================================

class Instrument(enum.Enum):
    OSC = "OSC"       # 示波器 LeCroy WR8208HD
    DIGI = "DIGI"     # Digitizer CAEN V1742


# ======================================================================
# 数据结构
# ======================================================================

@dataclass(frozen=True)
class ChannelKey:
    """
    唯一标识一个读出通道。

    Attributes
    ----------
    instrument : Instrument
        读出仪器。
    channel_id : int | str
        示波器时为字符串如 "C3", Digitizer 时为整数如 1。
    """
    instrument: Instrument
    channel_id: int | str

    def __hash__(self):
        return hash((self.instrument, str(self.channel_id)))

    def __str__(self):
        if self.instrument == Instrument.OSC:
            return f"OSC {self.channel_id}"
        return f"Digi ch{self.channel_id}"

    @property
    def is_osc(self) -> bool:
        return self.instrument == Instrument.OSC

    @property
    def is_digi(self) -> bool:
        return self.instrument == Instrument.DIGI


@dataclass(frozen=True)
class ElectrodeChannel:
    """
    一个电极的完整读出链路。

    Attributes
    ----------
    electrode : int
        传感器上的电极编号。
    signal : int
        放大板信号编号。
    channel : ChannelKey
        读出仪器 + 通道。
    """
    electrode: int
    signal: int
    channel: ChannelKey

    @property
    def instrument(self) -> Instrument:
        return self.channel.instrument

    @property
    def channel_id(self):
        return self.channel.channel_id


@dataclass
class SensorInfo:
    """
    一个传感器的完整信息。

    Attributes
    ----------
    name : str
        Wafer 上 sensor 的真实编号 (如 W3P35-6)。
    alias : str
        板子编号 (如 BT4, M4)。本次实验中板子与 sensor 绑定。
    draw : str
        抽屉编号 (如 Draw4)。用于定位板子在束线中的位置。
    pitch_um : int
        Strip pitch (μm)。
    role : str
        tracker / DUT。
    hv_channel : str
        HV 通道名。
    working_voltage : int
        典型工作电压 (V)。
    channels : List[ElectrodeChannel]
        所有参与读出的电极-通道映射。
    """
    name: str           # wafer ID, e.g. "W3P35-6"
    alias: str          # board ID, e.g. "BT4"
    draw: str           # drawer position, e.g. "Draw4"
    pitch_um: int
    role: str
    hv_channel: str
    working_voltage: int
    channels: List[ElectrodeChannel] = field(default_factory=list)

    @property
    def wafer_id(self) -> str:
        return self.name

    @property
    def n_electrodes(self) -> int:
        return len(self.channels)

    @property
    def electrodes(self) -> List[int]:
        return sorted(ch.electrode for ch in self.channels)

    @property
    def electrode_signals(self) -> Dict[int, int]:
        """返回 {electrode: signal} 映射。"""
        return {ch.electrode: ch.signal for ch in self.channels}

    def osc_channels(self) -> List[ElectrodeChannel]:
        return [ch for ch in self.channels if ch.instrument == Instrument.OSC]

    def digi_channels(self) -> List[ElectrodeChannel]:
        return [ch for ch in self.channels if ch.instrument == Instrument.DIGI]

    def get_channel(self, electrode: int) -> Optional[ElectrodeChannel]:
        for ch in self.channels:
            if ch.electrode == electrode:
                return ch
        return None

    def __repr__(self):
        return (f"Sensor({self.name}/{self.alias}, {self.draw}, "
                f"{self.role}, {self.pitch_um}μm, {self.n_electrodes} channels)")


# ======================================================================
# 硬编码映射表 — 唯一数据源
# ======================================================================

def _build_sensors() -> Dict[str, SensorInfo]:
    """构建三传感器完整映射。修改映射关系只需改此处。"""

    # ── Draw6: W6P18-6 (M4), upstream tracker ──
    draw6 = SensorInfo(
        name="W6P18-6",
        alias="M4",
        draw="Draw6",
        pitch_um=150,
        role="tracker",
        hv_channel="HV_4C.4",
        working_voltage=160,
        channels=[
            ElectrodeChannel(4, 1, ChannelKey(Instrument.OSC, "C3")),
            ElectrodeChannel(5, 2, ChannelKey(Instrument.OSC, "C4")),
            ElectrodeChannel(3, 5, ChannelKey(Instrument.OSC, "C5")),
            ElectrodeChannel(6, 3, ChannelKey(Instrument.DIGI, 7)),
        ],
    )

    # ── Draw4: W3P35-6 (BT4), DUT ──
    draw4 = SensorInfo(
        name="W3P35-6",
        alias="BT4",
        draw="Draw4",
        pitch_um=150,
        role="DUT",
        hv_channel="HV_4C.3",
        working_voltage=126,
        channels=[
            ElectrodeChannel(4, 1, ChannelKey(Instrument.DIGI, 2)),
            ElectrodeChannel(7, 2, ChannelKey(Instrument.DIGI, 3)),
            ElectrodeChannel(6, 3, ChannelKey(Instrument.DIGI, 4)),
            ElectrodeChannel(5, 4, ChannelKey(Instrument.DIGI, 5)),
            ElectrodeChannel(3, 5, ChannelKey(Instrument.DIGI, 6)),
        ],
    )

    # ── Draw1: W3P3-8 (BT2), downstream tracker ──
    draw1 = SensorInfo(
        name="W3P3-8",
        alias="BT2",
        draw="Draw1",
        pitch_um=200,
        role="tracker",
        hv_channel="HV_4C.0",
        working_voltage=150,
        channels=[
            ElectrodeChannel(3, 1, ChannelKey(Instrument.OSC, "C6")),
            ElectrodeChannel(4, 2, ChannelKey(Instrument.OSC, "C7")),
            ElectrodeChannel(5, 3, ChannelKey(Instrument.OSC, "C8")),
            ElectrodeChannel(2, 4, ChannelKey(Instrument.DIGI, 15)),
            ElectrodeChannel(1, 5, ChannelKey(Instrument.DIGI, 1)),
        ],
    )

    return {
        "W6P18-6": draw6,
        "W3P35-6": draw4,
        "W3P3-8":  draw1,
    }


# ======================================================================
# 全局单例
# ======================================================================

SENSORS: Dict[str, SensorInfo] = _build_sensors()


# ======================================================================
# 便捷查询函数
# ======================================================================

def get_sensor(query: str) -> Optional[SensorInfo]:
    """
    按 wafer ID、板子编号、或抽屉编号查找传感器。

    匹配优先级: wafer ID (key) > 板子编号 (alias) > 抽屉编号 (draw)
    """
    # 精确匹配 wafer ID (SENSORS 的 key)
    if query in SENSORS:
        return SENSORS[query]
    # 匹配 alias (板子编号) 或 draw (抽屉编号)
    for s in SENSORS.values():
        if s.alias == query or s.draw == query:
            return s
    return None


def lookup_by_instrument(instrument: Instrument) -> Dict[str, List[ElectrodeChannel]]:
    """
    按仪器分组返回所有通道。

    Returns
    -------
    dict: {sensor_name: [ElectrodeChannel, ...]}
    """
    result: Dict[str, List[ElectrodeChannel]] = {}
    for s in SENSORS.values():
        chs = [ch for ch in s.channels if ch.instrument == instrument]
        if chs:
            result[s.name] = chs
    return result


def lookup_electrode(electrode: int, sensor_name: str) -> Optional[ElectrodeChannel]:
    """在指定传感器中按电极编号查找读出通道。"""
    s = SENSORS.get(sensor_name)
    if s is None:
        return None
    return s.get_channel(electrode)


def lookup_by_channel_key(key: ChannelKey) -> Optional[Tuple[SensorInfo, ElectrodeChannel]]:
    """
    按读出通道反查传感器和电极。

    Returns
    -------
    (SensorInfo, ElectrodeChannel) 或 None
    """
    for s in SENSORS.values():
        for ch in s.channels:
            if ch.channel == key:
                return (s, ch)
    return None


# ======================================================================
# 打印表格 (方便核对)
# ======================================================================

def print_table():
    """打印完整的传感器-电极-信号-通道映射表。"""
    header = f"{'Wafer ID':<12} {'Board':<7} {'Draw':<7} {'Pitch':<7} {'Elec':<6} {'Sig':<5} {'Inst':<6} {'Ch':<6}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for s in SENSORS.values():
        for ch in s.channels:
            inst = "OSC" if ch.instrument == Instrument.OSC else "Digi"
            ch_str = str(ch.channel_id)
            print(f"{s.name:<12} {s.alias:<7} {s.draw:<7} {s.pitch_um}μm  "
                  f"{ch.electrode:<6} {ch.signal:<5} {inst:<6} {ch_str:<6}")
        print(sep)


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    print_table()
    print(f"\n共 {len(SENSORS)} 个传感器, "
          f"{sum(s.n_electrodes for s in SENSORS.values())} 个读出通道\n")

    # 演示查询
    print("查询示例:")
    for q in ["W3P35-6", "BT2", "M4", "Draw4", "Draw1"]:
        s = get_sensor(q)
        if s:
            print(f"  get_sensor('{q}') → {s}")
            print(f"    draw={s.draw}, electrodes={s.electrodes}")
            print(f"    osc: {[(ch.electrode, ch.channel_id) for ch in s.osc_channels()]}")
            print(f"    digi: {[(ch.electrode, ch.channel_id) for ch in s.digi_channels()]}")

    print("\n按仪器分组:")
    for inst in Instrument:
        grouped = lookup_by_instrument(inst)
        for sn, chs in grouped.items():
            print(f"  {inst.value} ← {sn}: {[(ch.electrode, ch.channel_id) for ch in chs]}")

    # 反查
    for key in [ChannelKey(Instrument.DIGI, 7), ChannelKey(Instrument.OSC, "C4")]:
        result = lookup_by_channel_key(key)
        if result:
            sn, ch = result
            print(f"  {key} → {sn.name} electrode {ch.electrode}, signal {ch.signal}")
