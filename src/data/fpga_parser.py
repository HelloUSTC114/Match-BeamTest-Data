"""
FPGA Data Packet Parser
========================
Pure data-layer module for parsing FPGA binary event packets.

Packet format (16 bytes = 128 bits):
    [Header: 2B LE] [ID: 4B LE] [Timestamp: 8B LE] [Tail: 2B LE]

Event types (distinguished by header/tail):
    0xABBA  — LGAD event (unused in this project)
    0x3553  — OSCI event (scope trigger, ID = trigger_id)

FPGA clock: 200 MHz → 5 ns/tick

This module has NO dependency on hardware interfaces (no socket, no threading).
It can be used safely in offline data analysis scripts.
"""

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ======================================================================
# Constants
# ======================================================================

# Packet size
FPGA_PACKET_SIZE = 16  # bytes

# Header/Tail values
HDR_LGAD = 0xABBA
HDR_OSCI = 0x3553

# FPGA clock
FPGA_CLOCK_HZ = 200e6  # 200 MHz

# Packet offset: [Header 2B] [ID 4B] [Timestamp 8B] [Tail 2B]
_OFF_HDR = 0
_OFF_ID = 2
_OFF_TS = 6
_OFF_TAIL = 14

_PACKET_FMT = "<HIQH"  # Header(2B) + ID(4B) + Timestamp(8B) + Tail(2B) = 16B
_PACKET_SIZE = struct.calcsize(_PACKET_FMT)  # = 16


# ======================================================================
# Data Structures
# ======================================================================

@dataclass
class OSCIEvent:
    """
    An OSCI trigger event from the FPGA board.

    Created when the scope AUX OUT trigger signal reaches the FPGA:
      - trigger_id increments by 1
      - timestamp records the FPGA clock value

    Fields:
        trigger_id:   FPGA counter (1-based, matches scope file index + 1)
        t_fpga:       FPGA clock at trigger arrival (ticks @ 200 MHz)
    """
    trigger_id: int
    t_fpga: int


@dataclass
class FPGAEvent:
    """
    A raw FPGA data packet (can be either LGAD or OSCI).

    Fields:
        header:     Packet header (0xABBA or 0x3553)
        event_id:   32-bit ID field
        timestamp:  64-bit timestamp (clock ticks @ 200 MHz)
        tail:       Packet tail (should match header)
        is_osci:    True if OSCI event (header=0x3553)
    """
    header: int
    event_id: int
    timestamp: int
    tail: int
    is_osci: bool

    def to_osci_event(self) -> Optional[OSCIEvent]:
        """Convert to OSCIEvent if this is an OSCI packet."""
        if self.is_osci:
            return OSCIEvent(trigger_id=self.event_id, t_fpga=self.timestamp)
        return None


# ======================================================================
# Packet Parsing
# ======================================================================

def parse_fpga_packet(data: bytes) -> Optional[FPGAEvent]:
    """
    Parse a 16-byte FPGA data packet.

    Args:
        data: Exactly 16 bytes.

    Returns:
        FPGAEvent if valid, None if parsing fails.
    """
    if len(data) < _PACKET_SIZE:
        return None

    try:
        hdr, eid, ts, tail = struct.unpack(_PACKET_FMT, data[:_PACKET_SIZE])
    except struct.error:
        return None

    # Validate header/tail
    if hdr == HDR_LGAD:
        if tail != HDR_LGAD:
            logger.warning(f"LGAD packet tail mismatch: hdr=0x{hdr:04X} tail=0x{tail:04X}")
            return None
        return FPGAEvent(header=hdr, event_id=eid, timestamp=ts, tail=tail, is_osci=False)

    elif hdr == HDR_OSCI:
        if tail != HDR_OSCI:
            logger.warning(f"OSCI packet tail mismatch: hdr=0x{hdr:04X} tail=0x{tail:04X}")
            return None
        return FPGAEvent(header=hdr, event_id=eid, timestamp=ts, tail=tail, is_osci=True)

    else:
        logger.debug(f"Unknown FPGA packet header: 0x{hdr:04X}")
        return None


def parse_fpga_packets(data: bytes) -> List[FPGAEvent]:
    """
    Parse all complete FPGA packets from a byte buffer.
    Handles alignment by searching for valid headers.

    Args:
        data: Raw byte buffer (may contain partial packets).

    Returns:
        List of parsed FPGAEvent objects.
    """
    events = []
    offset = 0
    while offset + _PACKET_SIZE <= len(data):
        chunk = data[offset:offset + _PACKET_SIZE]
        evt = parse_fpga_packet(chunk)
        if evt is not None:
            events.append(evt)
            offset += _PACKET_SIZE
        else:
            # Try next byte alignment
            offset += 1
    return events


# ======================================================================
# Data File I/O (offline processing)
# ======================================================================

def load_fpga_events(filepath: str) -> List[OSCIEvent]:
    """
    Load OSCI events from a saved FPGA binary file.

    Args:
        filepath: Path to saved FPGA .bin file.

    Returns:
        List of OSCIEvent sorted by trigger_id.
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"FPGA file not found: {path}")
        return []

    events = []
    skipped = 0
    with open(path, "rb") as f:
        data = f.read()

    # Find first valid packet alignment
    for offset in range(min(32, len(data))):
        if len(data) - offset >= _PACKET_SIZE:
            evt = parse_fpga_packet(data[offset:offset + _PACKET_SIZE])
            if evt is not None:
                pos = offset
                while pos + _PACKET_SIZE <= len(data):
                    evt = parse_fpga_packet(data[pos:pos + _PACKET_SIZE])
                    if evt is not None:
                        if evt.is_osci:
                            events.append(OSCIEvent(
                                trigger_id=evt.event_id,
                                t_fpga=evt.timestamp,
                            ))
                    else:
                        skipped += 1
                    pos += _PACKET_SIZE
                break

    logger.info(
        f"Loaded {len(events)} OSCI events from {filepath} "
        f"({skipped} invalid packets)"
    )
    return sorted(events, key=lambda e: e.trigger_id)


def load_fpga_raw_events(filepath: str) -> Tuple[List[OSCIEvent], List[FPGAEvent]]:
    """
    Load all events (both OSCI and LGAD) from a saved FPGA binary file.

    Args:
        filepath: Path to saved FPGA .bin file.

    Returns:
        (osci_events, lgad_raw_events)
    """
    path = Path(filepath)
    if not path.exists():
        return [], []

    osci = []
    lgad = []
    with open(path, "rb") as f:
        data = f.read()

    for offset in range(min(32, len(data))):
        if len(data) - offset >= _PACKET_SIZE:
            evt = parse_fpga_packet(data[offset:offset + _PACKET_SIZE])
            if evt is not None:
                pos = offset
                while pos + _PACKET_SIZE <= len(data):
                    evt = parse_fpga_packet(data[pos:pos + _PACKET_SIZE])
                    if evt is not None:
                        if evt.is_osci:
                            osci.append(OSCIEvent(
                                trigger_id=evt.event_id,
                                t_fpga=evt.timestamp,
                            ))
                        else:
                            lgad.append(evt)
                    pos += _PACKET_SIZE
                break

    return sorted(osci, key=lambda e: e.trigger_id), lgad
