#!/bin/bash
# compile_and_run.sh
# Step 1 (Windows): python npz_export_binary.py → export.bin
# Step 2 (WSL): root -l -q npz_export_binary_via_root.cxx → .root

SRC="$1"
if [ -z "$SRC" ]; then
    echo "Usage: $0 /mnt/d/Codes/Match-BeamTest-Data/data/BT.../run_XXXX"
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
WIN_SRC=$(echo "$SRC" | sed 's|/mnt/\(.\)/|\1:/|')

# Step 1: Python export (Windows)
echo "=== Step 1: Export binary ==="
python3 "$(echo "$SCRIPT_DIR" | sed 's|/mnt/\(.\)/|\1:/|')/npz_export_binary.py" --src "$WIN_SRC"
if [ $? -ne 0 ]; then
    echo "Export failed"
    exit 1
fi

# Step 2: ROOT convert (WSL)
echo "=== Step 2: ROOT TTree ==="
BIN="$SRC/temp/export.bin"
if [ ! -f "$BIN" ]; then
    echo "export.bin not found at $BIN"
    exit 1
fi

export ROOTSYS=/home/ypwang/root
export PATH=$ROOTSYS/bin:$PATH
export LD_LIBRARY_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH

root -l -q "$SCRIPT_DIR/npz_export_binary_via_root.cxx(\"$BIN\")"
echo "Done"
