#!/bin/bash
export ROOTSYS=/home/ypwang/root
export PATH=$ROOTSYS/bin:$PATH
export LD_LIBRARY_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH
echo "=== g++ ==="
g++ --version 2>&1 | head -1
echo "=== ROOT ==="
root-config --version
echo "=== root-config cflags ==="
root-config --cflags | head -3
echo "=== cmake / make ==="
which cmake make 2>/dev/null
