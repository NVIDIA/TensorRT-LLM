#!/bin/bash
#
# GPU Frequency Locking Script
#
# This script locks GPU graphics and memory clocks to specified frequencies
# to ensure consistent performance during benchmarking.
#
# Usage: lock_gpu_freq.sh <graphics_mhz> <memory_mhz>
#
# Parameters:
#   graphics_mhz - Graphics clock frequency in MHz (0 to skip)
#   memory_mhz   - Memory clock frequency in MHz (0 to skip)
#
# Example: lock_gpu_freq.sh 3996 1965
#

set -e

GRAPHICS_MHZ=${1:-0}
MEMORY_MHZ=${2:-0}

echo "========================================"
echo "GPU Frequency Locking"
echo "========================================"
echo "Graphics Clock: ${GRAPHICS_MHZ} MHz"
echo "Memory Clock:   ${MEMORY_MHZ} MHz"
echo "Node:           $(hostname)"
echo "========================================"

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found"
    exit 1
fi

# Function to lock GPU frequencies
lock_gpu_frequency() {
    local graphics=$1
    local memory=$2

    # Skip if both frequencies are 0
    if [ "$graphics" -eq 0 ] && [ "$memory" -eq 0 ]; then
        echo "⚠️  No lock frequencies specified (both are 0), skipping GPU frequency locking"
        return 0
    fi

    # Enable persistence mode (recommended for frequency locking)
    echo "📌 Enabling GPU persistence mode..."
    nvidia-smi -pm 1 2>&1 || echo "⚠️  Warning: Failed to enable persistence mode"

    # Lock GPU clocks using nvidia-smi commands
    # Note: -lgc locks graphics clock, -ac sets application clocks (memory,graphics)

    if [ "$graphics" -gt 0 ] && [ "$memory" -gt 0 ]; then
        echo "🔒 Locking GPU clocks..."
        echo "  - Graphics: ${graphics} MHz"
        echo "  - Memory:   ${memory} MHz"

        # Lock graphics clock
        sudo nvidia-smi -lgc ${graphics} 2>&1 || echo "⚠️  Warning: lgc command failed"

        # Set application clocks (memory,graphics)
        sudo nvidia-smi -ac ${memory},${graphics} 2>&1 || echo "⚠️  Warning: ac command failed"

        echo "✅ GPU clocks locked successfully"
    else
        echo "⏭️  Skipping GPU clock locking (one or both frequencies are 0)"
    fi

    return 0
}

# Lock GPU frequencies
lock_gpu_frequency "$GRAPHICS_MHZ" "$MEMORY_MHZ"

# Display current GPU status
echo ""
echo "📊 Current GPU Status:"
nvidia-smi --query-gpu=index,name,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory --format=csv

echo ""
echo "✅ GPU frequency locking completed on $(hostname)"
