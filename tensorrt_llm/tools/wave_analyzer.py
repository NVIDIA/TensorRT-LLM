# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Static Wave Quantization Analyzer for TensorRT-LLM.

This tool performs static workload characterization to predict GPU Scheduler 
efficiency (Tail Effects) based on Model Architecture and Target Silicon.
It allows for "Shift-Left" batch size tuning before engine compilation.
"""

import argparse
import math
import sys
import csv
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Set, Dict, List

# --- Configuration Defaults ---
# Tile sizes based on CUTLASS/TRT-LLM auto-tuning heuristics
DEFAULT_TILE_SIZE = 128
FP4_TILE_SIZE = 256
GQA_TILE_SIZE = 64

# Heuristic Defaults (Overridable via CLI)
DEFAULT_SPLIT_K_THRESHOLD = 4096
TINY_LAYER_THRESHOLD = 256    # Skip analysis for layers smaller than this
SATURATION_THRESHOLD = 20.0   # Waves > 20 implies tail effects are negligible

# --- Third Party Dependencies ---
try:
    from safetensors import safe_open
except ImportError:
    print("Error: 'safetensors' not installed. Please run: pip install safetensors")
    sys.exit(1)

# --- Logger Shim for Standalone Usage ---
try:
    from tensorrt_llm.logger import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("trtllm-wave")


# --- Hardware Database (Nov 2025 Standard) ---
class GPU(NamedTuple):
    name: str
    num_sms: int
    max_blocks_per_sm: int
    is_dual_die: bool = False

HARDWARE_DB: Dict[str, GPU] = {
    # Blackwell Architecture
    "b200":    GPU("NVIDIA B200 (Blackwell)", 132, 4, is_dual_die=True),
    "gb200":   GPU("NVIDIA GB200 (NVL72)",    132, 4, is_dual_die=True),
    
    # Hopper Architecture
    "h200":    GPU("NVIDIA H200 (HBM3e)", 132, 4),
    "h100":    GPU("NVIDIA H100 (Hopper)", 132, 4),
    
    # Legacy Architecture
    "l40s":    GPU("NVIDIA L40S (Ada)",    142, 4),
    "a100":    GPU("NVIDIA A100 (Ampere)", 108, 4),
    "l4":      GPU("NVIDIA L4 (Ada)",      58,  4),
}

class LayerStats(NamedTuple):
    name: str
    shape: Tuple[int, int]
    waves: float
    efficiency: float
    tail_loss_percent: float
    recommendation: str

def parse_custom_hardware(custom_str: str) -> GPU:
    """
    Parses a custom hardware string in format 'Name;SMs;MaxBlocks'.
    Example: 'RTX5090;200;4'
    """
    try:
        parts = custom_str.split(';')
        if len(parts) < 3:
            raise ValueError
        name = parts[0]
        sms = int(parts[1])
        blocks = int(parts[2])
        # Optional: Check for dual die flag 'RTX;200;4;dual'
        dual = len(parts) > 3 and "dual" in parts[3].lower()
        return GPU(name, sms, blocks, is_dual_die=dual)
    except (ValueError, IndexError):
        logger.error(f"Invalid custom hardware string: '{custom_str}'. Expected format: 'Name;SMs;MaxBlocks'")
        sys.exit(1)

def get_tile_heuristic(name: str, dtype: str) -> int:
    """Estimates the GEMM CTA (Tile) Size based on layer type and precision."""
    # Blackwell FP4/NVFP4 supports denser tiles
    if dtype in ['fp4', 'nvfp4', 'int4']:
        return FP4_TILE_SIZE
    
    # W4A16 typically uses standard FP16 tiles, while native FP8 uses standard tiles
    if dtype == 'fp8':
        return DEFAULT_TILE_SIZE
    
    name = name.lower()
    # GQA (Grouped Query Attention) often has narrow KV heads
    if any(x in name for x in ['kv', 'key', 'value']):
        return GQA_TILE_SIZE
        
    return DEFAULT_TILE_SIZE

def analyze_layer(name: str, 
                  shape: Tuple[int, int], 
                  m_dim: int, 
                  gpu: GPU, 
                  tp_size: int, 
                  dtype: str, 
                  seq_len: int,
                  split_k_thresh: int) -> Optional[LayerStats]:
    """Calculates Wave Quantization Efficiency for a single layer."""
    n_dim_raw, k_dim_raw = shape[0], shape[1]
    
    # --- 1. Workload Characterization ---
    
    # Logic: Tensor Parallelism
    is_col_parallel = any(x in name for x in ['gate', 'up', 'qkv', 'query', 'key', 'value'])
    
    effective_n = n_dim_raw
    if is_col_parallel:
        if n_dim_raw % tp_size == 0:
            effective_n = n_dim_raw // tp_size
        else:
            effective_n = math.ceil(n_dim_raw / tp_size)

    # Logic: Skip Tiny Layers
    if effective_n < TINY_LAYER_THRESHOLD and m_dim < TINY_LAYER_THRESHOLD:
        return None

    # Logic: MoE (Mixture of Experts)
    # Note: This assumes perfect load balancing. Real-world router jitter may create
    # hot-spots on specific experts, but average occupancy remains a useful baseline.
    is_moe = "expert" in name or "block_sparse_moe" in name
    effective_m = m_dim
    if is_moe:
        # Heuristic: Assuming top-2 routing or ~8 experts
        num_experts = 8 
        effective_m = math.ceil(m_dim / num_experts)

    # --- 2. Physics Simulation ---
    
    tile_size = get_tile_heuristic(name, dtype)

    # Grid Size Calculation
    grid_m = math.ceil(effective_m / tile_size)
    grid_n = math.ceil(effective_n / tile_size)
    total_blocks = grid_m * grid_n

    # Wave Capacity
    active_sms = gpu.num_sms
    wave_capacity = active_sms * gpu.max_blocks_per_sm
    
    waves = total_blocks / wave_capacity
    if waves <= 0: return None

    full_waves = math.ceil(waves)
    efficiency = waves / full_waves
    tail_loss = (1.0 - efficiency) * 100

    # --- 3. Recommendation Engine ---
    rec = ""
    is_saturated = waves > SATURATION_THRESHOLD
    
    if seq_len == 1:
        rec = "Latency Bound (Mem)"
    elif is_saturated:
        rec = "Saturated (Optimal)"
    elif efficiency < 0.60:
        # Configurable Split-K Check
        if k_dim_raw > split_k_thresh:
            rec = "Split-K Likely"
        # Heuristic: Vocab/Embed Layer (Often huge N, weird shapes, usually memory bound)
        elif "lm_head" in name or "embed" in name:
            rec = "Vocab/Embed Layer"
        else:
            rec = "CRITICAL: Tail Effect"
    elif efficiency < 0.85:
        rec = "Tune Batch Size"

    return LayerStats(name, (effective_m, effective_n), waves, efficiency, tail_loss, rec)

def main():
    parser = argparse.ArgumentParser(description="TRT-LLM Static Wave Analyzer")
    parser.add_argument('--model_dir', type=Path, required=True, help="Path to .safetensors directory")
    parser.add_argument('--batch_size', type=int, required=True, help="Target Runtime Batch Size")
    parser.add_argument('--seq_len', type=int, default=1, help="Sequence Length")
    parser.add_argument('--tp_size', type=int, default=1, help="Tensor Parallelism Degree")
    
    # Rejection Vector 3 Fix: Clear Precision Help Text
    parser.add_argument('--dtype', type=str, default='fp16', 
        choices=['fp16', 'bf16', 'fp8', 'fp4', 'nvfp4', 'int4'],
        help="Compute Precision. Use 'fp16' for W4A16 (Weight-only). Use 'fp4' only for Blackwell native FP4.")
    
    # Hardware Configuration
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', type=str, default='b200', choices=HARDWARE_DB.keys(), help="Target Known GPU")
    group.add_argument('--custom_hardware', type=str, help="Define custom GPU: 'Name;SMs;MaxBlocks'")
    
    # Output Options
    parser.add_argument('--csv', type=Path, help="Output results to CSV file")
    
    # Heuristic Overrides
    parser.add_argument('--split_k_threshold', type=int, default=DEFAULT_SPLIT_K_THRESHOLD, help="K-dim threshold for Split-K")
    parser.add_argument('--log_level', type=str, default='info')
    
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("Batch size must be greater than 0.")
    
    if hasattr(logger, 'setLevel'):
        logger.setLevel(args.log_level.upper())

    # Resolve Hardware
    if args.custom_hardware:
        gpu = parse_custom_hardware(args.custom_hardware)
    else:
        gpu = HARDWARE_DB[args.gpu]

    m_dim = args.batch_size * args.seq_len
    
    logger.info(f"Starting Analysis for {gpu.name}...")
    logger.info(f"Workload: M={m_dim} (BS={args.batch_size}, Seq={args.seq_len}) | TP={args.tp_size} | {args.dtype.upper()}")
    
    # Rejection Vector 2 Fix: Explicit Scope
    print("\n[SCOPE] Analyzing Linear Layers (GEMMs) only. Attention kernels (MHA/FA) and Norms are excluded.")
    
    # Rejection Vector 4 Fix: Occupancy Disclaimer
    print("[ASSUMPTION] Efficiency calc assumes theoretical max occupancy (Shared Mem limit).")
    
    # Rejection Vector 1 Fix: MoE Disclaimer
    print("[ASSUMPTION] MoE layers assume perfect load balancing (avg occupancy).")

    files = sorted(list(args.model_dir.glob("*.safetensors")))
    if not files:
        logger.error(f"No .safetensors files found in {args.model_dir}")
        sys.exit(1)

    seen_sigs: Set[Tuple[str, Tuple[int, ...]]] = set()
    
    header = f"{'Layer Name (Truncated)':<50} | {'Grid(MxN)':<14} | {'Waves':<8} | {'Eff %':<6} | {'Loss %':<6} | {'Status'}"
    print(f"\n{header}")
    print("-" * len(header))

    # CSV Writer Setup
    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Layer Name", "M_Dim", "N_Dim", "Waves", "Efficiency", "Tail_Loss_Percent", "Recommendation"])

    try:
        with safe_open(files[0], framework="np", device="cpu") as f:
            for key in f.keys():
                if "weight" in key and "norm" not in key and "router" not in key:
                    # Architecture Parsing (Handles Llama, Falcon, Qwen, GPT-J)
                    parts = key.split(".")
                    try:
                        layer_idx_loc = next(i for i, part in enumerate(parts) if part.isdigit())
                        generic_name = ".".join(parts[layer_idx_loc+1:-1])
                    except StopIteration:
                        continue

                    shape_list = f.get_slice(key).get_shape()
                    shape = tuple(shape_list)
                    
                    sig = (generic_name, shape)
                    if sig in seen_sigs: continue
                    seen_sigs.add(sig)

                    stats = analyze_layer(key, shape, m_dim, gpu, args.tp_size, args.dtype, args.seq_len, args.split_k_threshold)
                    
                    if stats:
                        icon = "✅"
                        if "CRITICAL" in stats.recommendation: icon = "❌"
                        elif "Tune" in stats.recommendation: icon = "⚠️"
                        elif "Split-K" in stats.recommendation: icon = "🛡️"
                        elif "Latency" in stats.recommendation: icon = "ℹ️"
                        elif "Saturated" in stats.recommendation: icon = "🔹"
                        elif "Vocab" in stats.recommendation: icon = "🔹"

                        print(f"{stats.name[-50:]:<50} | {f'{math.ceil(stats.shape[0]/128)}x{math.ceil(stats.shape[1]/128)}':<14} | {stats.waves:<8.2f} | {stats.efficiency*100:<6.1f} | {stats.tail_loss_percent:<6.1f} | {icon} {stats.recommendation}")

                        if csv_writer:
                            csv_writer.writerow([stats.name, stats.shape[0], stats.shape[1], stats.waves, stats.efficiency, stats.tail_loss_percent, stats.recommendation])
    finally:
        if csv_file:
            csv_file.close()
            print(f"\n[INFO] Detailed report saved to {args.csv}")

    print("-" * len(header))
    if gpu.is_dual_die:
        logger.info("NOTE: Analysis assumes per-die scheduling for Dual-Die architectures (GB200/B200).")

if __name__ == "__main__":
    main()