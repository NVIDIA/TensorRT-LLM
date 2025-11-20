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
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Set, Dict

# --- Configuration Constants ---
# Tile sizes based on CUTLASS/TRT-LLM auto-tuning heuristics
DEFAULT_TILE_SIZE = 128
FP4_TILE_SIZE = 256
GQA_TILE_SIZE = 64

# Heuristic Thresholds
SPLIT_K_THRESHOLD = 4096      # K-dimension size where Split-K is likely triggered by compiler
TINY_LAYER_THRESHOLD = 256    # Skip analysis for layers smaller than this (e.g. Routers)
SATURATION_THRESHOLD = 20.0   # Waves > 20 implies tail effects are negligible (<5% overhead)

# --- Third Party Dependencies ---
try:
    from safetensors import safe_open
except ImportError:
    print("Error: 'safetensors' not installed. Please run: pip install safetensors")
    sys.exit(1)

# --- Logger Shim for Standalone Usage ---
# Allows this tool to run on non-NVIDIA machines (e.g., during capacity planning) 
# without requiring the full TRT-LLM stack.
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

# Source: NVIDIA Whitepapers (Hopper, Blackwell, Ada)
HARDWARE_DB: Dict[str, GPU] = {
    # Blackwell Architecture (B200)
    # Note: B200 is dual-die. Wave quantization occurs per-die physically.
    "b200":    GPU("NVIDIA B200 (Blackwell)", 132, 4, is_dual_die=True),
    "gb200":   GPU("NVIDIA GB200 (NVL72)",    132, 4, is_dual_die=True),
    
    # Hopper Architecture (H100/H200)
    "h200":    GPU("NVIDIA H200 (HBM3e)", 132, 4),
    "h100":    GPU("NVIDIA H100 (Hopper)", 132, 4),
    
    # Ada / Ampere Architecture
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


def get_tile_heuristic(name: str, dtype: str) -> int:
    """
    Estimates the GEMM CTA (Tile) Size based on layer type and precision.
    
    Args:
        name: Layer name (e.g., 'mlp.down_proj')
        dtype: Precision (fp16, fp4, etc.)
    
    Returns:
        Estimated Tile Size (N-dimension)
    """
    # Blackwell FP4/NVFP4 supports denser tiles to feed 2nd Gen Transformer Engine
    if dtype in ['fp4', 'nvfp4', 'int4']:
        return FP4_TILE_SIZE
    
    # FP8 often uses 128x128 or 128x256
    if dtype == 'fp8':
        return DEFAULT_TILE_SIZE
    
    # Standard FP16/BF16 Heuristics
    name = name.lower()
    
    # GQA (Grouped Query Attention) often has narrow KV heads
    if any(x in name for x in ['kv', 'key', 'value']):
        return GQA_TILE_SIZE
        
    # MLPs are standard GEMMs
    return DEFAULT_TILE_SIZE


def analyze_layer(name: str, 
                  shape: Tuple[int, int], 
                  m_dim: int, 
                  gpu: GPU, 
                  tp_size: int, 
                  dtype: str, 
                  seq_len: int) -> Optional[LayerStats]:
    """
    Calculates Wave Quantization Efficiency for a single layer.
    
    Handles:
        - Tensor Parallelism (TP) slicing
        - Mixture of Experts (MoE) load balancing
        - Decode (Memory Bound) vs Prefill (Compute Bound) detection
    """
    n_dim_raw, k_dim_raw = shape[0], shape[1]
    
    # --- 1. Workload Characterization ---
    
    # Logic: Tensor Parallelism
    # Column Parallel: Output (N) is split. Row Parallel: Input (K) is split.
    is_col_parallel = any(x in name for x in ['gate', 'up', 'qkv', 'query', 'key', 'value'])
    
    effective_n = n_dim_raw
    if is_col_parallel:
        # TRT-LLM handles uneven splits via padding, but we assume divisibility for analysis
        if n_dim_raw % tp_size == 0:
            effective_n = n_dim_raw // tp_size
        else:
            effective_n = math.ceil(n_dim_raw / tp_size)

    # Logic: Skip Tiny Layers (Norms, Biases, Routers)
    if effective_n < TINY_LAYER_THRESHOLD and m_dim < TINY_LAYER_THRESHOLD:
        return None

    # Logic: MoE (Mixture of Experts)
    # If MoE, the batch is distributed across experts.
    is_moe = "expert" in name or "block_sparse_moe" in name
    effective_m = m_dim
    if is_moe:
        # Heuristic: Assuming top-2 routing or ~8 experts for Mixtral/DeepSeek class models
        # We divide M by 8 to simulate the reduced batch size seen by each GEMM.
        num_experts = 8 
        effective_m = math.ceil(m_dim / num_experts)

    # --- 2. Physics Simulation ---
    
    tile_size = get_tile_heuristic(name, dtype)

    # Grid Size Calculation (Ceiling Division)
    grid_m = math.ceil(effective_m / tile_size)
    grid_n = math.ceil(effective_n / tile_size)
    total_blocks = grid_m * grid_n

    # Wave Capacity Calculation
    # B200 Dual-Die: We analyze per-die saturation to capture worst-case tail effects.
    active_sms = gpu.num_sms
    wave_capacity = active_sms * gpu.max_blocks_per_sm
    
    waves = total_blocks / wave_capacity
    if waves <= 0: return None

    full_waves = math.ceil(waves)
    efficiency = waves / full_waves
    tail_loss = (1.0 - efficiency) * 100

    # --- 3. Recommendation Engine ---
    rec = ""
    
    # Check Saturation: If waves are high, tail effects are mathematically negligible.
    # Amdahl's Law: Optimizing the tail of a 50-wave kernel yields < 2% gain.
    is_saturated = waves > SATURATION_THRESHOLD
    
    # Context: Decode Phase (Memory Bound)
    if seq_len == 1:
        rec = "Latency Bound (Mem)"
    # Context: Fully Saturated
    elif is_saturated:
        rec = "Saturated (Optimal)"
    # Context: Low Efficiency Analysis
    elif efficiency < 0.60:
        # Heuristic: Split-K Candidate
        # If K-dimension is massive, TRT-LLM likely engages Split-K to increase occupancy.
        if k_dim_raw > SPLIT_K_THRESHOLD:
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
    parser.add_argument('--model_dir', type=Path, required=True, help="Path to directory containing .safetensors")
    parser.add_argument('--batch_size', type=int, required=True, help="Target Runtime Batch Size")
    parser.add_argument('--seq_len', type=int, default=1, help="Sequence Length (1 for Decode, >1 for Prefill)")
    parser.add_argument('--tp_size', type=int, default=1, help="Tensor Parallelism Degree")
    parser.add_argument('--gpu', type=str, default='b200', choices=HARDWARE_DB.keys(), help="Target GPU Architecture")
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp8', 'fp4', 'nvfp4', 'int4'])
    parser.add_argument('--log_level', type=str, default='info')
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("Batch size must be greater than 0.")
    
    if hasattr(logger, 'setLevel'):
        logger.setLevel(args.log_level.upper())

    gpu = HARDWARE_DB[args.gpu]
    m_dim = args.batch_size * args.seq_len
    
    logger.info(f"Starting Analysis for {gpu.name}...")
    logger.info(f"Workload Config: M={m_dim} (BS={args.batch_size}, Seq={args.seq_len}) | TP={args.tp_size} | Precision={args.dtype.upper()}")

    # File Discovery
    files = sorted(list(args.model_dir.glob("*.safetensors")))
    if not files:
        logger.error(f"No .safetensors files found in {args.model_dir}")
        sys.exit(1)

    # Analysis Loop
    # We scan the first shard only. Model architecture is identical across shards.
    seen_sigs: Set[Tuple[str, Tuple[int, ...]]] = set()
    
    # Header Formatting
    header = f"{'Layer Name (Truncated)':<50} | {'Grid(MxN)':<14} | {'Waves':<8} | {'Eff %':<6} | {'Loss %':<6} | {'Status'}"
    print(f"\n{header}")
    print("-" * len(header))

    with safe_open(files[0], framework="np", device="cpu") as f:
        for key in f.keys():
            # Filter for Weight tensors only (ignore metadata/norms)
            if "weight" in key and "norm" not in key and "router" not in key:
                # Robust Architecture Detection:
                # Iterate to find the layer index number (e.g. '0' in 'layers.0.mlp')
                # This handles Llama (layers.0), Falcon (h.0), GPT-J (h.0), etc.
                parts = key.split(".")
                try:
                    layer_idx_loc = next(i for i, part in enumerate(parts) if part.isdigit())
                    # Generic name is everything after the number until the end
                    # e.g. 'self_attn.q_proj'
                    generic_name = ".".join(parts[layer_idx_loc+1:-1])
                except StopIteration:
                    # Skip layers that don't have a numeric index (e.g. final norm, embeddings)
                    continue

                # Strict Type Casting: Safetensors returns list, Set requires Tuple for hashing
                shape_list = f.get_slice(key).get_shape()
                shape = tuple(shape_list)
                
                # Deduplication: Only analyze unique layer shapes
                sig = (generic_name, shape)
                if sig in seen_sigs: continue
                seen_sigs.add(sig)

                stats = analyze_layer(key, shape, m_dim, gpu, args.tp_size, args.dtype, args.seq_len)
                
                if stats:
                    # Status Icons
                    icon = "✅"
                    if "CRITICAL" in stats.recommendation: icon = "❌"
                    elif "Tune" in stats.recommendation: icon = "⚠️"
                    elif "Split-K" in stats.recommendation: icon = "🛡️"
                    elif "Latency" in stats.recommendation: icon = "ℹ️"
                    elif "Saturated" in stats.recommendation: icon = "🔹"
                    elif "Vocab" in stats.recommendation: icon = "🔹"

                    print(f"{stats.name[-50:]:<50} | {f'{math.ceil(stats.shape[0]/128)}x{math.ceil(stats.shape[1]/128)}':<14} | {stats.waves:<8.2f} | {stats.efficiency*100:<6.1f} | {stats.tail_loss_percent:<6.1f} | {icon} {stats.recommendation}")

    print("-" * len(header))
    if gpu.is_dual_die:
        logger.info("NOTE: B200/GB200 analysis assumes per-die scheduling. Dual-Die scaling may mitigate some local tail effects.")

if __name__ == "__main__":
    main()