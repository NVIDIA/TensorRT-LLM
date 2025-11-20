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

This utility performs static workload characterization to predict GPU Scheduler
Theoretical Occupancy (Tail Effects) based on Model Architecture and Target Silicon.
It is primarily used for "Shift-Left" batch size tuning and kernel optimization diagnosis
before engine compilation.
"""

import argparse
import math
import sys
import csv
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Set, Dict

# --- Dependency Management (Defensive Imports) ---
try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

try:
    # Use pynvml (or nvidia-ml-py) for robust, environment-independent GPU identification
    import pynvml
except ImportError:
    # Assign None so subsequent checks (if pynvml is not None) work cleanly
    pynvml = None

# --- Logger Shim ---
try:
    # Attempt to use TRT-LLM's internal logger
    import tensorrt_llm.logger as trt_logger_mod
    logger = trt_logger_mod.logger
except ImportError:
    # Fallback to standard logging library
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("trtllm-wave")
    if not hasattr(logger, "debug"):
        logger.debug = logger.info


# --- Configuration Constants & Heuristics ---
DEFAULT_TILE_SIZE = 128
FP4_TILE_SIZE = 256
GQA_TILE_SIZE = 64

# Heuristic Thresholds
DEFAULT_SPLIT_K_THRESHOLD = 4096
TINY_LAYER_THRESHOLD = 256
SATURATION_THRESHOLD = 20.0
DEFAULT_MOE_EXPERTS = 8


# --- Hardware Database (Profiles) ---
class GPU(NamedTuple):
    """Stores key physical characteristics of a target GPU for simulation."""
    name: str
    num_sms: int
    max_blocks_per_sm: int
    is_dual_die: bool = False

HARDWARE_DB: Dict[str, GPU] = {
    # Architecture profiles used for physics simulation (SM count, max occupancy blocks)
    "b200":    GPU("NVIDIA B200 (Blackwell)", 132, 4, is_dual_die=True),
    "gb200":   GPU("NVIDIA GB200 (NVL72)",    132, 4, is_dual_die=True),
    "h200":    GPU("NVIDIA H200 (HBM3e)", 132, 4),
    "h100":    GPU("NVIDIA H100 (Hopper)", 132, 4),
    "l40s":    GPU("NVIDIA L40S (Ada)",    142, 4),
    "a100":    GPU("NVIDIA A100 (Ampere)", 108, 4),
    "l4":      GPU("NVIDIA L4 (Ada)",      58,  4),
}

class LayerStats(NamedTuple):
    """Holds the computed metrics for a single layer analysis."""
    name: str
    shape: Tuple[int, int]
    waves: float
    occupancy: float
    tail_loss_percent: float
    recommendation: str
    weight_factor: int      # Total blocks, used for weighted average calculation


def get_gpu_from_env() -> Optional[GPU]:
    """
    Attempts to auto-detect the GPU using pynvml.

    Returns:
        Optional[GPU]: The matching GPU profile from HARDWARE_DB or None if detection fails.
    """
    if pynvml is None:
        logger.debug("pynvml not installed. GPU auto-detection skipped.")
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name_raw = pynvml.nvmlDeviceGetName(handle).decode('utf-8').lower()
        pynvml.nvmlShutdown()

        # Fuzzy matching logic to map raw names to DB keys
        if "b200" in gpu_name_raw: return HARDWARE_DB["b200"]
        if "h100" in gpu_name_raw: return HARDWARE_DB["h100"]
        if "h200" in gpu_name_raw: return HARDWARE_DB["h200"]
        if "a100" in gpu_name_raw: return HARDWARE_DB["a100"]
        if "l40" in gpu_name_raw:  return HARDWARE_DB["l40s"]
        if "l4" in gpu_name_raw:   return HARDWARE_DB["l4"]
        
        return None
    except pynvml.NVMLError as e:
        logger.debug(f"pynvml initialization failed: {e}. Cannot query CUDA device.")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error during pynvml check: {e}")
        return None

def parse_custom_hardware(custom_str: str) -> GPU:
    """
    Parses a custom hardware string into a GPU NamedTuple.
    Expected format: 'Name;SMs;MaxBlocks' (e.g., 'CustomA100;108;4').
    """
    try:
        parts = custom_str.split(';')
        if len(parts) < 3:
            raise ValueError("Missing fields")
        name = parts[0]
        sms = int(parts[1])
        blocks = int(parts[2])
        dual = len(parts) > 3 and "dual" in parts[3].lower()
        return GPU(name, sms, blocks, is_dual_die=dual)
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Invalid custom hardware string: '{custom_str}'. "
            "Expected format: 'Name;SMs;MaxBlocks' or 'Name;SMs;MaxBlocks;dual'"
        ) from exc

def get_tile_heuristic(name: str, dtype: str, tile_m: Optional[int], tile_n: Optional[int]) -> Tuple[int, int]:
    """
    Determines the GEMM CTA (Tile) Size, honoring user overrides with precedence.

    User-provided tile dimensions (\p tile_m, \p tile_n) take precedence over 
    internal heuristics. If an input is None, the heuristic is applied.
    """
    # 1. Determine the heuristic base size
    base_size = DEFAULT_TILE_SIZE
    if dtype in ['fp4', 'nvfp4', 'int4']:
        base_size = FP4_TILE_SIZE
    # Apply specialized heuristic for narrow KV heads (commonly 64 or 128)
    elif any(x in name.lower() for x in ['kv', 'key', 'value']):
        base_size = GQA_TILE_SIZE
    
    # 2. Apply user override: honor user input first, fall back to heuristic base size.
    final_m = tile_m if tile_m is not None else base_size
    final_n = tile_n if tile_n is not None else base_size
    
    return final_m, final_n

def analyze_layer(name: str,
                  shape: Tuple[int, int],
                  m_dim: int,
                  gpu: GPU,
                  tp_size: int,
                  dtype: str,
                  seq_len: int,
                  split_k_thresh: int,
                  num_experts: int,
                  tile_m: Optional[int],
                  tile_n: Optional[int]) -> Optional[LayerStats]:
    """
    Calculates Wave Quantization Occupancy based on workload dimensions and GPU profile.

    This models the potential for tail effects caused by limited thread blocks.

    Returns:
        Optional[LayerStats]: Computed metrics, or None if the layer is trivially small.
    """
    n_dim_raw, k_dim_raw = shape[0], shape[1]

    # --- 1. Workload Characterization (Parallelism Adjustments) ---
    effective_n = math.ceil(n_dim_raw / tp_size)
    
    if effective_n < TINY_LAYER_THRESHOLD and m_dim < TINY_LAYER_THRESHOLD:
        return None

    is_moe = "expert" in name or "block_sparse_moe" in name
    effective_m = m_dim
    if is_moe:
        # MoE layers distribute the batch M dimension across experts
        effective_m = math.ceil(m_dim / num_experts)

    # --- 2. Physics Simulation (Grid and Waves) ---
    tile_m_size, tile_n_size = get_tile_heuristic(name, dtype, tile_m, tile_n)

    # Calculate grid size based on effective dimensions and tile size
    grid_m = math.ceil(effective_m / tile_m_size)
    grid_n = math.ceil(effective_n / tile_n_size)
    total_blocks = grid_m * grid_n

    # Total capacity of the GPU for thread blocks
    wave_capacity = gpu.num_sms * gpu.max_blocks_per_sm
    waves = total_blocks / wave_capacity
    
    if waves <= 0: return None

    # Theoretical Occupancy calculation (how much of the final wave is filled)
    full_waves = math.ceil(waves)
    occupancy = waves / full_waves
    tail_loss = (1.0 - occupancy) * 100

    # --- 3. Recommendation Engine ---
    rec = ""
    is_saturated = waves > SATURATION_THRESHOLD

    if seq_len == 1 or waves < 1.0:
        # Low wave count or decoding workloads are often memory-bound
        rec = "Latency Bound (Mem)"
    elif is_saturated:
        rec = "Saturated (Optimal)"
    elif occupancy < 0.60:
        # Check K dimension against tunable threshold for Split-K recommendation
        if k_dim_raw > split_k_thresh:
            rec = "Split-K Likely"
        elif "lm_head" in name or "embed" in name:
            rec = "Vocab/Embed Layer"
        else:
            rec = "CRITICAL: Tail Effect"
    elif occupancy < 0.85:
        rec = "Tune Batch Size"

    return LayerStats(name, (effective_m, effective_n), waves, occupancy, tail_loss, rec, total_blocks)

def main():
    # --- CLI Argument Setup ---
    parser = argparse.ArgumentParser(
        description="TRT-LLM Static Wave Analyzer",
        formatter_class=argparse.RawTextHelpFormatter  # Allows for multi-line help text
    )
    parser.add_argument('--model_dir', type=Path, required=True, help="Path to .safetensors directory")
    parser.add_argument('--batch_size', type=int, required=True, help="Target Runtime Batch Size")
    parser.add_argument('--seq_len', type=int, default=1, help="Sequence Length")
    parser.add_argument('--tp_size', type=int, default=1, help="Tensor Parallelism Degree")

    parser.add_argument('--dtype', type=str, default='fp16',
        choices=['fp16', 'bf16', 'fp8', 'fp4', 'nvfp4', 'int4'],
        help="Compute Precision.")

    # Hardware Configuration - Explicitly documenting auto-detection failure fallback
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', type=str, choices=HARDWARE_DB.keys(),
                       help="Force specific GPU profile (e.g., h100). \n"
                            "If omitted and auto-detection fails (requires pynvml), \n"
                            "the tool defaults to H100.")
    group.add_argument('--custom_hardware', type=str, help="Define custom GPU: 'Name;SMs;MaxBlocks'")

    # Output Options
    parser.add_argument('--csv', type=Path, help="Output results to CSV file")
    parser.add_argument('--log_level', type=str, default='info', choices=['info', 'debug'])

    # Heuristic Overrides (Tunability)
    parser.add_argument('--moe_experts', type=int, default=DEFAULT_MOE_EXPERTS, help="Number of experts for MoE analysis")
    parser.add_argument('--split_k_threshold', type=int, default=DEFAULT_SPLIT_K_THRESHOLD, help="K-dim threshold for Split-K recommendation.")
    parser.add_argument('--tile_m', type=int, default=None, help="Override the M dimension of the kernel tile (row count)")
    parser.add_argument('--tile_n', type=int, default=None, help="Override the N dimension of the kernel tile (column count)")

    args = parser.parse_args()

    # Robust Dependency Check
    if safe_open is None:
        parser.error("'safetensors' is required for this tool. Please install it (e.g. `pip install safetensors`).")

    if args.batch_size <= 0 or args.seq_len <= 0 or args.tp_size <= 0:
        parser.error("Batch size, sequence length, and TP size must be greater than 0.")

    # Logger Configuration
    try:
        import tensorrt_llm.logger as trt_logger_mod
        if hasattr(trt_logger_mod, "set_level"):
            trt_logger_mod.set_level(args.log_level)
    except ImportError:
        if hasattr(logger, "setLevel"):
            logger.setLevel(args.log_level.upper())

    # Hardware Resolution (1. Custom, 2. Manual, 3. Auto-detect, 4. Default H100)
    gpu = None
    if args.custom_hardware:
        try:
            gpu = parse_custom_hardware(args.custom_hardware)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.gpu:
        gpu = HARDWARE_DB[args.gpu]
    else:
        # If no GPU is specified, attempt auto-detection; otherwise, default to H100.
        gpu = get_gpu_from_env()
        if gpu:
            logger.info(f"Auto-detected Hardware: {gpu.name}")
        else:
            logger.warning("Hardware could not be auto-detected and no --gpu specified. Defaulting to H100.")
            gpu = HARDWARE_DB["h100"]

    m_dim = args.batch_size * args.seq_len

    logger.info(f"Starting Analysis for {gpu.name}...")
    logger.info(f"Workload: M={m_dim} (BS={args.batch_size}, Seq={args.seq_len}) | TP={args.tp_size} | {args.dtype.upper()}")

    # Output Assumptions and Overrides
    print("\n[SCOPE] Analyzing Linear Layers (GEMMs) only. Attention kernels (MHA/FA) and Norms are excluded.")
    print("[ASSUMPTION] Metric is 'Theoretical Occupancy' (Shared Mem limited). Does not account for DRAM bandwidth.")
    print(f"[ASSUMPTION] MoE layers assume {args.moe_experts}-expert routing. For accuracy, check the model's config.json for 'num_local_experts'.")
    if args.tile_m or args.tile_n:
        tile_m_str = str(args.tile_m) if args.tile_m is not None else "Heuristic"
        tile_n_str = str(args.tile_n) if args.tile_n is not None else "Heuristic"
        print(f"[OVERRIDE] Forcing Tile Size: M={tile_m_str} x N={tile_n_str}")

    files = sorted(list(args.model_dir.glob("*.safetensors")))
    if not files:
        logger.error(f"No .safetensors files found in {args.model_dir}")
        sys.exit(1)

    # Use signatures (name and shape) to prevent duplicate analysis across sharded files
    seen_sigs: Set[Tuple[str, Tuple[int, ...]]] = set()
    header = f"{'Layer Name (Truncated)':<50} | {'Grid(MxN)':<14} | {'Waves':<8} | {'Occ %':<6} | {'Loss %':<6} | {'Status'}"
    print(f"\n{header}")
    print("-" * len(header))

    csv_file, csv_writer = None, None
    if args.csv:
        csv_file = open(args.csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Layer Name", "M_Dim", "N_Dim", "Waves", "Occupancy", "Tail_Loss_Percent", "Recommendation"])

    total_weighted_occupancy = 0.0
    total_weight = 0

    try:
        for path in files:
            with safe_open(path, framework="np", device="cpu") as f:
                for key in f.keys():
                    # Filter out non-GEMM layers (norms, biases, etc.)
                    if "weight" in key and "norm" not in key and "router" not in key:
                        
                        # --- Architecture Parsing (Defensive Logic) ---
                        parts = key.split(".")
                        generic_name = key
                        try:
                            # Heuristic: find the layer index (e.g., '0' in 'layers.0.mlp...')
                            layer_idx_loc = next(i for i, part in enumerate(parts) if part.isdigit())
                            # Short name is everything after the index up to the 'weight' ending
                            generic_name = ".".join(parts[layer_idx_loc+1:-1])
                        except StopIteration:
                            # Fallback: Used for flat checkpoints or simple keys (e.g., 'lm_head.weight').
                            logger.debug(f"Layer name parsing failed for '{key}'. Using truncated key.")
                            generic_name = key.split('.')[-2] if len(key.split('.')) > 1 else key

                        shape_list = f.get_slice(key).get_shape()
                        shape = tuple(shape_list)

                        sig = (generic_name, shape)
                        if sig in seen_sigs: continue
                        seen_sigs.add(sig)

                        stats = analyze_layer(key, shape, m_dim, gpu, args.tp_size, args.dtype, args.seq_len, 
                                            args.split_k_threshold, args.moe_experts, args.tile_m, args.tile_n)

                        if stats:
                            total_weighted_occupancy += (stats.occupancy * stats.weight_factor)
                            total_weight += stats.weight_factor

                            # Status Icons and Logic
                            icon = "✅"
                            if "CRITICAL" in stats.recommendation: icon = "❌"
                            elif "Tune" in stats.recommendation: icon = "⚠️"
                            elif "Split-K" in stats.recommendation: icon = "🛡️"
                            elif "Latency" in stats.recommendation or "Vocab" in stats.recommendation: icon = "🔹"

                            tile_m_eff, tile_n_eff = get_tile_heuristic(stats.name, args.dtype, args.tile_m, args.tile_n)
                            grid_str = f"{math.ceil(stats.shape[0]/tile_m_eff)}x{math.ceil(stats.shape[1]/tile_n_eff)}"

                            print(f"{stats.name[-50:]:<50} | {grid_str:<14} | {stats.waves:<8.2f} | {stats.occupancy*100:<6.1f} | {stats.tail_loss_percent:<6.1f} | {icon} {stats.recommendation}")

                            if csv_writer:
                                csv_writer.writerow([stats.name, stats.shape[0], stats.shape[1], stats.waves, stats.occupancy, stats.tail_loss_percent, stats.recommendation])
    finally:
        if csv_file:
            csv_file.close()
            print(f"\n[INFO] Detailed report saved to {args.csv}")

    print("-" * len(header))

    # Summary Block
    if total_weight > 0:
        avg_occ = (total_weighted_occupancy / total_weight) * 100
        print(f"MODEL SUMMARY FOR {gpu.name}:")
        print(f"  Weighted Average Theoretical Occupancy: {avg_occ:.2f}%")
        # Added comment to defend the summary metric
        print("  (Weighting based on Total Thread Blocks, acting as a proxy for layer compute volume.)")
        if avg_occ < 75.0:
             print("  Recommendation: ⚠️  Low utilization detected. Consider increasing Batch Size or TP degree.")
        else:
             print("  Recommendation: ✅  Workload appears well-saturated.")

    if gpu.is_dual_die:
        logger.info("NOTE: Analysis assumes per-die scheduling for Dual-Die architectures (GB200/B200).")

if __name__ == "__main__":
    main()