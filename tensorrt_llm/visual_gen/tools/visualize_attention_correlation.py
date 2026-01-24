# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Attention Correlation Visualization Script

This script reads saved io_tensors_per_step data, extracts query and key tensors from attention layers,
computes the correlation (attention scores) for each head, and generates heatmaps.

Usage:
    python tools/visualize_attention_correlation.py --input_dir /path/to/saved/tensors --output_dir /path/to/output
"""

import argparse
import glob
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

matplotlib.use("Agg")  # Use non-interactive backend for multiprocessing


def find_io_tensor_files(input_dir: str) -> List[str]:
    """Find all io_tensors files in the input directory.

    Args:
        input_dir: Directory containing saved tensor files

    Returns:
        List of tensor file paths
    """
    # Find all io_tensors chunk files
    chunk_files = glob.glob(os.path.join(input_dir, "io_tensors_chunk_*.pt"))

    if not chunk_files:
        # Try to load a single io_tensors.pt file
        single_file = os.path.join(input_dir, "io_tensors.pt")
        if os.path.exists(single_file):
            chunk_files = [single_file]
        else:
            raise FileNotFoundError(f"No io_tensors files found in {input_dir}")

    return sorted(chunk_files)


def process_chunk_attention_tensors(
    chunk_data: Dict[int, Dict[str, Any]],
    target_steps: Optional[Set[int]] = None,
    layer_patterns: Optional[List[str]] = None,
) -> List[Tuple[int, str, str, torch.Tensor, torch.Tensor]]:
    """Extract query and key tensors from attention layers in a single chunk.

    Args:
        chunk_data: Single chunk of io_tensors_per_step data
        target_steps: Optional set of steps to filter
        layer_patterns: Optional list of layer name patterns to filter

    Returns:
        List of tuples: (step, layer_name, cfg_type, query, key)
    """
    attention_data = []

    for step, step_data in chunk_data.items():
        # Filter by steps if specified
        if target_steps is not None and step not in target_steps:
            continue

        for cfg_type, cfg_data in step_data.items():
            for layer_name, tensors in cfg_data.items():
                # Only process attention layers
                if "attn" not in layer_name.lower():
                    continue

                # Filter by layer patterns if specified
                if layer_patterns is not None:
                    if not any(pattern in layer_name.lower() for pattern in layer_patterns):
                        continue

                if "query" in tensors and "key" in tensors:
                    query = tensors["query"]
                    key = tensors["key"]

                    # Verify tensor shapes
                    if len(query.shape) == 4 and len(key.shape) == 4:
                        print(
                            f"Found attention tensors: step={step}, layer={layer_name}, cfg={cfg_type}"
                        )
                        print(f"  Query shape: {query.shape}, Key shape: {key.shape}")
                        attention_data.append((step, layer_name, cfg_type, query, key))

    return attention_data


def compute_attention_correlation(
    query: torch.Tensor,
    key: torch.Tensor,
    temperature: float = 1.0,
    block_size: int = 128,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute attention correlation (softmax of query @ key^T) for each head.

    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        temperature: Temperature for softmax (default: 1.0 for no scaling)
        device: Device to compute on ('cpu' or 'cuda:X')

    Returns:
        Attention weights of shape [batch_size, num_heads, seq_len, seq_len] (on CPU)
    """
    # Move tensors to GPU for computation
    query_gpu = query.to(device).float()
    key_gpu = key.to(device).float()

    # Scale by sqrt(head_dim) as is standard in attention
    scale = 1.0 / np.sqrt(query_gpu.size(-1))

    # Compute attention scores: query @ key^T
    attention_scores = torch.matmul(query_gpu, key_gpu.transpose(-2, -1)) * scale / temperature

    # Apply softmax to get attention weights
    attention_weights = F.softmax(attention_scores, dim=-1)

    attention_weights = F.max_pool2d(attention_weights, kernel_size=block_size, stride=block_size)

    # Move back to CPU to save GPU memory
    attention_weights_cpu = attention_weights.cpu()

    return attention_weights_cpu


def create_raw_heatmaps_per_head(
    attention_weights: torch.Tensor, step: int, layer_name: str, cfg_type: str, output_dir: str
) -> None:
    """Create and save raw high-resolution heatmap images for each attention head.
    The image resolution matches the attention weight matrix shape exactly.

    Args:
        attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        step: Denoising step
        layer_name: Name of the attention layer
        cfg_type: Configuration type (positive/negative)
        output_dir: Output directory for saving heatmaps (flat structure)
    """
    batch_size, num_heads, seq_len, _ = attention_weights.shape

    # Take the first batch
    weights_batch = attention_weights[0].numpy()

    # Create output directory (flat structure)
    os.makedirs(output_dir, exist_ok=True)

    # Clean layer name for filename
    clean_layer_name = layer_name.replace("/", "_").replace(".", "_")

    # Create raw heatmaps for each head
    for head_idx in range(num_heads):
        head_weights = weights_batch[head_idx]  # Shape: [seq_len, seq_len]

        # Create figure with exact pixel size matching the attention matrix
        # Set DPI to 1 so that figure size directly corresponds to pixel size
        fig = plt.figure(figsize=(seq_len, seq_len), dpi=1)
        ax = fig.add_axes([0, 0, 1, 1])  # Use entire figure area

        # Remove all axes decorations
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        # Display the attention weights as image
        # Note: imshow expects (height, width) but our matrix is (seq_len, seq_len)
        # origin='upper' puts the origin at top-left (query=0 at top, key=0 at left)
        # extent controls the data coordinates of the image
        im = ax.imshow(
            head_weights,
            cmap="viridis",  # Use a good colormap
            aspect="equal",
            interpolation="nearest",
            origin="upper",
            extent=[0, seq_len, seq_len, 0],
        )  # [left, right, bottom, top]

        # Save with exact resolution
        head_filename = f"step_{step:03d}_{clean_layer_name}_{cfg_type}_head_{head_idx:02d}.jpg"
        head_path = os.path.join(output_dir, head_filename)

        # Save without any padding or borders
        plt.savefig(
            head_path,
            format="jpg",
            dpi=1,  # 1:1 pixel mapping
            bbox_inches="tight",
            pad_inches=0,
            facecolor="black",
            edgecolor="none",
        )
        plt.close()

        print(f"Saved raw heatmap ({seq_len}x{seq_len}): {head_filename}")

    print(f"Completed {num_heads} raw heatmaps for {layer_name} ({cfg_type}) at step {step}")


def process_chunk_on_gpu(
    chunk_file: str,
    gpu_id: int,
    output_dir: str,
    temperature: float,
    target_steps: Optional[Set[int]] = None,
    layer_patterns: Optional[List[str]] = None,
) -> int:
    """Process a single chunk file on a specific GPU.

    Args:
        chunk_file: Path to the chunk file
        gpu_id: GPU device ID
        output_dir: Output directory for heatmaps
        temperature: Temperature for softmax computation
        target_steps: Optional set of steps to filter
        layer_patterns: Optional list of layer name patterns to filter

    Returns:
        Number of attention tensor pairs processed
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu"

    print(f"[GPU {gpu_id}] Processing {os.path.basename(chunk_file)} on {device}")

    # Load current chunk
    chunk_data = torch.load(chunk_file, map_location="cpu")
    print(f"[GPU {gpu_id}] Loaded chunk with {len(chunk_data)} steps")

    # Extract attention tensors from current chunk
    attention_data = process_chunk_attention_tensors(chunk_data, target_steps, layer_patterns)
    print(f"[GPU {gpu_id}] Found {len(attention_data)} attention tensor pairs")

    if not attention_data:
        print(f"[GPU {gpu_id}] No attention tensors found, skipping...")
        return 0

    processed_count = 0

    # Process each attention tensor pair
    for step, layer_name, cfg_type, query, key in attention_data:
        print(f"[GPU {gpu_id}] Processing step {step}, layer {layer_name}, cfg {cfg_type}")
        print(f"[GPU {gpu_id}] Query shape: {query.shape}, Key shape: {key.shape}")

        # Compute attention correlation on GPU
        attention_weights = compute_attention_correlation(query, key, temperature, device=device)
        print(f"[GPU {gpu_id}] Computed attention weights shape: {attention_weights.shape}")

        # Create and save raw heatmaps for each head
        create_raw_heatmaps_per_head(attention_weights, step, layer_name, cfg_type, output_dir)
        processed_count += 1

    # Clear memory after processing chunk
    del chunk_data, attention_data
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(
        f"[GPU {gpu_id}] Finished processing {os.path.basename(chunk_file)} - {processed_count} pairs"
    )
    return processed_count


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        return []

    gpu_count = torch.cuda.device_count()
    return list(range(gpu_count))


def distribute_chunks_to_gpus(chunk_files: List[str], gpu_ids: List[int]) -> List[List[str]]:
    """Distribute chunk files evenly across available GPUs."""
    if not gpu_ids:
        # No GPUs available, return all chunks for CPU processing
        return [chunk_files]

    num_gpus = len(gpu_ids)
    chunks_per_gpu = [[] for _ in range(num_gpus)]

    for i, chunk_file in enumerate(chunk_files):
        gpu_idx = i % num_gpus
        chunks_per_gpu[gpu_idx].append(chunk_file)

    return chunks_per_gpu


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attention correlations from saved io_tensors"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing saved io_tensors_per_step files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for heatmap visualizations"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax computation (default: 1.0)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated list of specific steps to process (default: all)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layer name patterns to process (default: all attention layers)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="Number of GPUs to use (-1 for all available, 0 for CPU only)",
    )
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU-only processing")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse filtering criteria
    target_steps = None
    if args.steps:
        target_steps = set(int(s.strip()) for s in args.steps.split(","))
        print(f"Filtering for steps: {sorted(target_steps)}")

    layer_patterns = None
    if args.layers:
        layer_patterns = [p.strip().lower() for p in args.layers.split(",")]
        print(f"Filtering for layer patterns: {layer_patterns}")

    # Find all tensor files
    print(f"Searching for tensor files in {args.input_dir}...")
    tensor_files = find_io_tensor_files(args.input_dir)
    print(f"Found {len(tensor_files)} tensor files to process")

    if not tensor_files:
        print("No tensor files found!")
        return

    # Determine GPU usage
    if args.cpu_only:
        gpu_ids = []
        print("Using CPU-only processing")
    else:
        available_gpus = get_available_gpus()
        if args.num_gpus == -1:
            gpu_ids = available_gpus
        elif args.num_gpus == 0:
            gpu_ids = []
        else:
            gpu_ids = available_gpus[: min(args.num_gpus, len(available_gpus))]

        if gpu_ids:
            print(f"Using GPUs: {gpu_ids}")
        else:
            print("No GPUs available or requested, using CPU")

    # Distribute chunks across GPUs
    if gpu_ids:
        chunks_per_gpu = distribute_chunks_to_gpus(tensor_files, gpu_ids)
        print(f"Distributed {len(tensor_files)} chunks across {len(gpu_ids)} GPUs")
        for i, chunks in enumerate(chunks_per_gpu):
            print(f"  GPU {gpu_ids[i]}: {len(chunks)} chunks")
    else:
        # CPU processing
        chunks_per_gpu = [tensor_files]
        gpu_ids = [-1]  # -1 indicates CPU

    # Process chunks in parallel using multiprocessing
    if len(gpu_ids) > 1:
        print(f"\nStarting parallel processing with {len(gpu_ids)} workers...")
        mp.set_start_method("spawn", force=True)

        with mp.Pool(processes=len(gpu_ids)) as pool:
            # Create worker tasks
            tasks = []
            for gpu_id, chunk_list in zip(gpu_ids, chunks_per_gpu):
                for chunk_file in chunk_list:
                    task = pool.apply_async(
                        process_chunk_on_gpu,
                        (
                            chunk_file,
                            gpu_id,
                            args.output_dir,
                            args.temperature,
                            target_steps,
                            layer_patterns,
                        ),
                    )
                    tasks.append(task)

            # Wait for all tasks to complete and collect results
            total_processed = 0
            for task in tasks:
                try:
                    result = task.get(timeout=3600)  # 1 hour timeout per chunk
                    total_processed += result
                except Exception as e:
                    print(f"Error processing chunk: {e}")
    else:
        # Sequential processing (single GPU or CPU)
        print("\nStarting sequential processing...")
        total_processed = 0
        gpu_id = gpu_ids[0]

        for chunk_idx, chunk_file in enumerate(tensor_files):
            print(f"\n{'=' * 60}")
            print(
                f"Processing chunk {chunk_idx + 1}/{len(tensor_files)}: {os.path.basename(chunk_file)}"
            )
            print(f"{'=' * 60}")

            try:
                result = process_chunk_on_gpu(
                    chunk_file,
                    gpu_id,
                    args.output_dir,
                    args.temperature,
                    target_steps,
                    layer_patterns,
                )
                total_processed += result
            except Exception as e:
                print(f"Error processing chunk {chunk_file}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Visualization complete! Processed {total_processed} attention tensor pairs.")
    print(f"Check {args.output_dir} for generated heatmaps.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
