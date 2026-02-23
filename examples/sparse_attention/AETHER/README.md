# AETHER Sparse Attention

This document details enabling AETHER sparse attention within TensorRT-LLM.

AETHER (Adaptive Event-driven Threshold Hybrid Entangled Rendering) is a training-free, block-sparse attention mechanism that accelerates long-context LLM inference. It uses a lightweight "Event Radar" pre-scan to identify high-relevance KV-cache blocks, achieving significant speedups while maintaining attention quality.

For technical details, see: [AETHER Paper](https://doi.org/10.13141/RG.2.2.14811.27684)

## Overview

During LLM inference, computing full attention over long sequences becomes a bottleneck. AETHER addresses this with deviation-based block scoring:

1. **Metadata Precomputation**: Compute per-block statistics (mean direction, radius, variance, concentration) from the KV cache.
2. **Event Radar Scoring**: For each query, estimate attention potential per block using:
   ```
   score = (q · μ_block) × scale + ||q|| × radius × factors
   ```
3. **Block Selection**: Keep blocks exceeding the threshold; skip the rest.
4. **Sparse Attention**: Compute attention only over selected blocks.

## Algorithm Details

### Event Radar Scoring Formula

The core insight is that attention potential for a block can be upper-bounded without computing all individual attention scores:

```
AttentionPotential(q, block) ≤ (q · μ) + ||q|| × r × f_var × f_conc
```

Where:
- `μ` = normalized mean of keys in block
- `r` = maximum angular radius (deviation from mean)
- `f_var` = 1 + √variance (variance-aware bonus)
- `f_conc` = concentration factor ∈ (0, 1] (tighter bound)

### Concentration Factor (Tight Bounds)

The concentration factor measures how tightly keys cluster around the block mean:

```python
alignment = (keys_normalized · mean).mean()  # per block
concentration = clamp(alignment, 0.1, 1.0)
```

High concentration (≈1.0) means keys are tightly clustered, enabling tighter scoring bounds and fewer false positives.

### Causal Streaming Mode

For autoregressive decoding, AETHER supports:
- **Recency bias**: Recent blocks receive scoring bonus
- **Local window**: Last N blocks are always kept (guaranteed local attention)

```
score *= (1 + recency_bonus)
is_active |= (block_idx >= N_blocks - local_window)
```

## Support Matrix

| Feature | Status |
|---------|--------|
| GPU Compute Capability | >= 8.0 (Ampere+) |
| FP16 / BF16 / FP32 | ✓ |
| Triton Kernels | ✓ |
| TRT-LLM Integration | ✓ |
| Paged KV Cache | Planned |
| Tensor Parallel | Planned |
| CUDA Graph | Planned |

## Usage

### TensorRT-LLM API (Recommended)

The recommended way to use AETHER is through the official TensorRT-LLM API:

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import AetherSparseAttentionConfig

# Create AETHER configuration
aether_config = AetherSparseAttentionConfig(
    block_size=64,          # KV block size for partitioning
    threshold=0.05,         # Attention score threshold (lower = more blocks)
    local_window=8,         # Recent blocks to always keep
    use_variance=True,      # Enable variance-aware scoring
    use_concentration=True, # Enable concentration-based bounds
    min_seq_length=128,     # Min length to enable sparse attention
)

# Initialize LLM with AETHER sparse attention
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    sparse_attention_config=aether_config,
)

# Generate text - AETHER is automatically applied
output = llm.generate("The capital of France is", max_tokens=50)
print(output)
```

### Running the E2E Demo

```bash
# Basic usage with TinyLlama
python run_aether_e2e.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Compare with baseline (non-sparse) inference
python run_aether_e2e.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --compare-baseline

# Custom configuration
python run_aether_e2e.py \
    --model meta-llama/Llama-2-7b-hf \
    --block-size 64 \
    --threshold 0.05 \
    --local-window 8 \
    --compare-baseline
```

### Low-Level Kernel API

For advanced use cases, you can access the AETHER kernels directly:

```python
from tensorrt_llm._torch.attention_backend.sparse.aether_kernels import (
    run_aether_sparse, precompute_metadata
)

# Precompute block metadata from KV cache
means, radii, variances, concentrations = precompute_metadata(
    keys,  # (B, H, S, D)
    block_size=64,
    compute_variance=True,
    compute_concentration=True
)

# Run sparse attention prediction
mask, scores = run_aether_sparse(
    query,              # (B, H, D)
    means, radii,
    block_variances=variances,
    block_concentrations=concentrations,
    threshold=0.15,
    use_variance=True,
    use_concentration=True,
    is_causal=False,
)
# mask: (B, H, N_blocks) boolean - True = keep block
```

### Kernel Variants

| Variant | Flags | Use Case |
|---------|-------|----------|
| Basic | None | Fastest, baseline sparsity |
| Variance-aware | `USE_VARIANCE=True` | Better quality for diverse data |
| Tight bound | `USE_CONCENTRATION=True` | Fewer false positives |
| Causal | `IS_CAUSAL=True` | Autoregressive decoding |

### Running Benchmarks

```bash
# Basic benchmark
python benchmark_aether.py --batch_size 4 --seq_len 4096

# Full evaluation with quality metrics
python benchmark_aether.py --verify --all-variants

# Export results to CSV
python benchmark_aether.py --export-csv results.csv

# Sweep sparsity levels
python benchmark_aether.py --sparsity-sweep 0.5 0.6 0.7 0.8 0.9
```

## Configuration Arguments

- **`block_size`** (int, default=64): Size of each KV block. Sequence length must be divisible by this.
- **`threshold`** (float, default=0.15): Attention potential threshold. Higher = more sparsity, potentially lower quality.
- **`target_sparsity`** (float, default=0.8): For adaptive mode, target fraction of blocks to skip.
- **`local_window`** (int, default=4): For causal mode, number of recent blocks to always keep.
- **`recency_decay`** (float, default=0.95): For causal mode, exponential decay for recency bonus.
- **`use_variance`** (bool): Enable variance-aware scoring.
- **`use_concentration`** (bool): Enable concentration-based tight bounds.
- **`is_causal`** (bool): Enable causal streaming with recency bias.

## Evaluation Metrics

The benchmark suite reports comprehensive metrics beyond cosine similarity:

| Metric | Description |
|--------|-------------|
| Cosine Similarity | Similarity between sparse and dense outputs |
| L2 Distance | Normalized L2 distance of outputs |
| KL Divergence | KL(sparse_attn ‖ dense_attn) |
| Precision | % of kept blocks that are truly important |
| Recall | % of important blocks that are kept |
| Attention Entropy | Change in attention distribution entropy |
| Theoretical Speedup | 1 / (1 - sparsity) |
| Memory Saved | MB of KV cache memory avoided |

## Expected Performance

Typical results on Llama-7B style configurations (H=32, D=128):

| Sparsity | Cosine Sim | Precision | Recall | Speedup |
|----------|------------|-----------|--------|---------|
| 50% | 0.998+ | 0.95+ | 0.98+ | 2.0x |
| 70% | 0.995+ | 0.92+ | 0.95+ | 3.3x |
| 80% | 0.990+ | 0.88+ | 0.92+ | 5.0x |
| 90% | 0.970+ | 0.80+ | 0.85+ | 10.0x |

*Note: Actual results depend on data distribution and sequence length.*
