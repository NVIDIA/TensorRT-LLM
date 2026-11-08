<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# TE and Megatron-LM CUDA Graph APIs

## Overview

| API | Provider | Scope | Best For |
|-----|----------|-------|----------|
| TE `make_graphed_callables` | Transformer Engine | Per-callable (manual) | Any PyTorch model with FP8/PP needs |
| `CudaGraphManager` | Megatron-LM | Per-layer (automatic) | Megatron-LM with PP > 1 |
| `FullCudaGraphWrapper` | Megatron-LM | Full iteration (automatic) | Maximum performance, static workloads |

---

## Challenges: FP8 and Pipeline Parallelism

### FP8 Challenge

FP8 training introduces three complications for CUDA graphs:

1. **Global FP8 buffers**: TE maintains global scaling state (amax history, scale factors)
   that must have static memory addresses for graph capture. The buffers are originally
   dynamically constructed, which is incompatible with CUDA graphs.

2. **Dynamic scaling state**: `reduce_and_update_fp8_tensors()` performs all-reduce of amax
   across GPUs after each iteration. If captured per-layer, each replay uses outdated
   scaling factors.

3. **Weight quantization caching**: FP8 weight quantization and transposes are expensive.
   Without caching, each graph replay redundantly re-quantizes weights.

### Pipeline Parallelism Challenge

PP introduces interleaved microbatch execution. Graphs sharing memory pools must be
replayed in capture order — out-of-order replay corrupts intermediate tensors.

---

## TE `make_graphed_callables`

Extends PyTorch's `torch.cuda.make_graphed_callables` with FP8 handling. Works with
**any PyTorch model** (not limited to Megatron layers).

### Usage

```python
import transformer_engine.pytorch as te
from transformer_engine.pytorch.graph import make_graphed_callables
from transformer_engine.pytorch.fp8 import fp8_autocast

# Graph layers with FP8 support
graphed_layers = make_graphed_callables(
    tuple(layers),
    sample_args=sample_args,
    fp8_enabled=True,
    fp8_recipe=fp8_recipe,
    fp8_weight_caching=True,      # Cache FP8 weights across microbatches
    _order=layer_order,           # Pipeline schedule (None for no PP)
)

# Training loop — fp8_autocast required during replay
for batch in dataloader:
    optimizer.zero_grad()
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        for layer in graphed_layers:
            x = layer(x, is_first_microbatch=(mb_idx == 0))
    optimizer.step()
```

### Key Points

- **Capture timing**: AOT (ahead-of-time) — graphs captured before training starts
- **FP8 handling**: Uses `fp8_autocast(..., _graph=True)` internally to defer scaling
  updates. User must wrap replay with `fp8_autocast(enabled=True)`.
- **PP support**: Provide `_order` parameter with pipeline schedule (1-indexed chunk IDs,
  positive=forward, negative=backward). Replay order must match capture order.
- **Weight caching**: Set `fp8_weight_caching=True` and pass `is_first_microbatch` to
  control when weights are re-quantized.

### Limitations

- Manual PP scheduling required (user provides `_order`)
- Must provide correct sample inputs for all layers x microbatches
- RNG state registration required for custom generators

---

## CudaGraphManager (Per-Layer)

Automatic per-layer CUDA graph management for **Megatron-LM only**. Works with
`TransformerLayer` and `MambaLayer`.

### Usage — CLI

```bash
python pretrain_gpt.py \
    --enable-cuda-graph \
    --cuda-graph-num-warmup-steps 3
```

### Usage — Python Config

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    num_layers=24,
    hidden_size=1024,
    enable_cuda_graph=True,
    cuda_graph_num_warmup_steps=3,
)
```

### How It Works

Two-phase JIT (Just-in-Time) approach:

1. **Recording phase** (warmup iterations): Runs in eager mode, records execution order
   of each layer across all microbatches and forward/backward passes.

2. **Capture phase** (after warmup): Creates two graphs per layer (forward + backward).
   Captures in recorded execution order for correct memory pool sequencing.

### Memory Pool Strategy

| Setting | Pool | Graph Reuse | Graph Count | Best For |
|---------|------|-------------|-------------|----------|
| Default (`cuda_graph_use_single_mempool=False`) | Separate per microbatch | Yes | layers x 2 | PP > 1 (default) |
| `cuda_graph_use_single_mempool=True` | Single shared | No | layers x microbatches x 2 | Reducing fragmentation |
| PP=1 (automatic) | Single with reuse | Yes | layers x 2 | No pipeline parallelism |

### Advanced Options

**Buffer sharing** (`cuda_graph_share_io_buffers=True`):
- Reuses previous layer's output as next layer's input buffer
- Significantly reduces memory consumption
- Requires no operations between transformer layers

**External graph mode** (`external_cuda_graph=True`):
- Disables automatic CudaGraphManager creation
- User manages graphs manually via `model.cuda_graphs` list

---

## FullCudaGraphWrapper (Full-Iteration)

Captures all forward + backward passes across all microbatches as a **single graph**.
Maximum overhead reduction.

### Usage — CLI

```bash
python pretrain_gpt.py \
    --enable-cuda-graph \
    --cuda-graph-scope full_iteration \
    --cuda-graph-warmup-steps 1 \
    --te-rng-tracker \
    --no-check-for-nan-in-loss-and-grad
```

### Usage — Python (Custom Training Loop)

```python
from megatron.core.full_cuda_graph import FullCudaGraphWrapper

def forward_backward_func(data_iterator, model, num_microbatches,
                          seq_length, forward_only):
    data = next(data_iterator[0])
    y_pred = model(data['input'])
    loss = loss_fn(y_pred, data['target'])
    if not forward_only:
        loss.backward()
    return loss

# Wrap the function
forward_backward_func = FullCudaGraphWrapper(
    forward_backward_func, cuda_graph_warmup_steps=1
)
```

### Required Flags

| Flag | Why Required |
|------|-------------|
| `--te-rng-tracker` | Standard RNG uses CPU scalars; TE RNG uses device tensors compatible with graphs |
| `--no-check-for-nan-in-loss-and-grad` | NaN checking requires `.item()` sync, forbidden during capture |

### How It Works

Three-phase JIT approach:

1. **Warmup** (N iterations): Eager execution while `StaticBufferLoader` pre-allocates
   static buffers for all microbatch inputs.

2. **Capture** (iteration N+1): Reads all microbatch data, copies to static buffers,
   registers RNG states, captures entire `forward_backward_func` as single graph using
   `thread_local` capture mode. Separate graphs for training and validation.

3. **Replay** (all subsequent): Copies new data to static buffers, replays graph.
   Optimizer step, gradient clipping, LR scheduler remain in eager mode.

### Optimizer Inside vs Outside Graph

**Outside (default)**: Optimizer step in eager mode after graph replay. Simpler, allows
optimizer changes without recapture.

**Inside**: Include optimizer in `forward_backward_func` for maximum performance. Pass
`(model, optimizer)` tuple as the model argument.

---

## Comparison

| Aspect | TE `make_graphed_callables` | CudaGraphManager | FullCudaGraphWrapper |
|--------|---------------------------|------------------|---------------------|
| Scope | Per-callable (manual) | Per-layer (automatic) | Full iteration |
| Capture timing | AOT (before training) | JIT (during training) | JIT (during training) |
| Graph count | User-defined | Many (per layer) | 2 (train + val) |
| Overhead reduction | Variable | Moderate | Maximum |
| FP8 handling | Semi-automatic | Automatic | Automatic |
| PP support | Manual (`_order`) | Automatic | Automatic |
| Model compatibility | Any PyTorch model | Megatron layers only | Megatron training loop |
| Setup effort | High | Low (config flags) | Low (config flags) |
