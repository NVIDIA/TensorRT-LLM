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

# PyTorch CUDA Graph APIs

## API Selection Guide

| API | Effort | Control | Best For |
|-----|--------|---------|----------|
| `torch.compile(mode="reduce-overhead")` | Lowest | Automatic | Quick wins, unknown graph boundaries |
| `torch.cuda.make_graphed_callables()` | Medium | Per-callable | Training loops with autograd |
| `torch.cuda.graph()` | Highest | Full manual | Maximum control, custom pipelines |

---

## torch.compile with reduce-overhead

Automatic CUDA graph capture via CUDAGraph Trees. No manual capture code needed.

```python
@torch.compile(mode="reduce-overhead")
def train_step(model, x, target, criterion):
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    return loss

# Usage — drop-in replacement
for data, target in dataloader:
    optimizer.zero_grad()
    loss = train_step(model, data, target, criterion)
    optimizer.step()
```

**How it works:**
- Detects graph-compatible regions automatically
- Organizes multiple graphs in a tree structure for different execution paths
- Partitions incompatible ops (CPU ops, control flow) into non-graphed segments
- Shares a single memory pool across all graphs
- Triggers new captures on dynamic shapes

**Trade-offs:**
- Zero manual setup effort
- Often creates fragmented small graphs instead of few large ones
- Limited performance control vs manual capture
- Graph breaks from `.item()`, `print()`, data-dependent control flow reduce benefit

---

## torch.cuda.make_graphed_callables()

High-level API that graphs models/functions with automatic autograd support.

```python
model = MyModel().cuda()
sample_input = torch.randn(batch_size, *input_shape, device='cuda')

# Graph the model — handles warmup automatically
graphed_model = torch.cuda.make_graphed_callables(
    model,
    (sample_input,),
    num_warmup_iters=3,  # Use 11 for DDP
)

# Drop-in replacement in training loop
for data, target in dataloader:
    optimizer.zero_grad()
    output = graphed_model(data)   # Forward is graphed
    loss = criterion(output, target)
    loss.backward()                # Backward is also graphed
    optimizer.step()
```

**Key features:**
- Automatic warmup handling
- Creates separate graphs for forward and backward passes
- Returns callable replacing original module
- Manages memory pool sharing across multiple callables
- Autograd-compatible

**Limitations:**
- No double backward support
- No module hooks fire during replay (hooks on submodules are skipped)
- Cannot modify module structure after graphing
- DDP allreduce may serialize after backward, losing overlap (use `torch.cuda.graph()` for full-iteration capture if overlap matters)

### Graphing Multiple Callables

```python
# Graph multiple modules sharing a memory pool
graphed_m1, graphed_m2 = torch.cuda.make_graphed_callables(
    (module1, module2),
    ((sample1,), (sample2,)),
)
```

---

## torch.cuda.graph() Context Manager

Manual stream capture with full control. Requires explicit warmup and static tensor management.

### Basic Inference Pattern

```python
g = torch.cuda.CUDAGraph()
static_input = torch.randn(batch_size, *input_shape, device='cuda')

# Warmup on side stream (mandatory, minimum 3 iterations)
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):
        _ = model(static_input)
torch.cuda.current_stream().wait_stream(s)

# Capture
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay — update inputs in-place, never reassign
for data in dataloader:
    static_input.copy_(data)
    g.replay()
    result = static_output.clone()  # Clone before next replay overwrites
```

### Training Pattern with AMP

```python
g = torch.cuda.CUDAGraph()
static_input = torch.randn(batch_size, *input_shape, device='cuda')
static_target = torch.randint(0, num_classes, (batch_size,), device='cuda')

s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):
        with torch.amp.autocast("cuda", cache_enabled=False):
            out = model(static_input)
            loss = criterion(out, static_target)
        loss.backward()
torch.cuda.current_stream().wait_stream(s)

# Capture forward + backward together
with torch.cuda.graph(g):
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", cache_enabled=False):
        static_output = model(static_input)
        static_loss = criterion(static_output, static_target)
    static_loss.backward()

# Training loop
for data, target in dataloader:
    static_input.copy_(data)
    static_target.copy_(target)
    g.replay()
    optimizer.step()
```

**Critical: `cache_enabled=False`** — Autocast caches FP32-to-FP16 casts globally.
When the context exits, cache clears and may free tensors, leaving graph with stale
addresses.

### DDP Setup

```python
import os
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '0'  # Before init_process_group

torch.distributed.init_process_group(...)

s = torch.cuda.Stream()
with torch.cuda.stream(s):
    model = DistributedDataParallel(model)
    # 11 warmup iterations required for DDP (internal setup at ~iter 10)
    for _ in range(11):
        out = model(static_input)
        out.sum().backward()
torch.cuda.current_stream().wait_stream(s)
```

---

## CUDAGraph Class (Low-Level)

Direct wrapper around `cudaGraph_t` / `cudaGraphExec_t`. Used internally by
higher-level APIs but available when explicit control is needed.

| Method | Purpose |
|--------|---------|
| `capture_begin(pool=None)` | Start stream capture |
| `capture_end()` | End stream capture |
| `replay()` | Execute captured graph |
| `reset()` | Release graph resources |
| `pool()` | Return memory pool handle (for sharing) |
| `register_generator_state(gen)` | Register RNG generator for proper replay |

### Memory Pool Sharing

```python
g1 = torch.cuda.CUDAGraph()
# ... capture g1 ...

g2 = torch.cuda.CUDAGraph()
# Share g1's pool so sequential graphs reuse memory
with torch.cuda.graph(g2, pool=g1.pool()):
    static_output2 = model2(static_input2)
```

### RNG Generator Registration

```python
custom_gen = torch.Generator(device='cuda')
g = torch.cuda.CUDAGraph()
g.register_generator_state(custom_gen)  # Before capture

with torch.cuda.graph(g):
    output = F.dropout(input, p=0.5, training=True, generator=custom_gen)

# Each replay produces different random values (offset auto-advances)
```

**Note:** Default generator is automatically registered. Only custom generators
need explicit registration.

---

## Warmup Requirements

| Scenario | Minimum Warmup Iterations |
|----------|--------------------------|
| Standard | 3 |
| DistributedDataParallel | 11 |
| torch.compile functions | Run once before capture |

Warmup must occur on the **same side stream** used for capture. It triggers:
- All lazy memory allocations
- JIT compilation (cuBLAS, cuDNN autotuning)
- AccumulateGrad node initialization on correct stream
- Library internal state setup
