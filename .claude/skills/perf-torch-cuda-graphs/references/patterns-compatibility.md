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

# Writing CUDA Graph-Compatible PyTorch Code

## Three Principles

Every line inside a `torch.cuda.graph()` capture must satisfy all three:

1. **GPU-Only** — Only GPU ops are captured. CPU code runs at capture time but is
   eliminated during replay.
2. **Sync-Free** — No CPU-GPU synchronization. CPU continuously queues work without
   waiting for GPU results.
3. **Static** — Operation sequence, tensor shapes, memory addresses, and kernel
   parameters are fixed across replays.

---

## Principle 1: GPU-Only

### Incompatible Patterns

| Pattern | Why It Breaks |
|---------|---------------|
| `torch.load("file.pt")` | File I/O is CPU-only |
| `print(tensor)` | Forces GPU-to-CPU transfer |
| `random.randint()` | CPU RNG not captured |
| `buffer.append(tensor)` | Python list mutation not captured |
| Tokenization / data preprocessing | CPU computation |
| Logging, metrics collection | CPU side effects |

### Fix

Move all CPU-side work **outside** the graphed region. Data loading, preprocessing,
and logging happen before `g.replay()` or after reading `static_output`.

---

## Principle 2: Sync-Free

See the **perf-torch-sync-free** skill for comprehensive sync detection and
elimination. The sync-free skill covers all common PyTorch sync patterns
including device transfers, tensor creation, control flow, indexing, and
dynamic output shapes.

---

## Principle 3: Static

### What Must Be Static

| Aspect | Requirement |
|--------|------------|
| Control flow | Same branch every replay |
| Memory addresses | Same tensor pointers |
| Tensor shapes | Fixed dimensions |
| Kernel parameters | Fixed grid/block/args |

### Dynamic Control Flow

```python
# Breaks — branch depends on runtime value
if x.sum() > 0:
    y = path_a(x)
else:
    y = path_b(x)

# Works — execute both, select on GPU
y = torch.where(x.sum() > 0, path_a(x), path_b(x))
```

### Dynamic Tensors (Input Management)

```python
# Breaks — reassignment changes memory address
for data in dataloader:
    input_tensor = data        # New object each iteration!
    g.replay()                 # Graph reads stale address

# Works — in-place copy preserves address
static_input = torch.zeros(batch_size, *shape, device='cuda')
for data in dataloader:
    static_input.copy_(data)   # Same address, updated values
    g.replay()
```

### Dynamic Scalars

```python
# Breaks — Python float baked into kernel at capture time
temperature = 1.0
with torch.cuda.graph(g):
    output = logits / temperature   # 1.0 captured as constant

# Works — GPU tensor, update via .fill_()
temperature = torch.tensor(1.0, device='cuda')
with torch.cuda.graph(g):
    output = logits / temperature

# Before each replay:
temperature.fill_(new_value)
g.replay()
```

### Output Tensor Reuse

```python
# Bug — all entries point to same memory, overwritten each replay
results = []
for data in dataloader:
    static_input.copy_(data)
    g.replay()
    results.append(static_output)   # Same tensor every time!

# Fix — clone immediately
results = []
for data in dataloader:
    static_input.copy_(data)
    g.replay()
    results.append(static_output.clone())
```

---

## Quick Compatibility Checklist

Before attempting capture, verify your code satisfies:

- [ ] No `.item()`, `.cpu()`, `.numpy()`, `print(tensor)` inside graphed region
- [ ] No `torch.cuda.synchronize()` or `stream.synchronize()`
- [ ] No `if tensor_value:` conditionals (use `torch.where`)
- [ ] All input tensors pre-allocated and updated via `.copy_()`
- [ ] All shapes fixed (or using padding/bucketing)
- [ ] All Python scalars that change converted to GPU tensors
- [ ] Output tensors `.clone()`d before next replay if accumulated
- [ ] `cache_enabled=False` if using `torch.amp.autocast`
- [ ] Custom RNG generators registered with `g.register_generator_state()`
- [ ] Warmup completed (3 iters standard, 11 for DDP) on side stream
- [ ] Libraries use `torch.cuda.current_stream()`, not default stream
