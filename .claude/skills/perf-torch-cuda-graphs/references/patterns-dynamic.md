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

# Handling Dynamic Patterns in CUDA Graphs

Four categories of dynamism break CUDA graphs. Each has established workarounds.

---

## 1. Dynamic Control Flow

**Problem:** Python conditionals based on GPU values force CPU-GPU sync via `.item()`.

### Patterns and Solutions

| Pattern | Solution |
|---------|----------|
| `if loss.item() > threshold:` | `torch.where(loss > threshold, a, b)` |
| `if x.sum() > 0: path_a()` | Execute both paths, select with `torch.where` |
| NaN/Inf detection | `torch.where(torch.isfinite(loss), loss, fallback)` |
| Grad clipping with norm | `torch.nn.utils.clip_grad_norm_()` (sync-free since PT 1.13) |
| Early exit in inference | Separate graphs per path, select at runtime |

### Advanced: Partial Capture with Multiple Graphs

```python
# Capture separate graphs for distinct execution paths
g_path_a = torch.cuda.CUDAGraph()
g_path_b = torch.cuda.CUDAGraph()

with torch.cuda.graph(g_path_a):
    static_out_a = path_a(static_input)

with torch.cuda.graph(g_path_b):
    static_out_b = path_b(static_input)

# Select graph at runtime (decision made outside graph)
for data in dataloader:
    static_input.copy_(data)
    if should_use_path_a(data):  # CPU decision OK — outside graph
        g_path_a.replay()
        result = static_out_a.clone()
    else:
        g_path_b.replay()
        result = static_out_b.clone()
```

---

## 2. Dynamic Tensors

**Problem:** Graph inputs must maintain stable memory addresses. Reassignment creates
new objects at different addresses; the graph reads from the old (stale) address.

### Rules

- **Internal tensors** (created inside graph) — automatically stable, no action needed
- **External tensors** (cross graph boundary from outside) — must be pre-allocated and
  updated with `.copy_()`

### Common Dynamic Tensor Sources

| Source | Fix |
|--------|-----|
| Dataloader outputs | Pre-allocate static buffers, `.copy_()` each batch |
| Type casts (`x.half()`) | Cast once before capture; keep cast tensor alive |
| Global state variables | Use `.copy_()` or `.fill_()` to update |
| Externally created tensors | Store as persistent attributes |

### Dataloader Wrapper Pattern

```python
class StaticBufferLoader:
    """Wraps dataloader to copy into pre-allocated static tensors."""

    def __init__(self, loader, device='cuda'):
        self.loader = loader
        self.device = device
        self.static_buffers = None

    def __iter__(self):
        for batch in self.loader:
            if self.static_buffers is None:
                # Allocate on first batch
                self.static_buffers = tuple(
                    torch.empty_like(t, device=self.device) for t in batch
                )
            for static, new in zip(self.static_buffers, batch):
                static.copy_(new.to(self.device, non_blocking=True))
            yield self.static_buffers
```

### Global State Tensor

```python
# Breaks — reassignment changes address
running_mean = torch.zeros(features, device='cuda')
for batch in dataloader:
    running_mean = batch.mean(dim=0)  # New tensor!
    g.replay()

# Works — in-place update
for batch in dataloader:
    running_mean.copy_(batch.mean(dim=0))
    g.replay()
```

---

## 3. Dynamic Scalars

**Problem:** Python scalars become kernel constants baked at capture time. Changing
the Python variable has no effect on replay.

### Solution: Convert to GPU Tensor

```python
# Breaks — 2.0 baked in
with torch.cuda.graph(g):
    result = torch.pow(x, 2.0)

# Works — tensor dereferences pointer at replay
exponent = torch.tensor(2.0, device='cuda')
with torch.cuda.graph(g):
    result = torch.pow(x, exponent)

# Update before replay
exponent.fill_(3.0)
g.replay()
```

### Common Dynamic Scalars

| Scalar | Where It Appears | Fix |
|--------|-----------------|-----|
| Learning rate | Optimizer step | Use capturable optimizer (APEX FusedAdam) |
| Temperature | Softmax scaling | Convert to GPU tensor |
| Dropout rate | F.dropout `p` arg | Fixed at capture; vary outside graph |
| Loss scale | AMP GradScaler | Use `GradScaler(enabled=True)` with graph support |
| RNG offset | Random operations | `register_generator_state()` handles automatically |

### Capturable Optimizer Example

```python
from apex.optimizers import FusedAdam

optimizer = FusedAdam(model.parameters(), lr=0.001, capturable=True)

with torch.cuda.graph(g):
    optimizer.step()

# LR stored as GPU tensor — update via scheduler
for iteration in range(num_iters):
    lr_tensor = optimizer.param_groups[0]['lr']
    lr_tensor.fill_(scheduler.get_lr()[0])
    g.replay()
```

---

## 4. Dynamic Shapes

**Problem:** Different tensor dimensions trigger different kernel configs (grid/block
sizes), memory allocations, and algorithm selection. Graph replays with captured
config regardless of actual input shape.

### Approach A: Padding to Fixed Size

Single graph, pad all inputs to maximum. Simple but wastes compute on short inputs.

```python
max_seq_len = 2048
static_input = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device='cuda')

with torch.cuda.graph(g):
    static_output = model(static_input)

def forward(input_ids):
    seq_len = input_ids.shape[1]
    if seq_len < max_seq_len:
        input_ids = F.pad(input_ids, (0, max_seq_len - seq_len))
    static_input.copy_(input_ids)
    g.replay()
    return static_output[:, :seq_len].clone()
```

### Approach B: Bucketing

Multiple graphs for common shape ranges. More memory, less wasted compute.

```python
class BucketedGraphModel:
    def __init__(self, model, seq_lengths=[128, 256, 512, 1024, 2048]):
        self.graphs = {}
        for sl in seq_lengths:
            inp = torch.zeros(batch_size, sl, device='cuda')
            g = torch.cuda.CUDAGraph()

            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for _ in range(3):
                    _ = model(inp)
            torch.cuda.current_stream().wait_stream(s)

            with torch.cuda.graph(g):
                out = model(inp)
            self.graphs[sl] = {'graph': g, 'input': inp, 'output': out}

    def __call__(self, input_ids):
        seq_len = input_ids.shape[1]
        bucket = min(s for s in self.graphs if s >= seq_len)
        if seq_len < bucket:
            input_ids = F.pad(input_ids, (0, bucket - seq_len))
        entry = self.graphs[bucket]
        entry['input'].copy_(input_ids)
        entry['graph'].replay()
        return entry['output'][:, :seq_len].clone()
```

### Decision Guide

| Condition | Use |
|-----------|-----|
| Shapes cluster near max (e.g., fixed batch training) | Padding |
| Wide shape spread, memory available | Bucketing |
| 95% of shapes in 3 buckets | Bucket top 3, eager fallback for rest |
| Dynamic routing (MoE) | Partial graphing — graph static layers only |

---

## 5. Partial Graphing (MoE and Complex Models)

When full-model graphing is impractical, graph only the static portions.

```python
# Graph static components, keep dynamic layers in eager mode
pre_graphed = torch.cuda.make_graphed_callables(
    model.pre_layers,
    (torch.zeros(batch_size, seq_len, hidden_dim, device='cuda'),),
)
post_graphed = torch.cuda.make_graphed_callables(
    model.post_layers,
    (torch.zeros(batch_size, seq_len, hidden_dim, device='cuda'),),
)

for batch in dataloader:
    x = pre_graphed(batch)
    x = model.moe_layer(x)       # Eager — dynamic routing
    out = post_graphed(x)
```
