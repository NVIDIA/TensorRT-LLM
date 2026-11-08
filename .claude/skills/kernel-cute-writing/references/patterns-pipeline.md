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

# Pipeline & Synchronization Patterns

## Producer-Consumer Model

CUTLASS pipelines manage concurrent data movement and computation through
producer-consumer synchronization using circular buffers.

### State Machine

```
Producer: wait_empty → write_data → signal_full
Consumer: wait_full  → read_data  → signal_empty
```

Two barriers track each buffer stage:
- **Empty barrier**: Producer waits for empty, signals full after write
- **Full barrier**: Consumer waits for full, signals empty after read

## Pipeline Classes

### PipelineAsync (Base)

Generic asynchronous pipeline with configurable producers and consumers.

```python
from cutlass.cute.pipeline import PipelineAsync, CooperativeGroup, Agent

pipeline = PipelineAsync.create(
    num_stages=5,
    producer_group=CooperativeGroup(Agent.Thread, size=32),
    consumer_group=CooperativeGroup(Agent.Thread, size=32),
    barrier_storage=smem_ptr,
)

producer, consumer = pipeline.make_participants()

# Producer loop
for i in range(iterations):
    handle = producer.acquire_and_advance()
    # Write data to buffer[handle.stage]
    handle.commit()

# Consumer loop
for i in range(iterations):
    handle = consumer.wait_and_advance()
    # Read data from buffer[handle.stage]
    handle.release()
```

### PipelineTmaAsync (Hopper)

TMA-based producer with async thread consumer. TMA hardware handles data
movement — producer commit is a no-op since TMA updates transaction counts.

```python
from cutlass.cute.pipeline import PipelineTmaAsync

pipeline = PipelineTmaAsync.create(
    num_stages=num_stages,
    producer_group=producer_cg,
    consumer_group=consumer_cg,
    barrier_storage=smem_barrier,
)

# Producer: TMA handles data movement
state = pipeline.make_producer_start_state()
pipeline.producer_acquire(state, try_token)
# Issue TMA copy
pipeline.producer_commit(state)  # No-op for TMA

# Consumer
pipeline.consumer_wait(state)
# Use data
pipeline.consumer_release(state)
```

### PipelineTmaUmma (Blackwell)

TMA producer with UMMA (tensor core) consumer. Supports 2-CTA mode and
cluster multicast.

```python
from cutlass.cute.pipeline import PipelineTmaUmma

# Supports leader CTA selection for multi-CTA kernels
pipeline = PipelineTmaUmma.create(
    num_stages=num_stages,
    producer_group=producer_cg,
    consumer_group=consumer_cg,
    barrier_storage=smem_barrier,
)
```

### Other Pipeline Variants

| Class | Producer | Consumer | Architecture |
|-------|----------|----------|-------------|
| `PipelineAsync` | AsyncThread | AsyncThread | All |
| `PipelineCpAsync` | CpAsync | AsyncThread | SM80+ |
| `PipelineTmaAsync` | TMA | AsyncThread | SM90+ |
| `PipelineTmaUmma` | TMA | UMMA | SM100 |
| `PipelineAsyncUmma` | AsyncThread | UMMA | SM100 |
| `PipelineUmmaAsync` | UMMA | AsyncThread | SM100 |

## Pipeline State

Tracks position in circular buffer:

```python
from cutlass.cute.pipeline import PipelineState

state = PipelineState(stages=4, count=0, index=0, phase=0)
state.advance()   # Move to next stage (wraps around)
state.reverse()   # Go back one stage
state.clone()     # Independent copy
```

## Cooperative Groups

Define producer/consumer thread groups:

```python
from cutlass.cute.pipeline import CooperativeGroup, Agent

# Single warp producer
producer_cg = CooperativeGroup(Agent.Thread, size=32)

# Full CTA consumer
consumer_cg = CooperativeGroup(Agent.ThreadBlock, size=256)

# Cluster-level group
cluster_cg = CooperativeGroup(Agent.ThreadBlockCluster, size=512)
```

## Barrier Primitives

### sync_threads (CTA-level)

```python
cute.arch.sync_threads()  # __syncthreads() equivalent
```

### Warp Synchronization

```python
cute.arch.sync_warp(mask=0xFFFFFFFF)  # All lanes
```

### MBarrier (Hopper+)

Hardware barrier for async operations:

```python
from cutlass.cute.pipeline import MbarrierArray

barriers = MbarrierArray(
    barrier_storage=smem_ptr,
    num_stages=num_stages,
    agent=Agent.Thread,
    tx_count=expected_bytes,
)

barriers.mbarrier_init()           # Initialize (warp 0 only)
barriers.arrive(stage_idx)         # Signal arrival
barriers.wait(stage_idx, phase)    # Block until phase
barriers.try_wait(stage_idx, phase)  # Non-blocking check
```

### Named Barriers

16 hardware barriers (IDs 0-15):

```python
from cutlass.cute.pipeline import NamedBarrier

barrier = NamedBarrier(barrier_id=0, num_threads=128)
barrier.arrive()
barrier.wait()
barrier.arrive_and_wait()
```

## Cluster-Level Synchronization (Hopper+)

```python
cute.arch.cluster_arrive()
cute.arch.cluster_wait()
```

## Memory Fences

```python
cute.arch.fence_acq_rel_cta()      # CTA scope
cute.arch.fence_acq_rel_cluster()  # Cluster scope
cute.arch.fence_acq_rel_gpu()      # GPU scope
cute.arch.fence_acq_rel_sys()      # System scope
```

## Pipeline Order

Enforces execution order across groups (e.g., mainloop before epilogue):

```python
from cutlass.cute.pipeline import PipelineOrder

order = PipelineOrder.create(
    barrier_storage=smem_ptr,
    depth=2,       # Stages per group
    length=3,      # Number of groups
    group_id=0,
    producer_group=cg,
)
order.wait()     # Wait for previous group
order.arrive()   # Signal completion
```

## Warp Specialization Pattern

Hopper+ kernels split warps into producer and consumer roles:

```python
# Warp specialization: first N warps produce, rest consume
warp_id = cute.arch.warp_idx()
is_producer = warp_id < num_producer_warps

if cutlass.const_expr(is_producer):
    # Producer: issue TMA loads
    for k in range(K_TILES):
        pipeline.producer_acquire(state)
        cute.copy(tma_atom, gmem_tile[k], smem_buffer[state.index])
        pipeline.producer_commit(state)
        state.advance()
else:
    # Consumer: perform MMA
    for k in range(K_TILES):
        pipeline.consumer_wait(state)
        cute.gemm(tiled_mma, smem_A[state.index], smem_B[state.index], acc)
        pipeline.consumer_release(state)
        state.advance()
```

## Pipeline Operations Enum

```python
from cutlass.cute.pipeline import PipelineOp

PipelineOp.AsyncThread    # Async copy operations
PipelineOp.TCGen05Mma     # Blackwell MMA
PipelineOp.TmaLoad        # TMA loads
PipelineOp.TmaStore       # TMA stores
PipelineOp.ClcLoad        # Cluster Launch Control loads
PipelineOp.Composite      # Combined operations
```
