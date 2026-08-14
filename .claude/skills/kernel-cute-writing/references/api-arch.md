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

# API Reference: cute.arch Module

## Thread & Block Indexing

```python
tidx, tidy, tidz = cute.arch.thread_idx()   # Thread coords within CTA
bidx, bidy, bidz = cute.arch.block_idx()     # CTA coords within grid
bdim_x, bdim_y, bdim_z = cute.arch.block_dim()  # Threads per CTA dim

lane = cute.arch.lane_idx()                  # Lane within warp (0-31)
warp = cute.arch.warp_idx()                  # Warp within CTA
```

## Grid & Cluster (Hopper+)

```python
gdim_x, gdim_y, gdim_z = cute.arch.grid_dim()       # CTAs per grid dim
cidx, cidy, cidz = cute.arch.cluster_idx()            # Cluster in grid
cdim_x, cdim_y, cdim_z = cute.arch.cluster_dim()     # Clusters per dim
cute.arch.cluster_size()                               # CTAs in cluster
cute.arch.block_in_cluster_idx()                       # CTA in cluster (per dim)
cute.arch.block_idx_in_cluster()                       # Linearized CTA in cluster
```

## Synchronization

### Thread Block

```python
cute.arch.sync_threads()          # __syncthreads() — all threads in CTA
```

### Warp

```python
cute.arch.sync_warp(mask=0xFFFFFFFF)  # Warp-level sync with mask
```

### Cluster (Hopper+)

```python
cute.arch.cluster_arrive()        # Cluster-wide arrive
cute.arch.cluster_wait()          # Cluster-wide wait
```

### Memory Fences

```python
cute.arch.fence_acq_rel_cta()     # CTA scope acquire-release
cute.arch.fence_acq_rel_cluster() # Cluster scope
cute.arch.fence_acq_rel_gpu()     # GPU scope
cute.arch.fence_acq_rel_sys()     # System scope
```

## MBarrier (Hopper+)

Hardware barrier for asynchronous operations:

```python
cute.arch.mbarrier_init(mbar_ptr, count)     # Initialize with arrival count
cute.arch.mbarrier_arrive(mbar_ptr)          # Signal arrival
cute.arch.mbarrier_wait(mbar_ptr, phase)     # Block until phase
cute.arch.mbarrier_try_wait(mbar_ptr, phase) # Non-blocking wait (returns bool)
```

## Vote & Reduction

### Warp Vote

```python
ballot = cute.arch.vote_ballot_sync(mask, predicate)  # Predicate → bitmask
any_true = cute.arch.vote_any_sync(mask, predicate)   # Any thread true
all_true = cute.arch.vote_all_sync(mask, predicate)   # All threads true
```

### Warp Reduction

**High-level** (preferred — wraps butterfly shuffle internally):

```python
max_val = cute.arch.warp_reduction_max(local_max)  # Max across warp
sum_val = cute.arch.warp_reduction_sum(local_sum)   # Sum across warp
```

**Low-level** (hardware `redux` instruction — limited arch support):

```python
result = cute.arch.warp_redux_sync(value, op, mask)
```

Supported operations: `add`, `and_`, `max`, `min`, `or_`, `xor`, `fmin`, `fmax`

**Caveat:** `warp_redux_sync` with `fmax`/`fmin` is NOT supported on SM90
for float32. Use `warp_reduction_max` or manual `shuffle_sync_bfly` instead.

**Scalar max** (for per-element comparisons, not warp-wide):

```python
result = cute.arch.fmax(a, b)  # Returns max of two Float32 values
```

## Atomic Operations

All atomics operate on global or shared memory:

```python
cute.arch.atomic_add(ptr, value)    # Atomic addition
cute.arch.atomic_max(ptr, value)    # Atomic maximum
cute.arch.atomic_min(ptr, value)    # Atomic minimum
cute.arch.atomic_exch(ptr, value)   # Atomic exchange
cute.arch.atomic_and(ptr, value)    # Atomic bitwise AND
cute.arch.atomic_or(ptr, value)     # Atomic bitwise OR
cute.arch.atomic_xor(ptr, value)    # Atomic bitwise XOR
cute.arch.atomic_cas(ptr, compare, value)  # Compare-and-swap
```

## Memory Load/Store with Cache Hints

```python
val = cute.arch.load(ptr, cache_mode)     # Load with eviction policy
cute.arch.store(ptr, val, cache_mode)     # Store with coherence hints
```

## Shared Memory

```python
# Static allocation
smem = cute.arch.alloc_smem(dtype, layout)

# Dynamic allocation (from kernel launch smem parameter)
smem_ptr = cute.arch.get_dyn_smem(dtype, offset=0)
```

## Tensor Memory (Blackwell)

```python
tmem_ptr = cute.arch.alloc_tmem(num_columns)
cute.arch.dealloc_tmem(tmem_ptr)
```

## Async Copy

```python
cute.arch.cp_async_commit_group()      # Commit outstanding cp.async
cute.arch.cp_async_wait_group(n)       # Wait until ≤ n groups pending
```

## Thread Election

```python
is_elected = cute.arch.elect_one()     # One thread per warp returns True
```

## Type Conversion

```python
cute.arch.cvt_i8_bf16_intrinsic(val)   # Fast int8 → bfloat16
```

## Miscellaneous

```python
cute.arch.popc(val)                    # Population count (count set bits)
```
