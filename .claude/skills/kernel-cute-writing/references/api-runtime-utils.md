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

# API Reference: Runtime & Utilities

## cute.runtime Module

### Tensor Conversion

```python
from cutlass.cute.runtime import from_dlpack, make_ptr

# DLPack conversion (primary method)
tensor = from_dlpack(
    torch_tensor,
    assumed_align=16,         # Byte alignment (16 or 32)
    use_32bit_stride=False,   # Smaller strides for small tensors
    enable_tvm_ffi=False,     # TVM FFI compatibility
)

# Raw pointer creation
ptr = make_ptr(
    cutlass.Float16,          # Data type
    address,                  # Integer or ctypes address
    cute.AddressSpace.gmem,   # Memory space
    assumed_align=16,
)

# Null pointer (for compilation)
null = cute.runtime.nullptr(cutlass.Float16, cute.AddressSpace.gmem)
```

### Dynamic Layout Marking

```python
tensor.mark_layout_dynamic(leading_dim=1)
tensor.mark_compact_shape_dynamic(
    mode=0,              # Dimension to make dynamic
    divisibility=2,      # Alignment constraint
    stride_order=(1,0),  # Custom ordering
)
```

### Fake Tensors (For Compilation)

```python
# Compact fake tensor with symbolic shapes
n = cute.sym_int()
fake = cute.runtime.make_fake_compact_tensor(
    cutlass.Float32,
    shape=(n,),
    stride_order=None,
)

# Explicit fake tensor
fake = cute.runtime.make_fake_tensor(
    cutlass.Float16,
    shape=(M, N),
    stride=(N, 1),
)

# Fake stream
stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
```

### Module Loading (AOT)

```python
module = cute.runtime.load_module(
    "./artifacts/kernel.o",
    enable_tvm_ffi=True,
)
module.kernel_name(tensor_a, stream)

# Find required runtime libraries
libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
```

## cutlass.utils Module

### Shared Memory Management

```python
from cutlass.utils import SmemAllocator

# Get capacity
capacity = SmemAllocator.get_smem_capacity_in_bytes(compute_capability)

# Allocate tensors
allocator = SmemAllocator()
smem_A = allocator.allocate_tensor(
    cutlass.Float16, layout, swizzle=None)
smem_B = allocator.allocate_tensor(
    cutlass.Float16, layout, swizzle=None)
raw_bytes = allocator.allocate(num_bytes, alignment=128)
array = allocator.allocate_array(cutlass.Float16, count)
total = allocator.total_bytes()
```

### Tensor Memory Management (Blackwell)

```python
from cutlass.utils import TmemAllocator

tmem = TmemAllocator(...)
tmem.wait_for_alloc()          # Sync allocator warp
ptr = tmem.retrieve_ptr()      # Get allocated pointer
TmemAllocator.check_valid_num_columns(n)
```

### Tile Scheduling

```python
from cutlass.utils import (
    StaticPersistentTileScheduler,
    ClcDynamicPersistentTileScheduler,
    PersistentTileSchedulerParams,
)

# Static scheduler
params = PersistentTileSchedulerParams(cluster_shape, tile_shape)
scheduler = StaticPersistentTileScheduler.create(params, block_idx)

# Dynamic scheduler (Cluster Launch Control)
params = ClcDynamicPersistentTileSchedulerParams(...)
scheduler = ClcDynamicPersistentTileScheduler.create(params)
```

### Grouped GEMM

```python
from cutlass.utils import GroupedGemmTileSchedulerHelper, GroupSearchResult

result = GroupSearchResult(group_idx, tile_coord, problem_shape)
helper = GroupedGemmTileSchedulerHelper()
helper.delinearize_z(linear_idx)
helper.search_cluster_tile_count_k(...)
```

### MMA Helpers

```python
from cutlass.utils import (
    make_trivial_tiled_mma,
    make_blockscaled_trivial_tiled_mma,
    make_smem_layout_a,
    make_smem_layout_b,
    make_smem_layout_epi,
    compute_epilogue_tile_shape,
    compute_smem_layout,
)

tiled_mma = make_trivial_tiled_mma(
    mma_atom, a_dtype, b_dtype, acc_dtype,
    cta_group=CtaGroup.ONE,
)
```

### Layout Utilities

```python
from cutlass.utils import LayoutEnum

LayoutEnum.ROW_MAJOR
LayoutEnum.COL_MAJOR

# Query functions
LayoutEnum.is_k_major_a(layout)
LayoutEnum.is_m_major_a(layout)
LayoutEnum.is_n_major_b(layout)
LayoutEnum.is_k_major_b(layout)
```

### Scale & Transform

```python
from cutlass.utils import (
    scale_tma_partition,     # Partition scale tensors for TMA
    transform_partition,     # Configure transform pipelines
    scale_partition,         # Prepare scale tensors
    get_smem_layout_scale,   # Scale tensor SMEM layout
    get_gmem_layout_scale,   # Scale tensor GMEM layout
)
```

### TensorMap Management

```python
from cutlass.utils import TensorMapManager, TensorMapUpdateMode

manager = TensorMapManager(...)
# Update modes
TensorMapUpdateMode.GMEM   # Update in global memory
TensorMapUpdateMode.SMEM   # Update in shared memory

# Synchronization
cutlass.utils.fence_tensormap_initialization()
cutlass.utils.fence_tensormap_update()
```

### Hardware Info

```python
from cutlass.utils import HardwareInfo

hw = HardwareInfo()
hw.multiprocessor_count     # Number of SMs
hw.l2_cache_size            # L2 cache in bytes
hw.active_clusters          # Currently active clusters
```

### Data Type Utilities

```python
from cutlass.utils import (
    is_fp8_dtype,            # Check Float8 support
    create_cute_tensor_for_fp8,  # FP8 tensor creation
    get_divisibility,        # Power-of-2 alignment factor
    is_valid_scale_granularity,  # Validate quantization params
)
```

### Visualization

```python
from cutlass.utils import print_latex, print_latex_tv

print_latex(layout)                    # TiKZ layout diagram
print_latex_tv(layout, tile_shape)     # Thread-value diagram
```
