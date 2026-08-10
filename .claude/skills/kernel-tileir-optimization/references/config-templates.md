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

# TileIR Autotune Config Templates

Complete configuration templates for each kernel type. All functions accept
`pre_hook=None` to support TMA descriptor wiring.

## Dot-Related Kernels

```python
def get_dot_kernel_configs(pre_hook=None):
    """Configs for GEMM, BMM, FMHA, convolution."""
    return [
        # Single CTA configurations
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "occupancy": 1},
            num_stages=4, num_ctas=1, pre_hook=pre_hook
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "occupancy": 2},
            num_stages=4, num_ctas=1, pre_hook=pre_hook
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "occupancy": 2},
            num_stages=4, num_ctas=1, pre_hook=pre_hook
        ),

        # Extended num_stages for deeper pipelining
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "occupancy": 2},
            num_stages=6, num_ctas=1, pre_hook=pre_hook
        ),

        # 2CTA configurations (critical for Blackwell)
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "occupancy": 2},
            num_stages=4, num_ctas=2, pre_hook=pre_hook
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "occupancy": 2},
            num_stages=4, num_ctas=2, pre_hook=pre_hook
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "occupancy": 2},
            num_stages=6, num_ctas=2, pre_hook=pre_hook
        ),

        # Higher occupancy variants
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "occupancy": 4},
            num_stages=4, num_ctas=1, pre_hook=pre_hook
        ),
    ]
```

**Key params**: TMA descriptors, num_ctas (1, 2), occupancy (1, 2, 4), num_stages (4, 6).
Typical config count: 8-12.

## Norm-Like Kernels

```python
def get_norm_kernel_configs(pre_hook=None):
    """Configs for LayerNorm, RMSNorm, Softmax, GroupNorm."""
    return [
        triton.Config({"occupancy": 1}, num_warps=4, num_stages=3, pre_hook=pre_hook),
        triton.Config({"occupancy": 1}, num_warps=8, num_stages=3, pre_hook=pre_hook),
        triton.Config({"occupancy": 2}, num_warps=4, num_stages=3, pre_hook=pre_hook),
        triton.Config({"occupancy": 2}, num_warps=8, num_stages=3, pre_hook=pre_hook),
        triton.Config({"occupancy": 4}, num_warps=4, num_stages=3, pre_hook=pre_hook),
        triton.Config({"occupancy": 4}, num_warps=8, num_stages=3, pre_hook=pre_hook),
    ]
```

**Key params**: occupancy (1, 2, 4), num_warps (4, 8). No TMA needed.
Typical config count: 6-10.

## Element-Wise Kernels

```python
def get_elementwise_kernel_configs(pre_hook=None):
    """Configs for ReLU, GELU, Add, Mul, Exp, dropout."""
    return [
        # Standard configurations
        triton.Config({"occupancy": 1}, num_warps=4, num_stages=3, pre_hook=pre_hook),
        triton.Config({"occupancy": 2}, num_warps=4, num_stages=3, pre_hook=pre_hook),
        triton.Config({"occupancy": 2}, num_warps=8, num_stages=3, pre_hook=pre_hook),

        # num_stages variants
        triton.Config({"occupancy": 1}, num_warps=4, num_stages=2, pre_hook=pre_hook),
        triton.Config({"occupancy": 2}, num_warps=4, num_stages=4, pre_hook=pre_hook),
        triton.Config({"occupancy": 4}, num_warps=4, num_stages=4, pre_hook=pre_hook),

        # Extreme configs for small inputs
        triton.Config({"occupancy": 4}, num_warps=2, num_stages=2, pre_hook=pre_hook),
        triton.Config({"occupancy": 16}, num_warps=2, num_stages=3, pre_hook=pre_hook),
    ]
```

**Key params**: occupancy (1, 2, 4, 16), num_stages (2, 3, 4). No TMA needed.
Typical config count: 10-15.

## Reduction Kernels

Use the same configs as norm-like kernels:

```python
def get_reduction_kernel_configs(pre_hook=None):
    """Configs for sum, mean, max, argmax."""
    return get_norm_kernel_configs(pre_hook)
```

## Architecture Gating

Only add TileIR-specific configs on sm_100+:

```python
import torch

def get_configs_with_gating(pre_hook=None):
    configs = get_baseline_configs()  # PTX-compatible configs
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
        configs.extend(get_tileir_specific_configs(pre_hook))
    return configs
```

## Architecture Detection Helpers

```python
import torch

def is_blackwell_or_later():
    """Check if running on Blackwell (sm_100+)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10

def supports_tma():
    """Check if TMA is available (Hopper+)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9

def supports_2cta():
    """Check if 2CTA mode is available (Blackwell+)."""
    return is_blackwell_or_later()
```

## Config Cheat Sheet

| Kernel Type | Key Params | Configs | Priority |
|-------------|------------|---------|----------|
| Dot-Related | TMA, num_ctas, occupancy, num_stages | 8-12 | HIGH |
| Norm-Like | occupancy, num_warps | 6-10 | HIGH |
| Element-Wise | occupancy, num_stages | 10-15 | MEDIUM |
| Reduction | occupancy, num_warps | 6-10 | HIGH |

## Environment Variables

```bash
# TileIR backend control
export ENABLE_TILE=0  # PTX mode (compatibility)
export ENABLE_TILE=1  # TileIR mode (optimized)

# Numerical precision options
export TILEIR_ENABLE_APPROX=1  # Enable approximate math (exp/log)
export TILEIR_ENABLE_FTZ=1     # Enable flush-to-zero

# Optimization level (default: 3)
export TILEIR_OPT_LEVEL=3
```

## nvtriton Installation

```bash
pip uninstall -y pytorch-triton triton  # Remove existing triton packages
./scripts/install-nvtriton.sh           # Install nvtriton as 'triton'

# Verify
python -c "import triton; print(f'triton {triton.__version__}')"
```
