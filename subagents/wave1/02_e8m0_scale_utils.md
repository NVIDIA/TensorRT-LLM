<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Plan 02: E8M0 Scale Utilities

## Goal

Add small, well-tested utilities for handling E8M0 scale tensors correctly.
DeepSeek V4 uses E8M0 scales for both FP8 dense linears and packed FP4 routed
experts. Some kernels need FP32 scale values, while others need raw exponent
bytes.

This work is independent because it operates on tiny synthetic tensors.

## Owner Scope

Suggested files:

- `tensorrt_llm/_torch/auto_deploy/utils/e8m0.py`
- `tests/unittest/_torch/auto_deploy/unit/utils/test_e8m0.py`

Avoid editing quantization transforms directly unless adding import points or
small wrappers.

## Required APIs

Provide utilities equivalent to:

```python
def e8m0_to_uint8(scale: torch.Tensor) -> torch.Tensor:
    """Return raw exponent bytes without numeric conversion."""

def e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 exponent-only values to FP32 powers of two."""

def maybe_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 scales; return non-E8M0 scales as FP32."""
```

The implementation must preserve raw bytes with `view(torch.uint8)`, not
`to(torch.uint8)`.

## Implementation Notes

PyTorch exposes E8M0 locally as:

```python
torch.float8_e8m0fnu
```

Decode to FP32 by placing exponent bits into the IEEE-754 FP32 exponent field:

```python
exp_bits = scale.view(torch.uint8).to(torch.int32)
fp32_bits = exp_bits << 23
scale_fp32 = fp32_bits.view(torch.float32)
```

Handle environments where `torch.float8_e8m0fnu` is unavailable by checking
`hasattr(torch, "float8_e8m0fnu")` and failing with a clear message only when an
actual E8M0 tensor is required.

## Standalone Tests

Use tiny tensors and explicit bit patterns:

- Raw byte preservation:
  - construct bytes, view as E8M0, round trip to uint8.
- Numeric conversion guard:
  - prove `view(torch.uint8)` differs from `.to(torch.uint8)` for small scales.
- FP32 decode:
  - byte `127` decodes to `1.0`
  - byte `128` decodes to `2.0`
  - byte `126` decodes to `0.5`
- Non-E8M0 input:
  - `maybe_e8m0_to_fp32` returns FP32 values for BF16/FP32 scales.
- CPU-only execution.

## Done Criteria

- Utilities are importable without initializing CUDA.
- Tests pass on CPU.
- Utilities are documented enough for the FP8 and MXFP4 loader agents to use
  them without re-implementing scale handling.
