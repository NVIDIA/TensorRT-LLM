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

# Plan 04: Packed MXFP4 Expert Loader

## Goal

Load, validate, and stack DeepSeek V4 routed expert weights stored as packed
FP4 in `I8` tensors with E8M0 scales. This feature should produce backend-ready
expert tensors without needing the full MoE op to exist.

This work is independent because it can use synthetic packed expert tensors and
reference unpack/dequant helpers.

## Owner Scope

Suggested files:

- `tensorrt_llm/_torch/auto_deploy/transform/library/deepseek_v4_mxfp4.py`
- `tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/mxfp4_moe.py` only for
  small interface extensions
- `tensorrt_llm/_torch/auto_deploy/transform/library/sharding_ir.py` only for
  MXFP4 EP shape handling, if needed
- `tests/unittest/_torch/auto_deploy/unit/fused_moe/test_deepseek_v4_mxfp4_loader.py`

Do not implement the router or full `torch_deepseek_v4_moe` here.

## Checkpoint Layout

Representative routed expert tensors:

```text
w1.weight  I8 [2048, 2048]  -> logical FP4 [2048, 4096]
w3.weight  I8 [2048, 2048]  -> logical FP4 [2048, 4096]
w2.weight  I8 [4096, 1024]  -> logical FP4 [4096, 2048]

w1.scale   F8_E8M0 [2048, 128]
w3.scale   F8_E8M0 [2048, 128]
w2.scale   F8_E8M0 [4096, 64]
```

Two FP4 values are packed per byte. Scales use one E8M0 value per 32 logical K
elements.

## Deliverables

- A packed expert metadata validator.
- A loader that preserves:
  - packed weight bytes
  - E8M0 scale exponent bytes
- A stacker that emits:

```text
gate_up_blocks: [E_local, 2 * I, H / 32, 16] uint8
gate_up_scales: [E_local, 2 * I, H / 32]     uint8
down_blocks:    [E_local, H, I / 32, 16]     uint8
down_scales:    [E_local, H, I / 32]         uint8
```

or an equivalent backend-specific layout with explicit documentation.

## Implementation Steps

1. Add a parser for per-expert keys:
   - `layers.{i}.ffn.experts.{e}.w1.weight`
   - `layers.{i}.ffn.experts.{e}.w1.scale`
   - same for `w2` and `w3`
2. Add shape checks for hidden/intermediate dimensions.
3. Reinterpret `I8` weights as `uint8` raw bytes.
4. Reinterpret E8M0 scales as `uint8` raw bytes.
5. Stack `w3` and `w1` in the order required by the chosen MoE backend.
6. Keep `w2` as the down path.
7. Add an EP slicing helper that partitions the expert dimension without
   changing packed K layout.

## Standalone Tests

- Synthetic packed tensors:
  - construct small logical FP4 values
  - pack two values per byte
  - unpack in reference helper
- Verify shape transforms:
  - packed `[I, H/2]` -> blocks `[I, H/32, 16]`
  - packed `[H, I/2]` -> blocks `[H, I/32, 16]`
- Verify E8M0 scale byte preservation:
  - `view(torch.uint8)` round trip
  - `.to(torch.uint8)` is not used
- Verify per-expert stacking order.
- Verify EP split over experts for `E=8`, `ep_size=2`.

## Done Criteria

- Loader tests pass on CPU.
- The output layout is documented and stable.
- The full MoE agent can consume these tensors without needing to inspect the
  original checkpoint naming convention.
