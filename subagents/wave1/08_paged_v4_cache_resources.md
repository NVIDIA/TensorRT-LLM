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

# Plan 08: Paged V4 Cache Resources

## Goal

Add AutoDeploy runtime/cache support for DeepSeek V4's heterogeneous cache
needs. The feature should prove named or composite paged resources can be
allocated, indexed, and reused without requiring the full attention kernel.

This work is independent because it can use synthetic cache descriptors and
metadata tests.

## Owner Scope

Suggested files:

- `tensorrt_llm/_torch/auto_deploy/shim/interface.py`
- `tensorrt_llm/_torch/auto_deploy/transform/library/kvcache.py`
- `tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py`
- `tests/unittest/_torch/auto_deploy/unit/runtime/test_deepseek_v4_cache_resources.py`

Coordinate carefully with existing KV cache behavior. Do not change generic
cache semantics unless covered by tests.

## Required Resources

DeepSeek V4 needs:

```text
swa_kv_cache
  high-resolution local window cache

mhc_cache
  compressed KV entries for ratio-4 and ratio-128 layers

indexer_cache
  compact indexer KV for ratio-4 top-k selection

compressor_state
  per-sequence state for incomplete compression windows during decode
```

## Design Direction

Prefer named page tables per resource group:

```text
SequenceInfo.cache_loc["swa"]
SequenceInfo.cache_loc["mhc"]
SequenceInfo.cache_loc["indexer"]
SequenceInfo.cu_num_pages["swa"]
...
```

If dictionary-style metadata is too invasive for export or runtime code, use a
small fixed struct/list of named resource metadata with stable ordering.

Avoid unpaged `[max_batch, max_seq, ...]` caches for long-lived V4 cache data.

## Deliverables

- A cache resource descriptor for V4 attention.
- Allocation of multiple paged resource pools or a composite pool.
- Per-resource page metadata in `SequenceInfo` or equivalent.
- Prefix reuse and partial reuse behavior documented for V4 resources.
- A way for future V4 attention kernels to retrieve page tables by resource
  name.

## Standalone Tests

- Allocate synthetic V4 resources with small page sizes.
- Nest two or more sequences and verify page metadata.
- Verify SWA and MHC can have different logical lengths.
- Verify ratio-128 compressed cache uses fewer logical entries than token cache.
- Verify prefix reuse does not silently disable all cache reuse.
- Verify local/non-paged fallback is not used for long-lived V4 resources.

## Done Criteria

- Tests run without full model weights.
- Existing standard attention/MLA cache tests still pass.
- V4 attention kernel agents can consume stable metadata names and shapes.
