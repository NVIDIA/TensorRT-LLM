<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# KV Cache Page Table Spec Guide

## What it is

Every `KVCacheManagerV2` states, through `page_table_spec()`, what disaggregated transfer and KV
cache sharing need to know about its pages that its runtime config and storage do not. The result
is a `PageTableSpec` (`pyexecutor/kv_cache/page_table.py`):

- `role_layouts` and `default_layout`: how each role's bytes spread over ranks.
- `untransferred_roles`: roles that never move.
- `derived_roles`: roles the receiver rebuilds from the transferred ones
  (`rebuild_disagg_imported_cache`).
- `request_rebuilt_roles`: roles the receiver rebuilds for some requests.
- `layer_ids`: the global layer id and model layer of each layer where they differ from the
  default naming (`page_table.layer_ids_of`).
- `quant_mode` (one of `QUANT_MODES`): how the KV data is quantized, as configured.
- `mixed_precision`: reserved for the low and high element types of mixed-precision KV. No
  manager fills it today, and neither the page table nor a peer's check reads it.
- `retention_margins`: whether the manager keeps draft tokens, a rewind, or another margin
  beyond its attention window.

A role layout (`RoleLayoutV2`, `disaggregation/base/role_layout.py`) describes which pieces of a
role each rank holds. Managers import their layouts from `page_table.py`, which re-exports them.

## Current consumers

Native transfer and KV sharing still use the existing manager hooks. When a manager's roles or
layer naming change, update its spec and the corresponding hooks together.
