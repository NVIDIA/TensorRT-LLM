# KV Cache Page Table Spec Guide

## What it is

Every `KVCacheManagerV2` states, through `page_table_spec()`, what disaggregated transfer needs to
know about its pages that its runtime config and storage do not. The result is a `PageTableSpec`
(`pyexecutor/kv_cache/page_table.py`):

- `role_layouts` and `default_layout`: how each role's bytes spread over ranks.
- `untransferred_roles`: roles that never move.
- `derived_roles`: roles the receiver rebuilds from the transferred ones
  (`rebuild_disagg_imported_cache`).
- `request_rebuilt_roles`: roles the receiver rebuilds for some requests.
- `layer_ids`: the global layer id and model layer of every layer the storage holds, as the
  transfer names it (`_disagg_layer_names`).
- `quant_mode` (one of `QUANT_MODES`) and `mixed_precision`: how the KV data is quantized.

A role layout (`RoleLayoutV2`, `disaggregation/base/role_layout.py`) says which pieces of one
space of bytes each rank's buffer holds. The kinds are `ByHead` (head-major or token-major),
`Sectioned`, `Replicated`, `RingSlots` and `Own`. A layout in a spec is a template; the reader
resolves it against each buffer's facts (`BufferSite`). Two ranks' buffers correspond when their
resolved layouts give the same `space`, and the bytes to move are the overlaps of their `pieces`.

## Status

Nothing reads the spec yet. The native transfer, the extractor and KV cache sharing read the
`get_disagg_*` hooks. Until the hooks derive from the spec, a change to a manager's hooks needs the
same change to its `page_table_spec`. `test_page_table_spec_matches_disagg_hooks.py` checks that
each manager's spec agrees with its hooks.

## Changing a manager

- A role the manager registers either has a layout in its spec or is listed as untransferred,
  derived or rebuilt per request. Add it to the manager's case in the parity test.
- A manager that names its layers differently from its model layers overrides
  `_disagg_layer_names`, as the FP4 MLA cache layout does.
- Recurrent state splits over `recurrent_state_split(topology)`, which matches the native
  transfer's shard grid.
- In a class that marks its overrides with `@override`, an override of `page_table_spec` or
  `_disagg_layer_names` carries it too.

## What the spec does not state yet

- Layouts that differ between layers of one role. A manager whose `layer_specs` mix K-only and
  K/V caches raises `NotImplementedError`.
- Per-layer element types, such as FP4 MLA's FP8 fallback layers: `quant_mode` is the
  configured mode.
- Per-request rules, such as which layers of a request CSA2 rebuilds.
- How pages are assigned to context-parallel ranks.
