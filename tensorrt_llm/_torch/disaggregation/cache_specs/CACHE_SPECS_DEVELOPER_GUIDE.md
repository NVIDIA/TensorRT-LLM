# Cache Specs Developer Guide

`cache_specs/` is everything the native P/D transfer knows about *what* a cache is. Each cache
variant is a pluggable spec: one class in `specs/variants.py` that claims its cache-manager
classes, describes their memory, and owns the variant's rules. The rest of `disaggregation/` moves bytes and
keeps sessions; it never asks which variant it is moving.

The native transfer supports `KVCacheManagerV2`-based managers only; V1 managers use the C++
transceiver.

| File | Holds |
|------|-------|
| `page_table.py` | `PageTable` / `Pool` / `LayerGroup` / `Region` / `BufferDecl`: a rank's memory, with no spec-specific fields |
| `spec.py` | `CacheSpec`: what a spec answers (description, selection, request-wide rules, checks, memory) |
| `registry.py` | spec name -> spec; `claim(manager)`: the spec naming the class nearest in the manager's MRO |
| `build.py` | `build_page_table(manager)`: the claiming spec describes the manager |
| `correspondence.py` | the one mapping algorithm: region matching, byte runs, transfer groups, sender election, generic peer validation |
| `topology.py` | shard domains (`ATTENTION_TP`, `STATE_TP`) -> `ShardGrid`; which peer ranks overlap |
| `queries.py` | whole-table questions (registration, bounce sizing, chunk bytes, transfer length, exclusions, resume position) |
| `specs/_base.py` | `StandardSpec`: a V2 manager described life cycle by life cycle, with the default per-role layout rules |
| `specs/_paged.py`, `specs/_state.py` | rules shared by paged units (blocks by ordinal) and state units (one slot per request) |
| `specs/_ownership.py` | the physical-ownership bridge for NVFP4 key-only caches, asked of every V2 spec |
| `specs/_v2.py`, `specs/_virtual_layers.py` | reading a V2 manager's pools and life cycles; virtual-layer ids shared by DeepSeek-V4 and CSA2 |
| `specs/variants.py` | one registered spec per variant (see below) |

## The page table

```
PageTable
├─ pools:  [Pool]        base address, slot bytes, slot stride
└─ layer_groups: [LayerGroup]  one per V2 life cycle (index == life-cycle id)
     ├─ spec:    "csa2"           the spec owning the layer group's rules (peers must agree)
     ├─ kind:    PAGED | STATE    blocks by ordinal, or one slot per request; picks the shard
     │                            domain: attention TP or the recurrent-state grid
     ├─ params:  {...}            read only by the spec (window, ...)
     └─ regions: [Region]         one per (layer, role) buffer
          ├─ layer, role          global layer id + role: what it is, matched with the peer
          ├─ pool, offset         where it is inside a unit's slot
          └─ buffers: [BufferDecl] how its bytes are laid out and split across ranks
```

The shape is fixed: a new cache adds layer groups, regions, roles, axis names and params, never fields.
On the wire, regions are rows and declarations are sent once per distinct layout.

### Declaring a layout: `BufferDecl`

Each buffer names its axes outer to inner and the axis it is split along; its layer group's kind
names the shard domain that splits it (`topology.domain_of`):

```python
BufferDecl(nbytes, axes=(("token", tpb), ("head", kv)), split_axis="head")  # NHD key
BufferDecl(nbytes, axes=(("head", kv),), split_axis="head")                 # HND key
BufferDecl(section_bytes, axes=(("byte", section_bytes),), split_axis="byte")
BufferDecl(nbytes)                                                          # replicated
```

`nbytes` is what this rank holds. The algorithm derives the rest:

* **Spans.** Each rank of the domain's grid holds `units` consecutive units of a virtual axis of
  `min(units_a * grid_a, units_b * grid_b)` units. The side with fewer copies of each unit gets
  disjoint spans; the other side maps several ranks onto one span (GQA heads on more ranks than
  heads, an MLA latent on every rank). A replicated buffer is one unit.
* **Pairing.** Two ranks exchange a buffer only if their spans intersect.
* **Bytes.** The intersection becomes one run per row of the axes outside the split axis. Unit
  bytes come from `nbytes`, never from element sizes, so sub-byte dtypes stay exact and a slice
  that is not byte-aligned raises.
* **Election.** When `units * grid` is larger locally than on the peer, that many local ranks hold
  identical bytes; one is elected, rotated by the peer's DP rank.

Axis names are free strings; the algorithm only reads their extents and which one is split.

## How a transfer uses the module

1. **Build** (each side, once). `build_page_table(manager)`: `claim(manager)` finds the spec and
   its `describe` returns the page table, which travels to peers in `RankInfo`.
2. **Connect** (once per peer). `validate_peer_compatible` checks both sides run the same specs,
   matched regions declare the same layout, replicated roles are not one-sided, and then each
   spec's own `validate_peer`. `plan_transfers` matches regions by `(spec, layer, role)`, turns each
   pair into slot-relative byte runs, and merges them per `(layer_group, pool)` pair and sender election
   into `TransferGroup`s, cached per peer.
3. **Per request.** `local_units` lists the request's unit ids per layer group, `chunk_units` narrows
   them to a prompt chunk, `excluded_regions` names regions the request does not move; on the
   sender `pair_units` pairs the two sides' units, and each owned group's runs are added to the
   units' slot base addresses.

## The built-in specs

| Spec | Claims | What is its own |
|---|---|---|
| `attention` | `KVCacheManagerV2` (and any manager no spec claims, e.g. DSA) | K/V by KV head, head-major; replicated index-K; MLA; VSWA windows |
| `minimax_m3` | `MiniMaxM3KVCacheManagerV2` | HND or NHD K/V as the manager's backend stores it |
| `fp4_mla` | `Fp4MlaKVCacheManagerV2` | NHD K + block scales, replicated HP tail |
| `deepseek_v4` | `DeepseekV4CacheManager` | virtual layer ids and windows |
| `csa2` | `CSA2CacheManager` | every role replicated; virtual layer ids; remote tail: transfer length, excluded decoder SWA, no first tokens |
| `mamba` | `MambaHybridCacheManagerV2` (and subclasses, e.g. GLM-5 Next) | state life cycles and slot; sectioned conv, head-sharded SSM, replicated PLE state |
| `qsa` | `QSAMambaHybridCacheManagerV2` | replicated index positions |

A manager whose cache differs from its base class in nothing gets no spec of its own: it is claimed
by the spec of its nearest class.

How each role is laid out is the spec's knowledge, not the manager's: `StandardSpec.role_buffers`
answers per `(layer, role)` buffer from `replicated_roles`, `kv_layout` and `state_buffers`, which a
spec overrides with what it reads from its manager (section sizes, head geometry, the backend's
storage layout). Managers only name storage-only roles (`get_disagg_ignored_roles`).
`role_layout(manager, kind, role, nbytes, ...)` answers the same question for other readers of the
cache layout (named reuse).

## Adding a spec

1. Make sure the runtime manages the memory: a life cycle in a `KVCacheManagerV2` subclass. That
   part is outside `cache_specs/`.
2. Add a `@register`ed subclass of `StandardSpec` (or `CacheSpec`) to `specs/variants.py`,
   setting `name` and `managers` (dotted class paths).
3. Override only what differs, for example:
   * `role_buffers` (or `replicated_roles` / `kv_layout` / `state_buffers`) -- how each role is
     laid out and split across ranks;
   * `describe` / `describe_life_cycle` / `life_cycle_kind` / `global_layer_ids` /
     `window_size` -- what memory exists;
   * `local_units` / `chunk_units` / `pair_units` / `state_slot` -- which units move;
   * `transfer_length` / `excluded_regions` / `sends_first_tokens` / `resume_after_receive` --
     request-wide rules;
   * `validate_peer` / `check_rank_compatible` -- checks the declarations do not cover;
   * `requires_physical_ownership` -- a transfer mode the deployment needs;
   * `registration_regions` -- memory outside the manager's pools.
   A new shard domain is one entry in `topology._GRIDS` and its cache kind in `topology.domain_of`.
4. Add the spec's layouts to the resharding matrix in
   `tests/unittest/disaggregated/test_cache_specs.py`, which checks on synthetic page tables
   across TP / PP / CP layouts that the page table survives the wire and, byte by byte, that every
   receiver gets every byte exactly once from the source byte with the same
   `(layer, role, buffer, row, global unit, offset)`. The test adds a linear-attention state spec
   this way, with its own axis names.

Nothing under `native/`, the transceiver or the page table changes for a new spec. A peer that runs
a spec this build does not know is rejected when its regions are matched or validated, never
skipped.

## When the model itself must grow

The page table expresses memory at "pool base + unit * slot stride + offset", moved byte for byte,
split along one axis. A cache outside that -- variable-length or non-contiguous addressing, a buffer
split along two axes by two parallel dimensions, a unit spanning several slots, or bytes that must
be transformed in flight (dtype conversion, requantization) -- needs the model or the transfer
extended for every spec, not a special case in one.

## Not covered here

* Named reuse (the sharing lender) does not consume these specs yet.
* The C++ transceiver (`BindKvCacheTransceiver`) has its own formatters and does not use this
  module.
