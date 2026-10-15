# Decomposing `fuse_input_embeds` (multimodal placeholder splice)

Why the requested catalog entry `multimodal/fuse_input_embeds` was not written,
what the step actually is, which calls serve it, and the one call nobody in the
catalog can serve today.

Derived on: tensorrt_llm 1.3.0rc26 | torch 2.12.0a0+5aff3928d8.nv26.05 |
NVIDIA GB300, `torch.cuda.get_device_capability()` `(10, 3)` → **sm_103** — the
stamp on the observations below, not a pin to assert. Receipts are keyed by GPU
architecture and nothing else, and each constituent carries its own in its own
contract; this document carries none.

Checkpoint: `Qwen3.8-Flash-Next-NVFP4`, `model_type` `qwen4_exp`.

## 1. Why decompose it

The request was one entry for the step the run's `vocabulary.yaml` states as
"splice the vision embeddings into the text embedding rows at the placeholder
positions", sourced at tier 2 against the in-tree callable
`tensorrt_llm._torch.models.modeling_multimodal_utils.fuse_input_embeds`.

That callable is not one kernel-launching call. Its body composes, at the
Python level: a list join, an optional index derivation, **a module call**
(`embedding_layer(...)`), an optional buffer allocation, and one or two indexed
row assignments. `references/decomposition.md` names three of those shapes
explicitly — "module calls", "tensor glue beyond the single call", "a phase
dispatch" — and `entry-spec.md` forbids the result: "A wrapper that chains
split → norm → matmul … is not an entry."

**Measured, not argued.** Profiled on sm_103 at this checkpoint's widths
(`T = 2048` tokens, `M = 512` placeholder rows in 2 contiguous runs,
`H = 2560`, bf16, `vocab_size = 248320`), one `fuse_input_embeds` call on the
path this checkpoint takes launches **exactly 2 device kernels**:

```
1 x at::native::vectorized_gather_kernel<16, int>            <- the embedding lookup
1 x at::native::index_elementwise_kernel<...index_put_kernel_impl<OpaqueType<2>>...>
                                                             <- the row scatter
```

Two kernel-launching calls, each of which is a legal single entry on its own,
and **one of the two is already in the catalog**. A wrapper chaining them is
precisely the Python-level composition the entry spec bans.

The other branches of the same callable, measured in the same process, are
larger rather than smaller:

| branch | device kernels | why it is not this checkpoint's path |
|---|---|---|
| in-vocab, executor-supplied indices | **2** | **this is the path** |
| in-vocab, indices derived in-forward | 15 (incl. 2 `Memcpy DtoH`) | the executor ships the indices; see §4.2 |
| OOV placeholders, executor indices | 4 | placeholders are in vocabulary; see §4.1 |

### 1.1 No registered op serves this step

**The sweep had to be made exhaustive first, and the run's own ruling R2 says
why.** An op registers when the module holding its decorator is imported, and
several register outside `custom_ops/`. Sweeping after
`import tensorrt_llm._torch.custom_ops` alone yields **294** `trtllm::` /
`tensorrt_llm::` schemas and misses them — demonstrated on this host with
`trtllm::embedding_skip_forward`, which is declared by
`@torch.library.custom_op` in `_torch/modules/embedding.py` and is absent from
that 294. Walk-importing every module under `_torch/modules`,
`_torch/custom_ops`, `_torch/models`, `_torch/attention`,
`_torch/cute_dsl_kernels` and `_torch/cuda_tile_kernels` (474 imported, 6
failed on optional dependencies, 11 skipped as embargoed) raises the set to
**323**, of which **29** are new. **107** of the 323 match a name pattern
covering `scatter|index|put_|embed|fuse|splice|mask|place|insert|merge|copy|
gather|multimodal|vision|mm_|select`; every one was read, and none writes a row
subset of an activation buffer.

The single candidate worth refuting by name is
`trtllm::inplace_slice_copy(Tensor(a!) dest, Tensor src, int dim1_start,
int dim1_end) -> ()`. It copies into a **dim-1 (column)** range across *all*
rows — the Eagle3 hidden-state capture in `_torch/speculative/eagle3.py` uses it
to stack per-layer states at `i * hidden_size`. The splice is the transpose of
that: a row subset, all columns. It does not serve.

Spot checks, all `False` on this host after the full walk-import:
`hasattr(torch.ops.trtllm, x)` for `fuse_input_embeds`, `scatter_mm_embeds`,
`index_copy`, `index_put`, `masked_scatter`. The one op in the generated
inventory's `embedding` category, `embedding_skip_forward`, does resolve once
its module is imported, but it is a shape-only stand-in — its implementation
`embedding_skip_forward_impl` returns an uninitialised `new_empty` of the
output shape — not a lookup and not a splice.

### 1.2 No tier-2 kernel-plane callable serves it either

A tree-wide search (not a single directory — ruling R4's second-order note),
for `mm_token_indices|masked_scatter|index_copy_|index_put_` across
`tensorrt_llm/_torch/**.py`, returns no standalone callable under
`_torch/modules/` that performs this splice. The only in-tree servers are
`fuse_input_embeds` itself and inline code in per-model zoo files. Nothing under
the run's `source_embargo` was opened, and no constituent below routes through
it; the embargo is not what excludes tier 2 here.

**Transparency note, in the spirit of ruling R5.** The exhaustive sweep of §1.1
skipped the embargoed modules by name, but eight `trtllm::qwen4_exp_hc_*` /
`trtllm::qwen4_exp_ple_*` op *names* still appeared in the registry diff,
because a non-skipped module imports that package transitively. Only the
schema names were seen; no embargoed file was opened and none of those ops is a
multimodal splice, so nothing here depends on them — and ruling R1 would bar
wrapping them regardless. It is recorded because a sweep is the same hole a
tree-wide grep was.

## 2. Granularity and sourcing policy

Atomicity criterion: Python-level indivisible. Sourcing preference walked in
order — `torch.ops.trtllm.*` (§1.1, none), any other Python-callable atomic
surface (§1.2, none), torch-native glue as a thin mirror under `catalog/torch/`.
The third rung is where this step lands, and §3 constituent 3 is the
consequence.

## 3. The resulting vocabulary

Paths relative to `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/`.

| # | role | callable | state |
|---|---|---|---|
| 1 | join the per-request vision embeddings into one `[M, H]` block | `torch.cat` | **existing entry** `torch/concat.py` |
| 2 | text embedding lookup over the whole `input_ids` row | `torch.nn.functional.embedding` | **existing entry** `torch/embedding.py` |
| 3 | write the `[M, H]` block into the `M` placeholder rows | `torch.Tensor.index_copy_` | **gap** — see below |

Every symbol was resolved on this host before being cited. `torch.cat`,
`torch.nn.functional.embedding` and `torch.Tensor.index_copy_` are torch
callables and need no registration check; the three catalog modules
`torch/concat.py`, `torch/embedding.py` and `torch/copy_.py` were imported and
their functions printed. `fuse_input_embeds` and
`filter_mm_token_from_input_ids` both import from
`tensorrt_llm._torch.models.modeling_multimodal_utils`.

**Constituent 3 is the whole of the gap, and it is a placement question rather
than a capability one.** The computation exists and is one kernel; what does
not exist is a catalog entry for it. The natural home is `catalog/torch/`,
which already carries `copy_`, `concat`, `empty`, `expand`, `pad`, `reshape`,
`split`, `transpose` — movement and layout glue, which an indexed row write is.
smith may not write there (the role forbids it, and the reason given is that a
*computation* mirror would widen what the surface is for; an indexed row copy
is movement, not computation, so the stated reason does not bite — but the
prohibition is not smith's to read around). **The orchestrator decides whether
`torch/index_copy_.py` is added.** Nothing else in this decomposition is open.

Three interchangeable spellings of constituent 3 were measured, bitwise
identical to one another at `T = 2048, M = 512, H = 2560, bf16`:

| spelling | device kernels | kernel |
|---|---|---|
| `buf.index_copy_(0, idx, src)` | **1** | `index_copy_kernel_impl<OpaqueType<2>>` |
| `buf[idx, :] = src` (advanced index) | **1** | `index_put_kernel_impl<OpaqueType<2>>` |
| `buf.masked_scatter(mask2d, src)` | 6 | adds a cub scan, a size check and a DtoD copy |

`index_copy_` is the recommendation: fewest launches, and its signature takes
the index tensor the executor already ships. `masked_scatter` is the form the
HuggingFace reference for this architecture family uses
(`get_placeholder_mask` → `inputs_embeds.masked_scatter(...)` in
`transformers/models/qwen3_vl/modeling_qwen3_vl.py`); it is listed so the
difference is on the record, not because it should be preferred here — it needs
a boolean mask the runtime does not ship, and it costs 5 extra launches.

**The slice-copy shortcut does not generalise.** `catalog/torch/copy_.py`
already exists, and a contiguous placeholder run can be written with it — this
was measured bitwise equal to the scatter. But a batch with `n` runs needs `n`
calls, so `n` is data-dependent and the forward stops being flat. Measured at 3
runs: `scatter == 3 slice copies` is `True`, and the scatter is 1 call against
3. The multimodal skill's `references/vision-path.md` records the single-run
shape ("the write is a slice copy rather than a scatter") from a different
target's single-image measurement; it is true of that case and is not the
general form.

## 4. Assembly — the mainline path

At this checkpoint, with `text_token_indices` / `mm_token_indices` arriving
from the executor and `mm_embeds` already one pre-concatenated tensor:

```
mm_embed    = concat(mm_embeds)               # only when len(mm_embeds) > 1
inputs_embeds = embedding(input_ids, weight)  # [T, 2560] bfloat16
inputs_embeds.index_copy_(0, mm_token_indices, mm_embed)
```

Verified end to end on this host against an independently built native-torch
reference (`F.embedding(...).clone()`, then the scatter), **bitwise equal** to
`fuse_input_embeds`' own output at `T = 2048, M = 512` (2 runs) and
`T = 8192, M = 2048` (4 runs), for both the advanced-index and the
`index_copy_` spelling. Passing two `mm_embeds` tensors instead of one
pre-concatenated tensor is also bitwise equal.

### 4.1 Why the in-vocab branch is the one to assemble

`config.json` gives `image_token_id 248056`, `video_token_id 248057` and
`text_config.vocab_size 248320`. Both placeholders are **inside** the
vocabulary, so the embedding lookup over the whole `input_ids` row is in
bounds and the OOV branch — which gathers only text positions into a fresh
`torch.empty` buffer and scatters twice — is dead code for this target. The
OOV branch costs 4 kernels against 2 and would additionally need
`catalog/torch/empty.py` and an `index_select`.

Caveat the assembler owns: the target must declare its placeholder ids, because
`prepare_multimodal_indices` (`_torch/pyexecutor/engine/runners/common.py`)
reads `multimodal_token_ids` (falling back to `mm_token_ids`) off the model and
otherwise takes the `>= vocab_size` OOV predicate.

### 4.2 Why the index derivation is not in the forward

`ship_multimodal_indices` (same file) pins and async-copies the executor's
CPU-computed `mm_token_indices` / `text_token_indices` into the inputs dict, so
the forward receives them. Deriving them inside the forward instead costs 13
extra kernels **and two `Memcpy DtoH` host syncs** — measured: the standalone
`filter_mm_token_from_input_ids` call launches 13 kernels including 2
`Memcpy DtoH (Device -> Pinned)`, and the whole `fuse_input_embeds` call rises
from 2 launches to 15. A target that forwards the two index tensors explicitly
keeps the step at 2 launches and sync-free; one that relies on a `**kwargs`
splat silently takes the 15-launch syncing path.

This also means constituent 3's index tensor is `int64` and device-resident,
which is what `index_copy_` wants.

### 4.3 The deepstack branch is inert here

`vision_config.deepstack_visual_indexes` is `[]` in this checkpoint's
`config.json`, so `fuse_input_embeds`' `extra_embeds` path — which allocates a
`torch.zeros` buffer per extra feature and scatters into it — is never entered.
It is named here only so a later checkpoint in this family does not read its
absence as an oversight.

### 4.4 The one guard the callable raises, quoted

Passing a `mm_embeds` row count that disagrees with the index count is
rejected, not silently truncated. Observed verbatim on this host:

```
ValueError: Multimodal token count mismatch: found 512 image tokens in input_ids but received 10 image embeddings.
```

Whether the constituent calls carry that guard themselves is a question for
constituent 3's own contract, after it is onboarded; a bare `index_copy_` with
a short source does not raise this message.

## 5. Certification status

**Nothing here is certified.** No entry files were written, no test file, no
receipt. The launch counts, the bitwise equalities and the
quoted `ValueError` above are observations from two probe runs on sm_103; they
are evidence for the decomposition, not certification of any constituent.

- Constituents 1 and 2 are existing catalog entries and carry their own
  receipts in their own contracts. This document does not restate them.
- Constituent 3 has no entry and therefore no receipt, no certified dtype
  column and no certified shape column. What `index_copy_` does with duplicate
  indices, out-of-range indices, a non-contiguous source, or a dtype mismatch
  between source and destination is **unknown here** — it was not probed,
  because a macro report may not claim what only certification can establish.

## 6. Out of scope

- The vision tower that produces `mm_embeds` (queue items 17–21) and the
  patch-merger projection that maps it to `out_hidden_size 2560` (served by
  `gemm/cublas_mm`).
- Where the splice runs. `references/vision-path.md` of
  `staircase-modeling-multimodal` records that it belongs in the shell's
  forward, producing `inputs_embeds` with `input_ids=None`, because
  `DecoderModelForCausalLM.forward` forwards only the keywords it names. That
  is an assembly question, not a vocabulary one.
- Chunked prefill and KV-cache reuse slicing of `mm_embeds`
  (`find_input_mm_embeds`): it reads only CPU-resident counts off
  `MultimodalParams.multimodal_runtime` and launches nothing, so it is host
  bookkeeping rather than a catalog constituent — but it does decide which rows
  reach constituent 3, and a target that skips it will fail §4.4's guard.
