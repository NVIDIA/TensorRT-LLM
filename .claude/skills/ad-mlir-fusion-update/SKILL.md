---
name: ad-mlir-fusion-update
description: >
  Modify, extend, or debug the TensorRT-LLM AutoDeploy MLIR elementwise fusion pass
  (tensorrt_llm/_torch/auto_deploy/mlir/) — the FX→MLIR→decompose→discover→Triton-codegen→replace→FX
  pipeline built on xDSL. Use when adding a dialect op/primitive, editing the Triton emitter, changing
  subgraph discovery/splitting, the FX↔MLIR converters, or the rewrite/replacement plumbing; or when
  triaging fusion failures (KeyError/ValueError in codegen or back-conversion, illegal memory access in
  generated kernels, lost GraphModule methods). Covers the 4-touchpoint rule, fail-safe skip, a
  failure-mode triage table, the validation suite, and the agent_learnings.md log convention.
license: Apache-2.0
tags:
  - tensorrt-llm
  - autodeploy
  - mlir
  - xdsl
  - fusion
  - triton
metadata:
  author: NVIDIA Corporation
---

# AutoDeploy: Update the MLIR Elementwise Fusion Pass

## When to use this skill

Use this when the task touches the **MLIR elementwise fusion subsystem** under
`tensorrt_llm/_torch/auto_deploy/mlir/` — adding a primitive op, changing the
Triton emitter, subgraph discovery/splitting, the FX↔MLIR converters, the
rewrite plumbing, or triaging a fusion crash.

For adding a *generic* fusion transform under `transform/library/` (not the
MLIR pipeline), use **ad-add-fusion-transformation** instead. For dumping
before/after graphs, use **ad-graph-dump**. For checking whether the pass
actually ran in a serve log, use **ad-conf-check**.

> **Canonical log:** `tensorrt_llm/_torch/auto_deploy/mlir/agent_learnings.md`
> is the chronological issue/fix record (currently 15 entries). **Read it first**
> — this skill distills it. **After fixing a new bug, append a numbered entry**
> (Symptom / Root cause / Fix / Lesson) so the log stays the source of truth.

## Architecture map

The transform `mlir_elementwise_fusion` (in
`transform/library/mlir_elementwise_fusion.py`, registry key
`mlir_elementwise_fusion`) runs a 6-step pipeline. It is **fail-safe**: any
unsupported dtype/op/pattern must cause a graceful skip, never a crash.

| Step | Code | Notes |
|------|------|-------|
| 1. FX → MLIR | `mlir/fx_to_mlir.py` (`FXToMLIRConverter`) | Builds the `ad` dialect IR + a metadata side-table. Unmodeled ops → `ad.opaque`. |
| 2. Decompose | `mlir/decompose.py` (`run_decomposition`) + `mlir/decompose_rules/` | RMSNorm/gated-RMSNorm → elementwise primitives. xDSL `PatternRewriteWalker`. |
| 3+4. Discover + replace | `mlir/fusion/fuse.py` (`run_fusion`) → `subgraph_discovery.py` + `subgraph_replace.py` | Walker-driven. Discovery is custom union-find; replacement is rewriter-tracked. |
| (codegen) | `mlir/codegen/triton_emitter.py` (`_EMIT` table) + `kernel_cache.py` | Generates + caches one Triton kernel per subgraph. |
| 6. MLIR → FX | `mlir/mlir_to_fx.py` (`MLIRToFXConverter`) | Rebuilds the GM with fused kernel calls. |
| dialect | `mlir/dialect.py` (`AD_OPS`, dtype maps) | Op + type definitions; `_TORCH_TO_MLIR_DTYPE`. |

`default.yaml` entry: `mlir_elementwise_fusion` (stage `post_load_fusion`,
`enabled: false` by default, `run_shape_prop: false`). xDSL is optional —
everything is gated behind `HAS_XDSL`; the pass skips cleanly when absent.

## The 4-touchpoint rule (adding a primitive op)

When you add a new elementwise/reduction primitive, update **all four** or
back-conversion will silently drop it (agent_learnings #1):

1. `mlir/dialect.py` — define the `AdXxx` op class; add to `AD_OPS`. Add any new
   dtype to `_TORCH_TO_MLIR_DTYPE` (and the reverse map).
2. `mlir/fx_to_mlir.py` — route the aten op in `_convert_call_function`.
3. `mlir/mlir_to_fx.py` — add the reverse handler.
4. `mlir/codegen/triton_emitter.py` — add an `_EMIT` entry (or a special-case
   branch for attribute-carrying ops like `pow`/`splat`/`cast`/`floordiv`/`eq`).
   To make it fusible, also add the class to `FUSIBLE_OPS` in
   `subgraph_discovery.py`.

## Cardinal invariants (do not regress)

These are the load-bearing rules behind the log — violating one reintroduces a
known crash.

- **Fail-safe over correctness-at-all-costs.** Unsupported dtype/op/pattern →
  log a warning and skip (return original graph). The `converter.convert()` call
  is wrapped in `try/except ValueError` for exactly this (#13).
- **Discovery stays union-find — do NOT replace it with a root-anchored walk**
  (#15). A backward root cone fragments maximal multi-output components and
  can't express the 64-input dependent partitioning. Only iteration + mutation
  are xDSL-native; the discovery algorithm (union-find, `_has_placement_conflict`,
  `_split_subgraph`, Kahn topo-sort) is custom by design.
- **Safe erase via reverse-topological order** (#15). `subgraph.ops` is topo-sorted;
  erase in reverse (consumers first) after `replace_by` redirects external uses —
  then default `safe_erase=True` passes. Never reach for `safe_erase=False`.
- **Forward walk only.** `run_fusion` uses `PatternRewriteWalker(apply_recursively=False)`
  with a **forward** walk so split-partition anchors are visited in dependency
  order and `refresh_inputs()` (#10) still works. Do not set `walk_reverse=True`.
- **`run_shape_prop: false`** for this transform (#6) — shape-prop concretizes
  symbolic dims and breaks downstream transforms. Fused `register_fake` handles
  shapes via the reference-input mechanism.
- **Three input load categories in the emitter** (#3, #4): broadcast (lower rank →
  `offs` only), narrow (same rank, last-dim < `N_COLS` → scalar load), full-row
  (last-dim == `N_COLS` → `row_off + offs`). `N_COLS` = **widest** highest-rank input.
- **Separate load vs store offsets** (#5): loads use `row_off` (from input
  `stride(-2)`, may be non-contiguous from `aten.chunk`); stores use
  `out_row_off = pid * N_COLS` (outputs are always contiguous).
- **64-input limit is hard** (#9): `torch.library.custom_op` schemas cap at 64
  args. `_split_subgraph` must split oversized subgraphs **placement-aware** (#11):
  a partition is viable only if all input producers precede all output consumers.
- **`replace_subgraph_with_fused_op`'s `rewriter` param must stay optional** (#15) —
  unit tests call it standalone without a walker.
- **MLIR→FX must preserve GraphModule methods** (#14): copy missing public
  callables (e.g. `get_input_embeddings`) onto the reconstructed GM.
- **Pattern-matchers downstream are fragile** (#7): fusion absorbs `aten.add.Tensor`
  etc.; transforms like `multi_stream_moe` must match op-type-agnostically.

## Failure-mode triage table

| Symptom | Likely cause | See |
|---------|--------------|-----|
| `ValueError: FX node not found for MLIR value` (unfused op) | New op missing from a converter — 4-touchpoint violation | #1 |
| `ValueError: FX node not found ... 'ad.opaque' (all_reduce/dist)` | Fused op placed before an input producer in a spanning subgraph | #2, #11 |
| `RuntimeError: shape '[...]' is invalid` after fusion | `_get_ncols` not using the widest input | #3 |
| `Triton ... illegal memory access` | Narrow input loaded as full-row, or load/store stride mismatch | #4, #5 |
| `KeyError` in `triton_emitter` (`val_names` missing operand) | Stale subgraph inputs (missing `refresh_inputs`) or bad intra-subgraph topo order | #10, #12 |
| `RuntimeError: schema has N arguments but ... supports 64` | Oversized subgraph not split | #9 |
| Downstream sees concrete shapes instead of symbolic | `run_shape_prop: true` | #6 |
| `ValueError: Unsupported torch dtype ... complex64` | Unmapped dtype; should skip gracefully not crash | #13 |
| `AttributeError: 'GraphModule' has no attribute '...'` | Methods dropped on MLIR→FX reconstruction | #14 |
| `multi_stream_moe` produces 0 matches when fusion on | Merge-add absorbed by fusion; matcher too specific | #7 |

## Validation

Run the dedicated unit suite (xDSL required; CUDA tests auto-skip without a GPU):

```bash
python3 -m pytest tests/unittest/auto_deploy/singlegpu/mlir/ -q
```

What the suite covers (add to the matching file when you change behavior):

- `test_dialect.py` — op/type definitions.
- `test_fx_mlir_roundtrip.py` — FX↔MLIR conversion fidelity.
- `test_decompose.py` — decomposition rules.
- `test_subgraph_discovery.py` — grouping, splitting, **subgraph count** (parity invariant).
- `test_string_codegen.py` — generated kernel source + **hash stability** (kernel cache).
- `test_elementwise_fusion_e2e.py` — full pipeline incl. `run_fusion` walker path and CUDA numerical roundtrips.

Then end-to-end (the pass is off by default — enable it or use a config overlay):

```bash
python examples/auto_deploy/build_and_run_ad.py --args.model=<HF_MODEL>
```

Verify in logs: `Decomposed N high-level ops`, `Discovered N fusible subgraphs`,
`replaced M (skipped K low-rank)`, no `RuntimeError`/`KeyError`, correct output.
Always benchmark e2e **under CUDA graphs**, not standalone kernel speed (#8).

After Python edits: `pre-commit run --files <changed_files>`.

## Review checklist

- New primitive? All **4 touchpoints** updated and added to `FUSIBLE_OPS` if fusible.
- No `safe_erase=False`; erase order is reverse-topological.
- Discovery algorithm unchanged unless intentionally so (no root-walk substitution).
- `run_fusion` stays forward-walk; `rewriter` param on `replace_subgraph_with_fused_op` still optional.
- Unsupported input → graceful skip (covered by a test), not a crash.
- Subgraph-count and hash-stability tests still pass (no kernel-cache churn).
- New bug fixed? **Appended a numbered entry to `agent_learnings.md`.**

## Guardrails

- One logical change per patch; don't mix a converter fix with an emitter rewrite.
- If the symptom isn't in the triage table, reproduce with `AD_DUMP_GRAPHS_DIR`
  (see **ad-graph-dump**) before hypothesizing — the failure is usually a
  missing converter handler or a stale-input/placement issue, not "slow".
- Treat the pass as optional infrastructure: never let a fusion bug degrade a
  model that would otherwise run — prefer skip.
