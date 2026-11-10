---
name: trtllm-modeling-v2-entry
description: Load before implementing, modifying or certifying a TensorRT-LLM ModelingV2 catalog entry (OpWrapper) or its GPU test, including verification at new shapes/dtypes/architectures and target-local variants. Not needed for catalog discovery, assembling a target from unchanged entries, running model-level gates, or tuning serving settings.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ModelingV2 entry

## When to load

Load before entry work, including **verification without code changes**. In a
workflow, this means a `kernel` task (onboard, verify). A primer may preload it. Model
assembly, survey and serving-only work do not require it.

Read `tensorrt_llm/_torch/_experimental/modeling_v2/README.md`,
`cpp/tensorrt_llm/modeling_v2/README.md`, `catalog/_op.py` in the ModelingV2
package, and one same-category entry with its test. The two README guides
are the source of truth; this skill scopes one flow task and its certification,
not unrelated Python callables or the whole model's architecture. Follow the
assigned paths and cells and the tests in the current checkout.

Put operations at the narrowest scope that uses them: target, family, or
catalog for reuse across families. Family sharing is allowed; importing from
a sibling target is not. Keep `OpWrapper` stateless, pass weights, state and
configuration explicitly, and supply `reference`, `is_valid`, `CELLS` and
`ARCHS`. Validate a model using the operation with `modeling_v2="require"`.

## Implement

- **This flow's shared-entry dispatch uses a local variant:** use its
  `<target>/ops/<category>/<name>__<site>.py` and matching target test. Keep
  class/instance names; import the variant from the target. For verify, copy
  both shared files. Leave the shared files untouched. A new ops
  category needs a copyright-header `__init__.py`.
- For a single-kernel task, `__call__` makes one kernel-launching call;
  compositions follow the macro rule below. Reuse the installed callable that
  serves the operation: an engine op, a public library callable such as
  FlashInfer, or an existing family/target kernel adapted into the assigned
  scope. Classify its actual source using the dispatch's sourcing tiers; a
  Python callable is not excluded merely because it is outside FlashInfer.
  Mirror schema arguments and return what the call writes, including in-place/
  out variants. Import libraries at module scope, name their source in the
  docstring, and explicitly set `backend` where supported. This is a source
  classification and certification rule, not a separate approval requirement.
- `reference` uses plain torch with fp32 accumulation, accepts the cell's
  named arguments, and calls neither the kernel nor any entry.
- `is_valid` guards metadata only, for **measured silent wrong answers**;
  do not duplicate the op's loud validation. Record preconditions, arch
  quirks, measurements and hard limits in `note`.
- If Python must compose several calls, first look for a better replacement:
  one fused kernel in the tree (`torch.ops.trtllm.*`, the catalog, another
  model's ops), FlashInfer, or another engine's kernel. If one exists, end
  `BLOCKED` and name the entries and tiers to source it instead of writing an entry.
  If nothing better than the native PyTorch implementation exists, write the
  entry as that composition (a **macro entry**): `__call__` composes the native
  calls, `note` says so and lists what you searched, and the decision is
  `LANDED` with `macro:` (below). Never a fallback that hides a faster path.

## Native code

For C++/CUDA changes, follow the native README before editing. A family's
`csrc/CMakeLists.txt` declares its single library through
`add_modeling_v2_family`, `add_modeling_v2_shared` and `add_modeling_v2_target`;
put shared sources in family `csrc/` and target sources in target `csrc/`.
Use the `TRTLLM_MV2_NAMESPACE_*` namespace macros and
`TRTLLM_MV2_LIBRARY_FRAGMENT`/`TRTLLM_MV2_LIBRARY_IMPL` registration macros.
Load through `load_target_ops`/`load_family_ops`; register matching fakes in
`fake_ops.py` using `namespace_of`. Add op/reference/fake tests and the
README's CODEOWNERS paths for new code. Rebuild and install before testing:
the default installed library is a hard link, and relinking the build output
does not update it. Follow the guide's reinstall or symlink procedure and
verify which library the interpreter loads. An out-of-scope build or shared
file is named in your decision file; it is not silently omitted.

## Test and certify

- Parametrize `op.CELLS`; use `_validating.py`'s `certifying` fixture so the
  entry's own `__call__` and `reference` both run. Derive the comparison band;
  bands wider than torch's dtype-aware default need wrong-variant negative
  controls, with measured separation. Judge a cell by its output
  **distribution** against `reference` (cosine similarity, relative L2, a high
  percentile of the absolute error), not by the worst element alone: a cell
  whose distribution matches but whose worst element is past a strict
  `allclose` is certified, with the measured numbers in `note` and the report.
  A cell whose output is wrong (a different result, NaN, a diverged
  distribution) fails. Test each `is_valid` refusal with
  `validating(op)` and `pytest.raises` after demonstrating the unguarded error.
  Unsafe probes run in a subprocess or are recorded only in `note`.
- Use real stateful metadata/caches. **Pool slots are a `spec` axis sized from
  the cache manager, not batch size**; include slots != batch. Collectives use
  the `comm/` pattern: every rank asserts, a deadline, world size as an axis.
- Assert CUDA availability (no skip), seed randomness, import the entry by
  its absolute path, and share no fixture with another entry's test.
- A local entry's or variant's test counts toward its target's test budget
  (10-15 modules, 20 at most). Import a helper the catalog's tests or the
  target's already have (`_validating.py`, `_cells.py`, a `_`-prefixed module
  beside the target's tests) instead of pasting another copy of it.
- A normal cell fits **10 s / 40 GiB**. Boundary-sized probes use
  `spec.large=True` and `pytest.mark.large`; pre-merge excludes them. Report a
  missing marker registration rather than editing `tests/unittest/pytest.ini`.
- For index arguments, compute 2^31-element, 2^31/row-width and fp32 2^24+1
  boundaries. Probe those the runtime envelope reaches; record all limits,
  guarding measured silent failures. Size from measured values.
- Iterate with `pytest -k`; certify once after the last edit to entry **and**
  test: run the whole test file and `test_modeling_v2_no_stale_claims.py` under
  `tests/unittest/_torch/modeling_v2`.
  Batch compatible cells into one pytest process. Add the measured device to
  `ARCHS` only after passing; a missing `Arch` enum gets its own commit.
  Unmeasured cells/arches remain uncertified; report failures, never widen a
  band or claim coverage to hide them.
