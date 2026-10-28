---
name: trtllm-modeling-v2-entry
description: Load before implementing, modifying, certifying, or promoting a TensorRT-LLM ModelingV2 catalog entry (OpWrapper) or its GPU test, including verification at new shapes/dtypes/architectures and target-local variants. Not needed for catalog discovery, assembling a target from unchanged entries, running model-level gates, or tuning serving settings.
---

# ModelingV2 entry

## When to load

Load before entry work, including **verification without code changes**. In a
workflow, this means a `kernel` task (onboard, verify, promote); a tuner loads
it only when changing an entry or its test. A primer may preload it. Model
assembly, survey and serving-only work do not require it.

Read `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/_op.py`, the
ModelingV2 README, and one same-category entry with its test. Follow the
assigned paths and cells; this checklist supplies the additional rules.

## Implement

- **Shared entry → local variant first:** use the dispatch's
  `<target>/ops/<category>/<name>__<site>.py` and matching target test. Keep
  class/instance names; import the variant from the target. For verify, copy
  both shared files. Leave shared files untouched until promote. A new ops
  category needs a copyright-header `__init__.py`.
- `__call__` makes one kernel-launching call: prefer `torch.ops.trtllm.*`,
  otherwise an authorized FlashInfer public callable. Mirror schema argument
  names and return what the call writes, including in-place/out variants.
  Import FlashInfer at module scope, name it in the docstring, and explicitly
  set `backend` where supported. Other Python callables need authorization.
- `reference` uses plain torch with fp32 accumulation, accepts the cell's
  named arguments, and calls neither the kernel nor any entry.
- `is_valid` guards metadata only, for **measured silent wrong answers**;
  do not duplicate the op's loud validation. Record preconditions, arch
  quirks, measurements and hard limits in `note`.
- If Python must compose several calls, report `MACRO` and its missing
  constituents/tiers instead of writing an entry. No torch fallback wrapper.

## Test and certify

- Parametrize `op.CELLS`; use `_validating.py`'s `certifying` fixture so the
  entry's own `__call__` and `reference` both run. Derive the comparison band;
  bands wider than torch's dtype-aware default need wrong-variant negative
  controls, with measured separation. Test each `is_valid` refusal with
  `validating(op)` and `pytest.raises` after demonstrating the unguarded error.
  Unsafe probes run in a subprocess or are recorded only in `note`.
- Use real stateful metadata/caches. **Pool slots are a `spec` axis sized from
  the cache manager, not batch size**; include slots != batch. Collectives use
  the `comm/` pattern: every rank asserts, a deadline, world size as an axis.
- Assert CUDA availability (no skip), seed randomness, import the entry by
  its absolute path, and share no fixture with another entry's test.
- A normal cell fits **10 s / 40 GiB**. Boundary-sized probes use
  `spec.large=True` and `pytest.mark.large`; pre-merge excludes them. Report a
  missing marker registration rather than editing `tests/unittest/pytest.ini`.
- For index arguments, compute 2^31-element, 2^31/row-width and fp32 2^24+1
  boundaries. Probe those the runtime envelope reaches; record all limits,
  guarding measured silent failures. Size from measured `runtime_facts.yaml`.
- Iterate with `pytest -k`; certify once after the last edit to entry **and**
  test: run the whole test file and `test_modeling_v2_claims.py` /
  `test_modeling_v2_no_stale_claims.py` under `tests/unittest/_torch/modeling_v2`.
  Batch compatible cells into one pytest process. Add the measured device to
  `ARCHS` only after passing; a missing `Arch` enum gets its own commit.
  Unmeasured cells/arches remain uncertified; report failures, never widen a
  band or claim coverage to hide them.

## Promote

Merge the named variants into the shared entry in one commit:

1. Keep a common `__call__`/`reference` and update all callers together; block
   if no common form works.
2. Union cells (deduplicate `spec`, preserve `large`), arches, guards and
   measured notes. The shared test covers every spec key and retains each
   cell's band, choosing the tighter band for duplicates.
3. Switch all variant callers to the catalog; remove variants and their tests.
4. Run merged tests and claims gates together. A previously passing cell
   failing after merge is a conflict to report, not a change to land.

Report entry/op/tier, committed `ARCHS`/`CELLS`, hard limits, certifying
command and wall time, and **what remains uncertified**.
