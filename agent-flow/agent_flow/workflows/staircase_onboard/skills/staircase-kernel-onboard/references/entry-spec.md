# Entry spec — what a catalog entry is made of

Shared by `staircase-kernel-onboard` (building a new entry) and
`staircase-kernel-verify` (extending an existing one's coverage). Both write
the same three pieces under the same discipline; only the starting point
differs.

## Atomicity — what one entry is

One entry = one op invocation at the Python level.

- The wrapper body makes exactly one kernel-launching call into trtllm. Fusion is legal only inside that call: a fused kernel invoked once (silu-and-mul, allreduce+residual+rmsnorm) is one compliant entry.
- Python-level composition is forbidden. A wrapper that chains split → norm → matmul, or bundles qkv-proj + RoPE + SDPA + o-proj into one function, is not an entry — it hides exactly the bugs the catalog exists to expose.
- If inspection shows the requested op cannot be wrapped as one such call — its body is Python-level composition — it is a macro. Onboard nothing (no entry files, no index change); produce a decomposition report at `references/decompositions/<op>.md` instead, following the checklist in `references/decomposition.md`, "Writing a decomposition report for a new macro". Never onboard two entries in one run. <!-- PORT: verify — report location; see the marker in this skill's SKILL.md, "The macro path". -->

Category placement: by compute role, not dtype (an nvfp4 gemm lives in `gemm/`); fused ops by dominant cost (allreduce+norm lives in `comm/`). Categories: `gemm attention moe norm activation quantization comm ssm`. `torch/` is reserved for hand-added torch mirrors — never write there, even when the honest verdict is that a torch callable serves the step; report that and let the orchestrator place it, because a computation mirror widens what the surface is for. If no category fits, stop and report.

## The three pieces

An entry spans two trees. The contract and the wrapper live in
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/`; the GPU
test lives in `tests/unittest/_torch/modeling_v2/<category>/`, because that is
the tree this repo's CI collects from and an in-package test is on no list.
All three share one snake_case entry name. Name for the compute role; mirror
the trtllm op name when it is already descriptive.

### `<name>.md` — contract

The entry's sole interface document. Whoever calls this op reads the contract and nothing else — no trtllm source — so every fact needed to call it correctly must be in here. When mentioning sibling or variant ops, state that they exist — never whether they are onboarded; catalog membership is index.yaml's fact alone. YAML frontmatter, then sections:

```markdown
---
receipts:
  sm_<arch>: {status: passed, tests: <cases the run covered>}
---

# <name>

**Wraps** `torch.ops.trtllm.<op>` (one call).

## Semantics
What the single call computes, precisely — math where it disambiguates.
State the fusion boundary: what happens inside the call, what the caller
still owns (e.g. "rope is applied inside; qkv projection is not").

## Signature
The wrapper signature, plus a table of every argument and return:
shape, dtype, layout/contiguity, device.

## Metadata consumed
Every runtime/attention-metadata field the op reads and what it must
contain. "None" if stateless.

## Preconditions
Everything that must hold before the call: allocated buffers, prepared
metadata, weight formats, alignment, supported dtypes. A caller
violating none of these must get a correct result.

## Notes
Arch quirks observed in source or during testing.
```

### `<name>.py` — wrapper

Mirror the style of the `catalog/torch/` entries: module docstring, typed signature, short function docstring.

- Exactly one trtllm op invocation in the body. Glue that launches no kernel (e.g. building an `out` tensor for out-variant ops) is acceptable; tensor math is not.
- Guard asserts are allowed only when both hold: (a) the check is pure metadata (shape/stride/dtype/device), and (b) the violation was observed on this machine to fail silently (wrong results, silently ignored args) rather than raise. If the op validates its own inputs loudly, the wrapper stays assert-free.
- The default vocabulary is `torch.ops.trtllm.*`. Wrapping any other Python callable (e.g. a pybind binding) is allowed only when the run's request explicitly authorizes it; record the actual import path in the wrapper docstring, the contract, and the index `impl` field.
- The signature exposes the op honestly — no hidden defaults a contract reader would not expect.
- A first-call arch assertion is allowed, not required.

### `test_modeling_v2_<name>.py` — GPU test

At `tests/unittest/_torch/modeling_v2/<category>/`, under that name always: it is what keeps the file from colliding with upstream's test of the same op, and what the CI selector matches. `test_*` functions, importing the wrapper by its absolute package path — no fixtures shared with other entries, and no imports of another catalog entry.

- Assert `torch.cuda.is_available()` up front. No skip paths — a test that cannot run records no receipt.
- Build the reference on the spot from native torch ops (fp32 accumulation where the kernel accumulates in higher precision). Never build it from the wrapper, the op under test, or other catalog entries.
- Cover the shapes and dtypes the contract claims — include decode-like (few tokens) and prefill-like (many tokens) shapes where relevant. Seed all randomness.
- **Size persistent state — KV pages, recurrent and conv pools, any paged table — from the CACHE MANAGER, never from your batch.** These are two unrelated numbers. The batch is what the executor hands you; a pool's slot count is what the cache manager's capacity planning allocated, and the runtime picks it without reference to how many sequences are in flight. A test where `pool_slots == batch_size` is the one arrangement the runtime never produces, and it is worse than under-coverage: when the two are equal, "indexes the slot it was told to" and "indexes the row it happens to be" are the *same number*, so the test cannot see a slot-addressing defect at all. Take the width from a runtime-value dump if you have one and say so in the receipt; otherwise take one plainly too large for the batch and record that you chose it. **State the slot count in the certified column** — it is a column, not a free axis. One run certified nine entries this way, each sized to its own batch by a different smith over two days, and an audit found all nine outside the engine's real width of 461 at once.
- Compare with `torch.testing.assert_close` at default dtype-aware tolerances. Any loosening carries an inline comment justifying the number. Where the loosening is more than marginal — a chain whose intermediate is requantized, say — a justification in prose is not enough: **measure the discrimination.** Run the wrong variants too (swapped operand halves, un-swizzled scales, an omitted permutation) and record how far outside the loosened gate each lands. A gate 3x above the correct result and 25-500x below every wrong one is defensible; the same gate with no wrong-variant numbers is a guess.

<!-- PORT: verify — "Assert `torch.cuda.is_available()` up front. No skip
     paths" is what the in-tree entries do (a module-level `assert`), but these
     files are now collected by this repo's CI, where the house style for a
     GPU-only test is a skip marker. Confirm that a module-level assert is
     acceptable to the collectors these files run under, or the whole
     modeling_v2 directory errors on a CPU-only stage. -->

**When the surface you are certifying neighbours a known intermittent defect, prove the harness can see that defect before reporting a clean result.** Discrimination above answers "is my gate too loose"; this answers the different question "is my harness blind", and a clean sweep from a blind harness is worth nothing. Replay the documented arming sequence for the known defect in the same processes, confirm it fires, and only then report that the certified path ran through that state correctly. One run certified a workspace-backed collective this way — the neighbouring fused variant was measured wrong in 4 of 6 fresh processes by the same harness, and every call on the certified path that ran after it fired was still bitwise correct. Without the control, "no mismatches" would have been indistinguishable from "not looking".

Proven patterns — apply where the op's shape calls for them, not mechanically: for in-place ops, lock both sides (out-of-bounds write check via buffer snapshot; bitwise assertions on regions contracted as untouched); for quantized outputs, dual assertion (tolerance derived from the format's constants plus a bit-exact-fraction hard gate); for domains the contract declares unsupported, negative tests asserting rejection.

Stateful ops (the attention family) are tested the same way with minimal REAL state: construct the actual metadata and cache objects trtllm needs, never mocks of the op path. If minimal real state cannot be constructed in isolation, that is a symptom to report, not a license to mock.

Collective ops (`comm/`) cannot be exercised in one process, so the test spawns `N` ranks itself — `torch.multiprocessing`, world size read from `CUDA_VISIBLE_DEVICES` — and the entry keeps its single runnable command. Three things this shape gets wrong easily: the reference must stay arithmetic (the expected value of an all-reduce is computed, never obtained from a second collective); every rank asserts, and a failing rank has to make the command exit non-zero rather than print into a lost stream; and the run needs a deadline, because a broken collective **hangs** rather than raising — a test that can wedge forever records no receipt and blocks the run that dispatched you. Any workspace or communicator the op requires is real state under the rule above.

## Receipts

Receipts live in the contract frontmatter and are test outcomes, not claims:

- `sm_100: {status: passed, tests: 12}` — a filled-in example: the test passed on that arch, over that many cases. Receipts are keyed by **GPU architecture and nothing else**; there is no version pin in tree, because a target moves with the trunk.
- `status: failed` plus a short `note:` — wrapper and test are correct and the op demonstrably fails there. This is valuable knowledge; land the entry anyway.
- Key absent = unknown. Never pre-fill archs you did not run.
- A multi-rank entry adds `world_size: <N>` — the rank count the test actually ran at. Absent means 1, which is what every existing receipt asserts. Without it a four-rank receipt is indistinguishable from a single-GPU one, and rank count is exactly the axis such an op is uncertified across.

Derive the arch key from the device the test actually ran on (`torch.cuda.get_device_capability()`, e.g. `(10, 0)` → `sm_100`). **A receipt is valid only against the files of its own entry**: it must post-date the last write to every one of them, and an entry spans two trees, so the check looks in both. A write to any of the three makes the receipt stale — never rewrite a receipt without rerunning the test.

## index.yaml

Append to `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.yaml`,
matching the existing shape:

```yaml
  - path: norm/rms_norm.py
    impl: torch.ops.trtllm.<op>
    summary: "One-line compute-role summary"
```
