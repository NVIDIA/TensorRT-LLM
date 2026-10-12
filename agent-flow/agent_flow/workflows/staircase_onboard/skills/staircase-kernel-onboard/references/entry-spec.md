# Entry spec — what a catalog entry is made of

Shared by `staircase-kernel-onboard` (building a new entry) and
`staircase-kernel-verify` (extending an existing one's coverage). Both write
the same two pieces — the entry and its test — under the same discipline; only
the starting point differs.

## Atomicity — what one entry is

One entry = one op invocation at the Python level.

- The wrapper body makes exactly one kernel-launching call into trtllm — or, at sourcing tier 3, into FlashInfer's public API. Fusion is legal only inside that call: a fused kernel invoked once (silu-and-mul, allreduce+residual+rmsnorm) is one compliant entry.
- Python-level composition is forbidden. A wrapper that chains split → norm → matmul, or bundles qkv-proj + RoPE + SDPA + o-proj into one function, is not an entry — it hides exactly the bugs the catalog exists to expose.
- If inspection shows the requested op cannot be wrapped as one such call — its body is Python-level composition — it is a macro. Onboard nothing (no entry files, no index change); produce a decomposition report at `references/decompositions/<op>.md` instead, following the checklist in `references/decomposition.md`, "Writing a decomposition report for a new macro". Never onboard two entries in one run. <!-- PORT: verify — report location; see the marker in this skill's SKILL.md, "The macro path". -->

Category placement: by compute role, not dtype (an nvfp4 gemm lives in `gemm/`); fused ops by dominant cost (allreduce+norm lives in `comm/`). Categories: `gemm attention moe norm activation quantization comm ssm`, the same set in the shared catalog and in a target's `ops/`. **Native torch calls are not entries**: a target calls torch directly, and there is no `torch/` category. When the honest verdict is that plain torch serves the step, report that verdict and write no entry. If no category fits, stop and report.

## Where an entry goes

Two homes, decided by who calls the op:

| who calls it | wrapper | GPU test |
|---|---|---|
| more than one target, or an op of general use | `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.py` | `tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py` |
| only the target this run builds | `tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/ops/<category>/<name>.py` | `tests/unittest/_torch/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/<category>/test_modeling_v2_<name>.py` |

Your dispatch prompt's target says which: `ops/<category>/<name>` is the
run's own target's `ops/`, anything else is the shared catalog. A target
imports its own `ops/` relatively (`from .ops.moe.<name> import <Class>`) and
the shared catalog absolutely; it never imports another target's `ops/`. An op
a second target needs is **promoted**: the entry and its test move to the
shared homes in one commit, and the cells are driven again on this machine.
Each new `ops/<category>/` directory carries an `__init__.py` with the
copyright header, like its siblings.

## The pieces

An entry is **one Python file and its test**. There is no contract `.md`: the
`OpWrapper` subclass in the wrapper file *is* the contract, in executable form
(`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/_op.py`):

| member | what it states |
|---|---|
| `raw_call(...)` | the op call a target makes — exactly one kernel-launching call, every argument explicit |
| `reference(...)` | what the op is supposed to compute, in plain torch, accumulating in fp32 |
| `is_valid(...)` | raises on an input the op **accepts and answers wrongly**; never called on the way to the kernel |
| `compare(out, ref)` | how close the two must be, and why (default: torch's dtype-aware band) |
| `CELLS` | the input configurations the entry is certified over, one `Cell(why=..., spec=...)` each |
| `ARCHS` | the architectures those cells were driven on (`frozenset({Arch.SM_103})`) |
| `note` | what is left, and nothing that could have been one of the above |

The test is what keeps every member true: it drives each cell through
`raw_call` and `reference` and gates them with `compare`, so a claim that stops
holding stops the build. Whoever calls the op reads the class and nothing
else — no trtllm source — so every fact needed to call it correctly is in it.
When mentioning sibling or variant ops, state that they exist — never whether
they are onboarded; catalog membership is `index.yaml`'s fact alone.

### `<name>.py` — the entry

Follow the shape of the existing entries (`catalog/norm/flashinfer_rmsnorm.py`
is the smallest): SPDX header, a module docstring, the class, and a
module-level instance only if a target needs a shared one (targets normally
instantiate their own — instances are stateful).

```python
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class <Name>(OpWrapper):
    """What one call computes, and its fusion boundary: what happens inside the
    call, and what the caller still owns ("rope is applied inside; the qkv
    projection is not")."""

    ARCHS = frozenset({Arch.SM_<arch>})
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode-shaped, <the call site it covers>", spec=dict(...)),
        Cell(why="prefill-shaped: crosses the kernel's grid tiling", spec=dict(...)),
    )
    note = """
    Metadata consumed, preconditions, arch quirks, and the hard limits.
    """

    def raw_call(self, x: torch.Tensor, ...) -> torch.Tensor:
        return torch.ops.trtllm.<op>(x, ...)

    def reference(self, x: torch.Tensor, ...) -> torch.Tensor:
        ...
```

- **`raw_call`: exactly one op invocation — a `torch.ops.trtllm.*` op, or at
  tier 3 one FlashInfer public callable — and it reads nothing off `self`.** Glue that launches no kernel is acceptable; tensor math is not. A
  gate in `tests/unittest/_torch/modeling_v2/test_modeling_v2_claims.py`
  fails on any `self.` read inside it: the binding layer (`bind_const`,
  `bind_layered`) merges bound values in `__call__` before `raw_call` runs, so
  `CELLS` keeps covering the whole input space whatever a target binds.
- **Return what you write.** `out = op(inp)`: an out-variant or in-place op
  still returns the tensor it filled, so a call site reads its outputs from
  the call rather than from an argument it passed in.
- **Mirror the op's schema argument for argument**, by its own parameter
  names. A positional list that drifts silently is how an op-schema change
  turns into a wrong answer; a mirrored signature fails loudly instead.
- `reference` takes the arguments a cell supplies, by the same names as
  `raw_call`; it may narrow to the ones the computation needs (say which are
  ignored in its docstring). It never calls the op, the entry, or another
  entry. A collective's reference takes the group's inputs in rank order and
  says so.
- `is_valid`, when overridden, accepts every argument `raw_call` accepts —
  mirror the signature, or carry a `**kwargs` absorber; the claims test checks
  it. Guard only what is pure metadata (shape, stride, dtype, device) **and**
  was observed on this machine to fail silently. If the op validates its own
  inputs loudly, restating that only swaps its error for a worse one.
- `compare` is overridden only with a derivation: name the dtype and the
  accumulation order a wider band accounts for, and what was measured.
  `assert_within_ulp` in `_op.py` is the gate for a reference that differs
  only in accumulation order.
- The default vocabulary is `torch.ops.trtllm.*`. **FlashInfer's public API
  is the second**, and needs no authorization: `flashinfer-python` is pinned in
  the engine's `requirements.txt`, and most of its kernels have no trtllm op.
  Call it as `flashinfer.<module>.<name>(...)` with a module-scope
  `import flashinfer` (the catalog package imports nothing, so only a target
  that uses the entry pays for it), name that full path in the class docstring
  and the index `impl` field, and pass `backend=` explicitly when the callable
  takes one: `auto` resolves per architecture, so the cells certify the backend
  that ran, and the `note` says which. FlashInfer JIT-compiles many kernels on
  the first call; the test's first call includes it. When a trtllm op wraps the
  same kernel, wrap the op instead — it carries the engine's PDL and workspace
  handling.
- Wrapping any other Python callable (a pybind binding, say) is allowed only
  when the run's request explicitly authorizes it; record the import path in
  the class docstring and the index `impl` field.
- If the device's architecture is not yet an `Arch` member, add it to
  `catalog/_op.py` in a commit of its own containing only that line — every
  entry's `ARCHS` is checked against that enum — and say so in your report.

The `note` carries what the members cannot: the metadata the op consumes
(every runtime or attention-metadata field it reads, or "none"), the
preconditions a caller must establish (allocated buffers, prepared metadata,
weight formats, alignment), arch quirks, and the hard limits:

```text
Hard limits
  <argument>  <arithmetic>  boundary <b>  probed at <v>  <result>
```

The integer-width boundary of every index-typed argument (2^31 elements, 2^31
/ row width rows, 2^24 + 1 through fp32), probed and recorded whether or not
today's envelope reaches it. Recipe and the full row format: staircase-kernel-verify
`references/integer-width-probes.md` (section `## Hard limits`). A boundary
the op crosses *silently* is also an `is_valid` guard, with a test proving it
refuses.

### `test_modeling_v2_<name>.py` — the GPU test

At the test home in the table above, under that name always: it is what
keeps the file from colliding with upstream's test of the same op, and what
the CI selector matches. A model-local entry's test sits in the tests tree's
mirror of its target, one directory per category.

```python
from tensorrt_llm._torch._experimental.modeling_v2.catalog.<category>.<name> import <Name>

# ".." for a shared entry's test; "../../../.." under models/<family>/<target>/<category>/
__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

op = <Name>()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=...)
    op.compare(op(*args), op.reference(*args))
```

- **Parametrize off `op.CELLS`.** A cell list nothing executes reads as
  coverage; the claims test and delivery both refuse a test that never reads
  `.CELLS`. Turning a `spec` into real tensors is the test's job, not the
  entry's.
- **Arm `certifying`** (`tests/unittest/_torch/modeling_v2/_validating.py`) in
  every cell test: it interposes `is_valid` and refuses to finish unless the
  entry's own `raw_call`, `reference` and `compare` ran. A test that compares
  against a local copy of the reference certifies nothing the entry claims.
- **Guard tests** use `with validating(op): ...` and `pytest.raises` for each
  input `is_valid` exists to refuse — after first showing, unguarded, that the
  op really does return a plausible wrong answer there.
- Import the entry by its absolute package path. No fixtures shared with other
  entries' tests, and no imports of another entry.
- Assert `torch.cuda.is_available()` up front. No skip paths — a test that
  cannot run certifies nothing.
- Cover decode-like (few tokens) and prefill-like (many tokens) shapes where
  relevant. Seed all randomness.
- **Size persistent state — KV pages, recurrent and conv pools, any paged table — from the CACHE MANAGER, never from your batch.** These are two unrelated numbers. The batch is what the executor hands you; a pool's slot count is what the cache manager's capacity planning allocated, and the runtime picks it without reference to how many sequences are in flight. A test where `pool_slots == batch_size` is the one arrangement the runtime never produces, and it is worse than under-coverage: when the two are equal, "indexes the slot it was told to" and "indexes the row it happens to be" are the *same number*, so the test cannot see a slot-addressing defect at all. Take the width from a runtime-value dump if you have one and say so in the cell's `why`; otherwise take one plainly too large for the batch and record that you chose it. **Make the slot count a `spec` axis** — it is a cell dimension, not a free one. One run certified nine entries this way, each sized to its own batch by a different smith over two days, and an audit found all nine outside the engine's real width of 461 at once.
- A band wider than `compare`'s default is the entry's `compare`, with its derivation. Where the loosening is more than marginal — a chain whose intermediate is requantized, say — a justification in prose is not enough: **measure the discrimination.** Run the wrong variants too (swapped operand halves, un-swizzled scales, an omitted permutation) and record how far outside the loosened gate each lands. A gate 3x above the correct result and 25-500x below every wrong one is defensible; the same gate with no wrong-variant numbers is a guess.

**When the surface you are certifying neighbours a known intermittent defect, prove the harness can see that defect before reporting a clean result.** Discrimination above answers "is my gate too loose"; this answers the different question "is my harness blind", and a clean sweep from a blind harness is worth nothing. Replay the documented arming sequence for the known defect in the same processes, confirm it fires, and only then report that the certified path ran through that state correctly. One run certified a workspace-backed collective this way — the neighbouring fused variant was measured wrong in 4 of 6 fresh processes by the same harness, and every call on the certified path that ran after it fired was still bitwise correct. Without the control, "no mismatches" would have been indistinguishable from "not looking".

Proven patterns — apply where the op's shape calls for them, not mechanically: for in-place ops, lock both sides (out-of-bounds write check via buffer snapshot; bitwise assertions on regions contracted as untouched); for quantized outputs, dual assertion (tolerance derived from the format's constants plus a bit-exact-fraction hard gate); for domains the entry declares unsupported, negative tests asserting rejection.

Stateful ops (the attention family) are tested the same way with minimal REAL state: construct the actual metadata and cache objects trtllm needs, never mocks of the op path. If minimal real state cannot be constructed in isolation, that is a symptom to report, not a license to mock.

Collective ops (`comm/`) cannot be exercised in one process. Follow the shape the in-tree collectives use: the collected `test_modeling_v2_<name>_op_matrix.py` only starts the ranks through `_rank_job` (both started by file path, because the launcher must not import `tensorrt_llm`), and the rank body `_<name>_op_matrix.py` beside it drives `op.CELLS`; the entry keeps its single runnable command, and the world size is a cell axis. Three things this shape gets wrong easily: the reference must stay arithmetic (the expected value of an all-reduce is computed, never obtained from a second collective); every rank asserts, and a failing rank has to make the command exit non-zero rather than print into a lost stream; and the run needs a deadline, because a broken collective **hangs** rather than raising — a test that can wedge forever certifies nothing and blocks the run that dispatched you. Any workspace or communicator the op requires is real state under the rule above.

## Certification

There is no receipt file and no frontmatter. An entry is certified by two of
its own members, and kept certified by CI:

- `CELLS` — the input configurations, each with a `why` that says what it
  covers beyond the others. A cell that cannot say that costs CI time for
  nothing. A value outside every cell is **not certified**: for most ops that
  means "nobody measured it", not "known broken".
- `ARCHS` — the architectures the cells were driven on. **Add an arch only
  after watching the whole test file pass on that device**, deriving the key
  from the device the test ran on (`torch.cuda.get_device_capability()`,
  `(10, 3)` → `Arch.SM_103`). An arch not in `ARCHS` is **unknown** — never
  pre-fill one you did not run. Architecture is the whole key: two
  architectures can compute bit-different results from one entry.
- A multi-rank entry makes the world size a cell axis (`spec=dict(world_size=4,
  ...)`) and its test runs at that rank count. Rank count is exactly the axis
  such an op is uncertified across.

**The op may fail at a value.** That is knowledge, and it lands: the failing
value is not a cell; if the op *accepts* it and answers wrongly, `is_valid`
refuses it and a guard test proves the refusal; either way the `note` says
what was measured (accepted / raised, and what it then does). Report the turn
as `FAILED_RECEIPT` when the value the call site needs is the one that failed.

**Freshness.** Once merged, the entry's test runs in pre-merge CI, so the trunk
re-proves it on every commit; there is no version pin to drift against. Until
then, the run you watched is the only evidence, and it is valid only if it
**post-dates the last write to every file of the entry** — the wrapper and the
test, which sit in two trees. Check it mechanically against the files' mtimes,
not by recollection. A write to either after the run makes it stale; never add
to `ARCHS` without rerunning the test.

## Index fragment

Only a shared catalog entry has one; `index.yaml` lists the shared vocabulary,
and a target's `ops/` entries are found by its imports. Write the row as its
own file,
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.d/<category>/<name>.yaml`,
and commit it with the entry:

```yaml
path: norm/rms_norm.py
impl: torch.ops.trtllm.<op>
summary: "One-line compute-role summary"
```

`catalog/index.yaml` is generated from these fragments by the orchestrator and
must not be edited by hand. When several smiths run at once, each commits only
its own fragment, so no commit carries another entry's row.
