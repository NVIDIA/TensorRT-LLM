"""Shared prose blocks for staircase-bringup prompt extensions.

Each ``*_extra.py`` imports the subset relevant to its role and composes
``SYSTEM_PROMPT_EXTENSION``.

**This module imports nothing from another workflow, by design.** The
Slurm/container blocks below started as a borrow from ``modeling_bringup`` and
are now copies: self-containment beats deduplication here, the same way a
staircase target duplicates its sibling rather than sharing a helper. Two
copies that drift are a smaller problem than one shared block whose owner
changes it for the other workflow's reasons. The borrow also had a concrete
cost — ``modeling_bringup.prompts._common`` runs a skill probe at module scope,
so importing it spawned one backend session per configured backend, measured
at 5.07s against 1.16s for the ``agent_team`` prompts alone.

``EVALUATION_CRITERIA`` lives in ``qa.py``, not here: it is the base QA
prompt's own structure, and staircase replaces that prompt wholesale.

Nothing in this module probes or spawns at import. The skill probe is deferred
behind :func:`get_staircase_skill_invocation`, which the bundle builder calls.
"""

from __future__ import annotations

from functools import lru_cache

from agent_flow.utils import resolve_first_available_skill

# ---------------------------------------------------------------------------
# Domain priming — what staircase is, for every role
# ---------------------------------------------------------------------------

DOMAIN_PRIMING = """\
## Staircase bring-up frame

A **staircase target** is one self-contained modeling codebase per
(checkpoint, GPU arch, parallel topology) triple, living beside the built-in
model zoo at `tensorrt_llm/_torch/staircase/models/<family>/targets/<ckpt>/
<arch>/<parallel>/` rather than inside it. Where
`_torch/models/modeling_deepseekv3.py` is one class serving V3, V3-Lite, R1
and V3.2 across every GPU generation and parallel topology,
`_torch/staircase/models/deepseek_v3/` is one flat forward per triple —
assembled only from `catalog/` entries, sharing nothing with its siblings,
and trusted through accuracy gates instead of shared abstractions.

- Treat HuggingFace as the **semantic source of truth**: what the model
  computes (norm placement, residual structure, where rope and qk-norm apply,
  activation) comes from the checkpoint's own HF reference modeling code,
  because that is the code the checkpoint was trained against. vLLM and the
  in-tree TensorRT-LLM models are useful implementation references. None of
  them is the shape of target code: targets are written in the fused catalog
  vocabulary, not as a torch-op transliteration of HF's forward.
- Before relying on a TensorRT-LLM class or function, verify whether it is a
  real definition, a wrapper, an import alias, or a binding. This bites here
  specifically: a catalog wrapper *is* a thin binding layer, so a contract
  author who follows the name instead of the definition documents the wrong
  object.
- **Resolution is environment-gated.** `TRTLLM_STAIRCASE` selects the path:
  `off` (default) ignores the package entirely, `auto` uses a target when one
  matches and silently falls back to the built-in implementation when none
  does, `require` raises instead of falling back.
  **Use `require` for anything you will attribute to staircase.** Under
  `auto`, a configuration that misses a target's criteria gets the built-in
  implementation and its numbers then read as staircase's — the single most
  expensive mistake available here.
  **Export it before the ranks start**, not merely before `LLM(...)`: worker
  ranks receive the environment as it stood when MPI initialized, so a value
  set later reaches the driver and not them, and a driver resolving a target
  while its workers resolve the built-in is the silent split this package
  exists to prevent.
- Two artifact kinds are produced. **Catalog entries** (contract `.md` +
  wrapper `.py` + GPU test, closing on a receipt) grow the shared vocabulary.
  The **target** itself (`modeling.py`, `weights.py`, `smoke.py`,
  `TARGET.md`, plus `configs/` when a variant exists) is the deliverable.
- Pass-critical evidence runs on GPU. A skipped, optional, or CPU-only run is
  missing evidence, not pass evidence.
"""

# ---------------------------------------------------------------------------
# The closed vocabulary — staircase's central rule
# ---------------------------------------------------------------------------

CLOSED_VOCABULARY_RULE = """\
## The closed-vocabulary rule

Inside a target's forward, **every call that creates or transforms a tensor
must be a catalog entry**; only tensor-metadata reads and Python control flow
live outside. This is what makes a target auditable without reading it
closely, and it is the rule most of the other policies exist to protect.

**Scope is the whole trick.** The audit collects the calls in `forward` **and
the private methods it reaches** — not the whole file. A whole-file scan
false-positives on the rope table's `arange` / `cos` / `sin` and on load-time
`.t()` / `.to()`, which run at init and sit outside the rule; those false
positives then obscure a real violation.

Two escape hatches, with a hard line between them:

- `catalog/torch/` mirrors are **glue**: tensor layout, movement, allocation,
  and lookup (reshape, split, concat, `copy_`, `empty`, `expand`, embedding,
  residual add). A missing mirror can be added inline — a thin wrapper in the
  existing style plus its `index.yaml` entry. Torch mirrors carry no
  contract, no test, and no receipts by design; upstream owns their
  correctness.
- A **computation** the HF reference expresses as a named module or function
  (norm, rope, activation, linear, attention, quantization) is never composed
  from torch mirrors. A missing kernel there is a **vocabulary gap**: the
  fused entry has to be onboarded. Composing it from mirrors is exactly the
  torch-op transliteration the catalog exists to prevent, and it passes the
  gates while defeating their purpose.

Values matter as much as calls. Every argument the forward passes must land
inside the entry contract's **certified column** — and that includes the
shapes the engine varies at runtime, the decode batch sizes it captures and
the sequence lengths it serves. Those appear nowhere in the target, so
reading the forward does not surface them. Where a contract enumerates tested
shapes rather than stating a rule, that enumeration is a column too.
"""

# ---------------------------------------------------------------------------
# Required mechanisms — the outcome-bound escape hatch
# ---------------------------------------------------------------------------

PROJECT_REQUIRED_MECHANISMS = """\
## Project-level required mechanisms (treated as outcomes)

Acceptance criteria are normally outcome-bound, not means-bound: a criterion
naming a library, scheme, file path, function name, or backend choice that
`task.yaml` did not ask for is a leaked plan prescription and gets rephrased
as the underlying outcome.

Staircase carries a fixed set of mechanism names that **every** checklist must
prove regardless of how `task.yaml` is phrased, because they are the
project's contract rather than one plan's preference:

- **Closed vocabulary** — every tensor-creating call in the forward, and in
  the private methods it reaches, maps to an entry in `catalog/index.yaml`.
- **Receipt status and freshness** — each new or extended entry carries a
  receipt from an observed run that post-dates every file in the entry.
- **Certified column** — every argument value *and* every runtime-varied
  shape lands inside the contract's certified column.
- **Gate margin** — `measured >= reference - tol` on the accuracy gate,
  measured under `TRTLLM_STAIRCASE=require`.
- **Identity triple** — the (checkpoint, arch, parallel) triple the target
  actually runs matches the one its directory path claims.

Encoding these in `acceptance-criteria.md` is **not** a leaked prescription,
and a criterion must not be rejected for naming them even when `task.yaml`
does not. The no-leaked-prescriptions rule still applies to helper names,
file paths, function signatures, and other knobs the user did not ask for.
"""

# ---------------------------------------------------------------------------
# Catalog entry spec — the three pieces
# ---------------------------------------------------------------------------

CATALOG_ENTRY_SPEC = """\
## Catalog entry spec — the three pieces

One entry = **one op invocation at the Python level**. The wrapper body makes
exactly one kernel-launching call into trtllm. Fusion is legal only inside
that call: a fused kernel invoked once (silu-and-mul, allreduce+residual+
rmsnorm) is one compliant entry. Python-level composition is forbidden — a
wrapper that chains split -> norm -> matmul, or bundles qkv-proj + RoPE +
SDPA + o-proj into one function, is not an entry; it hides exactly the bugs
the catalog exists to expose.

If inspection shows the requested op cannot be wrapped as one such call, it is
a **macro**: onboard nothing and produce a decomposition report instead,
listing every constituent atomic op in call order. Never onboard two entries
under one Goal.

Category placement is by **compute role, not dtype** (an nvfp4 gemm lives in
`gemm/`); fused ops go by dominant cost (allreduce+norm lives in `comm/`).
Categories: `gemm attention moe norm activation quantization comm ssm`.
`torch/` is reserved for hand-added torch mirrors.

### `<name>.md` — contract

The entry's sole interface document: whoever calls this op reads the contract
and nothing else, so every fact needed to call it correctly must be in here.
Sections: **Semantics** (what the single call computes, and the fusion
boundary — what happens inside, what the caller still owns), **Signature**
(the wrapper signature plus a table of every argument and return: shape,
dtype, layout/contiguity, device), **Metadata consumed** (every runtime or
attention-metadata field the op reads and what it must contain),
**Preconditions** (everything that must hold before the call; a caller
violating none of these must get a correct result), **Notes** (version or
arch quirks observed).

**Write the path the receipt certifies, not the path the source describes.**
The one failure mode that survives review is a contract sentence that is true
of *some* execution path, just not the one the test drove. Real cases from
this catalog: a norm Preconditions section describing the CUDA-JIT path while
the receipt certified CuTe DSL; an fp8 `kv_scale_*` rule transcribed from an
upstream comment about XQA kernels, on an arch whose fp8 decode runs
trtllm-gen JIT; a "tactic choice does not change results" claim, true of the
cold autotuner the test runs in and false of the warm one serving runs in;
and a "mixed batches are rejected" claim that is a check in the Python
backend which the op in question bypasses. Before writing a Precondition,
name the path you measured — backend, kernel family, cache state — and say so
if it is not the only one.

**A claim that the op rejects something is the expensive kind to get wrong**,
because it promises a guard the caller then does not write. Every "raises" /
"is rejected" / "not survivable" sentence needs a probe that actually saw the
raise, in the certified configuration, with the error text quoted. Absent
that, write what you measured: accepted, and what it then does.

### `<name>.py` — wrapper

Module docstring, typed signature, short function docstring.

- Exactly one trtllm op invocation in the body. Glue that launches no kernel
  (building an `out` tensor for out-variant ops) is acceptable; tensor math
  is not.
- Guard asserts are allowed only when **both** hold: the check is pure
  metadata (shape / stride / dtype / device), **and** the violation was
  observed on this machine to fail silently — wrong results, silently ignored
  args — rather than raise. If the op validates its own inputs loudly, the
  wrapper stays assert-free.
- The signature exposes the op honestly: no hidden defaults a contract reader
  would not expect.

### `<name>_test.py` — GPU test

- Assert CUDA availability up front. **No skip paths** — a test that cannot
  run records no receipt.
- **Build the reference on the spot from native torch ops** (fp32
  accumulation where the kernel accumulates in higher precision). Never build
  it from the wrapper, the op under test, or another catalog entry, and never
  share a helper with the implementation: a reference correlated with the
  thing it checks lets a buggy implementation produce matching output.
- Cover the shapes and dtypes the contract claims, including decode-like (few
  tokens) and prefill-like (many tokens) where relevant. Seed all randomness.
- Compare at default dtype-aware tolerances. **Any loosening carries measured
  discrimination**, not prose: run the wrong variants too (swapped operand
  halves, un-swizzled scales, an omitted permutation) and record how far
  outside the loosened gate each lands. A gate 3x above the correct result
  and 25-500x below every wrong one is defensible; the same gate with no
  wrong-variant numbers is a guess.
- **Prove the harness can see a defect before reporting a clean result** when
  the surface neighbours a known intermittent one. Discrimination answers "is
  my gate too loose"; this answers the different question "is my harness
  blind", and a clean sweep from a blind harness is worth nothing. One run
  certified a workspace-backed collective this way: the neighbouring fused
  variant was measured wrong in 4 of 6 fresh processes by the same harness,
  and every call on the certified path after it fired was still bitwise
  correct. Without the control, "no mismatches" would have been
  indistinguishable from "not looking".
- **Stateful ops** (the attention family) are tested with minimal **real**
  state: construct the actual metadata and cache objects trtllm needs, never
  mocks of the op path. If minimal real state cannot be constructed in
  isolation, that is a symptom to report, not a license to mock.
- **Collective ops** (`comm/`) cannot be exercised in one process, so the
  test spawns `N` ranks itself and the entry keeps its single runnable
  command. Three things this shape gets wrong easily: the reference must stay
  arithmetic (the expected value of an all-reduce is computed, never obtained
  from a second collective); every rank asserts, and a failing rank has to
  make the command exit non-zero rather than print into a lost stream; and
  the run needs a **deadline**, because a broken collective *hangs* rather
  than raising, and a test that can wedge forever records no receipt and
  blocks everything behind it.
"""

# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------

RECEIPT_POLICY = """\
## Receipts — test outcomes, not claims

Receipts live in the contract frontmatter and are three-state:

- `sm_100: {status: passed, trtllm: 1.3.0rc21}` — the test passed on that arch
  under that installed version.
- `status: failed` plus a short `note:` — wrapper and test are correct and the
  op demonstrably fails there. **This is valuable knowledge; land the entry
  anyway.**
- Key absent = unknown. Never pre-fill an arch you did not run, and never
  infer one from plausibility.

A multi-rank entry adds `world_size: <N>` — the rank count the test actually
ran at. Absent means 1. Without it a four-rank receipt is indistinguishable
from a single-GPU one, and rank count is exactly the axis such an op is
uncertified across.

Derive the arch key from the device the test actually ran on
(`torch.cuda.get_device_capability()`, e.g. `(10, 0)` -> `sm_100`); read the
trtllm version from the installed package, not from memory. A version bump
makes old receipts visibly stale — never rewrite a receipt's version without
rerunning the test.

**A receipt is only valid if it post-dates the last write to every file in the
entry.** Recording the receipt, syncing the index, and linting all write to
the entry after the test last ran, and `ruff format` rewrites the test file
itself. So re-run the full test file as the *last* action and check the claim
mechanically against file mtimes rather than by recollection: a receipt that
predates its own test file has shipped here before, and it read as green.
"""

# ---------------------------------------------------------------------------
# Target products
# ---------------------------------------------------------------------------

TARGET_PRODUCT_SPEC = """\
## Target products

Under `_torch/staircase/models/<family>/targets/<ckpt>/<arch>/<parallel>/`:

1. **`modeling.py`** — an import-time static contract (the trtllm version pin
   plus every `torch.ops.trtllm.*` symbol and binding the forward calls; the
   list must **equal** the actual call set, not merely cover it); geometry
   read from `pretrained_config` with an assert per axis the assembly depends
   on; a flat `ParameterDict` declaration (`torch.empty` only, HF `[out, in]`
   storage); the step-args builder with its `_STEP_FIELDS` existence check and
   per-entry CUDA-graph classification; per-target call constants taken from
   the entry contract's certified column; **a single flat forward serving any
   batch composition**; loud asserts for inputs the target does not implement.
2. **`weights.py`** — the MANIFEST data table and load loop with
   **bidirectional** coverage asserts (every checkpoint key consumed, every
   declared parameter filled). Post-load derivations stay in the
   `post_load_weights` path.
3. **`smoke.py`** — self-locating, runnable, binary verdict. Author prompts
   whose greedy continuation is high-confidence for *this* model and **verify
   every keyword against the real model before freezing it**. A frozen case
   that later fails is a regression to investigate, not a case to relax.
4. **`TARGET.md`** — sections: Identity (triple, checkpoint hashes, host:path
   binding), Version (pins), Vocabulary (the forward's call set plus the audit
   statement), Verification (gate records with measured numbers and configs).
   A fifth section, Performance, belongs to a later perf campaign: leave it
   **absent**, not stubbed.
5. **`configs/`** — only when the target needs one. It holds three kinds of
   thing and the distinction matters: tuning variants (knobs), boot configs
   (for a reference the stock engine cannot otherwise start), and
   **capability variants** that select a second forward path. Speculative
   decoding is the third kind — it also changes the weight-loading path, so
   it is not "just a knob", it must say so in TARGET.md, and it carries its
   own gate records.

Registered class names are fixed fleet-wide; identity lives in the directory
path, never in a class name or a config key.

**Self-containment is the architecture.** A helper shared with a sibling
target is not a cleanup, it is the coupling this layout exists to remove.
Duplicate instead.
"""

# ---------------------------------------------------------------------------
# The gate ladder
# ---------------------------------------------------------------------------

GATE_LADDER = """\
## The gate ladder — cheapest signal first

Run the cheapest tier that covers your change, and never start an expensive
tier while a cheaper one is failing:

1. **Static** — format, import order, the bare lint, type check, and an
   import dry-run that loads the modeling file and asserts registration.
   Seconds; catches the whole class of "does not even load".
2. **Catalog entry test** — the touched entry's GPU test. Seconds to minutes.
3. **Smoke** — `smoke.py`: engine cold start, weight-manifest coverage
   asserts, greedy keyword checks. Binary. Catches catastrophes, not
   accuracy. A target whose `configs/` holds a variant that changes the
   forward gives smoke a `--config` so that path has a seconds-scale gate
   too.
4. **Module parity** — the reference ladder's rungs (below), on a handful of
   fixed prompts. Minutes.
5. **Accuracy canary** — a small deterministic slice of the configured
   benchmark, same protocol, run before the full gate. Minutes. This is the
   cheap catastrophic-regression check, not a substitute for the gate.
6. **Accuracy gate** — the release criterion: `measured >= reference - tol`,
   under the protocol the reference entry pins, with
   `TRTLLM_STAIRCASE=require`. About an hour.

**Bracket the failure surface.** When a constraint only manifests at scale or
after many steps, there must be a cheaper canary exercising the same failure
mode; the expensive gate must never be the only signal for it.

**A distribution-preserving feature needs a third kind of gate**, because the
first two are blind to it. Rejection sampling holds the emitted distribution
to the target model's, so a *miscomputed* speculative draft path produces
correct text more slowly — every draft rejected. Smoke passes, accuracy
passes, only speed moves. The detector is `acceptance_length` against a
reference at the same workload and the same speculative config: `1.0` means
total rejection, clearly above 1.0 but below the reference means subtly
wrong. The same reasoning applies to any feature whose output is
distribution-preserving by construction.
"""

# ---------------------------------------------------------------------------
# Reference construction
# ---------------------------------------------------------------------------

REFERENCE_LADDER = """\
## Build a verified reference ladder before claiming parity

Every parity comparison needs an already-trusted target, or it cannot
separate "my reference is wrong" from "my port is wrong". Build the rungs in
this order and pin each before the next:

1. **Native baseline.** Run the whole model end-to-end through the native HF
   interface (`AutoModelForCausalLM.from_pretrained`) on 3-5 fixed prompts.
   This is ground truth; capture its outputs.
2. **Pure-PyTorch module reference.** Implement each module as a local
   pure-PyTorch implementation and align its output to the native baseline on
   those prompts. Only once it matches is that module a trusted reference —
   this is what proves the reference itself is correct.
3. **Staircase parity.** Run the target's module against the verified
   pure-PyTorch implementation. Because rung 2 is already aligned to the
   native baseline, a remaining gap points at the target, not the reference.

Staircase has historically jumped from a keyword smoke test straight to a
14042-question benchmark, with nothing in between. Rungs 1 and 2 are that
missing middle, and they are minutes rather than an hour.

**When the HF reference must be driven by a hand-written prefill + greedy
loop** — because the checkpoint's on-disk modeling code does not run under the
pinned `transformers` — that loop is a reimplementation of generation: its
`position_ids`, attention mask, KV threading, and greedy tie-breaking can
silently diverge. And since the same loop usually produces both the reference
logits and the reference score, a delta-based check cannot catch a bug inside
the loop: both sides shift together. Anchor it first — generate the canonical
greedy token ids once with the source model's native `generate()` in a
throwaway environment whose `transformers` can actually run it, commit those
ids as a fixture, and assert the hand-written loop reproduces them
token-for-token. Never mutate the repo's pinned `transformers` to obtain a
reference, and never accept an in-process shim or monkeypatch as the golden
source.
"""

REFERENCE_INDEPENDENCE = """\
## Validation standard — validated, partially validated, not validated

Classify every high-risk contract:

- **`validated`** — checked against an **independent** reference, on the hard
  path, with the evidence produced by a command that ran.
- **`partially validated`** — evidence is indirect, correlated with the
  implementation (shares a helper with the reference, or is derived from it),
  or skips the hard path.
- **`not validated`** — no executed evidence at all.

Pass-critical contracts must be `validated`. **Partial validation is a
rejection, not partial credit** — name the missing reference or the skipped
path so it can be closed.

Passing tests prove agreement with *their reference*, not correctness. A
benchmark that passes proves agreement with itself. This is why the reference
ladder exists and why a catalog test may never build its expected value from
the op it is testing.
"""

# ---------------------------------------------------------------------------
# Accuracy debugging
# ---------------------------------------------------------------------------

ACCURACY_DEBUG_METHODOLOGY = """\
## Debugging an accuracy shortfall — stop using the full dataset as the debugger

A full-gate run costs about an hour and same-protocol rerun noise is 0.1-0.3
points, so a hypothesis-free rerun is expensive *and* information-free. Use
the small loop first:

1. **Export per-sample results** from both the reference path and the
   staircase path: prompt id and text, generated tokens, score labels,
   decoding config, and any logit or layer artifacts available.
2. **Pick the discriminating cases** — the samples where the reference is
   correct and staircase is wrong. Those drive the gap. Samples both paths
   get right, or both get wrong, carry little signal. Name the indices, so
   the comparison is over a handful of cases rather than the whole set.
3. **Compare under teacher forcing, not free-running generation.** Two
   free-running decodes fork at the first differing token, after which every
   downstream position has a different prefix and is no longer comparable.
   Instead drive both paths with the *reference's* tokens: at each step feed
   the reference's chosen token to both, so when their outputs disagree at a
   position both still continue on the reference token. With the prefix held
   identical, the per-step logits pin the exact step — and, with deeper
   replay, the layer or module — where staircase first diverges.
4. **Fix on that set**, then climb back up the ladder: module parity, then
   the canary, then the full gate.

Each re-gate is backed by **one explicit causal hypothesis**. After three
consecutive hypothesis-backed fixes without improvement, or when no new
hypothesis remains, stop and report the shortfall with the audit trail: the
per-axis re-derivation record and the remaining suspects (a protocol the
reference entry does not in fact match; entry precision, which is a
certification gap; a checkpoint peculiarity).

**The anchor and the tolerance are read-only.** A shortfall is first a debug
signal about the assembly, not a verdict about the bar. You fix the target,
never the bar.

Do not gate the *start* of dataset-accuracy testing on a perfectly tight
parity number. Parity max-abs can look loose purely from benign numerical
jitter (different kernels, accumulation order, dtype). Judge from parity
**and** dataset accuracy together: if accuracy lands where the model should
land, a loose-but-stable parity result is acceptable; if accuracy is low,
parity is the diagnostic that localizes where the divergence enters.
"""

SIGNAL_VS_NOISE = """\
## Signal versus noise

Same-protocol reruns of the accuracy gate move roughly **0.1-0.3 points** —
one target measured 94.7688 and 95.0720 on a bit-identical forward a day
apart. Three consequences:

- A sub-0.5-point difference is not a result. Say which side of the noise
  floor any number you cite sits on.
- A delta measured **across sessions** carries session variance into the
  judgement. Compare a variant against an identity run **in the same
  session**, or do not compare.
- Never call a construct load-bearing on a sub-floor difference. Downstream
  work takes the claim at face value and pays to disprove it.
"""

# ---------------------------------------------------------------------------
# Craft
# ---------------------------------------------------------------------------

LINT_RECIPE = """\
## Lint — all four commands, none redundant

```bash
ruff format <files>
ruff check --select I --fix <files>
ruff check <files>
ty check <files>
```

**Each catches what the others pass.** `ruff format` does not touch import
order, so an unsorted import block survives it *and* survives a clean
`ruff format --check` — only `--select I` reports it. And `--select I`
restricts ruff to *just* the import rules, so an unused variable or import
passes `format`, passes `--select I`, and passes `ty check`; only the bare
`ruff check` returns `F841`. A catalog entry reached review here with a dead
`F841` assignment because its author ran the two-ruff recipe.

Pass **file names**, not a directory: a sweep reformats entries you did not
touch, and directory sweeps have been observed to race on ignore rules and
reach files that are symlinks into shared checkpoints.
"""

STATUS_DONE_TODO_RUBRIC = """\
## `status.md` — the `Done / TODO` section

In addition to the rolling-state sections, `status.md` carries a
`## Done / TODO` section that both the Coder and the Reviewer keep current
when they call `update_status`. Use these exact headings:

```markdown
## Done / TODO

### Done
- <acceptance-criteria item or plan sub-task>: closed in iteration <n> — <evidence>

### TODO
- <acceptance-criteria item or plan sub-task>: <blocker, planned next step>
```

Rules:

- Every `- [ ]` line in `acceptance-criteria.md`, plus any high-risk sub-task
  `plan.md` calls out, appears in exactly one of `Done` or `TODO` **every
  turn**. No item silently disappears between iterations.
- An item moves to `Done` only with **executed evidence** observed this turn
  or a prior one: a command that ran, a test that passed, a receipt that
  landed. "Code written but not run" stays in `TODO`.
- Each `Done` row cites the iteration that closed it and a concrete evidence
  pointer. Each `TODO` row names the immediate blocker and the planned next
  move; "TBD" is not acceptable.
- The Coder drafts it each iteration by carrying the prior status forward;
  the Reviewer adjusts it post-review to reflect what was *actually*
  verified — demoting an item the Coder marked `Done` if the rerun failed.
"""

SELF_CONTAINMENT_POLICY = """\
## Design policy — self-containment, and no new entities

Two rules that pull in opposite directions from the usual advice, and both
matter here:

- **Do not add new schemas, runtime concepts, adapters, flags, or helper
  layers** unless existing abstractions cannot express the source semantics.
  Prefer narrowing or extending what exists. The catalog is a closed
  vocabulary; inventing a new entity to avoid onboarding an op is how that
  closure erodes.
- **Do not factor shared code across targets.** Self-containment *is* the
  architecture: each target duplicates rather than shares, and is trusted
  through its own gates instead of through an abstraction it has in common
  with a sibling. A shared helper "for future targets" is the coupling this
  layout exists to remove. The same applies between a target and the built-in
  model zoo — the one-to-one correspondence between a staircase target and
  its in-tree counterpart is the point of the exercise, and it only means
  something while the two are independent.

Do not accept local encoding tricks, padding tricks, fake config values, or
model-specific workarounds merely because focused tests or benchmarks pass.
Benchmarks validate behavior; they do not justify an architecturally rejected
direction.

**Layer separation.** Plan owns the architecture search space; implement
executes inside it, and may deviate from a specific *prescription* with
documented evidence when the criteria still hold; evaluate judges fit. An
architecture-level change invented outside any direction `plan.md`
enumerated exceeds the layer boundary — surface it as a blocker rather than
absorbing it as a routine deviation.
"""

# ---------------------------------------------------------------------------
# Slurm substrate — copied, not shared (see module docstring)
# ---------------------------------------------------------------------------

CONTAINER_BOOTSTRAP = """\
## Slurm container bootstrap for staircase bring-up

This task's `task.yaml` contains a `slurm-environment` section. Read these
fields from it and use them verbatim:

1. `slurm_partition` — the Slurm partition to submit GPU jobs to.
2. `docker_image` — the container image, typically an enroot/pyxis `.sqsh`
   image.
3. Top-level `trtllm_repo_path` — the TensorRT-LLM repo path on the Slurm
   host/login node. Treat this exact absolute path as the intended
   in-container path too: the Slurm job must bind-mount the host TensorRT-LLM
   checkout to the same path inside the container.

Do not invent a different partition, image, or repo path, and do not silently
fall back to a local non-Slurm run when `task.yaml` contains
`slurm-environment`. If any value is unusable at runtime, flag it as a
blocker for the human in the loop instead of guessing.

### Running a command: the checkout is what must be imported

Every command that exercises TensorRT-LLM runs inside the container, from the
mounted checkout, with that checkout ahead of the container's own installed
package:

```bash
cd <trtllm_repo_path>
export VENV=<trtllm_repo_path>/.venv-3.12
export PATH="$VENV/bin:$PATH"
export PYTHONPATH=<trtllm_repo_path>:${PYTHONPATH:-}
export TRTLLM_STAIRCASE=require
```

`PYTHONPATH` ahead of site-packages is what makes `import tensorrt_llm`
resolve to the checkout you edited rather than the image's copy — which is
the whole point, since your staircase code lives in the checkout. The venv
must be first on `PATH` because the multi-rank launcher calls a bare
`python3` for the leader and the workers.

There is no `pip install -e .` step and nothing to install into the
container: the image already carries the dependencies, and the checkout
carries the code.

### Rebuilding: required whenever you add or change a native op

**The checkout's `tensorrt_llm/libs/*.so` and `tensorrt_llm/*.so` are part of
what `PYTHONPATH` points at.** A catalog entry wraps `torch.ops.trtllm.<op>`,
and that symbol only exists if a built library in the checkout exports it. So:

- **Python-only change** (target `modeling.py` / `weights.py`, a wrapper whose
  op already exists) — no rebuild. `PYTHONPATH` picks it up immediately.
- **New or changed C++/CUDA/header/CMake** — rebuild before any run, or
  `import tensorrt_llm` raises before your test gets a chance to fail
  honestly. This checkout has already hit exactly that: a Python tree
  referencing `trtllm::silu_and_mul_fp8_quantize_1x128_packed_ue8m0` with no
  built library exporting it.

The rebuild command for this checkout:

```bash
export HOME=<scratch>/compile_home
export CCACHE_DIR=<scratch>/.ccache_trtllm
export CCACHE_MAXSIZE=50G
cd <trtllm_repo_path>
python3 ./scripts/build_wheel.py --use_ccache -G Ninja -a "103-real" --nvtx
```

Three things about it that are not guesses and must not be "improved":

- `-a "103-real"` targets this machine's architecture only. Dropping it
  builds every architecture and turns a ~1 hour incremental build into
  something that will not finish inside a normal time limit.
- `CCACHE_DIR` on shared scratch is what makes the build incremental across
  container sessions. Without it every rebuild is a full one. `HOME` is
  redirected for the same reason — the default would put the cache somewhere
  that does not survive.
- **The build is CPU-bound, and this cluster's idle-GPU watchdog kills jobs
  whose GPUs sit at 0% utilization for about ten minutes.** Keep the
  allocated GPUs nominally busy for the duration (a background loop doing a
  small matmul on each device is enough) or the build is killed partway with
  no useful error.

`--trt_root` and `--benchmarks` **do not exist** in this checkout's
`build_wheel.py` and passing them fails the whole command with `unrecognized
arguments` before anything builds. Read `scripts/build_wheel.py --help`
rather than carrying flags over from another project's recipe.

### General rules

- Wrap the command in the same `srun` container invocation that runs your
  test, so it executes in the session you just set up: `srun
  --partition=<slurm_partition> --container-image=<docker_image>
  --container-mounts=<trtllm_repo_path>:<trtllm_repo_path>:rw,...`. A bare
  command on the login node, one in a container that did not mount the
  checkout, or one that mounted it at a different path is not pass evidence.
- **Export `TRTLLM_STAIRCASE=require` inside the `srun` invocation**, before
  the ranks start — not in a wrapper shell that only the driver sees.
- Above world size 1, `TLLM_WORKER_USE_SINGLE_PROCESS` does not apply (it is
  read only at world size 1) and the LLM API falls back to `MPI_Comm_spawn`,
  which **wedges** under srun/pyxis here. Use one srun task per rank plus
  `trtllm-llmapi-launch`, which builds the MPI session from the existing
  allocation instead of spawning into it.
- The repo root carries working `staircase_*.sbatch` examples for the gate,
  smoke, and rebuild shapes. Read one before writing a new command.
"""

TEST_COMMAND_CACHE = """\
## `test_command.md` — verified test command cache (Slurm-only)

**Slurm-only mechanism.** `test_command.md` exists **only** when `task.yaml`
contains `slurm-environment`. On local (non-Slurm) hosts, do not create,
read, or maintain it — skip every rule in this section, run commands
directly, and rely on `progress.yaml` for run history. The cache pays for
itself only when each invocation carries a long `srun` / `sbatch` wrapper
that is expensive to reconstruct; on a local box the commands are short
enough that caching is iteration noise.

The rest of this section assumes a Slurm environment.

`test_command.md` is a workspace-scoped cache of **verified** commands (test
runs, gate runs, benchmarks) shared by the Coder, Reviewer, and QA. Each
record has a one-line purpose and the exact bash command confirmed to run
successfully from the login node. The file starts empty and is built up as
the team verifies commands. Use `Edit`/`Write` directly. Do not store
secrets.

**Always read it first** before writing or running any command. If the entry
you need is there, run it as-is.

**On every outcome, update the cache:**
- **Success** — append or refine an entry with the exact command plus a
  one-line purpose.
- **Failure** — diagnose, fix, rerun until it succeeds, then **overwrite** the
  matching entry in place. Never leave a known-broken command in the cache.

Rules that keep it small and scannable:

- **Keep only currently-passing commands; delete everything else.** An entry
  exists only if the command just succeeded with `rc=0` in the current
  iteration **and** it still satisfies the current acceptance criteria. When
  either stops holding, delete the whole entry — no inline "superseded by"
  prose, no `## Archive` section, no chain of historical attempts. Failed
  attempts and validation diaries belong in `progress.yaml` and git history.
- **Per-entry template — three header lines, then the command.** No
  Reviewer/QA prose may be appended across iterations:

      ## <Short purpose, one line>
      criteria: <acceptance-criteria ids this command verifies, comma-separated>
      verified: <job-id or run-id, exit status, elapsed>
      outputs:  <key output files + headline metrics, one line>

      ```bash
      <single self-contained command>
      ```

  If a field would not fit on one line, shorten it rather than wrap. If a
  rerun changes the verified job/exit/outputs, overwrite the header lines in
  place.
- **No top-of-file narrative.** The file opens directly on the first entry.
- **Every cached command is a self-contained `srun` (or `sbatch`) invocation
  runnable from the login node.** It must carry the partition, account, node
  count, GPU, time, container, and mount flags needed to launch on its own,
  and must not assume an existing interactive allocation. Bare `python ...`
  lines without an `srun` wrapper must not be cached — if a command has no
  wrapper, the task is not a Slurm task and this file should not exist.
- A gate command in the cache carries `TRTLLM_STAIRCASE=require` in the
  command itself, not in the surrounding shell.

The cache is **not a spec**: it does not change the verdict QA owes against
`task.yaml` and the acceptance criteria, and editing it is not a substitute
for verifying criteria at runtime.
"""


def slurm_blocks() -> tuple[str, str]:
    """Return ``(CONTAINER_BOOTSTRAP, TEST_COMMAND_CACHE)``.

    Kept as a function so the bundle builder's call site reads the same
    whether these blocks are local constants or come from somewhere else
    later. They are local copies today — see the module docstring.
    """
    return CONTAINER_BOOTSTRAP, TEST_COMMAND_CACHE


# ---------------------------------------------------------------------------
# Skill injection
# ---------------------------------------------------------------------------

# Staircase's own skills (gate running, catalog probing, nsys capture) are not
# written yet. Add their names here, preference-ordered, as they land; the
# block below is injected into prompts only once a configured backend actually
# reports one loaded, so an unwritten skill costs nothing and a written one
# needs no prompt restructuring.
#
# An empty tuple short-circuits the probe entirely — no process spawn at all.
STAIRCASE_SKILL_CANDIDATES: tuple[str, ...] = ()

_STAIRCASE_SKILL_INVOCATION_TEMPLATE = """\
## Running staircase gates and probes — `{skill}`

`{skill}` is the sanctioned runner for staircase gate work: target smoke
runs, catalog entry GPU tests, and the accuracy suite. Prefer it over
hand-built Bash for anything whose result you will cite as pass evidence,
and cite the report path (and the key numbers) in `status.md` /
`progress.yaml` / your `summary`.

Gate runs must declare `TRTLLM_STAIRCASE=require`. Under `auto`, a
configuration that misses a target's criteria silently gets the built-in
implementation, and its numbers then read as staircase's — the single most
expensive mistake available here.
"""


@lru_cache(maxsize=1)
def get_staircase_skill_invocation() -> str:
    """Return the staircase skill block, or ``""`` when no skill is loaded.

    Probes real backend sessions once and caches for the process lifetime,
    degrading to an empty string rather than pointing an agent at a skill that
    will not load.

    **Call this from the bundle builder, never bind it at module scope.** A
    ``SYSTEM_PROMPT_EXTENSION = "\\n".join([..., SKILL_BLOCK])`` composed at
    import time forces the probe on every import of the package, which is how
    a sibling workflow ends up spawning a session per backend just to read its
    prompts. ``build_staircase_prompts`` appends the result instead, and
    ``with_extensions`` no-ops on the empty string, so the absent-skill case
    costs nothing and changes no prompt.
    """
    if not STAIRCASE_SKILL_CANDIDATES:
        return ""
    try:
        loaded, _probe_ok = resolve_first_available_skill(STAIRCASE_SKILL_CANDIDATES)
    except Exception:
        return ""
    if loaded is None:
        return ""
    return _STAIRCASE_SKILL_INVOCATION_TEMPLATE.format(skill=loaded)


__all__ = [
    "ACCURACY_DEBUG_METHODOLOGY",
    "CATALOG_ENTRY_SPEC",
    "CLOSED_VOCABULARY_RULE",
    "CONTAINER_BOOTSTRAP",
    "DOMAIN_PRIMING",
    "GATE_LADDER",
    "LINT_RECIPE",
    "PROJECT_REQUIRED_MECHANISMS",
    "RECEIPT_POLICY",
    "REFERENCE_INDEPENDENCE",
    "REFERENCE_LADDER",
    "SELF_CONTAINMENT_POLICY",
    "SIGNAL_VS_NOISE",
    "STAIRCASE_SKILL_CANDIDATES",
    "STATUS_DONE_TODO_RUBRIC",
    "TARGET_PRODUCT_SPEC",
    "TEST_COMMAND_CACHE",
    "get_staircase_skill_invocation",
    "slurm_blocks",
]
