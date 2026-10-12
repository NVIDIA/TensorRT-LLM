# staircase-onboard

Brings one HuggingFace checkpoint to a gated **ModelingV2** target: a flat,
self-contained forward assembled only from catalog entries, trusted through
accuracy gates rather than through shared abstractions.

This workflow is the agent half of [staircase](https://github.com/NVIDIA/TensorRT-LLM).
The code half is already in tree at
`tensorrt_llm/_torch/_experimental/modeling_v2/` — the catalog, the targets,
the router and their tests. What lives here is what builds them: the
orchestrator, the six agent roles, and the ~7,000 lines of skills those agents
load.

```bash
staircase-onboard --task task.yaml --workspace workspace/staircase-onboard
```

Start with `--dry-run`. It runs preflight, resolves the launch environment and
stops before the first agent turn — the cheapest way to find out whether this
host can host this run.

Two companions to this file: **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** is keyed on
symptoms — the verbatim error you saw. **[BUILD_AND_TEST.md](BUILD_AND_TEST.md)** is
keyed on tasks — building, rebuilding, and running at CPU / 1 GPU / multi-GPU /
multi-node scale, collected from the places in TensorRT-LLM that already do each
of them correctly.

**When something breaks, read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) before
you debug the target.** It is keyed on verbatim error text, and the entries
share one property worth knowing in advance: every one of them read as a model
or target defect while it was happening. `MPI_ERR_SPAWN`, a kvCache that
cannot fit a sequence, NVRTC missing `cuda.h`, a 27-point accuracy regression
that was a protocol mismatch — none of those sentences mention the
environment, and all of them were the environment.

## The pipeline

```
preflight  ->  plan  ->  smith x N (concurrent)  ->  assemble x M (serial)  ->  deliver
  Python     capped         one per                   one per unit             Python
   only      1 round      catalog entry             core, then features       + 1 turn
```

| stage | who | what it produces |
|---|---|---|
| `preflight` | **nobody** — pure Python | the device claim, the slug, the branch, the resolved launch prefix, the substrate gate, the surface survey |
| `plan` | PlanDrafter ⇄ PlanReviewer | `plan.md`, `vocabulary.yaml` |
| `smith` | one smith per queue item | catalog entries + their GPU tests, committed per entry |
| `assemble` | one assembler per unit | `modeling.py`, `weights.py`, gated |
| `deliver` | Python + one Auditor turn | verification across six trees, then the merge |

Every stage checkpoints before advancing, so a crash resumes at the phase that
was running. Re-running the same command resumes; `--clean` starts over.

## Why it is shaped this way

The predecessor workflow for this job reliably failed the same way: runs spent
their budget deciding *how to launch a job* and re-litigating a plan, and never
reached model onboarding. Four things in this workflow exist to prevent that,
and they are the parts worth not "simplifying" later.

**Preflight is Python, not an agent turn.** The checkpoint parses, the arch
matches the host, the device set is claimed by `memory.free` and a
lone-process probe, `TMPDIR` is sane, and a multi-rank run's NCCL/MPI
substrate answers — all in seconds, before any model is invoked. A broken host
costs seconds instead of a plan phase and a first assembler turn.

**The launch environment is declared once and handed to every agent.**
`task.yaml`'s `environment` block resolves into a single command prefix that
every role's prompt carries verbatim. Agents never construct a launcher
command, never probe for one, and never call a test-runner skill. A prefix
that does not fit is a *symptom to report*, not a problem to solve.

**The plan phase is capped and non-blocking.** `--max-plan-rounds` defaults to
1. Reaching the cap is not a failure: the reviewer's unresolved items are
appended to `plan.md` as `## Open risks` — which the assembler reads — and the
build starts. A plan review never blocks the build.

**Gates are exit codes, not opinions.** There is no `--min-score` and no agent
scoring an artifact out of ten. `trtllm-eval` prints a number, the anchor comes
from TensorRT-LLM's own accuracy references, and the verdict is
`measured >= reference - tolerance`.

## The gates

| gate | when | cost |
|---|---|---|
| **boot** | after every assembler turn | minutes. `quickstart_advanced.py` with the target's topology flags |
| **accuracy** | at each unit boundary | the release criterion. `trtllm-eval` against the anchor. One-sided |
| **acceptance** | only for a distribution-preserving feature | `acceptance_length` under `=require` vs `=off` |

All three run with **`TRTLLM_MODELING_V2=require`**. Under `auto` a
configuration that misses a target's routing criteria silently gets the
built-in implementation, and a number measured that way reads as
modeling_v2's — the single most expensive mistake available in this package.

The **acceptance** gate exists because the first two are blind to speculative
decoding: rejection sampling holds the emitted distribution to the target
model's, so a *miscomputed* draft path produces correct text more slowly. Boot
passes, accuracy passes, only speed moves.

Under `parallel: unified` — the default — boot and accuracy run **once per
member**, where the members are the concrete configurations the one codebase
serves (`members:` in `task.yaml`, default `[tp<W>, tep<W>, dep<W>]`). A
unified target gated at a single topology leaves the expert-parallel and
attention-DP branches unexecuted while its Coverage claims them, so the
orchestrator enforces the member set rather than trusting the prose. Every
verdict names its member (`accuracy[gsm8k]@dep4`) and the unit advances only
when all of them pass. The witness gates run at the first member only: a
witness is a *contrast between two legs of one configuration*, not a number
per configuration. `--dry-run` prints the member set and the resulting gate
count before anything is spent.

## The six roles

| role | session | backend | one run does |
|---|---|---|---|
| `plan_drafter` | persistent | Claude Code | the checkpoint→catalog mapping and the plan |
| `plan_reviewer` | persistent | **Codex** | a paper review of both |
| `smith` | stateless | Claude Code | exactly one catalog entry |
| `assembler` | persistent *within a unit* | Claude Code | one unit: the core, or one feature |
| `reviewer` | stateless | Claude Code | reads one unit's diff against the gate evidence |
| `auditor` | stateless | Claude Code | the runtime-value dump vs the certified columns |

The PlanReviewer runs on a different model from the PlanDrafter on purpose: a
reviewer that shares the author's blind spots reviews its own reasoning back at
itself. The plan roles are stateful because they are asked to remember what
they decided; the work roles are not, because staircase's unit of work is one
entry or one unit, each in its own context.

## Workspace layout

```
<workspace>/
├── task.yaml             the resolved spec, defaults filled in
├── plan.md               PlanDrafter (+ `## Open risks` if the cap was reached)
├── vocabulary.yaml       the typed step→catalog mapping and dispatch queue
├── surface.md            the surface survey, unabridged (preflight)
├── progress.yaml         every agent's typed decision, by stage
├── journal.md            the run's timeline, written as events happen
├── reports/NN-<role>-<what>.md   each dispatch's prompt and result
├── decisions/<role>-N.yaml       typed decisions, when MCP tools are unavailable
├── boot.yaml             the boot gate's prompts and keywords (optional)
├── bin/commit.sh         the locking, self-verifying commit helper
├── .commit.lock          its lock file
├── .claude/skills/       this run's copy of the packaged skills
└── .staircase_onboard_state.json
```

The journal and the reports stay here rather than in the repository. They are
real evidence and worth keeping, but they are evidence *about a run*, and a
product tree is the wrong home for it — only a distilled record travels into
the repo at delivery.

`vocabulary.yaml` is the one artifact with a schema, because the orchestrator
walks it. `vocabulary.py` validates it and will reject a malformed one:
`OK`/`CERT` rows quote the contract line that decides coverage, `GAP` rows name
a candidate, `verify` dispatches name the surface, the two chapters do not
overlap, and a `ceiling` verdict carries its evidence.

A queue row may also name the `unit` that first needs its entry (`core` when
absent). Every row is dispatched in the smith phase regardless, but only a
failure core needs — by its tag, or as a transitive `depends_on` of a core row —
closes the run there (`smith-blocked`). Any other failure is recorded
`deferred` and retried once before its unit's first assembler turn; if the
retry fails too, the run closes `smith-blocked:<unit>`.

## Flags

| flag | default | |
|---|---|---|
| `--task` | required | the spec. Validated before any agent is constructed |
| `--workspace` | `workspace/staircase-onboard` | |
| `--dry-run` | off | preflight, then stop. Leaves the checkpoint at `preflight`, so the real run re-claims the devices |
| `--max-plan-rounds` | `1` | drafter, reviewer, at most one re-draft, then proceed regardless |
| `--max-assembler-turns-per-unit` | `6` | bounds the self-heal loop |
| `--max-parallel-smiths` | `10` | where extra compute buys wall-clock. Each dispatch is a full agent session and the upstream rate limit is reported without backoff, so this is bounded by default; `0` opens the whole batch and on a typical queue that is every entry at once |
| `--reviewer` | `light` | `light` reads the evidence and runs nothing; `full` re-runs the boot gate; `off` skips the turn |
| `--plan` / `--vocabulary` | — | skip the plan phase with pre-written artifacts |
| `--skip-substrate-gate` | off | skip the stock-path cold start in preflight. An escape hatch, not a speed-up — see the flag's own help |
| `--agent-transport` | `inprocess` | how a role turn is delivered. `external` writes it to `<workspace>/dispatch/` and waits — see below |
| `--agent-turn-timeout` | `10800` | seconds one externally-dispatched turn may take |
| `--clean` | off | wipe the checkpoint and managed files |

## Running where the orchestrator cannot authenticate an agent

On a sandboxed host the spawned CLI cannot write its credential lock, and the
turn fails with `EROFS ... credentials.lock` reported as
`stop_reason='stop_sequence'` and **zero tokens** — shaped exactly like a model
refusal when in fact no agent started. Two things happen, in order:

1. **`agent_home.py` runs automatically.** If `$HOME` is not writable it stages
   one under the workspace, seeds it from the real credential store and exports
   `HOME` / `XDG_CONFIG_HOME` / `CLAUDE_CONFIG_DIR` before any agent exists.
   Nothing to configure, and an ordinary workstation is untouched.
2. **If that authenticates and then stops working**, the cause is rotation, not
   permissions: the staged store holds a copy of a refresh token the parent
   also holds. Copying harder does not fix a single-use token. Use
   `--agent-transport external`.

External transport moves the **turn** and nothing else. The orchestrator writes
`dispatch/<id>.system.md`, `dispatch/<id>.prompt.md` and
`dispatch/<id>.request.json`, then blocks until a driver writes
`dispatch/<id>.result.md` followed by `dispatch/<id>.done`. The driver can be a
human, another session or a service; it is trusted with nothing, because every
*decision* still arrives through `collect_decision_file` — validated against the
same closed enum the MCP tool schema would have — and every *verdict* still
comes from a gate the orchestrator ran itself.

The one thing a driver must get right: **the role's system prompt travels with
the turn**. A driver that delivers only the user prompt is running a different
agent, and its answer still looks plausible.

## What this workflow does not do

- **Performance.** Perf is measured, never gated, and it is not part of
  onboarding. A target whose two gates are green is deliverable; the tuning
  campaign is separate work.
- **Isolation.** Staircase denied every agent read access to TensorRT-LLM's
  per-model definitions, so that a target could not be a transliteration of
  the implementation it is compared against. That hook is **not** ported: the
  zoo is an ordinary read. What carries the duty instead is the
  `modeling-core` invariants — a copied forward cannot pass the
  closed-vocabulary audit, is not flat, and leans on shared helpers a target
  may not have — plus the rule kept verbatim in all three writing roles:
  *read to understand an interface, never to copy a structure*. If a future
  run needs the stronger claim, a per-layer `PreToolUse` hook through
  `BackendConfig.hooks` is a small, self-contained change.

## Tests

```bash
pytest tests/workflows/staircase_onboard/
```

Every test runs on CPU with no model, no GPU and no cluster: the probes and
the shell are injected, and the agent layers are replaced by a fake that
writes the typed decision a real role would have written. Two exceptions earn
their keep by reading the real checkout — `test_delivery.py` verifies the
merged `gpt_oss / sm_103 / tp1` target passes every delivery check, and
`test_skills.py` resolves every repository path the skills name.

## Feature increments

`task.yaml`'s `features:` is a closed list, because a feature with no skill is
an assembler told to build something nobody wrote down how to build.

| feature | skill | gated by |
|---|---|---|
| `mtp` / `eagle3` / `dflash` / `dspark` | `staircase-modeling-speculator` | boot, accuracy, **acceptance** |
| `eplb` | `staircase-modeling-eplb` | boot, accuracy |
| `dwdp` | `staircase-modeling-dwdp` | boot, accuracy |
| `multimodal` | `staircase-modeling-multimodal` | boot, accuracy on a multimodal benchmark |
| `host_offload` | `staircase-modeling-host-offload` | boot, accuracy, **residency witness** |
| `disagg` | `staircase-modeling-disagg` | boot, accuracy, **equivalence witness** |
| `kv_quant` | `staircase-modeling-kv-quant` | boot, accuracy, **kv_quant witness** |
| `decoder_replay` / `encoder_replay` | `staircase-modeling-bounded-replay` | boot, accuracy, **replay witness** |

The speculator, `host_offload`, `disagg` and `kv_quant` rows share a property
worth stating plainly: they are **output-identical by construction**, so boot
and accuracy pass whether or not the feature is doing anything. Each therefore
carries a third measurement that can only come out one way if the feature is
real — `acceptance_length` under `=require` vs `=off`, device memory with and
without the offload, token-identical output across the disaggregation hop, or
stored KV bytes per token against a declared figure (plus token identity with
bf16 storage).

The bounded-replay row is the opposite case with the same need. DeepSeek-V4.1's
SWA Bounded Replay truncates sliding-window state on purpose, so accuracy is a
real check of it — but the truncation is designed to stay inside the accuracy
tolerance, and a unit that implements nothing stays inside it too. Its witness
reads per-request row counters the target prints (`MODELING_V2_WITNESS
key=<int> ...` under `TRTLLM_MODELING_V2_WITNESS=1`) against a window and a
workload the spec declares before assembly.

`dspark` is its own kind and never a spelling of `mtp`: DeepSeek-V4.1 stores
its DSpark block drafter as `mtp.*`, and `--spec_decode_algo=MTP` over those
weights builds the wrong drafter. The schema refuses the two together.
