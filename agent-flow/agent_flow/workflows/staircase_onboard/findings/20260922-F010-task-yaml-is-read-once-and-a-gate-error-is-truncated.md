# F010 — the spec is read once, and the gate cannot see why it failed

- **Found:** 2026-09-22 10:20-10:30 PDT, at the core unit's first boot gate
- **Components:** the orchestrator's task-data lifetime; `gates.py`'s
  `stdout_tail`; `preflight._reports_unsupported_architecture`
- **Severity:** cost one wasted assembler turn and two orchestrator restarts
- **Status:** worked around for this run; three fixes proposed

Two separate defects compounded into one confusing hour. Both are worth
fixing independently.

---

## Defect 1 — `task.yaml` is loaded once, so an agent cannot repair the spec

The core assembler discovered a genuine precondition: this checkpoint's
`auto_map` points at `configuration_mimo_v2.MiMoV2Config`, `transformers`
knows no `mimo_v2` model type, and `ModelConfig.from_pretrained` therefore
refuses to load it at all without `trust_remote_code`. It did the right thing
— added `--trust_remote_code` to `workspace/task.yaml`'s
`runtime.default.extra_flags` and `trust_remote_code: true` to the core
options file, and said so in its hand-back.

**The running orchestrator never saw it.** `task.yaml` is validated and loaded
into `self.task_data` when the workflow object is constructed; every gate
afterwards reads that in-memory copy. So an agent editing the file on disk is
editing something nothing will re-read, and the boot gate ran again with the
identical, still-broken command.

The workflow then did exactly what it should on a failed boot — `continue`, and
dispatch another assembler turn. That turn would have been asked to fix a
target that was already correct, against a gate that could not pass whatever
it did. The loop is bounded at `--max-assembler-turns-per-unit` (6), so the
unit would have burned six turns and ended `assembly-exhausted:core` with a
green target on disk.

**Why the agent's instinct was right and the system disagreed.** The
per-unit options files (`llm_api_options`) *are* re-read per gate — they are
paths, resolved at gate time. So the assembler correctly fixed one half in a
place that works, and the other half in a place that does not, with no signal
that the two behave differently.

### Proposed change

1. **Re-read `task.yaml` at each stage boundary**, or at minimum at each gate,
   and log when the on-disk copy differs from the loaded one. The spec is
   small and validation is fast; the cost is nothing against a wasted unit.
2. If that is judged too loose — a spec mutating mid-run is its own hazard —
   then the opposite, and say it: **make `workspace/task.yaml` read-only to
   agents**, and give the assembler a typed way to request a spec change
   (a `BLOCKED: spec_change` decision carrying the key and value), so the
   orchestrator applies it deliberately and records it.

Either is fine. What is not fine is the current state, where the edit is
allowed, looks successful, and silently does nothing.

---

## Defect 2 — the boot gate's failure detail is truncated to teardown noise

With the spec fixed and the orchestrator restarted, the substrate gate failed
again, and this time the reported cause was:

```
RuntimeError: Executor worker returned error
```

That is `GenerationExecutorProxy._fail_initialization` re-raising. The
worker's actual error — the thing that decides whether this is a host
problem, a spec problem, or an unsupported architecture — was raised in
another process and logged well before the end.

`GateResult.stdout_tail` keeps the **last 40 lines**. Under a 4-rank MPI
launch those 40 lines are reliably: UCX interface warnings, four ranks'
teardown messages, the proxy's generic re-raise, and Open MPI's
"Primary job terminated normally, but 1 process returned a non-zero exit
code" banner. The real exception is further up, and never reaches the
orchestrator.

**The direct consequence:** F001's fallback could not fire.
`_reports_unsupported_architecture` searches `result.detail` and
`result.stdout_tail` for the architecture markers. The marker was in the log
and not in the tail, so the run reported "this host cannot generate a token"
about a host that generates tokens fine — the exact misdiagnosis the substrate
gate exists to prevent, produced by the substrate gate.

### Proposed change

1. **Search the whole captured stdout for the discriminator, not the tail.**
   `stdout_tail` is a display budget; the classification should not inherit
   it. One-line change with a real payoff.
2. **Keep a first-exception excerpt alongside the tail.** Scan for the first
   `Traceback (most recent call last):` and keep that block. On a multi-rank
   failure it is far more diagnostic than the last 40 lines, and it is what a
   human scrolls up to find anyway.
3. **Teach `TROUBLESHOOTING.md` the shape**, keyed on the verbatim string
   `RuntimeError: Executor worker returned error` — "this is the proxy
   re-raising a worker's error; the cause is upstream in the log, search for
   the first traceback."

---

## What this run did

`--skip-substrate-gate`, with the substrate proven twice over and the evidence
on disk:

- by hand on the proxy checkpoint through the compute session: one leader,
  three workers, `mpi_world_size: 4`, generated `' Paris.'`, exit 0;
- by the core assembler on the **real** target, all three members, each
  answering `48 + 24 = 72 clips in April and May` and continuing `Paris`.

That is precisely the condition the flag's own help sanctions — "you have
already proven the substrate in this same environment" — rather than the
condition it warns against.
