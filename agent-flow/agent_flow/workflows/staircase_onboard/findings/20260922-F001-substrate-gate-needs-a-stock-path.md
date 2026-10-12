# F001 — the substrate gate assumes a stock path exists for the checkpoint

- **Found:** 2026-09-22 03:30 PDT, during preflight planning for MiMo-V2.6-Flash-RL
- **Component:** `agent-flow/agent_flow/workflows/staircase_onboard/preflight.py:385-444`
- **Severity:** blocks preflight for every genuinely new architecture
- **Status:** proposed fix drafted, not yet applied

## What happens

`substrate_gate()` proves the host can generate a token by running the boot
gate on **the run's own checkpoint** with `mode=MODE_OFF` — that is, the
stock TensorRT-LLM modeling path:

```python
result = gates_mod.run_boot_gate(
    environment,
    task_data=task_data,
    spec=gates_mod.BootSpec(prompts=("The capital of France is",), max_tokens=4),
    runner=shell_runner,
    timeout=SUBSTRATE_TIMEOUT_S,
    mode=gates_mod.MODE_OFF,
    features=[],
)
```

For MiMo-V2.6-Flash-RL there is no stock path — `grep -rl "MiMoV2\|mimo_v2"
tensorrt_llm/ tests/ examples/` returns nothing. The gate therefore fails
with an architecture-not-supported error, and `run_preflight` raises
`PreflightError` before any agent exists.

## Why this is the workflow's problem and not the run's

The docstring is explicit about what the gate is for: three host failures
(`MPI_ERR_SPAWN`, kvCache sizing, NVRTC's missing `cuda.h`) that "surfaced
mid-run, inside an agent's turn, reading as though the target were wrong".
All three are properties of **the host and the resolved runtime block**.

The gate reaches for the run's own checkpoint for a good reason — two of the
three failures depend on the checkpoint's geometry — but that reason only
holds for a checkpoint the stock path can load. The onboarding workflow's
entire purpose is checkpoints it cannot. So the gate is, by construction,
unavailable exactly when a run is doing the thing the workflow exists for.

The escape hatch today is `--skip-substrate-gate`, and its own help says to
use it "only when you have already proven the substrate in this same
environment". That is sound advice and it is also the wrong shape: it turns
the highest-value check into an all-or-nothing switch, and the run loses the
`MPI_ERR_SPAWN` and NVRTC checks — which do **not** depend on the
checkpoint — along with the kvCache check, which does.

## Proposed change (minimal)

Split the claim the gate makes into the part that needs this checkpoint and
the part that does not.

1. Try the stock path on the run's own checkpoint, as today. If it passes,
   nothing changes and the note is what it is now.
2. If it fails **because the architecture is unsupported** — a recognizable
   class of error, not any failure — fall back to a declared proxy
   checkpoint and re-run the same gate under the same environment and the
   same `runtime` block. Record the weaker claim verbatim in the notes:
   *"substrate proven on `<proxy>`; this checkpoint's own geometry is
   unverified, so a kvCache sizing fault will surface at the first boot gate
   rather than here."*
3. If the fallback is not declared, fail with a message that says to declare
   one — not with the raw architecture error, which reads as a target defect
   and is the exact failure mode the gate was written to prevent.

The proxy is a new optional key, `environment.substrate_checkpoint`. It is
optional because a checkpoint the stock path already supports needs no proxy,
and closed-schema validation means it has to be declared to be usable.

## Why not just widen the fallback to any failure

Because then a real kvCache fault on the target checkpoint would be answered
by silently measuring a different model and reporting green — the same class
of mistake as gating under `TRTLLM_MODELING_V2=auto`. The fallback has to be
keyed on "this path does not know this architecture" and nothing else.

## Interim action for this run

Prove the substrate by hand on the allocation with a proxy checkpoint, record
the evidence in `02-environment/`, then pass `--skip-substrate-gate` with a
pointer to that evidence. Apply the fix afterwards so the next run does not
need the manual step.
