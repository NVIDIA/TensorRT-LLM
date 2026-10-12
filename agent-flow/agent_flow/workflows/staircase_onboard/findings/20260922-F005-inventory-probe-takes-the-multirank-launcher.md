# F005 — the inventory probe takes the multi-rank launcher, and fails silently

- **Found:** 2026-09-22 05:27 PDT, on the first real `--dry-run` of this run
- **Component:** `agent-flow/agent_flow/workflows/staircase_onboard/survey.py`,
  `probe_command` / `run_inventory_probe`
- **Severity:** the plan drafter is handed an empty supply survey and drafts on
  assumption, with no error anywhere
- **Status:** reproduced both ways; fix proposed

## The symptom

`--dry-run` printed, in the middle of an otherwise green preflight:

```
✔ survey 12 structural demand(s), 14 unrecognized key(s); 0/0 entries certified
  on sm_103, 0 unwrapped op(s)
```

`0/0 entries certified` against a catalog that holds 76 entries, 61 of them
with a passing `sm_103` receipt. And it is reported with a **check mark**.

## The cause

`probe_command` builds its command through `RunEnvironment.command`, which
applies the same launch rules every gate gets. At world size > 1 that means:

```
mpirun -n 4 --allow-run-as-root trtllm-llmapi-launch <python> -c '<probe>'
```

Those rules are right for engine work and wrong here. The probe is read-only
introspection — it imports `tensorrt_llm`, walks `torch.ops.trtllm`, and
parses catalog frontmatter off disk. It needs one process and no MPI world.

Measured, same payload, same session, one variable:

| launch | result |
|---|---|
| single process | `trtllm_ops: 277`, `catalog: 76` (61 passed / 2 failed / 13 absent), `modules: 265`, `error: ""`, rc=0 |
| `mpirun -n 4 ... trtllm-llmapi-launch` | **rc=127**, no payload |

And the rc=127 is its own finding:

```
mpirun: command not found
```

This run's `environment.env` sets `PATH` explicitly, to pin the checkout's own
venv ahead of everything else:

```
PATH=<checkout>/.venv-3.12/bin:/usr/local/bin:/usr/bin:/bin
```

The container's MPI lives at `/usr/local/mpi/bin/mpirun`, which that list does
not contain. So the launcher the probe did not need is also the launcher the
environment cannot find.

## Why it is silent, and why that is the actual defect

`run_inventory_probe`'s docstring states the policy plainly, and the policy is
right:

> A failure here is reported, not raised. The plan can be drafted without an
> inventory — it just gets drafted the old way, on assumption — and losing the
> whole run because a probe could not import is a worse trade than losing the
> measurement.

But the reporting half does not happen. `parse_inventory` returns an empty
inventory with an `error`, and the caller then applies:

```python
landed = bool(inventory.catalog or inventory.trtllm_ops or inventory.modules)
if not landed and not getattr(result, "ok", True):
    return Inventory(arch=arch, error=f"{inventory.error} (probe exited ...)")
```

so the error *is* carried. It is the **summary line** that loses it: an
inventory with 0 entries and an error prints as `0/0 entries certified` with a
`✔`, indistinguishable from a catalog that genuinely has nothing certified.
An operator reading that line has no way to tell "nothing is certified" from
"nothing was measured".

For this run the consequence is concrete. The whole COVERED-vs-GAP decision —
which entries exist, which are certified on sm_103, which ops are unwrapped —
is what the smith queue is derived from. Drafted against an empty inventory,
every entry looks like a gap, and the run would dispatch smiths to build
things that already exist with passing receipts.

## Proposed change

Three parts, smallest first.

1. **Do not launch the probe as engine work.** `probe_command` should build a
   single-process command regardless of world size. The cleanest spelling is
   for `RunEnvironment` to expose the distinction it already makes internally
   — something like `command(argv, *, single_process=True)` — rather than
   `survey.py` assembling a prefix by hand, because assembling launcher lines
   outside `environment.py` is exactly what that module exists to prevent.
   Nothing about this probe wants a rank per device.

2. **Say when the measurement did not happen.** `format_surface` and the
   preflight summary must distinguish an empty inventory from a failed one:
   a `✘`/`⚠` and the error text, not a `✔`. The plan proceeding is fine —
   that is the documented policy — but it has to proceed *loudly*, because
   the drafter's prompt says whether it is reading a measurement or a guess.

3. **Make `PATH` a merge, not a replacement.** A spec that pins its own venv
   ahead of the system path should not have to enumerate the container's MPI,
   CUDA and system tool directories to avoid deleting them. Either
   `environment.env`'s `PATH` is prepended to the inherited one, or the schema
   rejects a `PATH` that does not end in `:$PATH` with a message saying why.
   The current behaviour turns a reasonable-looking spec line into a
   `command not found` several layers away from where it was written.

Part 3 is separable and is the one most likely to bite another run: every
task spec that pins an interpreter will write this same `PATH` line.

## Interim action for this run

The inventory is measured out of band and handed to the plan phase as a
pre-written artifact. The solo probe's payload is recorded in `04-catalog/`
so the drafter reads a measurement rather than an assumption.
