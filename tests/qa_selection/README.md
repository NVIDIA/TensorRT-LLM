# QA test selection

Picks the subset of `tests/integration/test_lists/qa/llm_function_core.txt` that can
actually run on a given machine, **before** Slurm allocates it.

The QA LLM functional cluster pipeline hands the same flat list to every architecture it
allocates. A test that cannot run on the allocated machine is still dispatched to it,
occupies the allocation, and only then skips — because pytest evaluates skip conditions
at run time, on the node. The cost being attacked is queue time, not runtime.

```
profiles.json ─┐
               ├─ machines.py ── MachineProfile ─┐
rules.json ────┴─ rules.py ───── SkipRuleTable ──┴─ selector.py ── Selector.decide()
```

`selector.py` imports neither pytest nor torch, so decisions are made from plain data and
this package loads on a login node with no GPU.

## Why rules are keyed on the reason string

`pytest.mark.skipif` evaluates its condition when the conftest is imported and freezes the
result to a boolean. On a login node that boolean describes the login node, so it cannot
answer "will this test run on the machine we are about to allocate".

The mark records nothing else useful: its `name` is always `"skipif"`, and the Python
variable name (`skip_pre_blackwell`) is not stored anywhere. That is why `-m` marker
filtering cannot do this job. What survives intact is `reason=`, which is distinct per
decorator — so it is the rule's identifier.

Keying on it means no new markers, no edits to test sources, and no hardware access. The
decorators in `conftest.py` remain the single source of architecture truth.

### The cost: an unversioned contract

Reason strings are prose in another file. Rewording one silently disables its rule, and the
affected tests are then kept for every machine. Two are already booby-trapped in ways that
look like typos:

| string | trap |
|---|---|
| `"This test is only  supported in Hopper architecture"` | double space after `only` |
| `"nvlink is inactive."` | trailing period |

`scripts/check_qa_selection_rules.py` is what makes this contract enforceable. It runs as a
pre-commit hook and reads the decorators with `ast` rather than importing them — importing
the conftest pulls in torch and tensorrt_llm and shells out to `nvidia-smi`, none of which a
hook has. It scans the whole `tests/integration/defs` tree, so it keeps working if the
decorators move out of `conftest.py`.

Its two directions are deliberately asymmetric:

- **A rule matching no decorator is an error.** The rule is dead, nothing says so, and
  selection silently over-selects.
- **A decorator with no rule is only reported.** Selection already keeps the test and
  records the reason, so the cost is a wasted slot, not a wrong answer.

## Marker precedence

The suite resolves its markers three different ways, and the inconsistency is real,
observable behaviour rather than an oversight:

| marker | resolution | consuming fixture |
|---|---|---|
| `skipif` | every level; any match skips | pytest itself |
| `skip_less_device` | closest marker only | `skip_by_device_count` |
| `skip_less_mpi_world_size` | closest marker only | `skip_by_mpi_world_size` |
| `skip_device_not_contain` | closest marker only | `skip_device_not_contain` |
| `skip_less_device_memory` | **every level** | `skip_by_device_memory` |

The odd one out is upstream and intentional — `skip_by_device_memory` carries the source
comment *"Get all markers, not just the closest one"*. So a method asking for 2 GPUs
**replaces** its class's 8, but a method asking for 80000 MiB does **not** replace its
class's 200000.

Modelling all of them uniformly would be simpler and wrong in both directions at once:
over-selecting on device memory, under-selecting on GPU count.

## Unknown reasons fail open

A reason string with no rule keeps the test and is recorded in `Decision.unknown_skipif`.
The asymmetry is intentional: failing closed would drop the test silently and report a
false green, while failing open costs one wasted slot and is visible in the report.
`Selector(..., strict_unknown=True)` turns it into an error, for auditing.

## What is not modelled

These bound achievable recall. They are limitations by construction, not defects.

| skip | why |
|---|---|
| `skip_no_nvls` | NVLS is a capability check across every *visible device*, so it depends on the allocation, not the machine type |
| `skip_nvlink_inactive` | NVLink activity depends on how many GPUs the job received; a 1-GPU allocation on an 8-GPU node has no peer |
| `skip_ray` | keys off `TLLM_DISABLE_MPI`, an execution mode chosen at run time |
| `skip_less_host_memory` | host memory is not in the catalogue; inside a container `psutil` reports the cgroup limit, not the node |
| `skip_fp8_pre_ada`, `skip_fp4_pre_blackwell` | imperative `pytest.skip()` inside test bodies — they carry no mark at all, so collection cannot see them |
| skips raised inside fixtures | same: nothing to read at collection time |

All of these are still skipped at run time on the allocated node. Selection simply does not
predict them.

## MPI world size

`MachineProfile` has no `mpi_world_size` field: `get_mpi_world_size()` reports how the test
*process* was launched, not what the machine is.

`skip_less_mpi_world_size` is therefore measured against `gpu_count`. That matches
production: `skip_by_mpi_world_size` falls back to the device count whenever the world size
is 1 — *"we can spawn mpi workers in the test itself"* — and the functional cluster job
builds its srun with `-N1` and no `--ntasks` (`LLMCluster.groovy:297`; the
`--ntasks-per-node` variants belong to `runMultiNodeTests*`, a different job type). So that
fallback is the only branch it ever takes.

A caller that does span nodes passes the rank count explicitly:

```python
Selector(profile, mpi_world_size=8).decide(test)   # e.g. 2 nodes x 4 GPUs
```

Nothing derives that number behind the caller's back — the caller who chose the launch is
the one that knows it.

## Changing a rule

1. Edit `rules.json`. `skip_when` accepts six `MachineProfile` fields and ten operators,
   both validated at load, so a typo fails immediately naming the legal set.
2. Run `python scripts/check_qa_selection_rules.py`. Exit 0 means no drift.

Copy reason strings **byte for byte** from the decorator, including anything that looks
like a typo.

## Machine profiles

`profiles.json` covers the six machines in selection scope: H100, B200, B300, GB200, GB300,
VR200. Values come from the in-production catalogue in `trt_jenkins`
(`src/com/nvidia/dlswqa/LLMCluster.groovy`, `MAKO_PROFILES` and `CLUSTER_CONFIGS`) and are
maintained by hand from there.

Memory fields are named `*_mib` because the unit is the failure mode: the value is what
`nvidia-smi` reports — framebuffer minus ECC and carveout — and is always below the marketed
board capacity. `skip_less_device_memory` is a `<` comparison, so a value that is too high
over-selects, wasting exactly the allocation this package exists to save.

An unknown machine name raises, listing the known names: a machine we cannot describe should
fail loudly rather than select wrongly.
