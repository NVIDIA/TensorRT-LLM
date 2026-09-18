# QA test selection

Picks the subset of `tests/integration/test_lists/qa/llm_function_core.txt` that can
actually run on a given machine, **before** Slurm allocates it.

The QA LLM functional cluster pipeline hands the same flat list to every architecture it
allocates. A test that cannot run on the allocated machine is still dispatched to it,
occupies the allocation, and only then skips — because pytest evaluates skip conditions
at run time, on the node. The cost being attacked is queue time, not runtime.

```
core/profiles.json ─┐
                    ├─ machines.py ── MachineProfile ─┐
core/rules.json ────┴─ rules.py ───── SkipRuleTable ──┴─ selector.py ── Selector.decide()
                                                            └─ allocation.py ── rung
```

## Layout

The package splits on what each half is allowed to depend on:

```text
qa_selection/
  core/          the decision layer -- stdlib only, never pytest
    machines.py    machine facts, read from profiles.json
    rules.py       skip rules, read from rules.json
    selector.py    feasibility: can this machine run this test
    allocation.py  demand and rung: which allocation a test belongs to
  collection.py  options, markers, the item adapter, the decisions
  report.py      the .ids files, the JSON record, the terminal summary
  plugin.py      the pytest hooks; the `-p qa_selection.plugin` entry point
```

Code inside `core/` imports neither pytest nor torch, so decisions are made from plain data
and this package loads on a login node with no GPU. The invariant is a one-liner:

```bash
grep -lE '^(import|from) pytest' tests/qa_selection/core/*.py   # prints nothing
```

`plugin.py` is deliberately short: a reader opening the entry point should see the hook
contract — `trylast`, non-wrapper, inert without `--machine` — with nothing else in the way.

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

### …and why every rule also names its decorator

Matching on prose is unavoidable. *Identifying* a rule by prose is not, so every entry carries
a required `decorator` field, listed first:

```json
{
  "decorator": "skip_no_hopper",
  "reason": "This test is only  supported in Hopper architecture",
  "skip_when": {"sm": {"ne": 90}}
}
```

Nothing in `core/selector.py` reads it — a collected mark carries only its `kwargs`, so `reason`
stays the only key selection can match on. Its purpose is the drift check: an identifier is
stable under the edit the check exists to catch, which turns *"this rule matches nothing"*
into *"`skip_no_hopper`'s reason is now **X** — paste it in"*.

> **The named-decorator invariant.** A rule must name a **module-level** decorator, never a
> one-off inline `skipif`: a skip worth curating is a skip worth naming. An inline `skipif`
> duplicating a named decorator's condition is a vocabulary bypass — adopt the name, never
> add a second rule for the synonym.

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

It resolves each rule through the `decorator` it names, then compares reasons. The failures
are kept separate because their remedies differ:

| what it finds | verdict | what it prints |
|---|---|---|
| decorator exists, reason differs | **error** | both strings, JSON-quoted and paste-ready |
| decorator gone (renamed or deleted) | **error** | remove the rule, or restore the decorator |
| reason found only on an inline `skipif` | **error** | name it at module level, or drop the rule |
| decorator has no literal `reason=` | reported | nothing to compare statically |
| decorator with no rule | reported | the decorator and its reason |

The first three are errors for one reason: the rule can never match a collected mark again, so
leaving it is a silent no-op and its tests ship to every machine. The third exists so that a
rule anchored to an inline `skipif` is diagnosed rather than mistaken for a deleted one — which
is what makes the invariant above enforceable rather than advisory.

The last row is the only asymmetry, and it is deliberate: **the table is curated, not an
inventory.** Selection already keeps an unruled test and records its reason, so the cost is a
wasted slot, not a wrong answer — see [What is not modelled](#what-is-not-modelled).

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

**Unknown is not the same as unkeyable.** A `skipif` written with no `reason=` at all is kept
silently and is *not* recorded there: `unknown_skipif` means "a string that could be given a
rule", and a keyless mark has none to give under any future edit. Recording it would make
`strict_unknown` demand a rule that cannot be written. No call site under
`tests/integration/defs` omits `reason=` today; surfacing one belongs to the reporting layer,
which sees the bare mark *and* the node id.

## What is not modelled

A rule answers one question: *can this machine type run this test?* So it must turn on a
permanent property of the hardware. Anything that depends on the allocation, the run-time
environment, or a bug that will be fixed does not belong in the table, however easy it would
be to express. These bound achievable recall — limitations by construction, not defects.

| skip | why |
|---|---|
| `skip_no_nvls` | NVLS is a capability check across every *visible device*, so it depends on the allocation, not the machine type |
| `skip_nvlink_inactive` | NVLink activity depends on how many GPUs the job received; a 1-GPU allocation on an 8-GPU node has no peer |
| `skip_ray` | keys off `TLLM_DISABLE_MPI`, an execution mode chosen at run time |
| `skip_no_mxfp4_swizzle` | a bug gate (nvbugs/5446119), not a capability gate — it describes a defect on H20 that will be fixed, so curating it would tie this table to a bug's lifecycle |
| `skip_less_host_memory` | host memory is not in the catalogue; inside a container `psutil` reports the cgroup limit, not the node |
| `skip_fp8_pre_ada`, `skip_fp4_pre_blackwell` | imperative `pytest.skip()` inside test bodies — they carry no mark at all, so collection cannot see them |
| skips raised inside fixtures | same: nothing to read at collection time |

All of these are still skipped at run time on the allocated node. Selection simply does not
predict them.

### Deferred: the SM107 sites

Four tests are gated by an inline `skipif` whose condition is byte-identical to
`skip_no_rubin` — a decorator that already has a rule — but whose reason string differs:

```python
@pytest.mark.skipif(get_sm_version() != 107, reason="fine-grained sync requires SM107")
```

| site in `accuracy/test_llm_api_pytorch.py` | test |
|---|---|
| `:1310` | `TestDeepSeekV3Lite::test_nvfp4_fine_grained_sync` |
| `:3965` | `TestQwen3_30B_A3B::test_w4a8_mxfp4_fine_grained_sync` |
| `:3994` | `TestQwen3_30B_A3B::test_w4a16_mxfp4_fine_grained_sync` |
| `:4654` | `TestGPTOSS::test_w4_1gpu_fine_grained_sync` |

Per the invariant the fix is to adopt `@skip_no_rubin`, not to add a rule for the synonym.
**That adoption is deliberately deferred**, because the two strings do not say the same thing
even though they select the same hardware today:

| | `skip_no_rubin` | `fine-grained sync requires SM107` |
|---|---|---|
| claim | this test is **only supported** on Rubin | this **feature** needs SM107 |
| why `!= 107` today | Rubin support is still landing | fine-grained sync is genuinely SM107-only |

Merging them now erases that distinction exactly when it starts to matter: as Rubin support
completes the support gate is expected to relax, while a genuinely SM107-only feature must not
relax with it. Which site is which is knowable today and unrecoverable after a merge.

**The cost, so it reads as a choice and not an oversight:** the 24 `*fine_grained_sync*` entries
of `llm_function_core.txt` stay undecidable — dispatched to every non-Rubin machine, occupying a
slot, skipping on the node. Bounded and reversible.

**The reminder regenerates itself.** That string has no rule, so every selection run lists it
under unknown reasons with its count; nothing depends on rereading this section. The drift check
stays green, because an inline `skipif` with no rule is not a failure.

**Revisit when Rubin support is complete**, deciding per site: support-gated sites adopt
`skip_no_rubin`; a genuinely feature-gated one keeps a distinct decorator, *named* at module
level so it can be curated.

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

A rule starts from a decorator name, not from a reason string.

1. **Pick the `skip_*` to curate.** It must already exist at module level under
   `tests/integration/defs`; if the skip is written inline at a test, name it there first.
2. **Add the entry to `core/rules.json`**, `decorator` first, then its `reason` copied **byte for
   byte** — including anything that looks like a typo. `skip_when` accepts six
   `MachineProfile` fields and ten operators, both validated at load, so a mistake fails
   immediately naming the legal set.
3. **Run `python scripts/check_qa_selection_rules.py`.** Exit 0 means no drift.

Repairing drift is the same loop in reverse: the check has already printed the decorator's
current reason, JSON-quoted, so paste it over the rule's `reason` and re-run.

## Machine profiles

`core/profiles.json` covers the six machines in selection scope: H100, B200, B300, GB200, GB300,
VR200. Values come from the in-production catalogue in `trt_jenkins`
(`src/com/nvidia/dlswqa/LLMCluster.groovy`, `MAKO_PROFILES` and `CLUSTER_CONFIGS`) and are
maintained by hand from there.

Memory fields are named `*_mib` because the unit is the failure mode: the value is what
`nvidia-smi` reports — framebuffer minus ECC and carveout — and is always below the marketed
board capacity. `skip_less_device_memory` is a `<` comparison, so a value that is too high
over-selects, wasting exactly the allocation this package exists to save.

An unknown machine name raises, listing the known names: a machine we cannot describe should
fail loudly rather than select wrongly.
