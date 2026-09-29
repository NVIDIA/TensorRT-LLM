# QA test selection

Pick the subset of a QA test list a machine can run, before Slurm allocates it, and split
the result across GPU allocation sizes.

The cluster pipeline sends the same flat list
([`llm_function_core.txt`](../integration/test_lists/qa/llm_function_core.txt)) to every
architecture. A test that cannot run on the allocated machine still occupies it and skips there,
because pytest evaluates skip conditions at run time. This plugin decides the same questions
during collection instead.

Functional tests only; perf runs are selected by their own configuration.

```bash
pytest --collect-only -q -p qa_selection.plugin \
       --machine=B200 --ladder=1,4,8 --selection-out-dir=out/
```

`pytest -p qa_selection.plugin --help` documents the five options. `-p` needs `tests/`
importable: the integration suite's ini `pythonpath` provides it, elsewhere set
`PYTHONPATH=tests`.

## Layout

```text
qa_selection/
  plugin.py        the options, the markers, the item adapter, the five hooks;
                   the `-p` entry point, and the only module importing pytest
  core/            the decisions -- stdlib only, never pytest
    ladder.py        what a legal ladder is
    machines.py      the machines selection can target
    rules.py         the curated skip rules, and what holds
    markers.py       the marks whose first argument is a need
    selector.py      can this machine run this test
    allocation.py    how much it wants, which rung takes it
    artifacts.py     what the output files are called
    request.py       what one run was asked for
    selection.py     what it decided about every test
    report.py        the .ids lists, the JSON record, the summary lines
  tests/           the behaviour suite -- no GPU, no container, no wheel
```

`core/` takes plain values and raises `SelectionError`; `plugin.py` reads them off pytest's
config and raises `pytest.UsageError`. So every decision can be exercised without pytest, and a
reader who only wants to know what the plugin *does to collection* has one file to read. Imports
point downward only, and nothing imports `plugin.py`.

## How the pieces connect

**At config time** `plugin.py` registers the options and resolves them once into a
`SelectionRequest`: a machine name becomes a `MachineProfile` read from `profiles.json`, a ladder
becomes a validated `Ladder`. An output directory is created here too, and refused if it already
holds files of this machine's that this run would not overwrite — a leftover list from a
different ladder. Every usage error surfaces here, before a test module is imported.

**At collection time** the hook is declared `trylast` and is not a wrapper, so it runs after
the integration conftest's own `hookwrapper` has applied `--test-list`, waives and regex
filtering. It therefore decides the filtered set rather than the whole suite.

Each item then crosses one boundary. `ItemView` copies a mark's name, a resource marker's first
argument and a `skipif`'s `reason=` — nothing else — producing the `CollectedTest` that `core/`
reads. A `skipif` condition was frozen against the collecting host when the conftest was
imported, so it describes the wrong machine and is never evaluated.

`core/` answers two independent questions about it:

| question | asked of | answer |
|---|---|---|
| can this machine run it? | `selector.py`, against the profile | `Decision` |
| how much does it want, which allocation? | `allocation.py`, from marks alone | `GpuDemand`, rung |

Feasibility is decided first, assignment second. Because demand never consults the profile, the
two cannot disagree, and that order cannot double-count. `selection.py` holds one answer per
test in collection order, so `partition` can split the item list by position: whatever is not
live goes to pytest's own deselection hook and is removed from the list in place.

**After collection** `report.py` turns those decisions into the per-rung `.ids` lists the
pipeline filters with `awk`, a JSON record, and a terminal summary. Nothing is written unless
`--selection-out-dir` was given.

## Acceptance tests

Six criteria, one module each. Every expected value was derived from the production decorators
before the plugin was run, so a test states a command line and the answer expected back — what
the plugin decides, and how to drive it. The criteria in full, with those derivations, are in
`openspec/changes/pytest-plugin-test/`.

```bash
pytest tests/qa_selection/tests      # 29 tests, no GPU, no container, no wheel
```

| | guarantee | proved by |
|---|---|---|
| **AC-1** | The target machine's **architecture** decides what is selected — `sm`, CPU arch and device memory, read from the profile and never from the collecting host | [`test_arch.py`](tests/test_arch.py) (5) |
| **AC-2** | The available **GPU count** decides what is selected: the machine's GPUs per node, unless `--gpus` names fewer | [`test_gpu_count.py`](tests/test_gpu_count.py) (5) |
| **AC-3** | The **ladder** routes each selected test to the smallest allocation that holds it, and publishes one list per rung | [`test_ladder.py`](tests/test_ladder.py) (5) |
| **AC-4** | Each **option answers one question**, and a rung run selects exactly what that rung published | [`test_options.py`](tests/test_options.py) (9) |
| **AC-5** | A ladder **shorter than the machine** strands feasible tests audibly — named, counted and warned, never folded into the largest rung | [`test_stranded.py`](tests/test_stranded.py) (3) |
| **AC-6** | Without `--machine`, loading the plugin **changes nothing**, so it can be loaded unconditionally | [`test_inert.py`](tests/test_inert.py) (2) |

## Config

Three JSON files under `core/`, each read by the module beside it. `rules.json` and
`markers.json` are copies of what the integration suite owns — selection must not share a source
of truth with the code it decides about. [`scripts/check_qa_selection_rules.py`](../../scripts/check_qa_selection_rules.py),
wired into pre-commit, fails when a copy drifts.

| file | holds |
|---|---|
| `profiles.json` | one entry per machine: `sm`, `device_name`, `device_memory_mib`, `max_gpu_per_node`, `cpu_arch`, and an optional `rungs` ladder — allocation policy, read by no rule |
| `rules.json` | the curated skip rules, each keyed by the exact `skipif` reason string it decides; a rule must turn on a permanent property of the machine |
| `markers.json` | the marks whose first argument is a *requirement* rather than a condition, with the description each is declared with |
