# QA test selection

Pick the subset of a QA test list a machine can run, before Slurm allocates it, and split
the result across GPU allocation sizes.

The cluster pipeline sends the same flat list (`llm_function_core.txt`) to every architecture. A test that cannot run on
the allocated machine still occupies it and skips there, because pytest evaluates skip
conditions at run time. This plugin decides the same questions during collection instead.

Functional tests only; perf runs are selected by their own configuration.

```bash
pytest --collect-only -q -p qa_selection.plugin \
       --machine=B200 --ladder=1,4,8 --selection-out-dir=out/
```

`pytest -p qa_selection.plugin --help` documents the four options. `-p` needs `tests/`
importable: the integration suite's ini `pythonpath` provides it, elsewhere set
`PYTHONPATH=tests`.

## Layout

Imports point downward only, and `plugin.py` is a leaf — nothing imports it.

```text
qa_selection/
  core/          the decision layer -- stdlib only, never pytest
    machines.py    profiles.json -> MachineProfile
    rules.py       rules.json    -> SkipRuleTable
    selector.py    can this machine run this test     -> Decision
    allocation.py  how much does it want, which rung  -> GpuDemand, Ladder, Assignment
  collection.py  the options, the markers, ItemView, Selection
  report.py      the .ids lists, the JSON record, the terminal summary
  plugin.py      the five pytest hooks; the `-p` entry point
```


## How the pieces connect

**At config time** `plugin.py` registers the options and resolves them once into a
`SelectionRequest`: a machine name becomes a `MachineProfile` read from `profiles.json`, a
ladder becomes a validated `Ladder`. Every usage error surfaces here, before a test module is
imported. 

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
two cannot disagree, and that order cannot double-count. Whatever is not live goes to pytest's
own deselection hook and is removed from the item list in place.

**After collection** `report.py` turns those decisions into the per-rung `.ids` lists the
pipeline filters with `awk`, a JSON record, and a terminal summary. Nothing is written unless
`--selection-out-dir` was given.

## Ideas worth knowing

**Feasibility and assignment are different questions.** `--gpus=4` alone selects everything
fitting in 4 GPUs. With `--ladder=1,4,8` it selects the tests *assigned to* the 4-GPU
allocation — those wanting 2, 3 or 4. The ladder, not the count, decides which.

**`required_gpus` is inferred, not declared.** `skip_less_device(4)` says "not fewer than 4",
a lower bound on the environment rather than a demand. So the number never travels without
`required_gpus_from`, the markers it came from; empty means the test said nothing and was
assumed to need 1.

**Rules key on the `skipif` reason string**, the only field a collected mark preserves. Reason
strings are prose in another file, so rewording one silently disables its rule —
`scripts/check_qa_selection_rules.py`, wired into pre-commit, is what makes that contract
enforceable. Add a rule by copying the decorator's reason byte for byte into `core/rules.json`
and running the check.

**A rule must turn on a permanent property of the machine.** Skips depending on the allocation
or the run-time environment get none, and a `pytest.skip()` raised in a body or a fixture
leaves no mark to read. All of them still skip on the node, as today.

**The table is a scope statement, so what it omits is not reported.** A `skipif` with no rule —
whatever its reason, declared in the conftest or written inline at a test, and including one
carrying no `reason=` at all — keeps the test for every machine. That is the whole answer, so
selection gives it no count and no line, and the drift check says nothing about it either. What
*is* reported is drift in the rules that exist: a reason reworded out from under one is an error
at commit time.

**Marker precedence mirrors the suite's**, inconsistency included: `skipif` and
`skip_less_device_memory` are read at every level, the other three resource markers only at the
closest. So a method asking for 2 GPUs replaces its class's 8, while one asking for 80000 MiB
does not replace its class's 200000.


## Hand-maintained inputs

`core/profiles.json` describes the machines selection can target, with the facts the rules ask
about; `core/rules.json` holds the curated skip rules.
