# QA test selection

Pick the subset of a QA test list a machine can run, before Slurm allocates it, and split
the result across GPU allocation sizes.

The cluster pipeline sends the same flat list
([`llm_function_core.txt`](../integration/test_lists/qa/llm_function_core.txt)) to every
architecture. A test that cannot run on the allocated machine still occupies it and skips there,
because pytest evaluates skip conditions at run time. This plugin decides the same questions
during collection instead.

Functional tests only; perf runs are selected by their own configuration.

## Usage

```bash
pytest --collect-only -q -p qa_selection.plugin --machine=B200 --ladder=1,4,8 --selection-out-dir=out/
```

The ladder chooses the question:

| question | options | writes |
|---|---|---|
| what can this machine run? | `--machine=B200` | `B200.json`, `B200-8gpu.ids` |
| what can one N-GPU allocation of it run? | `--machine=B200 --ladder=N` | `B200.json`, `B200-<N>gpu.ids` |
| how does an allocation policy divide it? | `--machine=B200 --ladder=1,4,8` | `B200.json`, one list per rung |

Without `--selection-out-dir` nothing is written.

`pytest -p qa_selection.plugin --help` documents the three options. `-p` needs `tests/`
importable: the integration suite's ini `pythonpath` provides it, elsewhere set
`PYTHONPATH=tests`.

## Layout

```text
qa_selection/
  plugin.py        the pytest hooks and options
  core/            the selection logic
    selection.py     one run's request, and its outcome per test
    report.py        the .ids lists and the JSON record
    ladder.py        --ladder: which rung holds a test
    machine.py       --machine: may the machine's card run a test
    marks.py         what a test asks for, read from its marks
  tests/           the acceptance tests
```

`core/` takes plain values and raises `SelectionError`; `plugin.py` reads the options from pytest
and reports that error as a `pytest.UsageError`.

## How the pieces connect

**Before collection** the options become a request: the machine's profile and a ladder,
`[max_gpu_per_node]` when `--ladder` is absent. An unknown machine, a rung above the node, or a
leftover list of this machine's in the output directory is a usage error, raised before any test
module is imported.

**During collection** the plugin runs after the integration conftest's `--test-list`, waive and
regex filtering, so it decides the filtered list. A test is kept when neither question blocks it:

| question | answered from |
|---|---|
| may the machine's card run it? | its `skipif` reasons, looked up in `rules.json`, and its device-name and memory markers, against the profile |
| does a rung hold it, and which? | its GPU-count markers: the smallest rung at least that large, or none above the largest |

A `skipif`'s condition is never evaluated: it describes the host that collected it. Deselected
tests go through pytest's own deselection hook.

**After collection** `<machine>.json` records the counts and each test's outcome, and each rung's
`.ids` list holds its selected node ids. The terminal shows the counts by rung and by reason; `-v`
adds node ids. Its `deselected` counts this plugin's decisions only, while pytest's own also
counts the test-list and waive filtering.

## Acceptance tests

```bash
pytest tests/qa_selection/tests      # 36 tests, no GPU, no container, no wheel
```

Each test states a command line and the answer expected back, derived from the production
decorators.

| | principle | criteria |
|---|---|---|
| **Before collection** | A machine and a ladder are the whole command, and without a machine the plugin changes nothing | AC-4 [`test_options.py`](tests/test_options.py) (5), AC-6 [`test_inert.py`](tests/test_inert.py) (2) |
| **During collection** | The target decides, never the collecting host: its architecture, wherever the mark sits, and its largest rung as the GPU count; a test above that rung is deselected with one reason | AC-1 [`test_arch.py`](tests/test_arch.py) (7), AC-2 [`test_gpu_count.py`](tests/test_gpu_count.py) (7), AC-5 [`test_largest_rung.py`](tests/test_largest_rung.py) (4) |
| **After collection** | Each kept test is in exactly one list, its smallest rung's, and the terminal explains the run | AC-3 [`test_ladder.py`](tests/test_ladder.py) (6), AC-7 [`test_summary.py`](tests/test_summary.py) (5) |

## Config

Three JSON files in `core/`. `rules.json` and `markers.json` mirror the integration suite's
`defs/conftest.py` and `defs/pytest.ini`;
[`scripts/check_qa_selection_rules.py`](../../scripts/check_qa_selection_rules.py) runs in
pre-commit and fails when they drift.

| file | holds |
|---|---|
| `profiles.json` | one entry per machine: `sm`, `device_name`, `device_memory_mib`, `max_gpu_per_node`, `cpu_arch` |
| `rules.json` | one rule per `skipif` reason: the profile values that skip it |
| `markers.json` | each resource marker's declaration, the profile fact it bounds, and whether the conftest reads its closest mark or every level's |
