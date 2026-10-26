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
cd tests/integration/defs
pytest --collect-only -q -p qa_selection.plugin --test-list=<list> --machine=B200 --selection-out-dir=out/
```

The plugin runs only when loaded with `-p qa_selection.plugin`. L0 CI never loads it; the QA
Jenkins pipelines pass it themselves.

| option | |
|---|---|
| `--machine=NAME` | the catalogue [profile](#profiles) to select for; needs no GPU. Absent: this host, recorded as `local` |
| `--ladder=RUNGS` | GPU allocation sizes, e.g. `1,4,8`; each test lands on the smallest rung that holds it. Default: the profile's `ladder`, else one rung of its GPUs per node |
| `--selection-out-dir=DIR` | write `<machine>.json`, the record, and one `<machine>-<rung>gpu.ids` per rung. Absent: nothing is written |

The ladder chooses the question:

| question | options | writes |
|---|---|---|
| what can this machine run? | `--machine=B200 --ladder=8` | `B200.json`, `B200-8gpu.ids` |
| what can one N-GPU allocation of it run? | `--machine=B200 --ladder=N` | `B200.json`, `B200-<N>gpu.ids` |
| how does an allocation policy divide it? | `--machine=B200 --ladder=1,4,8` | `B200.json`, one list per rung |

`pytest -p qa_selection.plugin --help` documents the options. Outside `tests/integration/defs`,
also set `PYTHONPATH=tests`.

## Profiles

A profile is what the selection rules ask about a machine:

```json
"B200": {
  "sm": 100,
  "device_name": "B200",
  "device_memory_mib": 183359,
  "max_gpu_per_node": 8,
  "cpu_arch": "x86_64",
  "ladder": [1, 4, 8]
}
```

`sm` is the compute capability (`10.0` is 100), `device_name` the `nvidia-smi` name without
`NVIDIA `, `device_memory_mib` its `memory.total`, and `cpu_arch` as `uname -m` prints it.
`ladder` is optional: the default `--ladder`, ending at `max_gpu_per_node`. Edit or add one to try
another allocation policy.

**Cluster machines** are in [`core/profiles.json`](core/profiles.json). The cluster selects in a
CPU container before any GPU is allocated, so every machine it schedules needs an entry there. To add one, run the plugin on the machine
without `--machine`, and copy the `profile` of its `local.json` under the machine's name.

**This host** needs no entry. Without `--machine`, the plugin reads the profile from `nvidia-smi`,
not counting display GPUs (`DGX Display`), and records it as `profile` in `local.json`. When
`DIR/local.json` already exists, its `profile` is used instead and nothing is probed, so a pipeline
that has measured the node saves it there first:

```json
{"profile": {"sm": 90, "device_name": "H20", "device_memory_mib": 97871, "max_gpu_per_node": 4, "cpu_arch": "x86_64"}}
```

Its `profile` holds exactly the five fields, without `ladder`; use `--ladder` instead. A file
without a valid profile is a usage error; delete it to probe again. A host without `nvidia-smi`
must name `--machine`.

## What it decides

A test is kept when the machine's card may run it and a rung holds it:

| question | answered from |
|---|---|
| may the machine's card run it? | its `skipif` reasons, looked up in `rules.json`, and its device-name and memory markers, against the profile |
| does a rung hold it, and which? | its GPU-count markers: the smallest rung at least that large, or none above the largest |

A `skipif`'s condition is never evaluated: it describes the host that collected the test. The
plugin decides the list left after `--test-list` and waive filtering, and deselects through
pytest's own hook. `<machine>.json` records the profile, the counts and each test's outcome; each
`.ids` list holds one rung's node ids. The terminal shows the counts by rung and by reason, and
`-v` adds node ids. An unknown machine, a rung above the node, or a leftover list of this
machine's in the output directory is a usage error.

## Layout

```text
qa_selection/
  plugin.py        the pytest hooks and options
  host.py          no --machine: this host's profile
  core/            the selection logic, plain values in
    selection.py     one run's request, and its outcome per test
    report.py        the .ids lists and the JSON record
    ladder.py        --ladder: which rung holds a test
    machine.py       --machine: may the machine's card run a test
    marks.py         what a test asks for, read from its marks
  tests/           the acceptance tests
```

## Acceptance tests

```bash
pytest tests/qa_selection/tests      # 70 tests, no wheel
```

Each test states a command line and the answer expected back, derived from the production
decorators. [`test_local_machine.py`](tests/test_local_machine.py) needs `nvidia-smi` and expects
the dev box's RTX PRO 6000 Blackwell Workstation Edition; one case of
[`test_unprobed_host.py`](tests/test_unprobed_host.py) needs a host without it. To cover both, run
the suite in a `docker --gpus all` and then a `docker --runtime=runc` container, each with
`--probed-out-dir=DIR`.

| | principle | criteria |
|---|---|---|
| **Loading** | Only `-p qa_selection.plugin` loads the plugin; the integration `pytest.ini` does not | [`test_integration_ini.py`](tests/test_integration_ini.py) (1) |
| **Before collection** | A machine and a ladder are the whole command; without a machine, this host is the target | AC-4 [`test_options.py`](tests/test_options.py) (5), [`test_local_machine.py`](tests/test_local_machine.py) (2), [`test_unprobed_host.py`](tests/test_unprobed_host.py) (2), [`test_saved_profile.py`](tests/test_saved_profile.py) (9) |
| **During collection** | The target decides, never the collecting host, so a named machine needs no GPU: its architecture, wherever the mark sits, and its largest rung as the GPU count; a test above that rung is deselected with one reason | AC-1 [`test_arch.py`](tests/test_arch.py) (7), AC-2 [`test_gpu_count.py`](tests/test_gpu_count.py) (7), AC-5 [`test_largest_rung.py`](tests/test_largest_rung.py) (4), [`test_named_machine.py`](tests/test_named_machine.py) (1) |
| **After collection** | Each kept test is in exactly one list, its smallest rung's, the record carries the profile, and the terminal explains the run | AC-3 [`test_ladder.py`](tests/test_ladder.py) (6), AC-7 [`test_summary.py`](tests/test_summary.py) (5), [`test_named_machine.py`](tests/test_named_machine.py) (1), [`test_local_machine.py`](tests/test_local_machine.py) (1) |
| **Catalogue** | Each profile holds its machine's measured facts, and its ladder ends at the node; a run without `--ladder` uses it | [`test_profiles.py`](tests/test_profiles.py) (19) |

## Config

Three JSON files in `core/`. `rules.json` and `markers.json` mirror the integration suite's
`defs/conftest.py` and `defs/pytest.ini`;
[`scripts/check_qa_selection_rules.py`](../../scripts/check_qa_selection_rules.py) runs in
pre-commit and fails when they drift.

| file | holds |
|---|---|
| `profiles.json` | the cluster machines' [profiles](#profiles) |
| `rules.json` | one rule per `skipif` reason: the profile values that skip it |
| `markers.json` | each resource marker's declaration, the profile fact it bounds, and whether the conftest reads its closest mark or every level's |
