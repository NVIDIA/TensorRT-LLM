# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the modeling_v2 whole-model test runner, ``accuracy/modeling_v2/_e2e.py``.

The runner is loaded from its file, without its package, so only its pure parts run:
cases loading, path expansion, request building, the judgments over recorded outputs,
the benchmark reference override, and the result lines, with fakes for the engine and
the accuracy task.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

pytestmark = [
    pytest.mark.cpu_only,
    # ``skip_less_mpi_world_size`` is registered by the integration tests' pytest.ini only.
    pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning"),
]

_REPO = Path(__file__).resolve().parents[4]
_RUNNER = _REPO / "tests/integration/defs/accuracy/modeling_v2/_e2e.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_mv2_e2e_runner", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e2e = _load_runner()


def _document(**overrides: Any) -> dict[str, Any]:
    document = {
        "version": 1,
        "model_name": "Org/Model-1B",
        "model_dir": "${LLM_MODELS_ROOT}/Model-1B",
        "cases": [
            {
                "id": "core-e2e-tp1",
                "feature": "core",
                "test": "e2e",
                "member": "tp1",
                "devices": 1,
                "llm": {"modeling_v2": "require", "max_num_tokens": 512},
                "sampling": {"temperature": 0.0, "top_k": 1},
                "datasets": [
                    {"name": "generate", "kind": "generate", "prompts": ["Hello"]},
                    {
                        "name": "gsm8k",
                        "kind": "benchmark",
                        "task": "GSM8K",
                        "extra_acc_spec": None,
                        "evaluator_kwargs": {},
                    },
                ],
            },
            {
                "id": "mtp-acceptance-tp4",
                "devices": 4,
                "llm": {"modeling_v2": "require"},
                "datasets": [
                    {
                        "name": "acceptance",
                        "kind": "acceptance",
                        "workload": {"prompts": ["Count to ten."], "max_tokens": 64},
                        "floor": 1.9,
                        "floor_source": "accuracy.acceptance.min_absolute",
                    }
                ],
            },
        ],
    }
    document.update(overrides)
    return document


def _write_cases(tmp_path: Path, document: Any) -> Path:
    (tmp_path / e2e.CASES_FILE).write_text(yaml.safe_dump(document), encoding="utf-8")
    return tmp_path / "test_e2e.py"


# --------------------------------------------------------------------------- cases


def test_load_cases_ids_marks_and_defaults(tmp_path):
    params = e2e.load_cases(_write_cases(tmp_path, _document()))
    assert [p.id for p in params] == ["core-e2e-tp1", "mtp-acceptance-tp4"]
    for param, devices in zip(params, (1, 4)):
        (mark,) = param.marks
        assert mark.name == "skip_less_mpi_world_size"
        assert mark.args == (devices,)
    first, second = (p.values[0] for p in params)
    assert first["model_name"] == "Org/Model-1B"
    assert first["model_dir"] == "${LLM_MODELS_ROOT}/Model-1B"
    generate, benchmark = first["datasets"]
    assert generate == {
        "name": "generate",
        "kind": "generate",
        "prompts": [{"prompt": "Hello", "images": [], "render": "loader"}],
        "max_tokens": 32,
        "chunk_probe_tokens": None,
        "reuse_probe": False,
        "apply_chat_template": False,
    }
    assert benchmark["extra_acc_spec"] is None
    assert benchmark["evaluator_kwargs"] == {}
    assert "min_tokens" not in benchmark and first["extra_checks"] == []
    assert second["sampling"] == {}
    (acceptance,) = second["datasets"]
    assert acceptance["min_draft_iterations"] == e2e.DEFAULT_MIN_DRAFT_ITERATIONS
    assert acceptance["floor"] == 1.9
    assert acceptance["workload"]["apply_chat_template"] is False


def test_generate_defaults_to_the_boot_prompts_and_the_image_prompt():
    document = _document()
    document["cases"][0]["datasets"] = [
        {"name": "generate", "kind": "generate"},
        {
            "name": "image",
            "kind": "generate",
            "prompts": [{"images": ["a.png"], "render": "raw"}],
            "apply_chat_template": True,
        },
    ]
    default, image = e2e.parse_cases(document)[0]["datasets"]
    assert [p["prompt"] for p in default["prompts"]] == list(e2e.DEFAULT_GENERATE_PROMPTS)
    assert image["prompts"] == [
        {"prompt": e2e.DEFAULT_IMAGE_PROMPT, "images": ["a.png"], "render": "raw"}
    ]
    assert image["apply_chat_template"] is True


def _case(document: dict[str, Any]) -> dict[str, Any]:
    return document["cases"][0]


def _bad_version(d):
    d["version"] = 2


def _unknown_kind(d):
    _case(d)["datasets"][0]["kind"] = "perplexity"


def _unknown_case_key(d):
    _case(d)["timeout"] = 60


def _unknown_dataset_key(d):
    _case(d)["datasets"][0]["temperature"] = 0.5


def _unknown_top_key(d):
    d["models"] = []


def _model_in_llm(d):
    _case(d)["llm"]["model"] = "/x"


def _no_modeling_v2(d):
    del _case(d)["llm"]["modeling_v2"]


def _repeated_id(d):
    d["cases"][1]["id"] = "core-e2e-tp1"


def _repeated_dataset(d):
    _case(d)["datasets"][1]["name"] = "generate"


def _missing_devices(d):
    del _case(d)["devices"]


def _beam_search(d):
    _case(d)["sampling"]["use_beam_search"] = True


def _both_workloads(d):
    d["cases"][1]["datasets"][0]["workload"]["dataset"] = {"source": "x.jsonl"}


def _benchmark_without_model_name(d):
    del d["model_name"]


def _check_outside_tests(d):
    _case(d)["extra_checks"] = [{"check": "lib/checks.py::longest_request"}]


def _check_named_like_a_dataset(d):
    _case(d)["extra_checks"] = [{"check": "tests/x/checks.py::gsm8k"}]


def _check_with_args(d):
    _case(d)["extra_checks"] = [{"check": "tests/x/checks.py::f", "args": {"tokens": 8}}]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_bad_version, "version 2 is not supported"),
        (_unknown_kind, "unknown kind 'perplexity'"),
        (_unknown_case_key, r"unknown key\(s\) \['timeout'\]"),
        (_unknown_dataset_key, r"unknown key\(s\) \['temperature'\]"),
        (_unknown_top_key, r"unknown key\(s\) \['models'\]"),
        (_model_in_llm, "`model` comes from the file's `model_dir`"),
        (_no_modeling_v2, "needs `modeling_v2: require`"),
        (_repeated_id, r"repeated case id\(s\) \['core-e2e-tp1'\]"),
        (_repeated_dataset, r"repeated dataset name\(s\) \['generate'\]"),
        (_missing_devices, r"missing required key\(s\) \['devices'\]"),
        (_beam_search, "without beam search"),
        (_both_workloads, "exactly one of `prompts` and `dataset`"),
        (_benchmark_without_model_name, "needs the file's `model_name`"),
        (_check_outside_tests, r"expected `tests/<path>.py::<function>`"),
        (_check_named_like_a_dataset, r"name\(s\) \['gsm8k'\] repeat a dataset's"),
        (_check_with_args, r"unknown key\(s\) \['args'\]"),
    ],
)
def test_load_cases_refuses_a_bad_file(tmp_path, mutate, message):
    document = _document()
    mutate(document)
    with pytest.raises(e2e.CasesError, match=message):
        e2e.load_cases(_write_cases(tmp_path, document))


def test_load_cases_refuses_a_missing_or_malformed_file(tmp_path):
    with pytest.raises(e2e.CasesError, match="cannot read the cases file"):
        e2e.load_cases(tmp_path / "test_e2e.py")
    (tmp_path / e2e.CASES_FILE).write_text("cases: [unclosed", encoding="utf-8")
    with pytest.raises(e2e.CasesError, match="not valid YAML"):
        e2e.load_cases(tmp_path / "test_e2e.py")


def test_expand_models_root_replaces_only_the_prefix():
    calls = []

    def root() -> str:
        calls.append(1)
        return "/models"

    value = {
        "a": "${LLM_MODELS_ROOT}/x",
        "b": ["${LLM_MODELS_ROOT}", "keep/${LLM_MODELS_ROOT}", 3],
        "c": {"d": "${LLM_MODELS_ROOT}/y/z"},
    }
    assert e2e.expand_models_root(value, root) == {
        "a": "/models/x",
        "b": ["/models", "keep/${LLM_MODELS_ROOT}", 3],
        "c": {"d": "/models/y/z"},
    }
    calls.clear()
    assert e2e.expand_models_root({"a": "/abs", "b": [1]}, root) == {"a": "/abs", "b": [1]}
    assert not calls


def test_expand_case_expands_model_llm_images_and_dataset_source():
    document = _document()
    document["cases"][0]["llm"]["speculative_config"] = {
        "speculative_model": "${LLM_MODELS_ROOT}/drafter"
    }
    document["cases"][0]["datasets"][0]["prompts"] = [
        {"prompt": "What is this?", "images": ["${LLM_MODELS_ROOT}/img.png", "rel.png"]}
    ]
    document["cases"][1]["datasets"][0]["workload"] = {
        "dataset": {"source": "${LLM_MODELS_ROOT}/datasets/openai/gsm8k", "config": "main"}
    }
    first, second = (e2e.expand_case(c, lambda: "/m") for c in e2e.parse_cases(document))
    assert first["model_dir"] == "/m/Model-1B"
    assert first["llm"]["speculative_config"]["speculative_model"] == "/m/drafter"
    assert first["datasets"][0]["prompts"][0]["images"] == ["/m/img.png", "rel.png"]
    assert second["datasets"][0]["workload"]["dataset"]["source"] == "/m/datasets/openai/gsm8k"


# --------------------------------------------------------------------------- generate


def _generate(**overrides: Any) -> dict[str, Any]:
    document = _document()
    document["cases"][0]["datasets"] = [{"name": "generate", "kind": "generate", **overrides}]
    return e2e.parse_cases(document)[0]["datasets"][0]


def test_generate_requests_resolve_images_and_append_the_chunk_probe(tmp_path):
    dataset = _generate(
        prompts=["Hi", {"prompt": "Look", "images": ["img/a.png", "/abs/b.png"]}],
        max_tokens=8,
        chunk_probe_tokens=1025,
    )
    assert e2e.generate_requests(dataset, repo_root=tmp_path) == [
        {"id": "p0", "prompt": "Hi", "max_tokens": 8},
        {
            "id": "p1",
            "prompt": "Look",
            "max_tokens": 8,
            "images": [str(tmp_path / "img/a.png"), "/abs/b.png"],
            "render": "loader",
        },
        {"id": "chunk", "probe_tokens": 1025, "max_tokens": 1},
    ]


def _record(text: str = "Paris.", prompt_tokens: int = 6, cached: int | None = 0) -> dict:
    return {"text": text, "prompt_tokens": prompt_tokens, "cached_tokens": cached}


_REUSE = {"modeling_v2": "require", "kv_cache_config": {"enable_block_reuse": True}}
_NO_REUSE = {"modeling_v2": "require", "kv_cache_config": {"enable_block_reuse": False}}


def test_generate_passes_on_non_empty_answers():
    requests = e2e.generate_requests(_generate(prompts=["a", "b"]))
    outcome = e2e.judge_generate(
        requests, [_record(), _record()], max_num_tokens=512, llm_options={}, probe=None
    )
    assert outcome.passed, outcome.detail
    assert outcome.score is None
    assert "2 prompt(s) answered with non-empty text" in outcome.detail


def test_generate_fails_on_an_empty_answer():
    requests = e2e.generate_requests(_generate(prompts=["a", "b"]))
    outcome = e2e.judge_generate(
        requests, [_record(), _record(" \n")], max_num_tokens=512, llm_options={}, probe=None
    )
    assert not outcome.passed
    assert "the engine generated nothing for p1" in outcome.detail


def test_generate_chunk_probe_needs_a_context_past_max_num_tokens():
    requests = e2e.generate_requests(_generate(prompts=["a"], chunk_probe_tokens=1025))
    chunked = [_record(), _record(text="", prompt_tokens=1025)]
    passed = e2e.judge_generate(requests, chunked, max_num_tokens=512, llm_options={}, probe=None)
    assert passed.passed, passed.detail
    assert "past max_num_tokens 512 was prefilled in chunks" in passed.detail

    short = e2e.judge_generate(requests, chunked, max_num_tokens=2048, llm_options={}, probe=None)
    assert not short.passed
    assert "no request was longer than max_num_tokens 2048" in short.detail

    resumed = [_record(), _record(text="", prompt_tokens=1025, cached=600)]
    cached = e2e.judge_generate(requests, resumed, max_num_tokens=512, llm_options={}, probe=None)
    assert not cached.passed
    assert "the longest computed 425 prompt tokens" in cached.detail

    unknown = e2e.judge_generate(requests, chunked, max_num_tokens=None, llm_options={}, probe=None)
    assert not unknown.passed
    assert "max_num_tokens is unknown" in unknown.detail


@pytest.mark.parametrize(
    "options, probe, fails",
    [
        (_REUSE, {"status": "OBSERVED", "tokens": 96, "first": 0, "reused": 0}, True),
        (_REUSE, {"status": "OBSERVED", "tokens": 96, "first": 0, "reused": 64}, False),
        (_REUSE, {"status": "UNKNOWN", "skipped": "does not fit"}, False),
        (_REUSE, {"status": "UNKNOWN", "error": "RuntimeError: boom"}, False),
        (_REUSE, None, False),
        (_NO_REUSE, {"status": "OBSERVED", "tokens": 96, "first": 0, "reused": 0}, False),
        ({"modeling_v2": "require"}, {"status": "OBSERVED", "reused": 0}, False),
    ],
)
def test_generate_reuse_fails_only_when_reuse_is_asked_and_the_probe_saw_none(
    options, probe, fails
):
    requests = e2e.generate_requests(_generate(prompts=["a"]))
    outcome = e2e.judge_generate(
        requests, [_record()], max_num_tokens=512, llm_options=options, probe=probe
    )
    assert outcome.passed is not fails, outcome.detail
    if fails:
        assert outcome.detail.startswith("the engine reused nothing")
        assert "96-token prompt" in outcome.detail
    elif probe and probe.get("status") == "UNKNOWN":
        assert "the reuse probe observed nothing" in outcome.detail


# --------------------------------------------------------------------------- acceptance


def _reading(*requests: tuple[int, int]):
    return e2e.read_acceptance({"tokens": t, "iterations": i} for t, i in requests)


def test_acceptance_value_is_iteration_weighted():
    reading = _reading((100, 41), (10, 9))
    assert reading.value == pytest.approx(110 / 50)
    assert reading.draft_iterations == 40 + 8
    assert e2e.read_acceptance([]) is None
    assert "iteration-weighted 2.20" in reading.breakdown()


def test_acceptance_fails_a_workload_with_too_few_draft_iterations():
    outcome = e2e.judge_acceptance(_reading((12, 4), (9, 3)), min_draft_iterations=32)
    assert not outcome.passed
    assert "too short to measure acceptance" in outcome.detail
    assert "property of the workload, not of the target" in outcome.detail
    assert outcome.score == pytest.approx(21 / 7)


def test_acceptance_fails_at_or_below_one():
    outcome = e2e.judge_acceptance(_reading((64, 64)), min_draft_iterations=32)
    assert not outcome.passed
    assert "every draft was rejected" in outcome.detail
    assert outcome.score == 1.0


def test_acceptance_fails_below_the_floor_and_passes_at_it():
    reading = _reading((150, 101))
    below = e2e.judge_acceptance(
        reading, min_draft_iterations=32, floor=1.9, floor_source="anchor run 7"
    )
    assert not below.passed
    assert "vs floor 1.9000 (from anchor run 7): NOT met" in below.detail
    assert (below.reference, below.threshold) == (1.9, 1.9)
    met = e2e.judge_acceptance(reading, min_draft_iterations=32, floor=1.4, workload="w")
    assert met.passed
    assert met.detail.endswith("Workload: w.")


def test_acceptance_without_a_floor_is_held_to_one_alone():
    outcome = e2e.judge_acceptance(_reading((150, 101)), min_draft_iterations=32)
    assert outcome.passed
    assert "held to 1.0 alone" in outcome.detail
    assert outcome.reference is None


def test_acceptance_with_no_streamed_iteration_fails():
    outcome = e2e.judge_acceptance(None, min_draft_iterations=32)
    assert not outcome.passed
    assert "no acceptance to read" in outcome.detail


def test_acceptance_prompts_from_a_snapshot_directory(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "gsm8k"
    for config in ("main", "socratic"):
        for split in ("test", "train"):
            (root / config).mkdir(parents=True, exist_ok=True)
            table = pyarrow.table(
                {"question": [f"{config} {split} q{n}" for n in range(3)], "answer": ["1"] * 3}
            )
            pq.write_table(table, root / config / f"{split}-00000-of-00001.parquet")
    (root / ".git").mkdir()
    (root / ".git" / "test.json").write_text('["git row"]', encoding="utf-8")

    rows = e2e.load_rows(root, split="test", config="main")
    assert [r["question"] for r in rows] == ["main test q0", "main test q1", "main test q2"]
    every_config = e2e.load_rows(root, split="test", config=None)
    assert [r["question"] for r in every_config][::3] == ["main test q0", "socratic test q0"]
    assert len(every_config) == 6

    workload = {
        "prompts": None,
        "dataset": {
            "source": str(root),
            "num": 2,
            "text_field": "question",
            "split": "test",
            "config": "main",
            "template": "Q: {text}",
        },
    }
    prompts, source = e2e.acceptance_prompts(workload)
    assert prompts == ["Q: main test q0", "Q: main test q1"]
    assert source == f"2 prompt(s) from {root}"


def test_load_rows_reads_local_files_and_select_prompts_skips_unusable_rows(tmp_path):
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('{"question": "one"}\n\n{"question": ""}\n{"prompt": "two"}\n')
    assert e2e.load_rows(jsonl) == [
        {"question": "one"},
        {"question": ""},
        {"prompt": "two"},
    ]
    listed = tmp_path / "rows.yaml"
    listed.write_text(yaml.safe_dump({"data": ["three", {"text": "four"}]}))
    rows = e2e.load_rows(jsonl) + e2e.load_rows(listed) + [7]
    assert e2e.select_prompts(rows, num=10) == ["one", "two", "three", "four"]
    assert e2e.select_prompts(rows, num=2) == ["one", "two"]
    with pytest.raises(e2e.DatasetError, match="does not fit row"):
        e2e.select_prompts([{"question": "q"}], num=1, template="{text} {missing}")
    with pytest.raises(e2e.DatasetError, match="does not exist"):
        e2e.load_rows(tmp_path / "absent.jsonl")
    workload = {
        "prompts": None,
        "dataset": {
            "source": str(jsonl),
            "num": 5,
            "text_field": "x",
            "split": None,
            "config": None,
            "template": "{text}",
        },
    }
    with pytest.raises(e2e.DatasetError, match="yields no usable prompt"):
        e2e.acceptance_prompts(workload)


def test_a_debugging_run_keeps_only_the_datasets_it_names():
    case = e2e.parse_cases(_document())[0]
    assert e2e.select_datasets(case, None) == case
    assert e2e.select_datasets(case, " ") == case
    only = e2e.select_datasets(case, "generate")
    assert [d["name"] for d in only["datasets"]] == ["generate"]
    assert [d["name"] for d in case["datasets"]] == ["generate", "gsm8k"]  # left as it was
    both = e2e.select_datasets(case, "gsm8k, generate")
    assert [d["name"] for d in both["datasets"]] == ["generate", "gsm8k"]  # the case's order
    with pytest.raises(e2e.CasesError, match=r"MV2FLOW_E2E_DATASETS names \['mmlu'\]"):
        e2e.select_datasets(case, "generate,mmlu")
    # Extra checks are named like datasets; a subset keeps those it names.
    case["extra_checks"] = [{"check": "tests/x/c.py::f", "name": "f"}]
    assert e2e.select_datasets(case, "generate")["extra_checks"] == []
    assert e2e.select_datasets(case, "generate,f")["extra_checks"] == case["extra_checks"]
    with pytest.raises(e2e.CasesError, match="names no dataset"):
        e2e.select_datasets(case, "f")


# --------------------------------------------------------------------------- benchmark


def test_references_override_parsing():
    assert e2e.parse_references(None) == {}
    assert e2e.parse_references("  ") == {}
    references = e2e.parse_references('{"gsm8k": 71.5, "mmlu": 80}')
    assert references.get("gsm8k") == 71.5
    assert references.get("mmlu") == 80.0
    assert references.get("gpqa_diamond") is None
    with pytest.raises(ValueError, match="MV2FLOW_E2E_REFERENCES is not valid JSON"):
        e2e.parse_references('{"gsm8k": 71.5')
    with pytest.raises(ValueError, match="not an object"):
        e2e.parse_references("[71.5]")
    with pytest.raises(
        ValueError, match=r"MV2FLOW_E2E_REFERENCES\['gsm8k'\] is '71.5', not a number"
    ):
        e2e.parse_references('{"gsm8k": "71.5"}')


@dataclass
class _Params:
    """Stands in for ``accuracy_core.HypothesisTestingParams``: threshold = ref - 5."""

    ref_accuracy: float
    num_samples: int
    metric_name: str = "accuracy"
    alpha: float = 0.05
    beta: float = 0.2
    sigma: float = 50.0
    higher_is_better: bool = True
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.threshold is None:
            self.threshold = self.ref_accuracy - 5

    def assert_passing(self, accuracy: float) -> None:
        assert accuracy >= self.threshold, f"got {accuracy}, threshold {self.threshold}"


class _Evaluator:
    def __init__(self, score: float) -> None:
        self.score = score

    def evaluate(self, llm: Any, *args: Any) -> float:
        return self.score


@dataclass
class _Task:
    """Stands in for an ``AccuracyTask``: the same calls, in the same order, as its evaluate."""

    reference: list[dict[str, Any]]
    score: float
    asked: list[dict[str, Any]] = field(default_factory=list)

    DATASET = "gsm8k"
    METRIC_NAME = "accuracy"
    ALPHA, BETA, SIGMA, NUM_SAMPLES, HIGHER_IS_BETTER = 0.05, 0.2, 50, 1319, True

    def get_hypothesis_testing_params(self, **acc_specs: Any) -> _Params:
        self.asked.append(acc_specs)
        for entry in self.reference:
            if entry.get("extra_acc_spec") == acc_specs.get("extra_acc_spec"):
                return _Params(
                    ref_accuracy=entry["accuracy"],
                    num_samples=entry.get("num_samples", self.NUM_SAMPLES),
                    sigma=entry.get("sigma", self.SIGMA),
                    threshold=entry.get("threshold"),
                )
        raise ValueError(f"Not registered specs: {acc_specs}.")

    def create_evaluator(self, num_samples: int, extra_evaluator_kwargs: Any = None) -> Any:
        return _Evaluator(self.score)

    def evaluate(self, llm: Any, extra_acc_spec: Any = None, extra_evaluator_kwargs: Any = None):
        params = self.get_hypothesis_testing_params(extra_acc_spec=extra_acc_spec)
        evaluator = self.create_evaluator(params.num_samples, extra_evaluator_kwargs)
        score = evaluator.evaluate(llm, None, False)
        params.assert_passing(score)
        return score


_BENCHMARK = {
    "name": "gsm8k",
    "kind": "benchmark",
    "task": "GSM8K",
    "extra_acc_spec": None,
    "evaluator_kwargs": {},
}


def test_benchmark_judged_by_the_references_row():
    task = _Task(reference=[{"accuracy": 70.0, "threshold": 66.0}], score=67.0)
    outcome = e2e.evaluate_benchmark(task, "llm", _BENCHMARK, override=None, params_cls=_Params)
    assert outcome.passed, outcome.detail
    assert (outcome.score, outcome.reference, outcome.threshold) == (67.0, 70.0, 66.0)
    assert "from references/gsm8k.yaml" in outcome.detail


def test_benchmark_records_the_score_of_a_failed_hypothesis_test():
    task = _Task(reference=[{"accuracy": 70.0}], score=60.0)
    outcome = e2e.evaluate_benchmark(task, "llm", _BENCHMARK, override=None, params_cls=_Params)
    assert not outcome.passed
    assert (outcome.score, outcome.reference, outcome.threshold) == (60.0, 70.0, 65.0)
    assert outcome.detail.endswith("FAILED")


def test_benchmark_override_keeps_the_row_parameters_and_drops_its_threshold():
    task = _Task(reference=[{"accuracy": 10.0, "threshold": 1.0, "sigma": 30}], score=60.0)
    outcome = e2e.evaluate_benchmark(task, "llm", _BENCHMARK, override=80.0, params_cls=_Params)
    assert not outcome.passed
    assert (outcome.reference, outcome.threshold) == (80.0, 75.0)
    assert "from MV2FLOW_E2E_REFERENCES" in outcome.detail
    row = _Task.get_hypothesis_testing_params(task, extra_acc_spec=None)
    params = e2e.overridden_params(_Params, row, _Task, 80.0)
    assert (params.ref_accuracy, params.sigma, params.num_samples) == (80.0, 30, 1319)
    assert params.threshold == 75.0


def test_benchmark_override_without_a_row_uses_the_task_defaults():
    task = _Task(reference=[], score=78.0)
    with pytest.raises(ValueError, match="Not registered specs"):
        e2e.evaluate_benchmark(task, "llm", _BENCHMARK, override=None, params_cls=_Params)
    task = _Task(reference=[], score=78.0)
    outcome = e2e.evaluate_benchmark(task, "llm", _BENCHMARK, override=80.0, params_cls=_Params)
    assert outcome.passed, outcome.detail
    assert (outcome.score, outcome.reference, outcome.threshold) == (78.0, 80.0, 75.0)
    assert "1319 samples" in outcome.detail


class _Output:
    """Stands in for a ``RequestOutput``: its prompt's and its completion's token ids."""

    def __init__(self, prompt: int, generated: int) -> None:
        self.prompt_token_ids = [0] * prompt
        self.outputs = [SimpleNamespace(token_ids=[0] * generated)]


class _Llm:
    """An engine whose requests come back with the (prompt, completion) lengths given."""

    def __init__(self, *lengths: tuple[int, int]) -> None:
        self.lengths = list(lengths)
        self.asked = 0

    def generate_async(self, prompt: Any, *args: Any, **kwargs: Any) -> _Output:
        self.asked += 1
        return _Output(*self.lengths[self.asked - 1])


def _sending(llm: _Llm, outcome: Any = None) -> Any:
    """A dataset's run that sends one request per length the engine has, as a dataset does."""

    def run() -> Any:
        for _ in llm.lengths:
            llm.generate_async("question")
        return outcome or e2e.DatasetOutcome(passed=True, detail="scored")

    return run


def test_a_dataset_is_observed_with_every_request_it_sent_and_the_engine_left_as_it_was():
    llm = _Llm((100, 20), (3000, 50))
    outcome = e2e.observed(llm, _sending(llm))
    assert outcome.passed and len(outcome.requests) == 2
    assert outcome.tokens == {"requests": 2, "longest_prompt": 3000, "longest": 3050}
    assert "generate_async" not in vars(llm)
    line = e2e.result_line("core-e2e-tp1", _BENCHMARK, outcome)
    assert line["tokens"] == outcome.tokens and "requests" not in line
    # An engine without generate_async is left alone, and nothing is counted.
    assert e2e.observed("llm", lambda: e2e.DatasetOutcome(True, "x")).tokens is None


def test_request_lengths_skip_what_is_not_a_finished_request():
    assert e2e.request_lengths([]) is None
    assert e2e.request_lengths([SimpleNamespace(prompt_token_ids=None, outputs=[])]) is None
    beams = SimpleNamespace(
        prompt_token_ids=[0] * 8,
        outputs=[SimpleNamespace(token_ids=[0] * 3), SimpleNamespace(token_ids=[0] * 5)],
    )
    assert e2e.request_lengths([beams]) == {"requests": 1, "longest_prompt": 8, "longest": 13}


def _case_result(
    *datasets: tuple[str, list[tuple[int, int]]],
    model_dir: str = "/no/checkpoint",
    options: dict[str, Any] | None = None,
) -> Any:
    """A case's result: each dataset by name, with the (prompt, output) lengths it sent."""
    return e2e.CaseResult(
        id="core-sparse-tp1",
        feature="core",
        test="sparse",
        member="tp1",
        model_dir=model_dir,
        options=options or {},
        sampling={},
        llm=None,
        datasets=tuple(
            e2e.DatasetResult(
                name,
                "benchmark",
                True,
                "",
                70.0,
                70.0,
                66.0,
                tuple(_Output(p, g) for p, g in lengths),
            )  # fmt: skip
            for name, lengths in datasets
        ),
    )


def test_a_case_result_gives_its_requests_lengths_and_its_checkpoints_config(tmp_path):
    result = _case_result(("generate", [(20, 32)]), ("longbench_v2", [(9000, 40)]))
    assert result.dataset("longbench_v2").lengths() == (e2e.RequestLength(9000, 9040),)
    assert result.lengths() == (e2e.RequestLength(20, 52), e2e.RequestLength(9000, 9040))
    assert result.dataset("mmlu") is None and result.model_config() == {}
    (tmp_path / "config.json").write_text('{"index_topk": 2048}')
    assert _case_result(model_dir=str(tmp_path)).model_config() == {"index_topk": 2048}


def test_a_case_records_each_datasets_result_for_its_extra_checks():
    case = e2e.parse_cases(_document())[0]
    llm = _Llm((10, 2))
    dataset_results: list[Any] = []

    def run_one(dataset):
        return e2e.observed(llm, _sending(llm)) if dataset["kind"] == "benchmark" else (
            e2e.DatasetOutcome(passed=True, detail="answered"))  # fmt: skip

    e2e.run_datasets(case, run_one, dataset_results)
    assert [(d.name, d.kind, len(d.requests)) for d in dataset_results] == [
        ("generate", "generate", 0),
        ("gsm8k", "benchmark", 1),
    ]


def test_extra_checks_run_in_order_and_each_fails_or_passes_on_its_own(capfd):
    def saw(result):
        return f"{result.dataset('longbench_v2').name} seen"

    def short(result):
        raise AssertionError("no request long enough")

    def broken(result):
        raise KeyError("datasets")

    checks = {"saw": saw, "short": short, "broken": broken}
    listed = [
        {"check": f"tests/x/c.py::{name}", "name": name}
        for name in ("saw", "short", "broken", "absent")
    ]

    def load(check):
        name = check.rpartition("::")[2]
        if name not in checks:
            raise e2e.CasesError(f"extra check {check}: tests/x/c.py defines no function {name}")
        return checks[name]

    lines = e2e.run_extra_checks(_case_result(("longbench_v2", [])), listed, load)
    assert [(line["dataset"], line["kind"], line["passed"]) for line in lines] == [
        ("saw", "extra_check", True),
        ("short", "extra_check", False),
        ("broken", "extra_check", False),
        ("absent", "extra_check", False),
    ]
    assert lines[0]["detail"] == "longbench_v2 seen"
    assert lines[1]["detail"] == "no request long enough"
    assert lines[2]["detail"] == "raised KeyError: 'datasets'"
    assert "defines no function absent" in lines[3]["detail"]
    with pytest.raises(pytest.fail.Exception, match=r"3 of 4 .*\n- short \(extra_check\)"):
        e2e.assert_all_passed("core-sparse-tp1", lines)


def test_an_extra_check_is_loaded_from_its_file_in_the_tree(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "checks.py").write_text("def seen(result):\n    return 'seen'\n")
    check = e2e.load_extra_check("tests/checks.py::seen", repo_root=tmp_path)
    assert check(None) == "seen"
    with pytest.raises(e2e.CasesError, match="defines no function other"):
        e2e.load_extra_check("tests/checks.py::other", repo_root=tmp_path)
    with pytest.raises(e2e.CasesError, match="is not a file of the tree"):
        e2e.load_extra_check("tests/none.py::seen", repo_root=tmp_path)


def _load_shared_checks():
    path = _REPO / "tests/integration/defs/accuracy/modeling_v2/_extra_checks.py"
    spec = importlib.util.spec_from_file_location("_mv2_shared_extra_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparse_topk_exceeded_holds_the_run_to_the_engines_own_budget(tmp_path):
    checks = _load_shared_checks()
    sent = (("generate", [(20, 32)]), ("longbench_v2", [(9000, 40), (3000, 10)]))
    # The engine's options set the budget.
    options = {"sparse_attention_config": {"algorithm": "dsa", "index_topk": 2048}}
    said = checks.sparse_topk_exceeded(_case_result(*sent, options=options))
    assert said == "the longest request 9040 tokens (prompt 9000) > TopK 2048, of 3"
    # Else the checkpoint does, at its top level or in its text config.
    (tmp_path / "config.json").write_text('{"text_config": {"index_topk": 16384}}')
    with pytest.raises(AssertionError, match="no request outgrew the TopK budget 16384"):
        checks.sparse_topk_exceeded(_case_result(*sent, model_dir=str(tmp_path)))
    # With no budget, or no length seen, nothing is shown.
    with pytest.raises(AssertionError, match="no TopK budget"):
        checks.sparse_topk_exceeded(_case_result(*sent))
    with pytest.raises(AssertionError, match="no request's length was seen"):
        checks.sparse_topk_exceeded(_case_result(("longbench_v2", []), options=options))


# --------------------------------------------------------------------------- results


def _two_dataset_case() -> dict[str, Any]:
    document = _document()
    return e2e.parse_cases(document)[0]


def test_result_lines_are_printed_and_appended_to_the_results_file(tmp_path, monkeypatch, capfd):
    results = tmp_path / "out" / "results.jsonl"
    monkeypatch.setenv(e2e.RESULTS_ENV, str(results))
    case = _two_dataset_case()
    outcomes = {
        "generate": e2e.DatasetOutcome(passed=True, detail="ok"),
        "gsm8k": e2e.DatasetOutcome(
            passed=True, detail="fine", score=71.0, reference=70.0, threshold=66.0
        ),
    }
    lines = e2e.run_datasets(case, lambda dataset: outcomes[dataset["name"]])
    expected = [
        {
            "case": "core-e2e-tp1",
            "dataset": "generate",
            "kind": "generate",
            "passed": True,
            "score": None,
            "reference": None,
            "threshold": None,
            "detail": "ok",
        },
        {
            "case": "core-e2e-tp1",
            "dataset": "gsm8k",
            "kind": "benchmark",
            "passed": True,
            "score": 71.0,
            "reference": 70.0,
            "threshold": 66.0,
            "detail": "fine",
        },
    ]
    assert lines == expected
    written = [json.loads(line) for line in results.read_text().splitlines()]
    assert written == expected
    printed = [
        json.loads(line[len(e2e.RESULT_PREFIX) + 1 :])
        for line in capfd.readouterr().out.splitlines()
        if line.startswith(e2e.RESULT_PREFIX + " ")
    ]
    assert printed == expected
    e2e.assert_all_passed(case["id"], lines)


def test_without_the_variable_results_are_only_printed(tmp_path, monkeypatch, capfd):
    monkeypatch.delenv(e2e.RESULTS_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    e2e.emit_result({"case": "c", "dataset": "d", "passed": True})
    assert capfd.readouterr().out.startswith(f"{e2e.RESULT_PREFIX} {{")
    assert not list(tmp_path.iterdir())


def test_every_dataset_runs_and_the_case_fails_naming_each_failure(tmp_path, monkeypatch):
    results = tmp_path / "results.jsonl"
    monkeypatch.setenv(e2e.RESULTS_ENV, str(results))
    case = _two_dataset_case()
    case["datasets"] = [
        {"name": name, "kind": "generate"} for name in ("first", "failing", "raising", "last")
    ]
    ran = []

    def run_one(dataset: dict[str, Any]) -> Any:
        ran.append(dataset["name"])
        if dataset["name"] == "failing":
            return e2e.DatasetOutcome(passed=False, detail="the engine generated nothing for p0")
        if dataset["name"] == "raising":
            raise RuntimeError("engine fell over")
        return e2e.DatasetOutcome(passed=True, detail="ok")

    lines = e2e.run_datasets(case, run_one)
    assert ran == ["first", "failing", "raising", "last"]
    assert [line["passed"] for line in lines] == [True, False, False, True]
    assert lines[2]["detail"] == "raised RuntimeError: engine fell over"
    assert len(results.read_text().splitlines()) == 4
    with pytest.raises(pytest.fail.Exception) as failure:
        e2e.assert_all_passed(case["id"], lines)
    message = str(failure.value)
    assert "core-e2e-tp1: 2 of 4 dataset(s) failed" in message
    assert "- failing (generate): the engine generated nothing for p0" in message
    assert "- raising (generate): raised RuntimeError: engine fell over" in message
    assert "first" not in message and "last" not in message
