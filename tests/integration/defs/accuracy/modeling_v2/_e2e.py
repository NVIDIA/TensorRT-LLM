# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner of the whole-model tests the trtllm_modeling_v2 agent flow generates for a target.

A target's generated ``test_e2e.py`` reads the ``e2e_cases.yaml`` beside it with
:func:`load_cases` and runs each case with :func:`run_case`: one ``LLM`` per case, and
every dataset of the case in order on it. A dataset is one of three kinds:

* ``generate``: prompts streamed one at a time (text ones optionally through the chat
  template), each of which must be answered with non-empty text; optionally a prompt the
  engine must prefill in chunks, and a probe that sends one prompt twice and reads what
  the KV cache reused.
* ``benchmark``: an ``accuracy_core`` task judged by its hypothesis test, against the
  references row it matches or against the score ``MV2FLOW_E2E_REFERENCES`` names.
* ``acceptance``: a drafter's acceptance length, the generated tokens of a workload over
  its streamed iterations, held above 1.0 and above an optional floor.

Every dataset writes one result line (:func:`emit_result`) and runs even after an earlier
one failed; the case then fails naming every failing dataset (:func:`assert_all_passed`).

Extra checks
------------

A case may name extra checks: functions of the tree that judge what its datasets did,
beyond what each dataset judges itself (that sparse attention's top-k selection ran,
that the cache was hit, ...). One is defined as::

    def sparse_topk_exceeded(result: CaseResult) -> str | None:

* **Where.** A public module-level function of one of two files: the shared
  ``_extra_checks.py`` beside this runner, or a target's own ``extra_checks.py``
  beside its generated ``test_e2e.py``. A spec names it by the function's name; the
  flow looks it up in the target's file, then the shared one, refuses a name both
  define, and writes where it is into the case (``check: <file>::<name>``). A
  function whose name starts with ``_`` is a helper, never a check. Each file is
  loaded by its path, so it imports by absolute names only.
* **Arguments.** The case's result (:class:`CaseResult`), and nothing else. A check
  derives what it holds the run to from the result (the engine's options, the
  checkpoint's config), and a target that needs another bar writes its own check.
  :class:`CaseResult` and :class:`DatasetResult` give what most checks read
  (:meth:`CaseResult.lengths`, :meth:`CaseResult.model_config`).
* **When.** Once per case, after every dataset of it, in the order the spec lists them,
  on the engine the datasets ran on, still open. Each writes one result line of kind
  ``extra_check``, named by the function.
* **It reads; it does not act.** It looks at :class:`CaseResult`: the datasets' outcomes,
  every request each sent (its ``RequestOutput``), the engine's options, the checkpoint
  and the engine (``get_stats()``, its tokenizer). It sends no request, changes no state of the engine,
  writes no file outside a temporary directory and opens no connection. A workload of its
  own is a dataset of the catalog, or a test of the tree run with ``pytest``.
* **Verdict.** It fails by raising ``AssertionError``, whose message says why; any other
  exception fails it as raised. It passes by returning ``None`` or a short text saying
  what it saw, which the result line carries as evidence. The same run gives the same
  verdict: no randomness, no clock.

At module level only the standard library, ``yaml`` and ``pytest`` are imported, so
loading cases and every judgment over recorded outputs run without a GPU. The engine,
``accuracy_core`` and ``conftest`` are imported by :func:`run_case` and what it calls.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import sys
import traceback
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import yaml

#: The cases file a generated ``test_e2e.py`` reads, beside it.
CASES_FILE = "e2e_cases.yaml"
#: The only ``version`` of the cases file this runner reads.
CASES_VERSION = 1
#: Names a file every result line is appended to, when set.
RESULTS_ENV = "MV2FLOW_E2E_RESULTS"
#: A JSON object ``{dataset name: score}`` that replaces a benchmark's reference score.
REFERENCES_ENV = "MV2FLOW_E2E_REFERENCES"
#: Comma-separated dataset names: run only these of each case (a debugging run's subset).
DATASETS_ENV = "MV2FLOW_E2E_DATASETS"
#: What each result line printed to stdout starts with.
RESULT_PREFIX = "MV2_E2E_RESULT"
#: A string starting with this has it replaced by ``llm_models_root()``.
MODELS_ROOT_VAR = "${LLM_MODELS_ROOT}"
#: The repository root, which relative image paths are resolved against.
REPO_ROOT = Path(__file__).resolve().parents[5]

#: What a ``generate`` dataset sends when it declares no prompts.
DEFAULT_GENERATE_PROMPTS = (
    "The capital of France is",
    "In one sentence, photosynthesis is",
)
#: What an image prompt asks when it names images and no text.
DEFAULT_IMAGE_PROMPT = "Describe this image in one sentence."
DEFAULT_GENERATE_MAX_TOKENS = 32
DEFAULT_ACCEPTANCE_MAX_TOKENS = 256
#: Generation iterations an acceptance workload must reach before its rate is read.
DEFAULT_MIN_DRAFT_ITERATIONS = 32
#: Prompts an acceptance workload takes from a dataset when it names no ``num``.
DEFAULT_DATASET_PROMPTS = 32

KINDS = ("generate", "benchmark", "acceptance")
RENDERS = ("loader", "raw")
_CASE_ID = re.compile(r"[\w.\-]+")


class CasesError(ValueError):
    """An ``e2e_cases.yaml`` outside the version-1 schema; the message names the place."""


class DatasetError(ValueError):
    """An acceptance workload's dataset could not be read; the message names what and where."""


@dataclass(frozen=True)
class DatasetOutcome:
    """The verdict on one dataset of a case.

    Attributes:
        passed: Whether the dataset passed.
        detail: What was measured and why it passed or failed.
        score: The benchmark score or the acceptance length; ``None`` for ``generate``.
        reference: The reference score or acceptance floor it was held to.
        threshold: The value the score had to reach.
        tokens: The lengths of the requests it sent (:func:`request_lengths`); ``None``
            when none were seen.
        requests: Every request it sent through the engine, as the outputs it returned.
    """

    passed: bool
    detail: str
    score: float | None = None
    reference: float | None = None
    threshold: float | None = None
    tokens: Mapping[str, int] | None = None
    requests: tuple[Any, ...] = field(default=(), repr=False, compare=False)


@dataclass(frozen=True)
class RequestLength:
    """One request's length in tokens.

    Attributes:
        prompt: Its prompt's.
        total: Its prompt's and its longest completion's together.
    """

    prompt: int
    total: int


def request_length(output: Any) -> RequestLength | None:
    """The length of the request ``output`` answers; ``None`` without its prompt or a completion."""
    prompt = getattr(output, "prompt_token_ids", None)
    completions = getattr(output, "outputs", None)
    if prompt is None or not completions:
        return None
    generated = max(len(getattr(c, "token_ids", None) or ()) for c in completions)
    return RequestLength(prompt=len(prompt), total=len(prompt) + generated)


@dataclass(frozen=True)
class DatasetResult:
    """One dataset's result in a case, for an extra check to read.

    Attributes:
        name: The dataset's name in the case.
        kind: ``generate``, ``benchmark`` or ``acceptance``.
        passed: Whether the dataset passed its own judgment.
        detail: What it measured and why it passed or failed.
        score: The benchmark score or acceptance length; ``None`` for ``generate``.
        reference: The reference or floor it was held to.
        threshold: The value the score had to reach.
        requests: The ``RequestOutput`` of every request it sent, in the order sent:
            ``prompt_token_ids``, ``outputs[i].token_ids``, the request's metrics.
    """

    name: str
    kind: str
    passed: bool
    detail: str
    score: float | None
    reference: float | None
    threshold: float | None
    requests: tuple[Any, ...]

    def lengths(self) -> tuple[RequestLength, ...]:
        """The length of each request it sent whose length is known, in the order sent."""
        return tuple(n for n in map(request_length, self.requests) if n is not None)


@dataclass(frozen=True)
class CaseResult:
    """A case's result, the one argument an extra check is given positionally.

    Attributes:
        id: The case's id.
        feature: The feature it proves, as the flow names it.
        test: The test of the spec it is.
        member: The member it runs at.
        model_dir: The checkpoint its engine loaded.
        options: The LLM-API options its engine was built with.
        sampling: Its sampling parameters.
        llm: The engine, still open: read it, send it nothing.
        datasets: Its datasets' results, in order.
    """

    id: str
    feature: str | None
    test: str | None
    member: str | None
    model_dir: str
    options: Mapping[str, Any]
    sampling: Mapping[str, Any]
    llm: Any
    datasets: tuple[DatasetResult, ...]

    def dataset(self, name: str) -> DatasetResult | None:
        """The result of the dataset named ``name``; ``None`` when it did not run."""
        return next((d for d in self.datasets if d.name == name), None)

    def lengths(self) -> tuple[RequestLength, ...]:
        """The length of every request its datasets sent whose length is known, dataset by dataset."""
        return tuple(n for d in self.datasets for n in d.lengths())

    def model_config(self) -> dict[str, Any]:
        """The checkpoint's ``config.json``; ``{}`` when it has none (a Mistral-native one)."""
        path = Path(self.model_dir) / "config.json"
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)


# --------------------------------------------------------------------------- cases


def _mapping(where: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CasesError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _keys(where: str, value: Any, required: Iterable[str], optional: Iterable[str]) -> None:
    data = _mapping(where, value)
    required = tuple(required)
    allowed = set(required) | set(optional)
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise CasesError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")
    missing = [key for key in required if key not in data]
    if missing:
        raise CasesError(f"{where}: missing required key(s) {missing}")


def _text(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CasesError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _optional_text(where: str, value: Any) -> str | None:
    return None if value is None else _text(where, value)


def _integer(where: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CasesError(f"{where}: expected an integer >= {minimum}, got {value!r}")
    return value


def _flag(where: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise CasesError(f"{where}: expected true or false, got {value!r}")
    return value


def _items(where: str, value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise CasesError(f"{where}: expected a non-empty list, got {value!r}")
    return value


def _parse_prompt(where: str, entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        return {"prompt": _text(where, entry), "images": [], "render": "loader"}
    _keys(where, entry, (), ("prompt", "images", "render"))
    images_value = entry.get("images")
    if images_value is None:
        images_value = []
    if not isinstance(images_value, list):
        raise CasesError(f"{where}.images: expected a list of paths, got {images_value!r}")
    images = [_text(f"{where}.images[{n}]", image) for n, image in enumerate(images_value)]
    if entry.get("prompt") is not None:
        prompt = _text(f"{where}.prompt", entry["prompt"])
    elif images:
        prompt = DEFAULT_IMAGE_PROMPT
    else:
        raise CasesError(f"{where}: a prompt mapping needs `prompt`, `images` or both")
    render = entry.get("render") or "loader"
    if render not in RENDERS:
        raise CasesError(f"{where}.render: expected one of {list(RENDERS)}, got {render!r}")
    return {"prompt": prompt, "images": images, "render": render}


def _parse_generate(where: str, data: Mapping[str, Any]) -> dict[str, Any]:
    _keys(
        where,
        data,
        ("name", "kind"),
        ("prompts", "max_tokens", "chunk_probe_tokens", "reuse_probe", "apply_chat_template"),
    )
    declared = data.get("prompts")
    entries = (
        list(DEFAULT_GENERATE_PROMPTS) if declared is None else _items(f"{where}.prompts", declared)
    )
    max_tokens = data.get("max_tokens")
    chunk = data.get("chunk_probe_tokens")
    reuse = data.get("reuse_probe")
    chat = data.get("apply_chat_template")
    return {
        "name": data["name"],
        "kind": "generate",
        "prompts": [_parse_prompt(f"{where}.prompts[{n}]", e) for n, e in enumerate(entries)],
        "max_tokens": DEFAULT_GENERATE_MAX_TOKENS
        if max_tokens is None
        else _integer(f"{where}.max_tokens", max_tokens, minimum=1),
        "chunk_probe_tokens": None
        if chunk is None
        else _integer(f"{where}.chunk_probe_tokens", chunk, minimum=1),
        "reuse_probe": False if reuse is None else _flag(f"{where}.reuse_probe", reuse),
        "apply_chat_template": False
        if chat is None
        else _flag(f"{where}.apply_chat_template", chat),
    }


def _parse_benchmark(where: str, data: Mapping[str, Any]) -> dict[str, Any]:
    _keys(where, data, ("name", "kind", "task"), ("extra_acc_spec", "evaluator_kwargs"))
    task = _text(f"{where}.task", data["task"])
    if not task.isidentifier():
        raise CasesError(f"{where}.task: expected an accuracy_core task class name, got {task!r}")
    kwargs = data.get("evaluator_kwargs")
    return {
        "name": data["name"],
        "kind": "benchmark",
        "task": task,
        "extra_acc_spec": _optional_text(f"{where}.extra_acc_spec", data.get("extra_acc_spec")),
        "evaluator_kwargs": None
        if kwargs is None
        else dict(_mapping(f"{where}.evaluator_kwargs", kwargs)),
    }


def _parse_workload(where: str, data: Any) -> dict[str, Any]:
    _keys(where, data, (), ("prompts", "dataset", "max_tokens", "apply_chat_template"))
    has_prompts = data.get("prompts") is not None
    if has_prompts == (data.get("dataset") is not None):
        raise CasesError(f"{where}: declare exactly one of `prompts` and `dataset`")
    dataset = None
    prompts = None
    if has_prompts:
        prompts = [
            _text(f"{where}.prompts[{n}]", p)
            for n, p in enumerate(_items(f"{where}.prompts", data["prompts"]))
        ]
    else:
        spec = data["dataset"]
        here = f"{where}.dataset"
        _keys(here, spec, ("source",), ("num", "text_field", "split", "config", "template"))
        num = spec.get("num")
        dataset = {
            "source": _text(f"{here}.source", spec["source"]),
            "num": DEFAULT_DATASET_PROMPTS
            if num is None
            else _integer(f"{here}.num", num, minimum=1),
            "text_field": _optional_text(f"{here}.text_field", spec.get("text_field")),
            "split": _optional_text(f"{here}.split", spec.get("split"))
            if "split" in spec
            else "test",
            "config": _optional_text(f"{here}.config", spec.get("config")),
            "template": _optional_text(f"{here}.template", spec.get("template")) or "{text}",
        }
    max_tokens = data.get("max_tokens")
    chat = data.get("apply_chat_template")
    return {
        "prompts": prompts,
        "dataset": dataset,
        "max_tokens": DEFAULT_ACCEPTANCE_MAX_TOKENS
        if max_tokens is None
        else _integer(f"{where}.max_tokens", max_tokens, minimum=1),
        "apply_chat_template": False
        if chat is None
        else _flag(f"{where}.apply_chat_template", chat),
    }


def _parse_acceptance(where: str, data: Mapping[str, Any]) -> dict[str, Any]:
    _keys(
        where,
        data,
        ("name", "kind", "workload"),
        ("min_draft_iterations", "floor", "floor_source"),
    )
    minimum = data.get("min_draft_iterations")
    floor = data.get("floor")
    if floor is not None and (isinstance(floor, bool) or not isinstance(floor, (int, float))):
        raise CasesError(f"{where}.floor: expected a number or null, got {floor!r}")
    return {
        "name": data["name"],
        "kind": "acceptance",
        "workload": _parse_workload(f"{where}.workload", data["workload"]),
        "min_draft_iterations": DEFAULT_MIN_DRAFT_ITERATIONS
        if minimum is None
        else _integer(f"{where}.min_draft_iterations", minimum, minimum=0),
        "floor": None if floor is None else float(floor),
        "floor_source": _optional_text(f"{where}.floor_source", data.get("floor_source")),
    }


_DATASET_PARSERS: dict[str, Callable[[str, Mapping[str, Any]], dict[str, Any]]] = {
    "generate": _parse_generate,
    "benchmark": _parse_benchmark,
    "acceptance": _parse_acceptance,
}


def _parse_dataset(where: str, data: Any) -> dict[str, Any]:
    data = _mapping(where, data)
    kind = data.get("kind")
    if kind not in KINDS:
        raise CasesError(f"{where}: unknown kind {kind!r}; expected one of {list(KINDS)}")
    _text(f"{where}.name", data.get("name"))
    return _DATASET_PARSERS[kind](where, data)


_EXTRA_CHECK = re.compile(r"(tests/[^:]+\.py)::([A-Za-z_]\w*)")


def _parse_extra_check(where: str, data: Any) -> dict[str, Any]:
    _keys(where, data, ("check",), ())
    check = _text(f"{where}.check", data["check"])
    match = _EXTRA_CHECK.fullmatch(check)
    if match is None or Path(match.group(1)).is_absolute() or ".." in Path(match.group(1)).parts:
        raise CasesError(
            f"{where}.check: expected `tests/<path>.py::<function>`, a function of the tree, "
            f"got {check!r}"
        )
    return {"check": check, "name": match.group(2)}


def _parse_sampling(where: str, value: Any) -> dict[str, Any]:
    sampling = {} if value is None else dict(_mapping(where, value))
    if sampling.get("n", 1) != 1 or sampling.get("use_beam_search"):
        raise CasesError(f"{where}: the checks read one sequence per request, without beam search")
    return sampling


def _parse_case(where: str, data: Any, model_name: str | None, model_dir: str) -> dict[str, Any]:
    _keys(
        where,
        data,
        ("id", "devices", "llm", "datasets"),
        ("feature", "test", "member", "sampling", "extra_checks"),
    )
    case_id = _text(f"{where}.id", data["id"])
    if not _CASE_ID.fullmatch(case_id):
        raise CasesError(f"{where}.id: {case_id!r} may hold only letters, digits, '_', '.', '-'")
    where = f"{where} ({case_id})"
    llm = dict(_mapping(f"{where}.llm", data["llm"]))
    if "model" in llm:
        raise CasesError(f"{where}.llm: `model` comes from the file's `model_dir`, not `llm`")
    if llm.get("modeling_v2") != "require":
        raise CasesError(
            f"{where}.llm: needs `modeling_v2: require`, so the target itself is measured "
            f"and not a fallback to the built-in model (got {llm.get('modeling_v2')!r})"
        )
    datasets = [
        _parse_dataset(f"{where}.datasets[{n}]", d)
        for n, d in enumerate(_items(f"{where}.datasets", data["datasets"]))
    ]
    checks = [
        _parse_extra_check(f"{where}.extra_checks[{n}]", c)
        for n, c in enumerate(data.get("extra_checks") or [])
    ]
    names = [d["name"] for d in datasets]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise CasesError(f"{where}.datasets: repeated dataset name(s) {repeated}")
    named = names + [c["name"] for c in checks]
    repeated = sorted({name for name in named if named.count(name) > 1})
    if repeated:
        raise CasesError(
            f"{where}.extra_checks: name(s) {repeated} repeat a dataset's or another check's"
        )
    if model_name is None and any(d["kind"] == "benchmark" for d in datasets):
        raise CasesError(f"{where}: a benchmark dataset needs the file's `model_name`")
    return {
        "id": case_id,
        "feature": _optional_text(f"{where}.feature", data.get("feature")),
        "test": _optional_text(f"{where}.test", data.get("test")),
        "member": _optional_text(f"{where}.member", data.get("member")),
        "devices": _integer(f"{where}.devices", data["devices"], minimum=1),
        "model_name": model_name,
        "model_dir": model_dir,
        "llm": llm,
        "sampling": _parse_sampling(f"{where}.sampling", data.get("sampling")),
        "datasets": datasets,
        "extra_checks": checks,
    }


def parse_cases(data: Any, where: str = CASES_FILE) -> list[dict[str, Any]]:
    """Validate a loaded cases document and return its cases with every default filled in.

    Args:
        data: The YAML document, as ``yaml.safe_load`` returns it.
        where: The name of the document in error messages.

    Returns:
        One mapping per case, carrying the file's ``model_name`` and ``model_dir``.

    Raises:
        CasesError: The document is not a version-1 cases file: an unsupported version,
            a missing or unknown key, an unknown dataset kind, or a value of the wrong type.
    """
    document = _mapping(where, data)
    version = document.get("version")
    if isinstance(version, bool) or version != CASES_VERSION:
        raise CasesError(
            f"{where}: version {version!r} is not supported; this runner reads version "
            f"{CASES_VERSION}"
        )
    _keys(where, document, ("version", "model_dir", "cases"), ("model_name",))
    model_dir = _text(f"{where}.model_dir", document["model_dir"])
    model_name = _optional_text(f"{where}.model_name", document.get("model_name"))
    cases = [
        _parse_case(f"{where}: cases[{n}]", case, model_name, model_dir)
        for n, case in enumerate(_items(f"{where}.cases", document["cases"]))
    ]
    ids = [case["id"] for case in cases]
    repeated = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if repeated:
        raise CasesError(f"{where}: repeated case id(s) {repeated}")
    return cases


def read_cases(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read and validate a cases file (:func:`parse_cases`).

    Raises:
        CasesError: The file cannot be read, is not YAML, or is not a version-1 cases file.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise CasesError(f"{source}: cannot read the cases file: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CasesError(f"{source}: not valid YAML: {exc}") from exc
    return parse_cases(data, where=str(source))


def load_cases(test_file: str | os.PathLike[str]) -> list[Any]:
    """The cases of the ``e2e_cases.yaml`` beside ``test_file``, as pytest parameters.

    Each parameter is the case mapping, with the case id as its pytest id and a
    ``skip_less_mpi_world_size`` mark for its ``devices`` (checks local GPUs on a single-rank
    run and MPI world size under ``trtllm-llmapi-launch``).

    Raises:
        CasesError: The cases file is missing or invalid, failing the collection.
    """
    path = Path(test_file).resolve().parent / CASES_FILE
    return [
        pytest.param(
            case, id=case["id"], marks=[pytest.mark.skip_less_mpi_world_size(case["devices"])]
        )
        for case in read_cases(path)
    ]


def expand_models_root(value: Any, models_root: Callable[[], str]) -> Any:
    """``value`` with every string starting with ``${LLM_MODELS_ROOT}`` rooted at ``models_root()``.

    Mappings and lists are expanded recursively; ``models_root`` is called only for a
    string that needs it.
    """
    if isinstance(value, str):
        if value.startswith(MODELS_ROOT_VAR):
            return models_root() + value[len(MODELS_ROOT_VAR) :]
        return value
    if isinstance(value, Mapping):
        return {key: expand_models_root(item, models_root) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_models_root(item, models_root) for item in value]
    return value


def expand_case(case: Mapping[str, Any], models_root: Callable[[], str]) -> dict[str, Any]:
    """A copy of ``case`` with ``${LLM_MODELS_ROOT}`` expanded where paths are named.

    That is ``model_dir``, every string under ``llm``, the image paths of ``generate``
    prompts and the dataset ``source`` of an acceptance workload.
    """
    expanded = copy.deepcopy(dict(case))
    expanded["model_dir"] = expand_models_root(case["model_dir"], models_root)
    expanded["llm"] = expand_models_root(case["llm"], models_root)
    for dataset in expanded["datasets"]:
        if dataset["kind"] == "generate":
            for entry in dataset["prompts"]:
                entry["images"] = expand_models_root(entry["images"], models_root)
        elif dataset["kind"] == "acceptance" and dataset["workload"]["dataset"]:
            source = dataset["workload"]["dataset"]["source"]
            dataset["workload"]["dataset"]["source"] = expand_models_root(source, models_root)
    return expanded


# --------------------------------------------------------------------------- results


def result_line(
    case_id: str, dataset: Mapping[str, Any], outcome: DatasetOutcome
) -> dict[str, Any]:
    """The result line of one dataset."""
    return {
        "case": case_id,
        "dataset": dataset["name"],
        "kind": dataset["kind"],
        "passed": outcome.passed,
        "score": outcome.score,
        "reference": outcome.reference,
        "threshold": outcome.threshold,
        "detail": outcome.detail,
        **({"tokens": dict(outcome.tokens)} if outcome.tokens else {}),
    }


def emit_result(line: Mapping[str, Any]) -> None:
    """Print ``MV2_E2E_RESULT <json>`` and append the JSON line to ``$MV2FLOW_E2E_RESULTS`` when set."""
    text = json.dumps(dict(line), ensure_ascii=True)
    print(f"{RESULT_PREFIX} {text}", flush=True)
    target = os.environ.get(RESULTS_ENV)
    if target:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def select_datasets(case: Mapping[str, Any], raw: str | None) -> dict[str, Any]:
    """``case`` with only the datasets and extra checks ``raw`` names, in order; ``case`` when unset.

    Args:
        case: One case.
        raw: The value of ``MV2FLOW_E2E_DATASETS``: comma-separated names of datasets and
            extra checks.

    Raises:
        CasesError: A name is none of the case's, or no dataset is left.
    """
    wanted = [name.strip() for name in (raw or "").split(",") if name.strip()]
    if not wanted:
        return dict(case)
    checks = case.get("extra_checks") or []
    known = [d["name"] for d in case["datasets"]] + [c["name"] for c in checks]
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise CasesError(
            f"{DATASETS_ENV} names {unknown}, which case {case['id']} does not run (its "
            f"datasets and extra checks: {known})"
        )
    datasets = [d for d in case["datasets"] if d["name"] in wanted]
    if not datasets:
        raise CasesError(f"{DATASETS_ENV} names no dataset of case {case['id']} to run")
    return {
        **case,
        "datasets": datasets,
        "extra_checks": [c for c in checks if c["name"] in wanted],
    }


def run_datasets(
    case: Mapping[str, Any],
    run_one: Callable[[Mapping[str, Any]], DatasetOutcome],
    record: list[DatasetResult] | None = None,
) -> list[dict[str, Any]]:
    """Run every dataset of ``case`` in order through ``run_one`` and emit one line each.

    A dataset whose run raises is recorded as failed with the exception, and the
    datasets after it still run. With ``record``, each dataset's result is appended to
    it, for the extra checks.

    Returns:
        The result lines, in dataset order.
    """
    lines = []
    for dataset in case["datasets"]:
        try:
            outcome = run_one(dataset)
        # Whatever one dataset raises is that dataset's failure: the case reports it with
        # the verdicts of the datasets after it, which still run on the same engine.
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            outcome = DatasetOutcome(passed=False, detail=f"raised {type(exc).__name__}: {exc}")
        line = result_line(case["id"], dataset, outcome)
        emit_result(line)
        lines.append(line)
        if record is not None:
            record.append(
                DatasetResult(
                    name=dataset["name"],
                    kind=dataset["kind"],
                    passed=outcome.passed,
                    detail=outcome.detail,
                    score=outcome.score,
                    reference=outcome.reference,
                    threshold=outcome.threshold,
                    requests=tuple(outcome.requests),
                )
            )
    return lines


def load_extra_check(check: str, repo_root: Path = REPO_ROOT) -> Callable[..., Any]:
    """The function ``check`` names, ``tests/<path>.py::<function>``, loaded from its file.

    Raises:
        CasesError: The file or the function is not there.
    """
    path, _, name = check.partition("::")
    source = repo_root / path
    module_name = "_mv2_extra_check_" + re.sub(r"\W", "_", path)
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None or not source.is_file():
            raise CasesError(f"extra check {check}: {path} is not a file of the tree")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[module_name]
            raise
    function = getattr(module, name, None)
    if not callable(function):
        raise CasesError(f"extra check {check}: {path} defines no function {name}")
    return function


def run_extra_checks(
    result: CaseResult,
    checks: Sequence[Mapping[str, Any]],
    load: Callable[[str], Callable[..., Any]] = load_extra_check,
) -> list[dict[str, Any]]:
    """Run each extra check of a case on its ``result`` and emit one line each, in order.

    A check fails by raising ``AssertionError``; one that raises anything else, or
    cannot be loaded, fails as raised. One that passes may return a short text of
    what it saw, which its line carries.

    Returns:
        The result lines, in the checks' order.
    """
    lines = []
    for check in checks:
        try:
            said = load(check["check"])(result)
        except AssertionError as exc:
            outcome = DatasetOutcome(passed=False, detail=str(exc) or "the assertion failed")
        # Whatever else a check raises fails that check only: the case reports it with
        # the checks after it, which still run.
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            outcome = DatasetOutcome(passed=False, detail=f"raised {type(exc).__name__}: {exc}")
        else:
            outcome = DatasetOutcome(passed=True, detail="" if said is None else str(said))
        line = result_line(result.id, {"name": check["name"], "kind": "extra_check"}, outcome)
        emit_result(line)
        lines.append(line)
    return lines


def assert_all_passed(case_id: str, lines: Sequence[Mapping[str, Any]]) -> None:
    """Fail the test, naming every failing dataset with its detail, unless all passed."""
    failed = [line for line in lines if not line["passed"]]
    if failed:
        listing = "\n".join(
            f"- {line['dataset']} ({line['kind']}): {line['detail']}" for line in failed
        )
        pytest.fail(
            f"{case_id}: {len(failed)} of {len(lines)} dataset(s) failed:\n{listing}",
            pytrace=False,
        )


# --------------------------------------------------------------------------- generate


#: The text the reuse and chunk probes tokenize and repeat to the length they need.
PROBE_TEXT = (
    "The river ran slowly past the mill, and the miller counted the sacks of grain "
    "that the farmers had brought in from the valley. Each sack was weighed twice, "
    "once when it arrived and once before it was ground, and the two numbers were "
    "written side by side in a ledger that had been kept in the same hand for years."
)


def resolve_image(path: str, repo_root: Path = REPO_ROOT) -> str:
    """An image path, a relative one resolved against the repository root."""
    image = Path(path)
    return str(image if image.is_absolute() else repo_root / image)


def generate_requests(
    dataset: Mapping[str, Any], repo_root: Path = REPO_ROOT
) -> list[dict[str, Any]]:
    """The requests of a ``generate`` dataset: its prompts in order, then the chunk probe.

    Prompts have ids ``p0``, ``p1``, ...; the chunk probe, when declared, is ``chunk``:
    ``chunk_probe_tokens`` probe tokens with ``max_tokens`` 1.
    """
    requests = []
    for n, entry in enumerate(dataset["prompts"]):
        request: dict[str, Any] = {
            "id": f"p{n}",
            "prompt": entry["prompt"],
            "max_tokens": dataset["max_tokens"],
        }
        if entry["images"]:
            request["images"] = [resolve_image(image, repo_root) for image in entry["images"]]
            request["render"] = entry["render"]
        requests.append(request)
    if dataset["chunk_probe_tokens"]:
        requests.append(
            {"id": "chunk", "probe_tokens": dataset["chunk_probe_tokens"], "max_tokens": 1}
        )
    return requests


def chunk_shortfall(records: Sequence[Mapping[str, Any] | None], max_num_tokens: int) -> str | None:
    """Why no request shows that chunked prefill ran; ``None`` when one does.

    A request whose prompt, less the tokens the KV cache resumed, is longer than
    ``max_num_tokens`` cannot be computed in one step, so serving it is the evidence.
    """
    longest = 0
    for record in records:
        if not record:
            continue
        prompt = record.get("prompt_tokens")
        if not isinstance(prompt, int):
            continue
        cached = record.get("cached_tokens")
        computed = prompt - (cached if isinstance(cached, int) else 0)
        if computed > max_num_tokens:
            return None
        longest = max(longest, computed)
    return (
        f"no request was longer than max_num_tokens {max_num_tokens} past what the KV cache "
        f"resumed (the longest computed {longest} prompt tokens), so the engine never split a "
        f"context into chunks and this check shows nothing of chunked prefill"
    )


def reuse_failure(llm_options: Mapping[str, Any], probe: Mapping[str, Any] | None) -> str | None:
    """Why the reuse probe fails the dataset; ``None`` when it does not.

    It fails only when ``llm_options`` asks for KV-cache block reuse and the probe ran
    and read no cached token on the second of two identical prompts. A probe that could
    not run, or whose responses reported no ``cached_tokens``, fails nothing.
    """
    kv = llm_options.get("kv_cache_config")
    if not (isinstance(kv, Mapping) and kv.get("enable_block_reuse")):
        return None
    if not isinstance(probe, Mapping):
        return None
    reused = probe.get("reused")
    if isinstance(reused, bool) or not isinstance(reused, (int, float)) or int(reused) != 0:
        return None
    return (
        f"the engine reused nothing: the reuse probe sent one {probe.get('tokens')}-token "
        "prompt twice after completion on the same cache, but cached_tokens was 0 on the "
        "second response. Check the declared snapshot policy and the engine's actual "
        "block-reuse configuration"
    )


def _probe_note(probe: Mapping[str, Any] | None) -> str:
    if probe is None:
        return ""
    if probe.get("reused") is not None:
        return f"; the reuse probe reused {probe['reused']} of {probe.get('tokens')} prompt tokens"
    why = probe.get("skipped") or probe.get("error") or "its responses reported no cached_tokens"
    return f"; the reuse probe observed nothing ({why})"


def judge_generate(
    requests: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    max_num_tokens: int | None,
    llm_options: Mapping[str, Any],
    probe: Mapping[str, Any] | None,
) -> DatasetOutcome:
    """Judge a ``generate`` dataset from the record of each of its requests.

    Every prompt must be answered with non-empty text; a chunk probe must show that a
    context longer than ``max_num_tokens`` was served (:func:`chunk_shortfall`); and the
    reuse probe must not show an engine asked for block reuse reusing nothing
    (:func:`reuse_failure`).

    Args:
        requests: The dataset's requests (:func:`generate_requests`).
        records: One record per request, in the same order (:func:`record_of`).
        max_num_tokens: The engine's ``max_num_tokens``; ``None`` when unknown.
        llm_options: The case's ``LLM`` options.
        probe: The reuse probe's record (:func:`run_reuse_probe`); ``None`` when it did not run.
    """
    if len(records) != len(requests):
        raise ValueError(f"{len(records)} record(s) for {len(requests)} request(s)")
    declared = [(req, rec) for req, rec in zip(requests, records) if "probe_tokens" not in req]
    problems = []
    empty = [req["id"] for req, rec in declared if not str(rec.get("text") or "").strip()]
    if empty:
        problems.append(f"the engine generated nothing for {', '.join(empty)}")
    chunked = any("probe_tokens" in req for req in requests)
    if chunked:
        if max_num_tokens is None:
            problems.append(
                "the engine's max_num_tokens is unknown, so the chunk probe cannot show "
                "that a context was split into chunks"
            )
        else:
            shortfall = chunk_shortfall(records, max_num_tokens)
            if shortfall:
                problems.append(shortfall)
    reuse = reuse_failure(llm_options, probe)
    if reuse:
        problems.append(reuse)
    if problems:
        return DatasetOutcome(passed=False, detail="; ".join(problems))
    images = sum(len(req.get("images") or ()) for req, _ in declared)
    detail = f"{len(declared)} prompt(s) answered with non-empty text"
    if images:
        detail += f", {images} image(s) sent"
    if chunked:
        detail += f"; a prompt past max_num_tokens {max_num_tokens} was prefilled in chunks"
    return DatasetOutcome(passed=True, detail=detail + _probe_note(probe))


# --------------------------------------------------------------------------- acceptance


@dataclass(frozen=True)
class AcceptanceReading:
    """One acceptance measurement, with the requests it came from.

    Attributes:
        value: Generated tokens over streamed iterations, summed over the requests.
        requests: ``(tokens, iterations)`` per request, in request order.
    """

    value: float
    requests: tuple[tuple[int, int], ...]

    @property
    def draft_iterations(self) -> int:
        """Generation iterations over the workload.

        The first streamed iteration of a request is its context step, which verifies
        no draft, so it is not counted.
        """
        return sum(max(0, iterations - 1) for _, iterations in self.requests)

    def breakdown(self) -> str:
        """Per-prompt rates and their plain mean, beside the iteration-weighted value."""
        rates = [(n, t, i) for n, (t, i) in enumerate(self.requests) if i]
        if not rates:
            return ""
        per_prompt = ", ".join(f"[{n}] {t / i:.2f} ({t} tok/{i} it)" for n, t, i in rates)
        mean = sum(t / i for _, t, i in rates) / len(rates)
        return (
            f"per prompt {per_prompt}; prompt mean {mean:.2f}, iteration-weighted "
            f"{self.value:.2f} over {self.draft_iterations} generation iterations"
        )


def read_acceptance(records: Iterable[Mapping[str, Any]]) -> AcceptanceReading | None:
    """The acceptance the records show, iteration-weighted; ``None`` when none streamed.

    Requests combine as ``sum(tokens) / sum(iterations)``, not as the mean of their
    ratios, which would let one short request outweigh a long one.
    """
    parts = tuple((int(r.get("tokens", 0)), int(r.get("iterations", 0))) for r in records)
    iterations = sum(i for _, i in parts)
    if not iterations:
        return None
    return AcceptanceReading(value=sum(t for t, _ in parts) / iterations, requests=parts)


def judge_acceptance(
    reading: AcceptanceReading | None,
    *,
    min_draft_iterations: int,
    floor: float | None = None,
    floor_source: str | None = None,
    workload: str = "",
) -> DatasetOutcome:
    """Judge an acceptance reading.

    It fails when the workload reached fewer than ``min_draft_iterations`` generation
    iterations (too short to measure, a property of the workload rather than of the
    target), when the value is at or below 1.0 (every draft was rejected), or when
    ``floor`` is set and the value is below it.

    Args:
        reading: The measurement (:func:`read_acceptance`).
        min_draft_iterations: Generation iterations the workload must reach.
        floor: The acceptance length the value must reach, if any.
        floor_source: Where ``floor`` was measured, named in the detail.
        workload: What the prompts were, named in the detail.
    """
    said = f" Workload: {workload}." if workload else ""
    if reading is None:
        return DatasetOutcome(
            passed=False,
            detail=f"the workload streamed no iterations, so there is no acceptance to read.{said}",
            reference=floor,
            threshold=floor,
        )
    value, drafts, prompts = reading.value, reading.draft_iterations, len(reading.requests)
    measured = (
        f"acceptance_length {value:.4f} over {drafts} generation iteration(s) of "
        f"{prompts} prompt(s)"
    )

    def outcome(passed: bool, detail: str) -> DatasetOutcome:
        return DatasetOutcome(
            passed=passed, detail=detail + said, score=value, reference=floor, threshold=floor
        )

    if drafts < min_draft_iterations:
        return outcome(
            False,
            f"workload too short to measure acceptance: {drafts} generation iteration(s) "
            f"over {prompts} prompt(s), fewer than the {min_draft_iterations} this check "
            f"needs, so its rate ({value:.2f}) says nothing about the draft path. This is a "
            f"property of the workload, not of the target.",
        )
    if value <= 1.0:
        return outcome(
            False,
            f"{measured}: at or below 1.0, every draft was rejected; the draft path is "
            f"computing the wrong thing.",
        )
    if floor is None:
        return outcome(True, f"{measured}, held to 1.0 alone: the dataset declares no floor.")
    source = f" (from {floor_source})" if floor_source else ""
    passed = value >= floor
    return outcome(
        passed, f"{measured} vs floor {floor:.4f}{source}: {'met' if passed else 'NOT met'}."
    )


# --------------------------------------------------------------------------- workloads


#: Fields tried, in order, when a dataset does not say which one holds the prompt.
TEXT_FIELDS = ("question", "prompt", "problem", "input", "text", "instruction", "query")
_LOCAL_SUFFIXES = (".jsonl", ".json", ".parquet", ".arrow", ".yaml", ".yml")
_LIST_KEYS = ("prompts", "data", "rows", "examples", "items")
_OFFLINE_ENV = ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")


def _rows_from(data: Any, source: Path) -> list[Any]:
    if isinstance(data, dict):
        for key in _LIST_KEYS:
            if isinstance(data.get(key), list):
                return list(data[key])
        raise DatasetError(f"{source} holds a mapping with none of {list(_LIST_KEYS)} as a list")
    if isinstance(data, list):
        return data
    raise DatasetError(f"{source} holds {type(data).__name__}, not a list of rows")


def _read_file(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            lines = path.read_text(encoding="utf-8").splitlines()
            return [json.loads(line) for line in lines if line.strip()]
        if suffix == ".json":
            return _rows_from(json.loads(path.read_text(encoding="utf-8")), path)
        if suffix in (".yaml", ".yml"):
            return _rows_from(yaml.safe_load(path.read_text(encoding="utf-8")), path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc
    if suffix in (".parquet", ".arrow"):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise DatasetError(f"reading {path} needs pyarrow, which is not installed") from exc
        try:
            if suffix == ".parquet":
                table = pq.read_table(path)
            else:
                try:
                    table = pa.ipc.open_stream(str(path)).read_all()
                except pa.ArrowInvalid:
                    table = pa.ipc.open_file(str(path)).read_all()
        except (OSError, pa.ArrowException) as exc:
            raise DatasetError(f"cannot read {path}: {exc}") from exc
        return table.to_pylist()
    raise DatasetError(f"{path}: unsupported dataset file type {suffix!r}")


def _local_files(directory: Path, split: str | None, config: str | None = None) -> list[Path]:
    """The data files of a local dataset directory for ``config`` and ``split``.

    Hidden directories (such as ``.git``) are skipped. A directory with a subdirectory
    named ``config`` (the Hugging Face snapshot layout, ``<config>/<split>-*.parquet``)
    contributes only the files under it; then the files naming ``split`` are kept, when
    any does.
    """
    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _LOCAL_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(directory).parts)
    )
    if config:
        chosen = [p for p in files if config in p.relative_to(directory).parts[:-1]]
        if chosen:
            files = chosen
    if split:
        # The split is looked for in the file name first, then anywhere under `directory`,
        # never in the directory's own ancestors, which say nothing about the files.
        for matches in (
            lambda p: split in p.name,
            lambda p: split in str(p.relative_to(directory)),
        ):
            named = [p for p in files if matches(p)]
            if named:
                return named
    return files


@contextmanager
def _hf_offline() -> Iterator[Any]:
    """Import ``datasets`` and hold the Hugging Face libraries offline for the block.

    The libraries read their offline switches into module attributes when first
    imported, so the environment alone does not reach a process that imported them
    already. ``datasets`` is imported first under the caller's own settings; the
    switches and the environment are set for the block and restored after it.
    """
    import datasets
    import huggingface_hub.constants as hub_constants

    switches = (
        (hub_constants, "HF_HUB_OFFLINE"),
        (datasets.config, "HF_HUB_OFFLINE"),
        (datasets.config, "HF_DATASETS_OFFLINE"),
    )
    saved = [
        (module, name, getattr(module, name)) for module, name in switches if hasattr(module, name)
    ]
    saved_env = {name: os.environ.get(name) for name in _OFFLINE_ENV}
    try:
        os.environ.update({name: "1" for name in _OFFLINE_ENV})
        for module, name, _ in saved:
            setattr(module, name, True)
        yield datasets
    finally:
        for module, name, value in saved:
            setattr(module, name, value)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_hf_offline(source: str, config: str | None, split: str | None) -> list[Any]:
    try:
        with _hf_offline() as datasets:
            loaded = datasets.load_dataset(source, config, split=split or "test")
    except ImportError as exc:
        raise DatasetError(
            f"{source!r} is not a local path, and reading it as a Hugging Face dataset id "
            f"needs the `datasets` package, which is not installed"
        ) from exc
    # An offline miss surfaces as a ConnectionError, a FileNotFoundError or a
    # DatasetNotFoundError; a bad config or split as a ValueError.
    except (OSError, ConnectionError, ValueError) as exc:
        raise DatasetError(
            f"{source!r} (config {config!r}, split {split!r}) is not in the local Hugging Face "
            f"cache and nothing is downloaded: {type(exc).__name__}: {exc}. Declare a local "
            f"path instead"
        ) from exc
    return [dict(row) for row in loaded]


def load_rows(
    source: str | os.PathLike[str], *, split: str | None = "test", config: str | None = None
) -> list[Any]:
    """Every row of ``source``, which is read locally and never downloaded.

    ``source`` is a JSON, JSON-lines, YAML, Parquet or Arrow file; a directory of them,
    such as a Hugging Face snapshot ``<config>/<split>-*.parquet`` (those under
    ``config`` and naming ``split``, when any are); or a Hugging Face dataset id, read
    from the local cache with the Hugging Face libraries held offline.

    Raises:
        DatasetError: The source cannot be read, or is an id missing from the local cache.
    """
    path = Path(str(source))
    if path.is_file():
        return _read_file(path)
    if path.is_dir():
        files = _local_files(path, split, config)
        if not files:
            raise DatasetError(f"{path} holds no {list(_LOCAL_SUFFIXES)} file")
        rows: list[Any] = []
        for file in files:
            rows.extend(_read_file(file))
        return rows
    if path.is_absolute() or str(source).startswith("."):
        raise DatasetError(f"dataset path does not exist: {path}")
    return _load_hf_offline(str(source), config, split)


def select_prompts(
    rows: Iterable[Any], *, num: int, text_field: str | None = None, template: str = "{text}"
) -> list[str]:
    """The first ``num`` usable rows, rendered through ``template``.

    A row is a string, the text itself, or a mapping holding the text under
    ``text_field`` (or the first of :data:`TEXT_FIELDS` present). ``template`` names the
    text ``{text}``; every other field of the row is available under its own name.

    Raises:
        DatasetError: ``template`` names a field a row lacks.
    """
    selected: list[str] = []
    for index, row in enumerate(rows):
        if len(selected) >= num:
            break
        if isinstance(row, str):
            values: dict[str, Any] = {"text": row}
        elif isinstance(row, Mapping):
            name = text_field or next((n for n in TEXT_FIELDS if n in row), None)
            if name is None or not isinstance(row.get(name), str) or not row[name].strip():
                continue
            values = {str(key): value for key, value in row.items()}
            values["text"] = row[name]
        else:
            continue
        try:
            selected.append(template.format(**values))
        except (KeyError, IndexError, ValueError) as exc:
            raise DatasetError(
                f"prompt template {template!r} does not fit row {index}: {exc}"
            ) from exc
    return selected


def acceptance_prompts(
    workload: Mapping[str, Any], load: Callable[..., list[Any]] = load_rows
) -> tuple[list[str], str]:
    """The prompts of an acceptance workload, and where they came from.

    Raises:
        DatasetError: The workload's dataset cannot be read or yields no usable prompt.
    """
    if workload["prompts"] is not None:
        return list(workload["prompts"]), f"{len(workload['prompts'])} declared prompt(s)"
    dataset = workload["dataset"]
    rows = load(dataset["source"], split=dataset["split"], config=dataset["config"])
    prompts = select_prompts(
        rows, num=dataset["num"], text_field=dataset["text_field"], template=dataset["template"]
    )
    if not prompts:
        raise DatasetError(f"{dataset['source']}: the dataset yields no usable prompt")
    return prompts, f"{len(prompts)} prompt(s) from {dataset['source']}"


# --------------------------------------------------------------------------- benchmark


def parse_references(raw: str | None) -> dict[str, float]:
    """The reference scores ``MV2FLOW_E2E_REFERENCES`` holds, by dataset name.

    Args:
        raw: The variable's value; ``None`` or blank when it is not set.

    Raises:
        ValueError: The value is not a JSON object of numbers.
    """
    if raw is None or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{REFERENCES_ENV} is not valid JSON; expected an object {{dataset name: score}}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{REFERENCES_ENV} holds {type(data).__name__}, not an object {{dataset name: score}}"
        )
    references = {}
    for name, score in data.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{REFERENCES_ENV}[{name!r}] is {score!r}, not a number")
        references[str(name)] = float(score)
    return references


def overridden_params(params_cls: Any, base: Any, task_cls: Any, score: float) -> Any:
    """Hypothesis-test parameters judging against ``score``.

    The parameters are those of ``base``, the references row's, or the task class
    defaults when ``base`` is ``None``, with ``ref_accuracy`` replaced by ``score`` and
    the threshold computed from them rather than taken from the row.

    Args:
        params_cls: ``accuracy_core.HypothesisTestingParams``.
        base: The parameters of the matching references row, or ``None``.
        task_cls: The ``AccuracyTask`` subclass, whose defaults apply without a row.
        score: The reference score.
    """
    if base is None:
        values = {
            "metric_name": task_cls.METRIC_NAME,
            "alpha": task_cls.ALPHA,
            "beta": task_cls.BETA,
            "sigma": task_cls.SIGMA,
            "num_samples": task_cls.NUM_SAMPLES,
            "higher_is_better": task_cls.HIGHER_IS_BETTER,
        }
    else:
        values = {
            "metric_name": base.metric_name,
            "alpha": base.alpha,
            "beta": base.beta,
            "sigma": base.sigma,
            "num_samples": base.num_samples,
            "higher_is_better": base.higher_is_better,
        }
    return params_cls(ref_accuracy=float(score), threshold=None, **values)


@dataclass
class BenchmarkObservation:
    """What an instrumented task used and measured.

    Attributes:
        params: The hypothesis-test parameters it judged with; ``None`` when it asked for none.
        score: The score its evaluator returned; ``None`` before the evaluator ran.
    """

    params: Any = None
    score: float | None = None


def instrument_task(task: Any, *, override: float | None, params_cls: Any) -> BenchmarkObservation:
    """Make ``task`` record its parameters and score, and judge against ``override`` if set.

    ``task`` is an ``AccuracyTask`` instance; its ``get_hypothesis_testing_params`` and
    ``create_evaluator`` are shadowed by instance attributes that call the originals.
    With ``override``, the parameters come from :func:`overridden_params`, the matching
    references row's or, when no row matches, the task class defaults.

    Returns:
        The observation the task fills in while it evaluates.
    """
    seen = BenchmarkObservation()
    original_params = task.get_hypothesis_testing_params
    original_create = task.create_evaluator

    def get_hypothesis_testing_params(**acc_specs: Any) -> Any:
        if override is None:
            params = original_params(**acc_specs)
        else:
            try:
                base = original_params(**acc_specs)
            # accuracy_core raises ValueError when no references row matches the specs.
            except ValueError:
                base = None
            params = overridden_params(params_cls, base, type(task), override)
        seen.params = params
        return params

    def create_evaluator(num_samples: int, extra_evaluator_kwargs: dict | None = None) -> Any:
        evaluator = original_create(num_samples, extra_evaluator_kwargs)
        original_evaluate = evaluator.evaluate

        def evaluate(*args: Any, **kwargs: Any) -> Any:
            score = original_evaluate(*args, **kwargs)
            seen.score = None if score is None else float(score)
            return score

        evaluator.evaluate = evaluate
        return evaluator

    task.get_hypothesis_testing_params = get_hypothesis_testing_params
    task.create_evaluator = create_evaluator
    return seen


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


@contextmanager
def counting_requests(llm: Any) -> Iterator[list[Any]]:
    """Every request ``llm.generate_async`` takes while open, as the outputs it returned.

    Every ``accuracy_core`` evaluator sends its requests through ``generate_async``;
    it is shadowed by an instance attribute for the duration. An ``llm`` without one
    is left as it is, and nothing is counted.
    """
    sent: list[Any] = []
    original = getattr(llm, "generate_async", None)
    if original is None:
        yield sent
        return
    own = getattr(llm, "__dict__", {})
    had, previous = "generate_async" in own, own.get("generate_async")

    def generate_async(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        sent.append(output)
        return output

    llm.generate_async = generate_async
    try:
        yield sent
    finally:
        if had:
            llm.generate_async = previous
        else:
            del llm.generate_async


def observed(llm: Any, run: Callable[[], DatasetOutcome]) -> DatasetOutcome:
    """``run()``'s outcome, with every request it sent through ``llm`` and their lengths."""
    with counting_requests(llm) as sent:
        outcome = run()
    return replace(
        outcome,
        requests=tuple(sent),
        tokens=outcome.tokens if outcome.tokens is not None else request_lengths(sent),
    )


def request_lengths(outputs: Iterable[Any]) -> dict[str, int] | None:
    """How many requests ``outputs`` hold, and the longest prompt and request, in tokens.

    A request's length is its prompt and its longest completion together. An output
    with no prompt token ids or no completion is not counted; ``None`` when none is.
    """
    lengths = [n for n in map(request_length, outputs) if n is not None]
    if not lengths:
        return None
    return {
        "requests": len(lengths),
        "longest_prompt": max(n.prompt for n in lengths),
        "longest": max(n.total for n in lengths),
    }


def evaluate_benchmark(
    task: Any,
    llm: Any,
    dataset: Mapping[str, Any],
    *,
    override: float | None,
    params_cls: Any,
) -> DatasetOutcome:
    """Run ``task.evaluate`` on ``llm`` and turn its hypothesis test into an outcome.

    The task judges exactly as ``accuracy_core`` does; its ``AssertionError`` after the
    evaluator scored is the failed hypothesis test, recorded with the score.
    """
    seen = instrument_task(task, override=override, params_cls=params_cls)
    try:
        task.evaluate(
            llm,
            extra_acc_spec=dataset["extra_acc_spec"],
            extra_evaluator_kwargs=dataset["evaluator_kwargs"],
        )
    except AssertionError:
        if seen.score is None or seen.params is None:
            raise
        passed = False
    else:
        passed = True
    params = seen.params
    if params is None:
        return DatasetOutcome(
            passed=passed,
            detail=f"{dataset['task']} scored {_number(seen.score)}, judged against no "
            f"reference (accuracy_core's INTEGRATION_TEST mode)",
            score=seen.score,
        )
    source = REFERENCES_ENV if override is not None else f"references/{task.DATASET}.yaml"
    direction = ">=" if params.higher_is_better else "<="
    detail = (
        f"{dataset['task']} {params.metric_name} {_number(seen.score)} {direction} threshold "
        f"{_number(params.threshold)} (reference {_number(params.ref_accuracy)} from {source}, "
        f"{params.num_samples} samples): {'passed' if passed else 'FAILED'}"
    )
    return DatasetOutcome(
        passed=passed,
        detail=detail,
        score=seen.score,
        reference=params.ref_accuracy,
        threshold=params.threshold,
    )


# --------------------------------------------------------------------------- engine


def sampling_for(base: Any, max_tokens: int, *, ignore_eos: bool = False) -> Any:
    """A copy of the case's ``SamplingParams`` with this request's token budget."""
    params = copy.deepcopy(base)
    params.max_tokens = int(max_tokens)
    if ignore_eos:
        params.ignore_eos = True
    return params


def cached_tokens(output: Any) -> int | None:
    """The response's ``cached_tokens``; ``None`` when absent or not a count."""
    value = getattr(output, "cached_tokens", None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def probe_prompt(tokenizer: Any, tokens: int) -> list[int]:
    """``tokens`` ordinary text tokens: :data:`PROBE_TEXT`, tokenized and repeated."""
    ids = list(tokenizer.encode(PROBE_TEXT, add_special_tokens=False))
    if not ids:
        raise ValueError("the tokenizer encoded the probe text as no tokens")
    return (ids * (tokens // len(ids) + 1))[:tokens]


def is_lfs_pointer(path: str | os.PathLike[str]) -> bool:
    """Whether ``path`` is a git-lfs pointer file rather than the content it names."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs")


def record_of(request_id: str, output: Any, iterations: int) -> dict[str, Any]:
    """The record of one finished request: its first sequence and its streamed iterations."""
    if not output.outputs:
        raise RuntimeError(f"the engine returned no sequence for request {request_id}")
    first = output.outputs[0]
    prompt_ids = getattr(output, "prompt_token_ids", None)
    return {
        "id": request_id,
        "text": first.text or "",
        "tokens": len(first.token_ids),
        "iterations": iterations,
        "prompt_tokens": len(prompt_ids) if prompt_ids is not None else None,
        "cached_tokens": cached_tokens(output),
    }


def _make_input(llm: Any, request: Mapping[str, Any], model_dir: str) -> Any:
    """The engine input of a request: text, probe tokens, or text with local images."""
    if request.get("probe_tokens") is not None:
        return {"prompt_token_ids": probe_prompt(llm.tokenizer, int(request["probe_tokens"]))}
    prompt = str(request["prompt"])
    images = [str(image) for image in request.get("images") or ()]
    if not images:
        return prompt
    for image in images:
        if not Path(image).is_file():
            raise FileNotFoundError(f"image not found: {image}")
        if is_lfs_pointer(image):
            raise RuntimeError(f"{image} is a git-lfs pointer, not an image")
    if request.get("render") == "raw":
        from PIL import Image

        return {
            "prompt": prompt,
            "multi_modal_data": {"image": [Image.open(p).convert("RGB") for p in images]},
        }
    from tensorrt_llm.inputs import default_multimodal_input_loader

    config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    return default_multimodal_input_loader(
        tokenizer=llm.tokenizer,
        model_dir=model_dir,
        model_type=config["model_type"],
        modality="image" if len(images) == 1 else "multiple_image",
        prompts=[prompt],
        media=images if len(images) == 1 else [images],
        image_data_format="pt",
        device="cpu",
    )[0]


def _chat_prompt(tokenizer: Any, text: str) -> str:
    """``text`` as one user turn of the tokenizer's chat template, ready for generation."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def _stream(llm: Any, prompt: Any, params: Any) -> tuple[Any, int]:
    """Stream one request to completion; its final output and the iterations it took."""
    iterations, output = 0, None
    for output in llm.generate_async(prompt, params, streaming=True):
        iterations += 1
    if output is None:
        raise RuntimeError("the engine streamed no output for a request")
    return output, iterations


def _kv_capacity(llm: Any) -> dict[str, Any]:
    """Rank 0's primary KV-cache pool, as its executor reports it; empty when it cannot."""
    try:
        capacity = llm.get_kv_cache_capacity()
    except (AttributeError, RuntimeError, ValueError, TypeError, OSError):
        return {}
    return dict(capacity) if isinstance(capacity, Mapping) else {}


def run_reuse_probe(llm: Any, params: Any, requested: Mapping[str, Any]) -> dict[str, Any]:
    """Send one prompt twice on a fresh cache salt and read what the second one reused.

    The prompt is two KV blocks past the first snapshot point the options declare (or
    past one block). A probe that does not fit the declared limits or the KV capacity,
    or cannot run, is recorded with ``status`` ``UNKNOWN`` and why.

    Args:
        llm: The engine.
        params: The sampling parameters to copy for the probe's one-token requests.
        requested: The case's ``LLM`` options.
    """
    record: dict[str, Any] = {"status": "UNKNOWN"}
    try:
        capacity = _kv_capacity(llm)
        block = capacity.get("tokensPerBlock")
        if not isinstance(block, int) or block <= 0:
            return {**record, "skipped": "the KV capacity did not report tokensPerBlock"}
        kv = requested.get("kv_cache_config") or {}
        state = kv.get("mamba_state_config") or {}
        points = [
            int(x) for x in state.get("additional_snapshot_offsets_from_start") or () if int(x) > 0
        ]
        interval = int(
            state.get("periodic_snapshot_interval", kv.get("mamba_state_cache_interval", 0)) or 0
        )
        if interval > 0:
            points.append(interval)
        if not points:
            points = [
                int(x)
                for x in state.get("additional_snapshot_offsets_from_end") or ()
                if int(x) > 0
            ]
        tokens = (min(points) if points else block) + 2 * block
        record["tokens"] = tokens
        seq = requested.get("max_seq_len")
        budget = requested.get("max_num_tokens")
        if (seq and tokens + 1 > seq) or (
            budget and not requested.get("enable_chunked_prefill") and tokens > budget
        ):
            return {**record, "skipped": "the repeated prompt does not fit the declared limits"}
        if capacity.get("maxNumTokens", 0) < tokens + 1:
            return {**record, "skipped": "the repeated prompt does not fit the KV capacity"}
        params = copy.deepcopy(params)
        params.n = 1
        params.use_beam_search = False
        params.temperature = 0.0
        params.top_k = 1
        params.max_tokens = 1
        pinned: dict[str, Any] = {"cache_salt": f"mv2-e2e-reuse-{uuid.uuid4().hex}"}
        if requested.get("enable_attention_dp"):
            from tensorrt_llm.scheduling_params import SchedulingParams

            pinned["scheduling_params"] = SchedulingParams(
                attention_dp_rank=0, attention_dp_relax=False
            )
        prompt = {"prompt_token_ids": probe_prompt(llm.tokenizer, tokens)}
        first = llm.generate_async(prompt, params, **pinned).result()
        second = llm.generate_async(prompt, params, **pinned).result()
        record.update(first=cached_tokens(first), reused=cached_tokens(second))
        if record["reused"] is not None:
            record["status"] = "OBSERVED"
        return record
    except (AttributeError, ImportError, TypeError, ValueError, RuntimeError) as exc:
        return {**record, "error": f"{type(exc).__name__}: {exc}"[:300]}


def _max_num_tokens(llm: Any, options: Mapping[str, Any]) -> int | None:
    """The engine's resolved ``max_num_tokens``, else the declared one; ``None`` when neither."""
    for value in (
        getattr(getattr(llm, "args", None), "max_num_tokens", None),
        options.get("max_num_tokens"),
    ):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _run_generate(
    llm: Any, case: Mapping[str, Any], base: Any, dataset: Mapping[str, Any]
) -> DatasetOutcome:
    requests = generate_requests(dataset)
    records = []
    for request in requests:
        prompt = _make_input(llm, request, case["model_dir"])
        if dataset["apply_chat_template"] and isinstance(prompt, str):
            prompt = _chat_prompt(llm.tokenizer, prompt)
        output, iterations = _stream(llm, prompt, sampling_for(base, request["max_tokens"]))
        records.append(record_of(request["id"], output, iterations))
    probe = (
        run_reuse_probe(llm, sampling_for(base, 1, ignore_eos=True), case["llm"])
        if dataset["reuse_probe"]
        else None
    )
    return judge_generate(
        requests,
        records,
        max_num_tokens=_max_num_tokens(llm, case["llm"]),
        llm_options=case["llm"],
        probe=probe,
    )


def _run_acceptance(
    llm: Any, case: Mapping[str, Any], base: Any, dataset: Mapping[str, Any]
) -> DatasetOutcome:
    workload = dataset["workload"]
    prompts, source = acceptance_prompts(workload)
    if workload["apply_chat_template"]:
        prompts = [_chat_prompt(llm.tokenizer, prompt) for prompt in prompts]
    records = []
    for n, prompt in enumerate(prompts):
        output, iterations = _stream(llm, prompt, sampling_for(base, workload["max_tokens"]))
        records.append(record_of(f"a{n}", output, iterations))
    reading = read_acceptance(records)
    if reading is not None:
        print(f"{case['id']}/{dataset['name']}: {reading.breakdown()}", flush=True)
    return judge_acceptance(
        reading,
        min_draft_iterations=dataset["min_draft_iterations"],
        floor=dataset["floor"],
        floor_source=dataset["floor_source"],
        workload=f"{source}, max_tokens {workload['max_tokens']}",
    )


def _run_benchmark(
    llm: Any,
    case: Mapping[str, Any],
    references: Mapping[str, float],
    dataset: Mapping[str, Any],
) -> DatasetOutcome:
    from ..accuracy_core import HypothesisTestingParams, get_accuracy_task

    task = get_accuracy_task(dataset["task"])(case["model_name"])
    return evaluate_benchmark(
        task,
        llm,
        dataset,
        override=references.get(dataset["name"]),
        params_cls=HypothesisTestingParams,
    )


def _run_dataset(
    llm: Any,
    case: Mapping[str, Any],
    base: Any,
    references: Mapping[str, float],
    dataset: Mapping[str, Any],
) -> DatasetOutcome:
    if dataset["kind"] == "generate":
        return observed(llm, lambda: _run_generate(llm, case, base, dataset))
    if dataset["kind"] == "benchmark":
        return observed(llm, lambda: _run_benchmark(llm, case, references, dataset))
    return observed(llm, lambda: _run_acceptance(llm, case, base, dataset))


def run_case(case: Mapping[str, Any]) -> None:
    """Run one case: build its ``LLM``, run every dataset on it and its extra checks, then judge.

    ``${LLM_MODELS_ROOT}`` is expanded (:func:`expand_case`), ``MV2FLOW_E2E_DATASETS``
    keeps a subset of the datasets and extra checks (:func:`select_datasets`), and the
    engine is ``LLM(model=model_dir, **llm)``, used as a context manager. The extra
    checks run after the datasets, on the same engine (:func:`run_extra_checks`). A
    dataset or check that fails or raises does not stop the ones after it; the test
    fails at the end naming every failing one. An ``LLM`` that fails to build fails the
    test at once.

    Args:
        case: One case from :func:`load_cases`.
    """
    from tensorrt_llm import LLM, SamplingParams

    from ...conftest import llm_models_root

    references = parse_references(os.environ.get(REFERENCES_ENV))
    expanded = select_datasets(expand_case(case, llm_models_root), os.environ.get(DATASETS_ENV))
    base = SamplingParams(**expanded["sampling"])
    with LLM(model=expanded["model_dir"], **expanded["llm"]) as llm:
        dataset_results: list[DatasetResult] = []
        lines = run_datasets(
            expanded, partial(_run_dataset, llm, expanded, base, references), dataset_results
        )
        case_result = CaseResult(
            id=expanded["id"],
            feature=expanded["feature"],
            test=expanded["test"],
            member=expanded["member"],
            model_dir=expanded["model_dir"],
            options=expanded["llm"],
            sampling=expanded["sampling"],
            llm=llm,
            datasets=tuple(dataset_results),
        )
        lines += run_extra_checks(case_result, expanded.get("extra_checks") or [])
    assert_all_passed(expanded["id"], lines)


__all__ = [
    "AcceptanceReading",
    "BenchmarkObservation",
    "CASES_FILE",
    "CaseResult",
    "CASES_VERSION",
    "DATASETS_ENV",
    "CasesError",
    "DatasetError",
    "DatasetOutcome",
    "DatasetResult",
    "DEFAULT_GENERATE_PROMPTS",
    "DEFAULT_IMAGE_PROMPT",
    "KINDS",
    "MODELS_ROOT_VAR",
    "PROBE_TEXT",
    "REFERENCES_ENV",
    "RESULTS_ENV",
    "RESULT_PREFIX",
    "RequestLength",
    "acceptance_prompts",
    "assert_all_passed",
    "chunk_shortfall",
    "counting_requests",
    "emit_result",
    "evaluate_benchmark",
    "expand_case",
    "expand_models_root",
    "generate_requests",
    "instrument_task",
    "judge_acceptance",
    "judge_generate",
    "load_cases",
    "load_extra_check",
    "load_rows",
    "overridden_params",
    "parse_cases",
    "parse_references",
    "read_acceptance",
    "observed",
    "read_cases",
    "record_of",
    "result_line",
    "request_length",
    "request_lengths",
    "reuse_failure",
    "run_case",
    "run_datasets",
    "run_extra_checks",
    "run_reuse_probe",
    "sampling_for",
    "select_datasets",
    "select_prompts",
]
