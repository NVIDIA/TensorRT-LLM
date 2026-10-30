# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model-key selection and conservative fallbacks for perf path-table changes."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import pytest
import yaml
from cbts.blocks import Stage, YAMLIndex, compute_stage_test_counts, write_filtered_test_db
from cbts.command.main import _load_pr_inputs
from cbts.rules.base import PRInputs, RuleResult
from cbts.rules.perf_model_paths import MODEL_PATHS, changed_model_keys
from cbts.rules.tests_def import TestsDefRule as CbtsTestsDefRule

pytestmark = pytest.mark.cpu_only

_BEFORE = 'MODEL_PATH_DICT = {"old": "Old"}\n'
_AFTER = 'MODEL_PATH_DICT = {"old": "Old", "new": "New"}\n'
_LOOKUP = """from ._model_paths import MODEL_PATH_DICT
AGG_CONFIG_FOLDER = os.environ.get("AGG_CONFIG_FOLDER", "tests/scripts/perf-sanity/aggregated")
DISAGG_CONFIG_FOLDER = os.environ.get("DISAGG_CONFIG_FOLDER", "tests/scripts/perf-sanity/disaggregated")
def get_model_dir(model_name):
    if model_name in MODEL_PATH_DICT.keys():
        return MODEL_PATH_DICT[model_name]
    return model_name
"""


def _diff(before: str, after: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))


def _write(root: Path, path: str, source: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def _fixture(root: Path, source: str = _AFTER) -> tuple[CbtsTestsDefRule, dict[str, Stage]]:
    _write(root, MODEL_PATHS, source)
    for filename in ("test_perf.py", "test_perf_sanity.py"):
        _write(root, f"tests/integration/defs/perf/{filename}", _LOOKUP)
    _write(root, "tests/scripts/perf-sanity/aggregated/old.yaml", "metadata:\n  model_name: old\n")
    _write(root, "tests/scripts/perf-sanity/aggregated/new.yaml", "metadata:\n  model_name: new\n")
    lists = {
        "l0_perf": [
            "perf/test_perf.py::test_perf[old-serve-pytorch]",
            "perf/test_perf.py::test_perf[new-serve-pytorch] TIMEOUT(120)",
        ],
        "l0_old_multi_gpus_perf_sanity": [
            "perf/test_perf_sanity.py::test_e2e[aggr_upload-old-server]",
        ],
        "l0_new_multi_gpus_perf_sanity": [
            "perf/test_perf_sanity.py::test_e2e[aggr_upload-new-server]",
        ],
        "l0_other": ["other/test_other.py::test_other"],
        "l0_cpu": ["cpu/test_cpu.py::test_cpu"],
    }
    stages = {}
    for stem, tests in lists.items():
        _write(root, f"test-db/{stem}.yml", yaml.safe_dump({stem: [{"tests": tests}]}))
        stages[stem] = Stage(stem, stem, "x86_64", 1, 1)
    return CbtsTestsDefRule(YAMLIndex.load(root / "test-db"), stages, root), stages


def _apply(rule: CbtsTestsDefRule, before: str = _BEFORE, after: str = _AFTER) -> RuleResult:
    return rule.apply(PRInputs([MODEL_PATHS], {MODEL_PATHS: _diff(before, after)}))


@pytest.mark.parametrize(
    ("before", "after", "keys"),
    [
        (_BEFORE, _AFTER, {"new"}),
        (_AFTER, _BEFORE, {"new"}),
        (_BEFORE, _BEFORE.replace('"Old"', '"Updated"'), {"old"}),
        (_BEFORE, _BEFORE + "# comment\n", set()),
    ],
)
def test_changed_keys(before: str, after: str, keys: set[str]) -> None:
    assert changed_model_keys(after, _diff(before, after)) == keys


@pytest.mark.parametrize(
    "after",
    [
        _AFTER + "OTHER = 1\n",
        'MODEL_PATH_DICT = {"old": "Old", "new": make_path()}\n',
        'MODEL_PATH_DICT = {"old": "Old", "old": "Other"}\n',
        'MODEL_PATH_DICT = {**BASE, "new": "New"}\n',
        _AFTER + "ALIASES = list(MODEL_PATH_DICT)\n",
    ],
)
def test_unknown_module_edits_fall_back(after: str) -> None:
    assert changed_model_keys(after, _diff(_BEFORE, after)) is None


def test_selects_exact_cases_and_keeps_counts_and_rendering_consistent(tmp_path: Path) -> None:
    rule, stages = _fixture(tmp_path)
    result = _apply(rule)
    assert result.scope == "testdefonly"
    assert result.affected_stages == {"l0_perf", "l0_new_multi_gpus_perf_sanity"}
    counts = compute_stage_test_counts(
        rule.yaml_index, stages, result.affected_stages, result.block_filters
    )
    assert counts == {"l0_perf": 1, "l0_new_multi_gpus_perf_sanity": 1}
    output = tmp_path / "filtered"
    write_filtered_test_db(tmp_path / "test-db", output, result.block_filters)
    data = yaml.safe_load((output / "l0_perf.yml").read_text())
    assert data["l0_perf"][0]["tests"] == [
        "perf/test_perf.py::test_perf[new-serve-pytorch] TIMEOUT(120)"
    ]
    assert not (output / "l0_old_multi_gpus_perf_sanity.yml").exists()


def test_removing_or_modifying_an_existing_alias_keeps_its_consumers(tmp_path: Path) -> None:
    rule, _ = _fixture(tmp_path, _BEFORE)
    removed = _apply(rule, _AFTER, _BEFORE)
    assert removed.affected_stages == {"l0_perf", "l0_new_multi_gpus_perf_sanity"}
    changed = _BEFORE.replace('"Old"', '"Updated"')
    _write(tmp_path, MODEL_PATHS, changed)
    modified = _apply(rule, _BEFORE, changed)
    assert modified.affected_stages == {"l0_perf", "l0_old_multi_gpus_perf_sanity"}


@pytest.mark.parametrize(
    "config", ["not: [valid", "[]", "metadata: {}", "metadata: {model_name: 42}"]
)
def test_unknown_config_keeps_the_case(tmp_path: Path, config: str) -> None:
    rule, _ = _fixture(tmp_path)
    _write(tmp_path, "tests/scripts/perf-sanity/aggregated/old.yaml", config)
    assert "l0_old_multi_gpus_perf_sanity" in _apply(rule).affected_stages


def test_missing_config_and_environment_override_keep_the_case(tmp_path: Path, monkeypatch) -> None:
    rule, _ = _fixture(tmp_path)
    (tmp_path / "tests/scripts/perf-sanity/aggregated/old.yaml").unlink()
    assert "l0_old_multi_gpus_perf_sanity" in _apply(rule).affected_stages
    monkeypatch.setenv("AGG_CONFIG_FOLDER", "/custom/config")
    assert "l0_new_multi_gpus_perf_sanity" in _apply(rule).affected_stages


def test_server_override_is_read_even_if_metadata_is_unaffected(tmp_path: Path) -> None:
    rule, _ = _fixture(tmp_path)
    _write(
        tmp_path,
        "tests/scripts/perf-sanity/aggregated/old.yaml",
        """metadata:
  model_name: old
server_configs:
- name: server
  model_name: new
""",
    )
    assert "l0_old_multi_gpus_perf_sanity" in _apply(rule).affected_stages


@pytest.mark.parametrize(
    "source", [_LOOKUP + "ALL = list(MODEL_PATH_DICT)\n", _LOOKUP + 'MODEL_PATH_DICT["x"] = "y"\n']
)
def test_unrecognized_table_usage_keeps_all_consumer_cases(tmp_path: Path, source: str) -> None:
    rule, _ = _fixture(tmp_path)
    _write(tmp_path, "tests/integration/defs/perf/test_perf_sanity.py", source)
    assert "l0_old_multi_gpus_perf_sanity" in _apply(rule).affected_stages


def test_legacy_sanity_alias_and_local_override_are_supported(tmp_path: Path) -> None:
    rule, _ = _fixture(tmp_path)
    source = _LOOKUP.replace("import MODEL_PATH_DICT", "import MODEL_PATH_DICT as BASE")
    source = source.replace(
        "def get_model_dir", 'MODEL_PATH_DICT = {**BASE, "local": "Local"}\ndef get_model_dir'
    )
    _write(tmp_path, "tests/integration/defs/perf/test_perf_sanity.py", source)
    assert "l0_old_multi_gpus_perf_sanity" not in _apply(rule).affected_stages


def test_new_unrecognized_importer_restores_existing_fallback(tmp_path: Path) -> None:
    rule, _ = _fixture(tmp_path)
    _write(tmp_path, "tests/integration/defs/perf/test_other.py", _LOOKUP)
    assert "l0_old_multi_gpus_perf_sanity" in _apply(rule).affected_stages


def test_coarse_test_entries_are_preserved(tmp_path: Path) -> None:
    rule, _ = _fixture(tmp_path)
    next(b for b in rule.yaml_index.blocks if b.yaml_stem == "l0_perf").tests = [
        "perf/test_perf.py -k new"
    ]
    result = _apply(rule)
    assert "l0_perf" in result.affected_stages
    assert "perf/test_perf.py" in result.block_filters[("l0_perf", 0)]


def test_other_changed_file_contributions_are_not_subtracted(tmp_path: Path) -> None:
    rule, _ = _fixture(tmp_path)
    helper = "tests/integration/defs/perf/test_perf_sanity.py"
    result = rule.apply(PRInputs([MODEL_PATHS, helper], {MODEL_PATHS: _diff(_BEFORE, _AFTER)}))
    assert "l0_old_multi_gpus_perf_sanity" in result.affected_stages


def test_cli_uses_raw_diff_for_verified_reconstruction(tmp_path: Path) -> None:
    source = _AFTER + "# new explanatory comment\n"
    rule, _ = _fixture(tmp_path, source)
    payload = tmp_path / "input.json"
    payload.write_text(
        json.dumps({"changed_files": [MODEL_PATHS], "diffs": {MODEL_PATHS: _diff(_BEFORE, source)}})
    )
    result = rule.apply(_load_pr_inputs(payload, tmp_path))
    assert "l0_old_multi_gpus_perf_sanity" not in result.affected_stages


@pytest.mark.parametrize(
    "parameter",
    [
        "disagg_upload-e2e-config-with-dashes",
        "disagg_upload-gen_only-config-with-dashes",
        "aggr_upload-ctx_only-config-with-dashes",
        "aggr_upload-gen_only_no_context-config-with-dashes",
    ],
)
def test_disaggregated_configs_are_resolved(tmp_path: Path, parameter: str) -> None:
    rule, _ = _fixture(tmp_path)
    _write(
        tmp_path,
        "tests/scripts/perf-sanity/disaggregated/config-with-dashes.yaml",
        "metadata: {model_name: old}\n",
    )
    block = next(
        b for b in rule.yaml_index.blocks if b.yaml_stem == "l0_old_multi_gpus_perf_sanity"
    )
    block.tests = [f"perf/test_perf_sanity.py::test_e2e[{parameter}]"]
    assert "l0_old_multi_gpus_perf_sanity" not in _apply(rule).affected_stages


@pytest.mark.parametrize(
    "source",
    [
        _LOOKUP.replace("tests/scripts/perf-sanity/aggregated", "custom/config"),
        _LOOKUP + '\nALL = globals()["MODEL_PATH_DICT"]\n',
        _LOOKUP + '\nALL = getattr(module, "MODEL_PATH_DICT")\n',
    ],
)
def test_unsupported_consumer_contract_keeps_its_cases(tmp_path: Path, source: str) -> None:
    rule, _ = _fixture(tmp_path)
    _write(tmp_path, "tests/integration/defs/perf/test_perf_sanity.py", source)
    assert "l0_old_multi_gpus_perf_sanity" in _apply(rule).affected_stages


def test_unverified_patch_and_non_table_edits_restore_fallback(tmp_path: Path) -> None:
    rule, _ = _fixture(tmp_path)
    missing = rule.apply(PRInputs([MODEL_PATHS], {}))
    assert "l0_old_multi_gpus_perf_sanity" in missing.affected_stages
    mismatch = _apply(rule, _BEFORE, _AFTER.replace('"New"', '"Mismatch"'))
    assert "l0_old_multi_gpus_perf_sanity" in mismatch.affected_stages
    source = _AFTER + "OTHER = 42\n"
    _write(tmp_path, MODEL_PATHS, source)
    assert "l0_old_multi_gpus_perf_sanity" in _apply(rule, _BEFORE, source).affected_stages
