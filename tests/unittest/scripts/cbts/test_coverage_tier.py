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
"""Single-node coverage narrowing, including multi-GPU stage families."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts import blocks as block_utils
from cbts.command import main as main_module
from cbts.coverage import tier
from cbts.coverage.selection import selector
from cbts.rules.base import PRInputs

pytestmark = pytest.mark.cpu_only


def test_single_node_multi_gpu_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    safe, impacted, untrusted, missing = [
        f"test_x.py::test_{s}" for s in ("safe", "impacted", "untrusted", "missing")
    ]
    names = [
        "H100-PyTorch-1",
        "H100-4_GPUs-PyTorch-1",
        "H100-4_GPUs-PyTorch-2",
        "H100-8_GPUs-PyTorch-1",
        "H100-4_GPUs-PyTorch-Post-Merge-1",
        "GB200-8_GPUs-2_Nodes-PyTorch-1",
        "A10-PyTorch-1",  # No instrumented family at all.
    ]
    stages = {name: block_utils.Stage(name, str(i), "x86", 1, 4) for i, name in enumerate(names)}
    blocks = [block_utils.Block(s.yaml_stem, 0, {}, [safe]) for s in stages.values()]
    blocks[1].tests = blocks[2].tests = [safe, impacted, untrusted, missing]
    index = SimpleNamespace(blocks=blocks)
    family = "H100-4_GPUs-PyTorch"
    known = {name.rsplit("-", 1)[0]: {safe} for name in names[:-1]}
    known[family] = {safe, impacted, untrusted}
    cov = selector.CoverageResult(
        ok=True, reason="", impacted={family: {impacted}}, skippable=known
    )
    monkeypatch.setattr(
        tier,
        "CoverageSelector",
        lambda *a, **kw: SimpleNamespace(
            decide=lambda *a, **kw: cov,
            untrusted_families=lambda: {f"{family}/{untrusted}"},
        ),
    )
    db = SimpleNamespace(known_by_family=lambda: known)
    result, _ = tier.apply_coverage_tier(
        PRInputs(["tensorrt_llm/example.py"], {}),
        [],
        set(),
        stages,
        index,
        tmp_path,
        db,
    )
    assert result is not None
    assert result.dropped == {names[0], names[3], names[4]}
    assert {names[1], names[2], names[5], names[6]} <= result.affected_stages
    assert ("5", 0) not in result.removed  # Multi-node stays at baseline.
    assert ("6", 0) not in result.removed  # Missing family stays at baseline.
    assert result.removed[("1", 0)] == result.removed[("2", 0)] == {safe}
    assert result.must_run_reasons == {"impacted": 2, "untrusted": 2, "no_data": 2}

    counts, splits = tier.compute_coverage_stage_counts(
        result.affected_stages,
        stages,
        index,
        result.removed,
        {},
    )
    assert counts[names[1]] == counts[names[2]] == 3
    assert splits[names[1]] == splits[names[2]] == 1
    assert names[5] not in counts
    src = tmp_path / "src"
    src.mkdir()
    (src / "1.yml").write_text(yaml.safe_dump({"1": [{"tests": blocks[1].tests}]}))
    tier.write_coverage_test_db(src, tmp_path / "out", result.removed)
    assert yaml.safe_load((tmp_path / "out/1.yml").read_text())["1"][0]["tests"] == [
        impacted,
        untrusted,
        missing,
    ]


@pytest.mark.parametrize("other", ["GB200-8_GPUs-2_Nodes-PyTorch-1", "A10-PyTorch-1"])
def test_shared_block_requires_all_served_stages_supported(other: str) -> None:
    single = "H100-4_GPUs-PyTorch-1"
    stages = {n: block_utils.Stage(n, "shared", "x86", 1, 1) for n in (single, other)}
    known = {single.rsplit("-", 1)[0]: {"test_x.py::test_safe"}}
    if "_Nodes" in other:
        known[other.rsplit("-", 1)[0]] = {"test_x.py::test_safe"}
    cov = selector.CoverageResult(ok=True, reason="", skippable=known)
    index = SimpleNamespace(blocks=[block_utils.Block("shared", 0, {}, ["test_x.py::test_safe"])])
    removed, dropped, _ = tier._build_narrowing(cov, stages, index, {}, known, set())
    assert removed == {}
    assert dropped == set()


def test_main_does_not_force_dropped_multi_gpu_back_to_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    single, multi = "H100-PyTorch-1", "H100-4_GPUs-PyTorch-1"
    stages = {n: block_utils.Stage(n, "sample", "x86", 1, 1) for n in (single, multi)}
    test_db = tmp_path / "test-db"
    test_db.mkdir()
    (test_db / "sample.yml").write_text("sample: []\n")
    groovy = tmp_path / "test.groovy"
    groovy.touch()
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"changed_files": ["tensorrt_llm/example.py"]}))
    meta = tmp_path / "meta.json"
    meta.write_text('{"drift": 0}')
    monkeypatch.setattr(main_module, "parse_stages_from_groovy", lambda *a, **kw: stages)
    monkeypatch.setattr(main_module, "build_rules", lambda *a: [])
    monkeypatch.setattr(main_module, "open_db", lambda *a: None)
    monkeypatch.setattr(
        main_module,
        "apply_coverage_tier",
        lambda *a, **kw: (tier.CoverageTierResult(affected_stages={single}, dropped={multi}), ""),
    )
    assert (
        main_module._run(
            main_module._Args(
                input_json=str(input_path),
                repo_root=str(tmp_path),
                test_db=str(test_db),
                groovy_file=str(groovy),
                coverage_db="unused.sqlite",
                coverage_db_meta=str(meta),
                list_needed_diffs=False,
                no_data_policy="file",
            )
        )
        == 0
    )
    decision = json.loads(capsys.readouterr().out)
    assert decision["scope"] == "coverage"
    assert decision["affected_stages"] == [single]
    assert decision["coverage_dropped_stages"] == [multi]
    assert decision["enable_multi_gpu"] is False
