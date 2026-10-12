# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the cleanup path of the MX qualification harness (`mx_harness.py`).

The qualification test writes its timing report and archives its artifacts
from a `finally` block, so those helpers must cope with whatever a failed run
leaves behind: absent or truncated payloads, manifest files that are not JSON
or not manifests at all, and roles that never started. These tests lay out
exactly that and check that `timing.json` still lands in the archive without
raising, which would otherwise replace the failure that ended the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.weight_sharing import (
    build_weight_manifest,
    manifest_file_name,
    write_weight_manifest,
)

__extra_import_path__ = ["~/tests/integration"]
from defs.model_express.mx_harness import (  # noqa: E402
    ROLES,
    MxE2ECase,
    MxRunLayout,
    archive_run_artifacts,
    collect_available_manifests,
    collect_available_payloads,
    report_timings,
)

pytestmark = pytest.mark.cpu_only


class _TwoTensorModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(2, 4))
        self.register_buffer("scale", torch.tensor([0.5, 0.25]))


def _case() -> MxE2ECase:
    return MxE2ECase(
        model_env="TRTLLM_MX_UNIT_MODEL",
        default_model_subdir="tiny",
        repository_cache_prefix="models--trtllm-mx-unit--tp1",
        tp_size=1,
    )


def _failed_run(root: Path) -> MxRunLayout:
    """Lay out what a run that died in the receiver leaves behind."""
    layout = MxRunLayout(root)
    layout.manifest_dir.mkdir(parents=True)
    layout.transfer_logs("receiver").mkdir()
    (layout.transfer_logs("receiver") / "rank0.log").write_text("partial\n", encoding="utf-8")
    # The baseline finished, the donor's payload was cut off mid-write, and the
    # receiver died before writing one.
    layout.output("baseline").write_text(
        json.dumps({"role": "baseline", "load_seconds": 1.5, "generate_seconds": 0.25}),
        encoding="utf-8",
    )
    layout.output("donor").write_text('{"role": "donor", "load_seconds": ', encoding="utf-8")
    layout.log("receiver").write_text("MX E2E worker exited with status 1\n", encoding="utf-8")
    # One complete manifest, one object without manifest keys, one that is
    # valid JSON but not an object, and one file that is not a manifest at all.
    write_weight_manifest(
        build_weight_manifest(_TwoTensorModule()),
        layout.manifest_dir / manifest_file_name("final", "baseline", 0),
    )
    (layout.manifest_dir / manifest_file_name("final", "donor", 0)).write_text(
        "{}\n", encoding="utf-8"
    )
    (layout.manifest_dir / manifest_file_name("transfer", "donor", 0)).write_text(
        "[]\n", encoding="utf-8"
    )
    (layout.manifest_dir / "notes.txt").write_text("not a manifest\n", encoding="utf-8")
    return layout


def test_collectors_tolerate_partial_and_malformed_run_state(tmp_path: Path) -> None:
    layout = _failed_run(tmp_path / "run")

    payloads = collect_available_payloads(layout)
    manifests = collect_available_manifests(layout.manifest_dir)

    assert set(payloads) == {"baseline"}
    assert payloads["baseline"]["load_seconds"] == 1.5
    assert set(manifests) == {("final", "baseline", 0)}
    assert len(manifests[("final", "baseline", 0)].entries) == 2


def test_timing_report_and_archive_survive_a_failed_run(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    layout = _failed_run(tmp_path / "run")

    rows = report_timings(
        _case(),
        collect_available_payloads(layout),
        collect_available_manifests(layout.manifest_dir),
        layout,
    )
    destination = archive_run_artifacts(tmp_path / "output", "llama-bf16-tp1", layout)

    assert [row["role"] for row in rows] == list(ROLES)
    by_role = {row["role"]: row for row in rows}
    assert by_role["baseline"]["load_seconds"] == 1.5
    assert by_role["baseline"]["entry_count"] == 2
    assert by_role["baseline"]["manifest_final_seconds"] is not None
    assert by_role["donor"]["load_seconds"] is None
    assert by_role["donor"]["manifest_transfer_seconds"] is None
    assert by_role["receiver"]["generate_seconds"] is None
    assert json.loads(layout.timing_path.read_text(encoding="utf-8")) == {
        "tp_size": 1,
        "rows": rows,
    }
    # capfd, not capsys: tests/unittest/conftest.py has an autouse fixture
    # (cuda_error_early_quit) that takes capfd, and pytest forbids holding
    # capsys and capfd in the same test.
    assert capfd.readouterr().out.count("MX E2E timing role=") == len(ROLES)

    assert destination == tmp_path / "output" / "model_express" / "llama-bf16-tp1"
    assert (destination / "timing.json").read_text(
        encoding="utf-8"
    ) == layout.timing_path.read_text(encoding="utf-8")
    # Payloads and logs are archived as they are, including the truncated one.
    assert (destination / "baseline.json").is_file()
    assert (destination / "donor.json").is_file()
    assert (destination / "receiver.log").is_file()
    archived_manifests = sorted(path.name for path in (destination / "weight-manifests").iterdir())
    assert archived_manifests == sorted(path.name for path in layout.manifest_dir.glob("*.json"))
    assert (destination / "receiver-transfer-logs" / "rank0.log").is_file()


def test_archive_is_skipped_without_an_output_dir(tmp_path: Path) -> None:
    layout = _failed_run(tmp_path / "run")

    assert archive_run_artifacts(None, "llama-bf16-tp1", layout) is None
    assert archive_run_artifacts("", "llama-bf16-tp1", layout) is None
    assert not (tmp_path / "model_express").exists()
