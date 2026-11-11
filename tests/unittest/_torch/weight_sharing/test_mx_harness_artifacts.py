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
"""Unit tests for the MX qualification harness (`mx_harness.py`) and its worker arguments.

Cleanup path: the qualification test writes its timing report and archives
its artifacts from a `finally` block, so those helpers must cope with whatever
a failed run leaves behind: absent or truncated payloads, manifest files that
are not JSON or not manifests at all, and roles that never started. These
tests lay out exactly that and check that `timing.json` still lands in the
archive without raising, which would otherwise replace the failure that ended
the run.

Probe arguments: the remaining tests check how a row's probe becomes
`mx_e2e_worker.py` arguments, including missing probe images, and that the
worker rejects inconsistent probe arguments and sizes every role of an
image-probe row alike.
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
from defs.model_express import mx_e2e_worker  # noqa: E402
from defs.model_express.mx_harness import (  # noqa: E402
    IMAGE_PROBE_FILES,
    MX_ROLES,
    ROLES,
    MxE2ECase,
    MxRunLayout,
    archive_run_artifacts,
    assert_weight_manifests,
    collect_available_manifests,
    collect_available_payloads,
    collect_weight_manifests,
    probe_worker_args,
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


def _write_mx_manifests(manifest_dir: Path, *, donor_preloaded: bool = False) -> None:
    """Write the final and transfer manifests of a TP=1 MX donor/receiver run without a baseline."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    module = _TwoTensorModule()
    contexts = {
        ("final", "donor"): {
            "checkpoint_format": "MX",
            "weights_preloaded": donor_preloaded,
            "world_size": 1,
        },
        ("final", "receiver"): {
            "checkpoint_format": "MX",
            "weights_preloaded": True,
            "world_size": 1,
        },
        ("transfer", "donor"): {"boundary": "donor_publish"},
        ("transfer", "receiver"): {"boundary": "receiver_p2p_success"},
    }
    for (family, role), extra in contexts.items():
        context = {"family": family, "role": role, "rank": 0, **extra}
        write_weight_manifest(
            build_weight_manifest(module, context=context),
            manifest_dir / manifest_file_name(family, role, 0),
        )


def test_mx_role_manifests_pass_without_a_baseline(tmp_path: Path) -> None:
    _write_mx_manifests(tmp_path)

    manifests = collect_weight_manifests(tmp_path, _case(), roles=MX_ROLES)
    assert sorted(manifests) == [
        ("final", "donor", 0),
        ("final", "receiver", 0),
        ("transfer", "donor", 0),
        ("transfer", "receiver", 0),
    ]
    assert_weight_manifests(_case(), manifests, roles=MX_ROLES)

    # The default roles still require the HF baseline's final manifest.
    with pytest.raises(pytest.fail.Exception, match=r"absent=\[\('final', 'baseline', 0\)\]"):
        collect_weight_manifests(tmp_path, _case())


def test_mx_role_manifests_still_reject_a_preloaded_donor(tmp_path: Path) -> None:
    # A donor that itself received weights from another live MX source.
    _write_mx_manifests(tmp_path, donor_preloaded=True)
    manifests = collect_weight_manifests(tmp_path, _case(), roles=MX_ROLES)

    with pytest.raises(AssertionError, match="donor must load from disk"):
        assert_weight_manifests(_case(), manifests, roles=MX_ROLES)


def test_manifest_roles_must_include_both_mx_roles(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="include"):
        collect_weight_manifests(tmp_path, _case(), roles=("baseline", "donor"))
    with pytest.raises(ValueError, match="include"):
        assert_weight_manifests(_case(), {}, roles=("donor", "donor", "receiver"))


def _image_probe_case() -> MxE2ECase:
    """Return a unit-test row that requests the image probe."""
    return MxE2ECase(
        model_env="TRTLLM_MX_UNIT_MODEL",
        default_model_subdir="tiny",
        repository_cache_prefix="models--trtllm-mx-unit--tp1",
        tp_size=1,
        probe="image",
    )


def test_image_probe_case_builds_worker_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image-probe row passes the shared test images to every role; a text row passes nothing."""
    test_data = tmp_path / "multimodals" / "test_data"
    test_data.mkdir(parents=True)
    for file_name in IMAGE_PROBE_FILES:
        (test_data / file_name).write_bytes(b"")
    monkeypatch.setenv("LLM_MODELS_ROOT", str(tmp_path))

    assert probe_worker_args(_case()) == ()
    assert probe_worker_args(_image_probe_case()) == (
        "--probe",
        "image",
        "--image",
        str(test_data / "seashore.png"),
        "--image",
        str(test_data / "inpaint.png"),
    )


@pytest.mark.parametrize("required", [False, True], ids=["optional", "required"])
def test_image_probe_without_test_images_skips_or_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, required: bool
) -> None:
    """Missing probe images skip the row, or fail it when qualification is required."""
    monkeypatch.setenv("LLM_MODELS_ROOT", str(tmp_path))
    if required:
        monkeypatch.setenv("TRTLLM_MX_E2E_REQUIRED", "1")
    else:
        monkeypatch.delenv("TRTLLM_MX_E2E_REQUIRED", raising=False)

    expected = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(expected, match="probe image does not exist"):
        probe_worker_args(_image_probe_case())


def test_mx_e2e_case_rejects_unknown_probe() -> None:
    """A row cannot request a probe that the worker does not implement."""
    with pytest.raises(ValueError, match="Unknown MX E2E probe"):
        MxE2ECase(
            model_env="TRTLLM_MX_UNIT_MODEL",
            default_model_subdir="tiny",
            repository_cache_prefix="models--trtllm-mx-unit--tp1",
            tp_size=1,
            probe="video",
        )


_WORKER_ARGS = ["--role", "baseline", "--model", "m", "--tp-size", "1", "--output", "o.json"]


@pytest.mark.parametrize(
    "extra_args",
    [
        pytest.param(["--probe", "image"], id="image-probe-without-images"),
        pytest.param(["--image", "a.png"], id="images-without-image-probe"),
        pytest.param(
            [
                "--probe",
                "image",
                "--image",
                "a.png",
                "--eval-task",
                "GSM8K",
                "--eval-num-samples",
                "1",
                "--eval-dataset-path",
                "d",
                "--eval-max-input-len",
                "1",
                "--eval-max-output-len",
                "1",
            ],
            id="image-probe-with-eval",
        ),
    ],
)
def test_worker_rejects_inconsistent_probe_args(extra_args: list[str]) -> None:
    """The worker refuses probe arguments that would not produce the requested probe."""
    with pytest.raises(SystemExit):
        mx_e2e_worker._parse_args(_WORKER_ARGS + extra_args)


def test_worker_image_probe_sizes_every_role_alike() -> None:
    """Image-probe engine limits do not depend on the role, so `SourceIdentity` can match."""
    image_args = ["--probe", "image", "--image", "a.png", "--image", "b.png"]
    kwargs = {}
    for role, role_args in (
        ("baseline", []),
        ("donor", ["--mx-url", "http://mx"]),
        ("receiver", ["--mx-url", "http://mx"]),
    ):
        args = mx_e2e_worker._parse_args(
            ["--role", role, "--model", "m", "--tp-size", "1", "--output", "o.json"]
            + role_args
            + image_args
        )
        assert args.image == [Path("a.png"), Path("b.png")]
        kwargs[role] = mx_e2e_worker._llm_kwargs(args)

    for role_kwargs in kwargs.values():
        assert role_kwargs["skip_tokenizer_init"] is False
        assert role_kwargs["max_batch_size"] == 2 * len(mx_e2e_worker._IMAGE_PROBE_QUESTIONS)
        for key in ("max_batch_size", "max_num_tokens", "max_seq_len"):
            assert role_kwargs[key] == kwargs["baseline"][key]
