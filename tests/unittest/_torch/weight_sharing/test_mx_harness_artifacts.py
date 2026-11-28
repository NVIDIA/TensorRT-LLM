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

Transfer bound: `mx_transfer_upper_bound` counts parameters that view a
shared storage at the same offset once, at the largest of their sizes, and
`assert_transfer_within_manifests` rejects a receiver rank whose RDMA record
moves more than its manifest holds.

Probe arguments: the probe tests check how a row's probe becomes
`mx_e2e_worker.py` arguments, including missing probe images, and that the
worker rejects inconsistent probe arguments and sizes every role of an
image-probe row alike.

Donor snapshot: a row that loads only the first decoder layers of a sharded
checkpoint links only the shards those layers need, and its snapshot is named
by the identity of exactly the linked files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.weight_sharing import (
    ArtifactIdentity,
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
    SAFETENSORS_INDEX,
    MxE2ECase,
    MxRunLayout,
    archive_run_artifacts,
    assert_transfer_within_manifests,
    assert_weight_manifests,
    build_canonical_snapshot,
    collect_available_manifests,
    collect_available_payloads,
    collect_weight_manifests,
    first_layers_checkpoint_files,
    mx_transfer_upper_bound,
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


class _SharedStorageModule(nn.Module):
    """A tied head, a runtime alias module, two empty parameters, and a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 2)
        self.norm = nn.LayerNorm(2)
        self.next_layer_layernorm = self.norm
        self.lm_head = nn.Linear(2, 4, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.empty = nn.Parameter(torch.empty(0))
        self.other_empty = nn.Parameter(torch.empty(0))
        self.register_buffer("scale", torch.ones(2))


def test_mx_transfer_upper_bound_counts_each_shared_view_once() -> None:
    # The MX catalog holds embed_tokens.weight (which the head shares),
    # norm.weight, norm.bias, and one empty parameter: 32 + 8 + 8 + 0 bytes.
    # The bound counts the second empty parameter as well.
    assert mx_transfer_upper_bound(build_weight_manifest(_SharedStorageModule())) == (5, 48)


class _SharedStartModule(nn.Module):
    """Two parameters that start at the same element of one storage, the larger registered first."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(8, dtype=torch.float32))
        self.a_prefix = nn.Parameter(self.weight.data[:2])


def test_mx_transfer_upper_bound_counts_a_shared_start_at_its_largest_view() -> None:
    # ModelExpress registers `weight` (32 bytes), the first in module order, at
    # the shared data pointer, while the manifest lists `a_prefix` (8 bytes)
    # first because its entries are sorted by name.
    assert mx_transfer_upper_bound(build_weight_manifest(_SharedStartModule())) == (1, 32)


@pytest.mark.parametrize(
    "logged, within",
    [
        pytest.param("4 tensors, 0.00 GB", True, id="empty-parameters-merged"),
        pytest.param("5 tensors, 0.00 GB", True, id="bound"),
        pytest.param("6 tensors, 0.00 GB", False, id="shared-storage-sent-twice"),
        pytest.param("5 tensors, 0.02 GB", False, id="too-many-bytes"),
    ],
)
def test_receiver_transfer_must_stay_within_its_manifest(
    tmp_path: Path, logged: str, within: bool
) -> None:
    manifests = {("transfer", "receiver", 0): build_weight_manifest(_SharedStorageModule())}
    receiver_log = tmp_path / "receiver.log"
    receiver_log.write_text(f"[Worker 0] INFO: [TIMING] RDMA transfer complete: {logged}, 1.0s\n")

    if within:
        assert_transfer_within_manifests(_case(), receiver_log, manifests)
    else:
        with pytest.raises(pytest.fail.Exception, match="moved more than"):
            assert_transfer_within_manifests(_case(), receiver_log, manifests)


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


# A three-layer checkpoint in four shards: layer 0 shares a shard with the
# embedding, layer 1 has its own shard, layer 2 shares one with the final norm
# and LM head, and the MTP layer 3 has its own.
_SHARDED_WEIGHT_MAP = {
    "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00004.safetensors",
    "model.layers.1.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    "model.layers.1.mlp.gate.weight": "model-00002-of-00004.safetensors",
    "model.layers.2.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    "model.norm.weight": "model-00003-of-00004.safetensors",
    "lm_head.weight": "model-00003-of-00004.safetensors",
    "model.layers.3.eh_proj.weight": "model-00004-of-00004.safetensors",
}


def _sharded_checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"num_hidden_layers": 3}), encoding="utf-8")
    (root / SAFETENSORS_INDEX).write_text(
        json.dumps({"metadata": {"total_size": 8}, "weight_map": _SHARDED_WEIGHT_MAP}),
        encoding="utf-8",
    )
    for shard in sorted(set(_SHARDED_WEIGHT_MAP.values())):
        (root / shard).write_bytes(shard.encode())
    return root


def _snapshot_files(snapshot: Path) -> set[str]:
    return {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*")}


def test_first_layers_keep_only_the_shards_those_layers_need(tmp_path: Path) -> None:
    model_path = _sharded_checkpoint(tmp_path / "model")
    files = tuple(sorted(model_path.iterdir()))

    kept_files, index = first_layers_checkpoint_files(model_path, files, num_hidden_layers=1)

    # Layer 0 and the embedding share shard 1; shard 3 holds the norm and LM
    # head beside layer 2, which the load ignores.
    assert {path.name for path in kept_files} == {
        "config.json",
        "model-00001-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
    }
    restricted = json.loads(index)
    assert restricted["metadata"] == {"total_size": 8}
    assert restricted["weight_map"] == {
        name: shard
        for name, shard in _SHARDED_WEIGHT_MAP.items()
        if shard in ("model-00001-of-00004.safetensors", "model-00003-of-00004.safetensors")
    }


def test_snapshot_of_the_first_layers_holds_and_hashes_only_their_shards(tmp_path: Path) -> None:
    model_path = _sharded_checkpoint(tmp_path / "model")
    case = MxE2ECase(
        model_env="TRTLLM_MX_UNIT_MODEL",
        default_model_subdir="tiny",
        repository_cache_prefix="models--trtllm-mx-unit--first-layers-tp1",
        tp_size=1,
        model_kwargs={"num_hidden_layers": 2},
    )

    snapshot = build_canonical_snapshot(case, model_path, tmp_path / "run")

    assert _snapshot_files(snapshot) == {
        "config.json",
        SAFETENSORS_INDEX,
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
    }
    assert set(json.loads((snapshot / SAFETENSORS_INDEX).read_text())["weight_map"]) == {
        name for name in _SHARDED_WEIGHT_MAP if not name.startswith("model.layers.3.")
    }
    # The snapshot is named by the identity of exactly its files.
    expected_identity = ArtifactIdentity.from_checkpoint(_copy_of(snapshot, tmp_path / "copy"))
    assert snapshot.name == expected_identity.digest
    assert not list((tmp_path / "run" / "donor-hf-cache").glob("staging-*"))


def test_snapshot_of_a_full_row_links_every_checkpoint_file(tmp_path: Path) -> None:
    model_path = _sharded_checkpoint(tmp_path / "model")

    snapshot = build_canonical_snapshot(_case(), model_path, tmp_path / "run")

    assert _snapshot_files(snapshot) == {path.name for path in model_path.iterdir()}
    assert all((snapshot / path.name).is_symlink() for path in model_path.iterdir())
    assert snapshot.name == ArtifactIdentity.from_checkpoint(model_path).digest


def _copy_of(snapshot: Path, destination: Path) -> Path:
    """Copy a snapshot's files, following its symlinks, into a plain directory."""
    destination.mkdir()
    for path in snapshot.iterdir():
        (destination / path.name).write_bytes(path.read_bytes())
    return destination
