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
"""Unit tests for how the MX qualification harness passes MoE settings to its workers.

Every role of one donor/receiver row must load the model with the same MoE
partition and finalize setting, and rows without MoE settings pass no MoE
keyword arguments to `LLM`. The harness then reads the final weight manifests
to check that every rank realized the partition the row names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorrt_llm._torch.weight_sharing.weight_manifest import WeightManifest, WeightManifestEntry
from tensorrt_llm.llmapi import MoeConfig

__extra_import_path__ = ["~/tests/integration"]
from defs.model_express.mx_e2e_worker import _llm_kwargs, _parse_args  # noqa: E402
from defs.model_express.mx_harness import (  # noqa: E402
    ROLES,
    MxE2ECase,
    assert_moe_partition,
    moe_worker_args,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "case_overrides, expected_args, expected_moe_kwargs",
    [
        pytest.param({}, (), {}, id="dense-row"),
        pytest.param(
            {"moe_disable_finalize_fusion": True},
            ("--moe-disable-finalize-fusion",),
            {"moe_config": MoeConfig(disable_finalize_fusion=True)},
            id="moe-tp-row",
        ),
        pytest.param(
            {"moe_ep_size": 2, "moe_disable_finalize_fusion": True},
            ("--moe-ep-size", "2", "--moe-disable-finalize-fusion"),
            {
                "moe_expert_parallel_size": 2,
                "moe_config": MoeConfig(disable_finalize_fusion=True),
            },
            id="moe-ep-row",
        ),
    ],
)
@pytest.mark.parametrize("role", ROLES)
def test_moe_row_settings_reach_every_role(
    case_overrides: dict[str, object],
    expected_args: tuple[str, ...],
    expected_moe_kwargs: dict[str, object],
    role: str,
) -> None:
    case = MxE2ECase(
        model_env="TRTLLM_MX_TEST_MODEL",
        default_model_subdir="model",
        repository_cache_prefix="models--trtllm-mx-e2e--test-tp2",
        tp_size=2,
        **case_overrides,
    )
    worker_args = moe_worker_args(case)
    argv = ["--role", role, "--model", "/model", "--tp-size", "2", "--output", "/out.json"]
    if role != "baseline":
        argv.extend(("--mx-url", "http://mx:8001"))

    kwargs = _llm_kwargs(_parse_args([*argv, *worker_args]))

    assert worker_args == expected_args
    moe_kwargs = {
        key: kwargs[key] for key in ("moe_expert_parallel_size", "moe_config") if key in kwargs
    }
    assert moe_kwargs == expected_moe_kwargs
    assert kwargs == {**_llm_kwargs(_parse_args(argv)), **expected_moe_kwargs}


# A small MoE checkpoint: 8 experts of intermediate size 64 over hidden size 32.
_MOE_CONFIG = {"num_experts": 8, "moe_intermediate_size": 64, "hidden_size": 32}
# Per-rank (`w3_w1_weight`, `w2_weight`) shapes of each partition of that checkpoint.
_TP1_SHAPES = ((8, 128, 32), (8, 32, 64))
_MOE_TP2_SHAPES = ((8, 64, 32), (8, 32, 32))
_MOE_EP2_SHAPES = ((4, 128, 32), (4, 32, 64))


def _moe_row(tp_size: int, moe_ep_size: int = 1) -> MxE2ECase:
    return MxE2ECase(
        model_env="TRTLLM_MX_TEST_MODEL",
        default_model_subdir="model",
        repository_cache_prefix=f"models--trtllm-mx-e2e--test-tp{tp_size}",
        tp_size=tp_size,
        moe_ep_size=moe_ep_size,
    )


def _final_manifests(
    tp_size: int, expert_shapes: tuple[tuple[int, ...], ...] | None
) -> dict[tuple[str, str, int], WeightManifest]:
    """Final manifests of every role and rank: a dense tensor plus two layers of routed experts."""
    layers = () if expert_shapes is None else (0, 1)
    shapes = {"model.embed_tokens.weight": (16, 32)}
    for layer in layers:
        prefix = f"model.layers.{layer}.mlp.experts.backend."
        shapes[prefix + "w3_w1_weight"], shapes[prefix + "w2_weight"] = expert_shapes
    entries = tuple(
        WeightManifestEntry(
            fqn=fqn,
            kind="param",
            dtype="torch.bfloat16",
            shape=shape,
            stride=(),
            storage_offset=0,
            nbytes=0,
            sha256="0",
        )
        for fqn, shape in sorted(shapes.items())
    )
    manifest = WeightManifest(
        manifest_format_version=1,
        entries=entries,
        skipped=(),
        alias_groups=(),
        manifest_sha256="0",
    )
    return {("final", role, rank): manifest for role in ROLES for rank in range(tp_size)}


def _checkpoint(tmp_path: Path, config: dict[str, int]) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.mark.parametrize(
    "tp_size, moe_ep_size, expert_shapes",
    [
        pytest.param(1, 1, _TP1_SHAPES, id="tp1"),
        pytest.param(2, 1, _MOE_TP2_SHAPES, id="moe-tp2"),
        pytest.param(2, 2, _MOE_EP2_SHAPES, id="moe-ep2"),
    ],
)
def test_moe_partition_accepts_the_partition_a_row_names(
    tmp_path: Path,
    tp_size: int,
    moe_ep_size: int,
    expert_shapes: tuple[tuple[int, ...], ...],
) -> None:
    assert_moe_partition(
        _moe_row(tp_size, moe_ep_size),
        _final_manifests(tp_size, expert_shapes),
        _checkpoint(tmp_path, _MOE_CONFIG),
    )


@pytest.mark.parametrize(
    "moe_ep_size, realized_shapes",
    [
        pytest.param(2, _MOE_TP2_SHAPES, id="ep2-row-realized-moe-tp2"),
        pytest.param(1, _MOE_EP2_SHAPES, id="tp2-row-realized-moe-ep2"),
    ],
)
def test_moe_partition_rejects_a_partition_the_row_did_not_name(
    tmp_path: Path,
    moe_ep_size: int,
    realized_shapes: tuple[tuple[int, ...], ...],
) -> None:
    """A MoE size that never reached the model leaves every role on the same other partition."""
    with pytest.raises(AssertionError, match=f"did not realize MoE EP {moe_ep_size}"):
        assert_moe_partition(
            _moe_row(2, moe_ep_size),
            _final_manifests(2, realized_shapes),
            _checkpoint(tmp_path, _MOE_CONFIG),
        )


@pytest.mark.parametrize("tp_size, moe_ep_size", [(1, 2), (2, 0), (2, 3)])
def test_moe_row_rejects_an_ep_size_that_does_not_divide_tp(tp_size: int, moe_ep_size: int) -> None:
    """A row's MoE split is checked when the row is defined, before any worker runs."""
    with pytest.raises(
        ValueError, match=f"MoE EP size {moe_ep_size} does not divide TP size {tp_size}"
    ):
        _moe_row(tp_size, moe_ep_size)


def test_moe_partition_asks_to_be_extended_for_another_moe_layout(tmp_path: Path) -> None:
    """A routed-expert layout other than the 3-D CUTLASS one fails as unknown, not as mismatched."""
    block_major_shapes = ((8, 4, 32, 32), (8, 32, 2, 32))
    with pytest.raises(AssertionError, match="teach `assert_moe_partition` this MoE layout"):
        assert_moe_partition(
            _moe_row(2),
            _final_manifests(2, block_major_shapes),
            _checkpoint(tmp_path, _MOE_CONFIG),
        )


def test_moe_partition_asks_to_be_extended_for_another_moe_config(tmp_path: Path) -> None:
    config = {"num_experts": 8, "intermediate_size": 64, "hidden_size": 32}
    with pytest.raises(AssertionError, match="teach `assert_moe_partition` this MoE config"):
        assert_moe_partition(
            _moe_row(2),
            _final_manifests(2, _MOE_TP2_SHAPES),
            _checkpoint(tmp_path, config),
        )


def test_moe_partition_requires_routed_experts_of_a_moe_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="holds no routed-expert `w3_w1_weight` tensor"):
        assert_moe_partition(
            _moe_row(2), _final_manifests(2, None), _checkpoint(tmp_path, _MOE_CONFIG)
        )


def test_moe_partition_skips_dense_rows(tmp_path: Path) -> None:
    dense_checkpoint = _checkpoint(tmp_path, {"hidden_size": 32})

    assert_moe_partition(_moe_row(2), _final_manifests(2, None), dense_checkpoint)
    with pytest.raises(AssertionError, match="config has no `num_experts`"):
        assert_moe_partition(_moe_row(2), _final_manifests(2, _MOE_TP2_SHAPES), dense_checkpoint)
