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
"""CPU tests of the dense DWDP settings: ``DenseConfig``, ``DwdpConfig`` and the launch rules.

``DwdpConfig.enable_dense`` and ``copy_mode`` are the public settings of dense DWDP, which runs
only without expert DWDP (``enable_experts=False``); ``TorchLlmArgs.validate_dwdp_launch`` refuses
the launches dense DWDP cannot serve, except that a DSpark or DFlash drafter with
``context_kv_only`` is accepted. The dense group is an attention-DP instance or a Ulysses
context-parallel one at tensor parallel size 1.
"""

import re
from typing import Any

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.types import DEFAULT_MIN_BYTES, CopyMode, SegmentKind
from tensorrt_llm.llmapi.llm_args import (
    CpConfig,
    CudaGraphConfig,
    DFlashDecodingConfig,
    DSparkDecodingConfig,
    DwdpConfig,
    LoraConfig,
    MTPDecodingConfig,
    PrefillCudaGraphBackend,
    SleepConfig,
    TorchCompileConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import CpType

pytestmark = pytest.mark.cpu_only

_STANDALONE = dict(enable_experts=False, enable_dense=True)
_ATTENTION_DP = dict(tensor_parallel_size=16, enable_attention_dp=True)
_ULYSSES = dict(context_parallel_size=8, cp_config=CpConfig(cp_type=CpType.ULYSSES))

# ---------------------------------------------------------------------------
# DenseConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dwdp_config,enabled",
    [(None, False), (DwdpConfig(dwdp_size=2), False), (DwdpConfig(**_STANDALONE), True)],
    ids=["no-dwdp-config", "expert-only", "dense"],
)
def test_a_launch_sets_only_whether_dense_dwdp_is_on(
    dwdp_config: DwdpConfig | None, enabled: bool
) -> None:
    """``from_dwdp_config`` takes ``enable_dense``; every other field keeps its default."""
    assert DenseConfig.from_dwdp_config(dwdp_config) == DenseConfig(enabled=enabled)
    config = DenseConfig.from_dwdp_config(dwdp_config)
    assert config.segments == (SegmentKind.ATTN, SegmentKind.MOE)
    assert (config.min_bytes, config.store_chunk_mb, config.strict) == (DEFAULT_MIN_BYTES, 0, False)


@pytest.mark.parametrize(
    "copy_mode,expected",
    [(None, CopyMode.BATCHED), ("batched", CopyMode.BATCHED), ("gather", CopyMode.GATHER)],
)
def test_a_launch_sets_the_standalone_transfer(copy_mode: str | None, expected: CopyMode) -> None:
    """``from_dwdp_config`` also takes ``copy_mode``; unset, the transfer is batched."""
    config = DenseConfig.from_dwdp_config(DwdpConfig(**_STANDALONE, copy_mode=copy_mode))
    assert config.standalone_copy_mode() is expected


def test_segments_are_canonical_and_start_with_attention() -> None:
    """Strings are converted and put in canonical order; repeats collapse."""
    assert DenseConfig(segments=("moe", "attn", "moe")).segments == (
        SegmentKind.ATTN,
        SegmentKind.MOE,
    )
    assert DenseConfig(segments=("attn",)).segments == (SegmentKind.ATTN,)


@pytest.mark.parametrize(
    "segments,message",
    [(("moe",), "must include the attention segment"), (("ffn",), "has an unknown segment")],
)
def test_invalid_segments_raise(segments: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DenseConfig(segments=segments)


@pytest.mark.parametrize("name", ["min_bytes", "store_chunk_mb"])
def test_sizes_are_non_negative(name: str) -> None:
    with pytest.raises(ValueError, match=f"DenseConfig.{name} must be non-negative"):
        DenseConfig(**{name: -1})


# ---------------------------------------------------------------------------
# DwdpConfig
# ---------------------------------------------------------------------------


def test_the_defaults_keep_expert_dwdp_alone() -> None:
    config = DwdpConfig()
    assert (config.enable_experts, config.enable_dense, config.copy_mode) == (True, False, None)


@pytest.mark.parametrize(
    "fields",
    [
        dict(_STANDALONE),
        dict(_STANDALONE, copy_mode="batched"),
        dict(_STANDALONE, copy_mode="per_slice"),
        dict(_STANDALONE, copy_mode="gather"),
        dict(dwdp_size=4, contention_opt=True),
    ],
    ids=["dense", "dense-batched", "dense-per_slice", "dense-gather", "expert"],
)
def test_accepted_combinations(fields: dict) -> None:
    DwdpConfig(**fields)


def test_nothing_to_distribute_is_rejected() -> None:
    with pytest.raises(ValidationError, match="enable_experts and enable_dense are both"):
        DwdpConfig(enable_experts=False)


def test_dense_dwdp_runs_without_expert_dwdp() -> None:
    with pytest.raises(ValidationError, match="set enable_experts=False"):
        DwdpConfig(enable_dense=True)


def test_copy_mode_needs_dense_dwdp() -> None:
    with pytest.raises(ValidationError, match="copy_mode='batched' selects the transfer"):
        DwdpConfig(dwdp_size=4, copy_mode="batched")


def test_copy_mode_takes_the_three_transfers_only() -> None:
    with pytest.raises(ValidationError, match="copy_mode"):
        DwdpConfig(**_STANDALONE, copy_mode="fast")


@pytest.mark.parametrize(
    "fields",
    [
        dict(dwdp_size=4),
        dict(num_groups=2),
        dict(num_experts_per_worker=8),
        dict(num_prefetch_experts=8),
        dict(contention_opt=True),
    ],
    ids=lambda fields: ",".join(fields),
)
def test_expert_fields_need_expert_dwdp(fields: dict) -> None:
    """Without expert DWDP the expert fields keep their defaults."""
    with pytest.raises(
        ValidationError,
        match=f"{next(iter(fields))}=.* configure expert DWDP, but enable_experts=False turns it "
        "off",
    ):
        DwdpConfig(**_STANDALONE, **fields)


# ---------------------------------------------------------------------------
# TorchLlmArgs.validate_dwdp_launch
# ---------------------------------------------------------------------------


def _check(dwdp: dict, **fields: Any) -> None:
    """Run the validator alone on a launch's arguments, applied unvalidated."""
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        skip_tokenizer_init=True,
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
    )
    args.model_copy(update={"dwdp_config": DwdpConfig(**dwdp), **fields}).validate_dwdp_launch()


def _dspark(context_kv_only: bool) -> DSparkDecodingConfig:
    """A DSpark drafter config, built without its validators: the launch rules read the type and
    ``context_kv_only`` only."""
    return DSparkDecodingConfig.model_construct(max_draft_len=7, context_kv_only=context_kv_only)


def _dflash(context_kv_only: bool) -> DFlashDecodingConfig:
    """A DFlash drafter config, built without its validators (see ``_dspark``)."""
    return DFlashDecodingConfig.model_construct(max_draft_len=7, context_kv_only=context_kv_only)


def test_an_unset_copy_mode_gathers_on_a_ulysses_instance() -> None:
    """Unset, the transfer is batched on an attention-DP instance and the SM gather on a Ulysses
    context-parallel one; a set copy_mode wins on both."""
    unset = DwdpConfig(**_STANDALONE)
    assert DenseConfig.from_dwdp_config(unset).standalone_copy_mode() is CopyMode.BATCHED
    assert DenseConfig.from_dwdp_config(unset, cp_size=8).standalone_copy_mode() is CopyMode.GATHER
    batched = DwdpConfig(**_STANDALONE, copy_mode="batched")
    assert (
        DenseConfig.from_dwdp_config(batched, cp_size=8).standalone_copy_mode() is CopyMode.BATCHED
    )


@pytest.mark.parametrize(
    "fields",
    [
        dict(_ATTENTION_DP),
        dict(_ATTENTION_DP, disable_overlap_scheduler=False),
        dict(_ATTENTION_DP, speculative_config=_dspark(context_kv_only=True)),
        dict(_ULYSSES),
        dict(context_parallel_size=8),
        dict(_ULYSSES, speculative_config=_dflash(context_kv_only=True)),
        dict(_ATTENTION_DP, speculative_config=_dflash(context_kv_only=True)),
    ],
    ids=[
        "attention_dp",
        "overlap_scheduler",
        "dspark-context_kv_only",
        "ulysses_cp",
        "cp_without_cp_config",
        "ulysses_cp-dflash-context_kv_only",
        "dflash-context_kv_only",
    ],
)
def test_accepted_dense_launches(fields: dict) -> None:
    _check(_STANDALONE, **fields)


def test_expert_dwdp_launches_are_left_to_the_executor() -> None:
    """Expert DWDP keeps its launch checks in the executor creator; the validator passes it."""
    _check(dict(dwdp_size=16))


@pytest.mark.parametrize(
    "fields,error",
    [
        (dict(tensor_parallel_size=16), "requires enable_attention_dp=True"),
        (dict(enable_attention_dp=True), "requires tensor_parallel_size > 1"),
        (
            dict(_ATTENTION_DP, cuda_graph_config=CudaGraphConfig()),
            "requires cuda_graph_config=None, but cuda_graph_config is set",
        ),
        (
            dict(_ATTENTION_DP, torch_compile_config=TorchCompileConfig()),
            "requires torch_compile_config=None, but torch_compile_config is set",
        ),
        (
            dict(_ATTENTION_DP, prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE),
            "requires prefill_cuda_graph_backend='disabled', but "
            "prefill_cuda_graph_backend='breakable'",
        ),
        (
            dict(_ATTENTION_DP, sleep_config=SleepConfig()),
            "requires sleep_config=None, but sleep_config is set",
        ),
        (
            dict(_ATTENTION_DP, lora_config=LoraConfig()),
            "requires lora_config=None, but lora_config is set",
        ),
        (
            dict(_ATTENTION_DP, speculative_config=MTPDecodingConfig(max_draft_len=1)),
            "or a DSpark or DFlash drafter with context_kv_only=True, but speculative_config is "
            "a MTPDecodingConfig",
        ),
        (
            dict(_ATTENTION_DP, speculative_config=_dspark(context_kv_only=False)),
            "but speculative_config is a DSparkDecodingConfig without context_kv_only",
        ),
        (
            dict(_ATTENTION_DP, speculative_config=_dflash(context_kv_only=False)),
            "but speculative_config is a DFlashDecodingConfig without context_kv_only",
        ),
        (
            dict(_ATTENTION_DP, enable_locality_domains=True),
            "requires enable_locality_domains=False, but enable_locality_domains=True",
        ),
        (
            dict(_ATTENTION_DP, pipeline_parallel_size=2),
            "requires pipeline_parallel_size=1, but pipeline_parallel_size=2",
        ),
        (
            dict(_ATTENTION_DP, context_parallel_size=2),
            "requires enable_attention_dp=False with Ulysses context parallelism",
        ),
        (
            dict(_ULYSSES, tensor_parallel_size=2),
            "requires tensor_parallel_size=1 with Ulysses context parallelism, but "
            "tensor_parallel_size=2",
        ),
        (
            dict(context_parallel_size=8, cp_config=CpConfig(cp_type=CpType.HELIX)),
            "requires cp_config.cp_type=ULYSSES with context_parallel_size > 1",
        ),
        (
            dict(_ULYSSES, speculative_config=_dflash(context_kv_only=False)),
            "but speculative_config is a DFlashDecodingConfig without context_kv_only",
        ),
    ],
    ids=[
        "without_attention_dp",
        "tensor_parallel_1",
        "cuda_graph_config",
        "torch_compile_config",
        "prefill_cuda_graph_backend",
        "sleep_config",
        "lora_config",
        "speculative_config",
        "dspark-drafting",
        "dflash-drafting",
        "enable_locality_domains",
        "pipeline_parallel",
        "context_parallel-attention_dp",
        "ulysses_cp-tensor_parallel_2",
        "helix_cp",
        "ulysses_cp-dflash-drafting",
    ],
)
def test_dense_dwdp_requirements(fields: dict, error: str) -> None:
    with pytest.raises(ValueError, match=re.escape(error)) as raised:
        _check(_STANDALONE, **fields)
    message = str(raised.value)
    assert message.startswith("Dense DWDP (dwdp_config.enable_dense) ")
    assert message.endswith(
        "Change the launch configuration or set dwdp_config.enable_dense=False."
    )


def test_building_the_arguments_runs_it() -> None:
    """The validator runs when the arguments are built: a launch without attention DP fails."""
    with pytest.raises(ValidationError, match="requires enable_attention_dp=True"):
        TorchLlmArgs(
            model="/tmp/dummy_model",
            skip_tokenizer_init=True,
            dwdp_config=DwdpConfig(**_STANDALONE),
        )


def test_arguments_without_dense_dwdp_pass_it() -> None:
    """Without ``dwdp_config`` the validator changes nothing."""
    args = TorchLlmArgs(model="/tmp/dummy_model", skip_tokenizer_init=True)
    assert args.dwdp_config is None
    assert args.validate_dwdp_launch() is args
