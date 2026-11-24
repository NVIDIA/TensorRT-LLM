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
"""DFlash speculative_config.context_kv_only / context_kv_dtype and the MLA drafter's draft cache.

context_kv_only validates as DSpark's does. context_kv_dtype takes DSpark's
values and also None, its DFlash default: no context transfer unless set. Both
reach the KV cache setup through the LLM API arguments. context_kv_dtype='fp8'
with an MLA drafter that keeps its latents in its weight dtype fails at setup
instead of storing BF16.

Helix puts every MLA DFlash drafter's context KV in the target manager's unified
draft cache, and disaggregation does with ``speculative_config.context_kv_dtype``
set. Otherwise a drafter keeps its private ring unless it sets
``unified_draft_cache_when_aggregated = True``. A GQA DFlash drafter never uses it.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm.llmapi import DFlashDecodingConfig, DSparkDecodingConfig
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, TorchLlmArgs
from tensorrt_llm.mapping import CpType, Mapping

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("config_cls", [DFlashDecodingConfig, DSparkDecodingConfig])
def test_context_kv_fields_default_and_set(config_cls):
    config = config_cls(max_draft_len=5)
    assert config.context_kv_only is False
    assert config.context_kv_dtype == (None if config_cls is DFlashDecodingConfig else "auto")

    config = config_cls(max_draft_len=5, context_kv_only=True, context_kv_dtype="fp8")
    assert config.context_kv_only is True
    assert config.context_kv_dtype == "fp8"


@pytest.mark.parametrize(
    "config_cls, dtype",
    [
        *(
            (cls, dtype)
            for cls in (DFlashDecodingConfig, DSparkDecodingConfig)
            for dtype in ("bf16", "fp8_e4m3", "FP8")
        ),
        (DSparkDecodingConfig, None),
    ],
)
def test_context_kv_dtype_rejects_other_values(config_cls, dtype):
    with pytest.raises(ValueError):
        config_cls(max_draft_len=5, context_kv_dtype=dtype)


def test_dflash_context_kv_fields_match_dspark():
    """context_kv_only is DSpark's field; context_kv_dtype takes DSpark's values or None."""
    dflash = DFlashDecodingConfig.model_fields
    dspark = DSparkDecodingConfig.model_fields
    assert dflash["context_kv_only"].annotation == dspark["context_kv_only"].annotation
    assert dflash["context_kv_only"].default == dspark["context_kv_only"].default
    dspark_dtype = dspark["context_kv_dtype"].annotation
    assert dflash["context_kv_dtype"].annotation == Optional[dspark_dtype]
    assert dflash["context_kv_dtype"].default is None


def test_dflash_context_kv_fields_round_trip_through_a_dump():
    config = DFlashDecodingConfig(max_draft_len=5, context_kv_only=True, context_kv_dtype="fp8")
    restored = DFlashDecodingConfig.model_validate(config.model_dump())
    assert (restored.context_kv_only, restored.context_kv_dtype) == (True, "fp8")


def _mla_drafter_dir(tmp_path: Path) -> Path:
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlash2DraftModel"],
                "dflash_config": {
                    "attention_mode": "mla",
                    "target_layer_ids": [0, 1],
                    "conv_kernel_size": 2,
                    "selector_rank": 8,
                },
            }
        )
    )
    return draft


@pytest.mark.parametrize("context_kv_only", [False, True])
def test_dflash_context_kv_fields_reach_the_llm_args(tmp_path: Path, context_kv_only: bool):
    """The fields survive the dict form of speculative_config, as a YAML file gives it."""
    args = TorchLlmArgs(
        model=tmp_path,
        gpus_per_node=1,
        max_batch_size=1,
        max_num_tokens=64,
        max_seq_len=128,
        kv_cache_config={"use_kv_cache_manager_v2": True, "enable_block_reuse": False},
        speculative_config={
            "decoding_type": "DFlash",
            "speculative_model": str(_mla_drafter_dir(tmp_path)),
            "max_draft_len": 4,
            "context_kv_only": context_kv_only,
            "context_kv_dtype": "fp8",
        },
        cache_transceiver_config=CacheTransceiverConfig(backend="NIXL"),
    )
    spec = args.speculative_config
    assert isinstance(spec, DFlashDecodingConfig)
    assert spec.context_kv_only is context_kv_only
    assert spec.context_kv_dtype == "fp8"


def _drafter(*, supports=True, aggregated=None, kv_factor=1, context_kv_fp8=False):
    """An MLA drafter's attributes the KV cache setup reads (five 576-wide latents)."""
    fields = dict(
        supports_unified_draft_cache=supports,
        context_kv_fp8=context_kv_fp8,
        _num_attn_layers=5,
        _head_dim=576,
        dflash_attention_backend="VANILLA",
    )
    if kv_factor is not None:
        fields["_kv_factor"] = kv_factor
    if aggregated is not None:
        fields["unified_draft_cache_when_aggregated"] = aggregated
    return SimpleNamespace(**fields)


def _creator(
    *,
    draft,
    attention_mode="mla",
    disagg=False,
    v2=True,
    mapping=None,
    context_kv_only=False,
    context_kv_dtype=None,
):
    """A KvCacheCreator for a standalone DFlash drafter, without its constructor."""
    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dspark=lambda: False, is_dflash=lambda: True),
        draft_is_embedded_in_target=False,
        _use_shared_kv_cache=False,
        context_kv_only=context_kv_only,
        context_kv_dtype=context_kv_dtype,
        max_draft_len=7,
        attention_backend="AUTO",
    )
    creator._model_engine = SimpleNamespace(model=SimpleNamespace(draft_model=draft))
    creator._is_kv_cache_manager_v2 = v2
    creator._is_disagg = disagg
    creator._mapping = mapping or Mapping()
    creator._draft_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(dflash_config={"attention_mode": attention_mode}),
        mapping=Mapping(),
    )
    return creator


_DISTRIBUTED = ["disagg", "helix", "helix_repurposed"]


def _distributed(kind):
    """The creator arguments of a distributed run: disaggregated with the drafter's context carried
    (context_kv_dtype set), or Helix CP over 8 ranks (the target's mapping, or the one with CP
    repurposed to TP)."""
    if kind == "disagg":
        return {"disagg": True, "context_kv_dtype": "auto"}
    helix = Mapping(world_size=8, cp_size=8, cp_config={"cp_type": CpType.HELIX})
    return {"mapping": helix if kind == "helix" else helix.repurpose_helix_cp_to_tp()}


def test_aggregated_mla_drafter_that_opts_in_is_unified():
    creator = _creator(draft=_drafter(aggregated=True))
    assert creator._uses_unified_standalone_draft_cache()
    # The target manager then owns the drafter's context KV: no separate draft manager.
    assert not creator._should_create_separate_draft_kv_cache()
    creator._validate_standalone_draft_cache()
    kwargs = creator._get_unified_draft_cache_kwargs()
    assert (kwargs["draft_num_layers"], kwargs["draft_kv_factor"]) == (5, 1)
    assert (kwargs["draft_cp_size"], kwargs["draft_head_dim"]) == (1, 576)


@pytest.mark.parametrize(
    "draft",
    [
        _drafter(),  # the general target's drafter: unified-capable, no aggregated opt-in
        _drafter(aggregated=False),
        # Only a literal True opts in, not any truthy attribute.
        _drafter(aggregated=1),
        SimpleNamespace(
            supports_unified_draft_cache=True, unified_draft_cache_when_aggregated=MagicMock()
        ),
    ],
    ids=["no_attribute", "false", "truthy_int", "mock"],
)
def test_aggregated_mla_drafter_without_the_opt_in_keeps_its_private_ring(draft):
    creator = _creator(draft=draft)
    assert not creator._uses_unified_standalone_draft_cache()
    creator._validate_standalone_draft_cache()


@pytest.mark.parametrize("aggregated", [None, True])
@pytest.mark.parametrize("kind", _DISTRIBUTED)
def test_distributed_mla_drafter_is_unified_either_way(kind, aggregated):
    creator = _creator(draft=_drafter(aggregated=aggregated), **_distributed(kind))
    assert creator._uses_unified_standalone_draft_cache()
    creator._validate_standalone_draft_cache()


@pytest.mark.parametrize(
    "aggregated, context_kv_dtype, unified",
    [(None, "auto", True), (True, "auto", True), (True, None, True), (None, None, False)],
)
def test_disaggregated_mla_drafter_is_unified_with_context_kv_dtype_or_its_opt_in(
    aggregated, context_kv_dtype, unified
):
    creator = _creator(
        draft=_drafter(aggregated=aggregated), disagg=True, context_kv_dtype=context_kv_dtype
    )
    assert creator._uses_unified_standalone_draft_cache() is unified
    creator._validate_standalone_draft_cache()


@pytest.mark.parametrize("context_kv_dtype", ["auto", "fp8"])
def test_disaggregated_context_server_with_dflash2_history_needs_the_v2_manager(context_kv_dtype):
    """A disaggregated context server that carries a DFlash2 drafter's history (context_kv_only)
    keeps it in the unified draft cache of KV cache manager V2, and is refused without one."""
    kwargs = dict(
        draft=_drafter(context_kv_fp8=True),
        disagg=True,
        context_kv_only=True,
        context_kv_dtype=context_kv_dtype,
    )
    creator = _creator(**kwargs)
    assert creator._carries_dflash_context()
    assert creator._uses_unified_standalone_draft_cache()
    creator._validate_standalone_draft_cache()
    with pytest.raises(ValueError, match="Distributed MLA DFlash requires"):
        _creator(v2=False, **kwargs)._validate_standalone_draft_cache()


@pytest.mark.parametrize("disagg", [False, True])
def test_gqa_dflash_never_uses_the_unified_cache(disagg):
    """The opt-in is read only for an MLA drafter; a GQA DFlash run never goes unified."""
    creator = _creator(draft=_drafter(aggregated=True), attention_mode="gqa", disagg=disagg)
    assert not creator._needs_unified_mla_dflash()
    assert not creator._uses_unified_standalone_draft_cache()
    creator._validate_standalone_draft_cache()


@pytest.mark.parametrize(
    "kwargs",
    [{"v2": False}, {"draft": _drafter(supports=False, aggregated=True)}],
    ids=["v1", "unsupported"],
)
def test_opted_in_drafter_without_a_unified_cache_fails_at_setup(kwargs):
    kwargs.setdefault("draft", _drafter(aggregated=True))
    with pytest.raises(ValueError, match="unified_draft_cache_when_aggregated"):
        _creator(**kwargs)._validate_standalone_draft_cache()


@pytest.mark.parametrize("kind", _DISTRIBUTED)
def test_distributed_mla_drafter_without_a_unified_cache_is_refused(kind):
    """KV cache manager V1 has no unified draft cache for a distributed MLA drafter."""
    creator = _creator(draft=_drafter(), v2=False, **_distributed(kind))
    with pytest.raises(ValueError, match="Distributed MLA DFlash requires"):
        creator._validate_standalone_draft_cache()


def test_fp8_context_kv_with_a_bf16_drafter_raises():
    """A drafter that does not read the field (its FP8 switch off) would store BF16."""
    creator = _creator(draft=_drafter(), disagg=True, context_kv_dtype="fp8")
    with pytest.raises(ValueError, match="context_kv_dtype='fp8'") as error:
        creator._validate_standalone_draft_cache()
    assert "context_kv_fp8" in str(error.value)


@pytest.mark.parametrize("context_kv_only", [False, True])
def test_fp8_context_kv_with_an_fp8_drafter_is_accepted(context_kv_only):
    creator = _creator(
        draft=_drafter(context_kv_fp8=True),
        disagg=True,
        context_kv_dtype="fp8",
        context_kv_only=context_kv_only,
    )
    creator._validate_standalone_draft_cache()


@pytest.mark.parametrize(
    "kv_factor, dtype",
    [
        # 'auto' never asks for FP8.
        (1, "auto"),
        # A drafter whose storage is not an MLA latent is not covered by the check.
        (None, "fp8"),
        (2, "fp8"),
    ],
    ids=["auto", "no_kv_factor", "gqa"],
)
def test_fp8_check_covers_only_mla_latents_asked_for_fp8(kv_factor, dtype):
    creator = _creator(draft=_drafter(kv_factor=kv_factor), disagg=True, context_kv_dtype=dtype)
    creator._validate_standalone_draft_cache()


def test_fp8_context_kv_without_a_unified_cache_is_refused():
    """Without a unified draft cache the context_kv_dtype check refuses first."""
    creator = _creator(draft=_drafter(), context_kv_dtype="fp8")
    with pytest.raises(ValueError, match="context_kv_dtype sets the storage"):
        creator._validate_standalone_draft_cache()
