# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends import interface as interface_module
from tensorrt_llm._torch.attention.backends import trtllm as trtllm_module
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import KVCacheLayerSpec, ModelConfig
from tensorrt_llm._torch.pyexecutor.engine.metadata import (
    _get_max_num_heads_per_kv,
    build_attention_metadata,
)
from tensorrt_llm.mapping import CpType, Mapping

pytestmark = pytest.mark.cpu_only


class _AttentionMetadata:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _AttentionBackend:
    Metadata = _AttentionMetadata


def test_build_attention_metadata_resolves_model_derived_values() -> None:
    sparse_metadata_params = object()
    sparse_attention_config = SimpleNamespace(
        to_sparse_metadata_params=Mock(return_value=sparse_metadata_params)
    )
    pretrained_config = SimpleNamespace(
        num_attention_heads=8,
        num_key_value_heads=[4, 2],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    model_config = SimpleNamespace(
        pretrained_config=pretrained_config,
        sparse_attention_config=sparse_attention_config,
        enable_flash_mla=True,
    )
    runtime_features = AttentionRuntimeFeatures(cache_reuse=True)

    metadata = build_attention_metadata(
        model_config,
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=2,
        attention_backend=_AttentionBackend,
        attention_runtime_features=runtime_features,
        mapping=object(),
        cache_indirection=None,
    )

    assert metadata.max_num_heads_per_kv == 4
    assert metadata.enable_flash_mla
    assert metadata.enable_context_mla_with_cached_kv
    assert metadata.sparse_metadata_params is sparse_metadata_params
    sparse_attention_config.to_sparse_metadata_params.assert_called_once_with(
        pretrained_config=pretrained_config
    )


def test_build_attention_metadata_forwards_shared_and_cache_inputs() -> None:
    mapping = object()
    cache_indirection = object()
    kv_cache_manager = object()
    draft_kv_cache_manager = object()
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(),
        sparse_attention_config=None,
        enable_flash_mla=True,
    )

    metadata = build_attention_metadata(
        model_config,
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=2,
        attention_backend=_AttentionBackend,
        attention_runtime_features=AttentionRuntimeFeatures(),
        mapping=mapping,
        cache_indirection=cache_indirection,
        kv_cache_manager=kv_cache_manager,
        draft_kv_cache_manager=draft_kv_cache_manager,
        enable_context_mla_with_cached_kv=True,
        max_num_heads_per_kv=4,
    )

    assert metadata.max_num_requests == 4
    assert metadata.max_num_tokens == 16
    assert metadata.max_num_sequences == 8
    assert metadata.mapping is mapping
    assert metadata.cache_indirection is cache_indirection
    assert metadata.kv_cache_manager is kv_cache_manager
    assert metadata.draft_kv_cache_manager is draft_kv_cache_manager
    assert metadata.enable_context_mla_with_cached_kv
    assert metadata.max_num_heads_per_kv == 4


@pytest.mark.parametrize(
    ("mla", "cp_size", "override", "expected"),
    [
        pytest.param(True, 2, None, True, id="fresh-helix-mla"),
        pytest.param(True, 1, None, False, id="ordinary-mla"),
        pytest.param(False, 2, None, False, id="helix-non-mla"),
        pytest.param(True, 1, True, True, id="explicit-enable"),
    ],
)
def test_build_attention_metadata_fresh_helix_mla_default(
    mla: bool, cp_size: int, override: bool | None, expected: bool
) -> None:
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            kv_lora_rank=512 if mla else None,
            qk_rope_head_dim=64 if mla else None,
        ),
        sparse_attention_config=None,
        enable_flash_mla=False,
    )
    metadata = build_attention_metadata(
        model_config,
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=1,
        attention_backend=_AttentionBackend,
        attention_runtime_features=AttentionRuntimeFeatures(
            cache_reuse=False, chunked_prefill=False
        ),
        mapping=Mapping(world_size=cp_size, cp_size=cp_size, cp_config={"cp_type": CpType.HELIX}),
        cache_indirection=None,
        enable_context_mla_with_cached_kv=override,
    )

    assert metadata.enable_context_mla_with_cached_kv is expected


@pytest.mark.parametrize("algorithm", ["dsa", "deepseek_v4"])
@pytest.mark.parametrize("override", [None, False, True])
@pytest.mark.parametrize(
    ("cache_reuse", "chunked_prefill"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_build_attention_metadata_sparse_helix_preserves_cached_kv_policy(
    algorithm: str, override: bool | None, cache_reuse: bool, chunked_prefill: bool
) -> None:
    sparse_metadata_params = SimpleNamespace(algorithm=algorithm)
    sparse_config = SimpleNamespace(
        algorithm=algorithm,
        to_sparse_metadata_params=Mock(return_value=sparse_metadata_params),
    )
    pretrained_config = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
    model_config = SimpleNamespace(
        pretrained_config=pretrained_config,
        sparse_attention_config=sparse_config,
        enable_flash_mla=False,
    )
    metadata = build_attention_metadata(
        model_config,
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=1,
        attention_backend=_AttentionBackend,
        attention_runtime_features=AttentionRuntimeFeatures(
            cache_reuse=cache_reuse, chunked_prefill=chunked_prefill
        ),
        mapping=Mapping(world_size=2, cp_size=2, cp_config={"cp_type": CpType.HELIX}),
        cache_indirection=None,
        enable_context_mla_with_cached_kv=override,
    )

    expected = cache_reuse or chunked_prefill if override is None else override
    assert metadata.enable_context_mla_with_cached_kv is expected
    assert metadata.sparse_metadata_params is sparse_metadata_params
    sparse_config.to_sparse_metadata_params.assert_called_once_with(
        pretrained_config=pretrained_config
    )


@pytest.mark.parametrize(
    ("cache_reuse", "chunked_prefill"), [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize(
    ("mla", "cp_size", "cp_type"),
    [
        pytest.param(True, 2, CpType.HELIX, id="helix-mla"),
        pytest.param(True, 1, CpType.HELIX, id="ordinary-mla"),
        pytest.param(False, 2, CpType.HELIX, id="helix-non-mla"),
        pytest.param(True, 2, CpType.ULYSSES, id="non-helix-cp-mla"),
    ],
)
def test_build_attention_metadata_explicit_disable(
    mla: bool, cp_size: int, cp_type: CpType, cache_reuse: bool, chunked_prefill: bool
) -> None:
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            kv_lora_rank=512 if mla else None,
            qk_rope_head_dim=64 if mla else None,
        ),
        sparse_attention_config=None,
        enable_flash_mla=False,
    )
    constructor = Mock(side_effect=_AttentionMetadata)
    kwargs = dict(
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=1,
        attention_backend=SimpleNamespace(Metadata=constructor),
        attention_runtime_features=AttentionRuntimeFeatures(
            cache_reuse=cache_reuse, chunked_prefill=chunked_prefill
        ),
        mapping=Mapping(world_size=cp_size, cp_size=cp_size, cp_config={"cp_type": cp_type}),
        cache_indirection=None,
        enable_context_mla_with_cached_kv=False,
    )
    if mla and cp_size > 1 and cp_type == CpType.HELIX:
        with pytest.raises(
            ValueError, match="Helix MLA requires enable_context_mla_with_cached_kv"
        ):
            build_attention_metadata(model_config, **kwargs)
        constructor.assert_not_called()
    else:
        metadata = build_attention_metadata(model_config, **kwargs)
        assert metadata.enable_context_mla_with_cached_kv is False
        constructor.assert_called_once()


@pytest.mark.parametrize("override", [None, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_build_attention_metadata_helix_mla_preparation_lifecycle(
    monkeypatch: pytest.MonkeyPatch, override: bool | None, rank: int
) -> None:
    """Run real allocation/preparation logic with only CUDA storage replaced."""
    monkeypatch.setattr(
        interface_module.AttentionMetadata,
        "get_empty",
        staticmethod(
            lambda buffers, tensor_shape, dtype, cache_name, capture_graph=False: torch.zeros(
                tensor_shape, dtype=dtype
            )
        ),
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self.clone())
    monkeypatch.setattr(interface_module, "maybe_pin_memory", lambda tensor: tensor)
    monkeypatch.setattr(trtllm_module, "maybe_pin_memory", lambda tensor: tensor)
    monkeypatch.setattr(trtllm_module, "prefer_pinned", lambda: False)
    monkeypatch.setattr(trtllm_module, "get_model_extra_attrs", lambda: object())
    mapping = Mapping(world_size=2, rank=rank, cp_size=2, cp_config={"cp_type": CpType.HELIX})
    manager = SimpleNamespace(
        mapping=mapping,
        tokens_per_block=16,
        num_pools=1,
        max_blocks_per_seq=8,
        max_seq_len=128,
        host_kv_cache_block_offsets=torch.zeros((1, 3, 2, 8), dtype=torch.int32),
        copy_batch_block_offsets=Mock(),
    )
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        sparse_attention_config=None,
        enable_flash_mla=False,
    )
    metadata = build_attention_metadata(
        model_config,
        max_batch_size=3,
        max_num_tokens=32,
        max_beam_width=1,
        attention_backend=SimpleNamespace(
            Metadata=partial(
                TrtllmAttentionMetadata,
                workspace=torch.empty(0),
                cuda_graph_workspace=torch.empty(0),
            )
        ),
        attention_runtime_features=AttentionRuntimeFeatures(),
        mapping=mapping,
        cache_indirection=None,
        kv_cache_manager=manager,
        enable_context_mla_with_cached_kv=override,
    )
    assert metadata.enable_context_mla_with_cached_kv is True
    # Fresh mixed batch, decode-only, then changed fresh context on the same object.
    for lengths, num_contexts, cached in [
        ([17, 3, 1], 2, [0, 0, 9]),
        ([1], 0, [10]),
        ([3, 19, 1], 2, [0, 0, 11]),
    ]:
        metadata.num_contexts = num_contexts
        metadata.seq_lens = torch.tensor(lengths, dtype=torch.int32)
        metadata.prompt_lens = lengths
        metadata.request_ids = list(range(len(lengths)))
        metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached)
        metadata.update_helix_param([cached[-1]], [False])
        metadata.prepare()

        assert metadata.num_ctx_cached_tokens == 0
        assert metadata.max_ctx_cached_token_len == 0
        assert metadata.max_ctx_seq_len == max(lengths[:num_contexts], default=0)
        assert metadata.max_ctx_kv_len == metadata.max_ctx_seq_len
        if not num_contexts:
            assert metadata.helix_context_local_slots is None
            continue
        expected_slots = [
            position // 32 * 16 + position % 16 if position // 16 % 2 == rank else -1
            for length in lengths[:num_contexts]
            for position in range(length)
        ]
        assert metadata.helix_context_local_slots.tolist() == expected_slots
        expected_indptr = [0, lengths[0], sum(lengths[:num_contexts])]
        for prefix in ("", "host_"):
            assert getattr(metadata, f"{prefix}ctx_cached_token_indptr")[:3].tolist() == [0, 0, 0]
            for name in ("ctx_uncached_token_indptr", "ctx_kv_indptr"):
                indptr = getattr(metadata, prefix + name)
                assert indptr.dtype == torch.int64
                assert indptr[:3].tolist() == expected_indptr
    assert manager.copy_batch_block_offsets.call_count == 3


@pytest.mark.parametrize(
    ("head_pairs", "expected_ratio"),
    [([(16, 8), (16, 1)], 16), ([(16, 8), (8, 2)], 4), ([(16, 1), (16, 8)], 16)],
)
def test_metadata_gqa_ratio_uses_model_provided_per_layer_heads(
    head_pairs: list[tuple[int, int]], expected_ratio: int
) -> None:
    class StrictConfig:
        num_hidden_layers = 2

        def __getattribute__(self, name: str) -> object:
            if name in {"num_attention_heads", "num_key_value_heads"}:
                raise RuntimeError(f"global geometry must not be read: {name}")
            return super().__getattribute__(name)

    model_config = ModelConfig(pretrained_config=StrictConfig())
    model_config.set_kv_cache_layer_specs(
        [
            KVCacheLayerSpec(head_dim=256, num_q_heads=q_heads, num_kv_heads=kv_heads)
            for q_heads, kv_heads in head_pairs
        ]
    )
    metadata = build_attention_metadata(
        model_config,
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=1,
        attention_backend=_AttentionBackend,
        attention_runtime_features=AttentionRuntimeFeatures(),
        mapping=object(),
        cache_indirection=None,
    )

    assert metadata.max_num_heads_per_kv == expected_ratio


@pytest.mark.parametrize(
    ("config", "expected_ratio"),
    [
        (SimpleNamespace(num_attention_heads=8, num_key_value_heads=2), 4),
        (SimpleNamespace(num_attention_heads=8, num_key_value_heads=[4, 2]), 4),
        (SimpleNamespace(num_attention_heads=8, num_key_value_heads=[None, 0]), 1),
        (SimpleNamespace(), 1),
    ],
)
def test_metadata_gqa_ratio_preserves_flat_config_fallback(
    config: SimpleNamespace, expected_ratio: int
) -> None:
    assert _get_max_num_heads_per_kv(SimpleNamespace(pretrained_config=config)) == expected_ratio
