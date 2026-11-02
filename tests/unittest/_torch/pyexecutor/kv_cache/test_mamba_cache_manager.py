# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Python, Cpp, and V2 Mamba cache managers."""

import contextlib
import math
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.attention.backends.fp4_mla import (
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_TOKENS_PER_BLOCK,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaV2CacheLayoutPolicy
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.cache_manager import (
    Glm5NextCacheManager,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.page import (
    MAMBA_CONV_ROLE,
    MAMBA_SSM_ROLE,
    AttentionLayerGroup,
    MambaLayerGroup,
    MapperKind,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.kimi_k3_cache_policy import KIMI_K3_FP8_KV_LAYERS_ENV
from tensorrt_llm._torch.modules.fla.cache_manager import (
    GDNReplayState,
    Qwen35HybridCacheManager,
    _use_flashinfer_ring_replay,
)
from tensorrt_llm._torch.modules.fla.flashinfer_cached_replay import FLASHINFER_REPLAY_RING_SIZE
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    _KDA_BETA_CACHE_ALIGNMENT_BYTES,
    KDAIntermediateState,
    KDAReplayLayerCache,
    KDAReplayState,
    KimiK3HybridCacheManager,
    _kda_replay_layout,
)
from tensorrt_llm._torch.modules.mamba.cache_manager import (
    MIN_REPLAY_HISTORY_SIZE,
    Mamba2State,
    NemotronHybridCacheManager,
    ReplayHistory,
    ReplayStateUpdateMetadata,
    _advance_replay_state,
)
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor._util import (
    KvCacheCreator,
    _create_kv_cache_manager,
    get_kv_cache_manager_cls,
)
from tensorrt_llm._torch.pyexecutor.config_utils import (
    MambaKVCacheParams,
    extract_mamba_kv_cache_params,
)
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    CppMambaHybridCacheManager,
    InforwardSnapshotEntry,
    MambaCacheManager,
    MambaHybridCacheManagerV2,
    MambaRole,
    MixedMambaHybridCacheManager,
    PythonMambaCacheManager,
    _estimate_mamba_hybrid_cache_cost,
    _get_local_mamba_cache_layout,
    _get_mamba_hybrid_pool_size,
    _get_num_cuda_graph_padding_dummy_slots,
    _mamba_snapshot_rule_counts,
    _promote_mamba_state_triton,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaAcceptanceBatch,
    _stack_state_views,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    ATTENTION_DP_DUMMY_REQUEST_ID,
    LlmRequest,
    SamplingConfig,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    KVCacheManager,
    get_pp_layers,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings.executor import KvCacheRetentionConfig
from tensorrt_llm.bindings.internal.batch_manager import LinearCacheType
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
from tensorrt_llm.llmapi.llm_args import (
    BlockReuseConfig,
    CacheTransceiverConfig,
    KvCacheConfig,
    MambaStateConfig,
    MTPDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.llmapi.llm_utils import (
    _resolve_kv_cache_manager_v2_auto,
    _resolve_transceiver_runtime_auto,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    GPU_LEVEL,
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    DataRole,
    GpuCacheTierConfig,
    KVCacheDesc,
    KVCacheManagerConfig,
    LayerId,
    SsmLayerConfig,
    _introspection,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager as RuntimeKVCacheManager
from tensorrt_llm.sampling_params import SamplingParams

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_advance_replay_state_uses_checkpoint_predicate_and_skips_dummies():
    metadata = ReplayStateUpdateMetadata(
        prev_num_accepted_tokens=torch.tensor([11, 12, 13], dtype=torch.int32),
        cache_buf_idx=torch.tensor([0, 1, 1], dtype=torch.int32),
        replay_step_width=5,
        replay_history_size=16,
    )

    _advance_replay_state(
        metadata,
        state_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        accepted_tokens=torch.tensor([2, 3, 4], dtype=torch.int32),
        is_dummy_request=torch.tensor([False, False, True]),
    )

    # Equality does not write a checkpoint; overflow does. Dummy slots do not
    # advance either piece of replay bookkeeping.
    assert metadata.prev_num_accepted_tokens.tolist() == [13, 3, 13]
    assert metadata.cache_buf_idx.tolist() == [0, 0, 1]


def test_cuda_graph_padding_dummy_slot_count_tracks_reachable_draft_lengths():
    assert _get_num_cuda_graph_padding_dummy_slots(None, 64) == 1

    static = MTPDecodingConfig(max_draft_len=4)
    assert _get_num_cuda_graph_padding_dummy_slots(static, 64) == 1

    gated = MTPDecodingConfig(
        max_draft_len=4,
        acceptance_rate_window_size=8,
        acceptance_rate_threshold=0.5,
    )
    assert _get_num_cuda_graph_padding_dummy_slots(gated, 64) == 2

    dynamic = MTPDecodingConfig(
        max_draft_len=4,
        draft_len_schedule={4: 4, 8: 2, 32: 1},
    )
    assert _get_num_cuda_graph_padding_dummy_slots(dynamic, 5) == 2
    assert _get_num_cuda_graph_padding_dummy_slots(dynamic, 32) == 3
    assert _get_num_cuda_graph_padding_dummy_slots(dynamic, 64) == 4

    repeated = MTPDecodingConfig(
        max_draft_len=4,
        draft_len_schedule={4: 4, 8: 4, 32: 2},
    )
    assert _get_num_cuda_graph_padding_dummy_slots(repeated, 64) == 3


def _hybrid_model_config():
    config = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    return SimpleNamespace(
        pretrained_config=config,
        sparse_attention_config=None,
        spec_config=None,
        get_num_mamba_layers=lambda: 1,
        mapping=Mapping(),
    )


def _hybrid_cache_sizing_model_config(layer_types):
    config = SimpleNamespace(
        architectures=["Qwen3_5ForCausalLM"],
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
        linear_key_head_dim=8,
        linear_conv_kernel_dim=4,
        linear_num_value_heads=4,
        linear_num_key_heads=1,
        linear_value_head_dim=8,
        num_key_value_heads=2,
        num_attention_heads=4,
        hidden_size=32,
        torch_dtype=torch.float16,
    )
    return SimpleNamespace(pretrained_config=config, quant_config=None)


def test_mamba_kv_cache_params_separate_target_and_draft_masks():
    model_config = _hybrid_cache_sizing_model_config(
        [
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ]
    )
    params = extract_mamba_kv_cache_params(
        model_config.pretrained_config,
        spec_config=MTPDecodingConfig(max_draft_len=1),
    )

    assert params.mamba_layer_mask == [True, False, True, False]
    assert params.target_full_attention_layer_mask == [False, True, False, True]
    assert params.num_draft_layers == 1

    assert params.get_layer_masks() == (
        [True, False, True, False, False],
        [False, True, False, True, True],
    )
    assert params.get_layer_masks(use_separate_draft_kv_cache=True) == (
        [True, False, True, False],
        [False, True, False, True],
    )
    assert params.get_layer_masks(is_draft=True) == (
        [False, False, False, False, False],
        [False, False, False, False, True],
    )


def test_kimi_kda_cache_params_preserve_qkv_and_fp32_state_geometry() -> None:
    config = SimpleNamespace(
        model_type="kimi_linear",
        num_hidden_layers=4,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
        dtype=torch.bfloat16,
    )

    params = extract_mamba_kv_cache_params(config)

    assert params.state_size == 8
    assert params.conv_kernel == 4
    assert params.num_heads == 4
    assert params.n_groups == 4
    assert params.head_dim == 8
    assert params.mamba_layer_mask == [True, False, True, False]
    assert params.target_full_attention_layer_mask == [
        False,
        True,
        False,
        True,
    ]
    assert params.num_mamba_layers == 2
    assert params.dtype is torch.bfloat16
    assert params.mamba_ssm_cache_dtype is torch.float32


def _kimi_kda_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="kimi_linear",
        num_hidden_layers=4,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
        dtype=torch.bfloat16,
    )


def test_kimi_kda_state_dtype_bf16_is_opt_in() -> None:
    """kv_cache_config.mamba_ssm_cache_dtype=bfloat16 reaches the state pool.

    The bf16 staging paths in the KDA mixer and the fused decode kernel are
    only reachable when the explicit request survives cache-param extraction.
    """
    params = extract_mamba_kv_cache_params(
        _kimi_kda_hf_config(),
        quant_config=SimpleNamespace(mamba_ssm_cache_dtype=torch.bfloat16),
    )

    assert params.mamba_ssm_cache_dtype is torch.bfloat16


def test_kimi_kda_state_dtype_rejects_unsupported_request() -> None:
    with pytest.raises(ValueError, match="float32 .default. or"):
        extract_mamba_kv_cache_params(
            _kimi_kda_hf_config(),
            quant_config=SimpleNamespace(mamba_ssm_cache_dtype=torch.float16),
        )


def _kimi_model_config() -> SimpleNamespace:
    config = SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        vocab_size=163840,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=4,
        kv_lora_rank=32,
        qk_rope_head_dim=8,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
        dtype=torch.bfloat16,
    )
    return SimpleNamespace(
        pretrained_config=config,
        mapping=Mapping(),
        quant_config=None,
        sparse_attention_config=None,
        spec_config=None,
        get_num_mamba_layers=lambda: 2,
    )


def _capture_kimi_v2_manager_ctor(
    monkeypatch: pytest.MonkeyPatch,
    spec_config=None,
    model_engine=None,
    kv_dtype: str = "auto",
) -> tuple[tuple, dict]:
    """Route a Kimi config through _create_kv_cache_manager with an explicit
    V2 manager and capture the constructor arguments."""
    captured: dict[str, object] = {}

    class RecordingV2Manager(KimiK3HybridCacheManager):
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

    model_config = _kimi_model_config()
    tokens_per_block = FP4_MLA_TOKENS_PER_BLOCK if kv_dtype == "nvfp4" else 64
    kv_cache_config = KvCacheConfig(
        dtype=kv_dtype, tokens_per_block=tokens_per_block, use_kv_cache_manager_v2=True
    )
    if kv_dtype == "nvfp4":
        model_config.quant_config = _fp4_kv_quant_config()
        model_config.attn_backend = "TRTLLM"
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)

    assert get_kv_cache_manager_cls(model_config, kv_cache_config) is KimiK3HybridCacheManager

    _create_kv_cache_manager(
        model_engine=model_engine,
        kv_cache_manager_cls=RecordingV2Manager,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        kv_cache_config=kv_cache_config,
        tokens_per_block=tokens_per_block,
        max_seq_len=2048,
        max_batch_size=4,
        spec_config=spec_config,
        sparse_attention_config=None,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=False,
    )
    return captured["args"], captured["kwargs"]


def test_kimi_explicit_v2_manager_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    args, kwargs = _capture_kimi_v2_manager_ctor(monkeypatch)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    assert args[:9] == (
        8,
        4,
        4,
        4,
        8,
        2,
        [True, False, True, False],
        torch.bfloat16,
        torch.float32,
    )
    assert kwargs["num_layers"] == 2
    assert kwargs["layer_mask"] == [False, True, False, True]
    assert kwargs["num_kv_heads"] == 1
    assert kwargs["head_dim"] == 40
    assert kwargs["max_num_tokens"] == 256
    assert kwargs["vocab_size"] == 163840
    assert "kda_replay_num_spec" not in kwargs


def _fp4_kv_quant_config() -> SimpleNamespace:
    return SimpleNamespace(
        quant_mode=SimpleNamespace(has_fp4_kv_cache=lambda: True, has_fp8_kv_cache=lambda: False),
        mamba_ssm_cache_dtype=torch.float32,
    )


def test_kimi_fp4_mla_routes_to_hybrid_v2_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    model_config = _kimi_model_config()
    model_config.quant_config = _fp4_kv_quant_config()
    model_config.attn_backend = "TRTLLM"

    manager_cls = get_kv_cache_manager_cls(
        model_config,
        KvCacheConfig(
            dtype="nvfp4",
            tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
            use_kv_cache_manager_v2=True,
        ),
    )

    # The Kimi K3 hybrid V2 manager owns the KDA state, its replay caches,
    # and the FP4 MLA layout of the MLA layers.
    assert manager_cls is KimiK3HybridCacheManager


def test_kimi_fp4_mla_rejects_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    model_config = _kimi_model_config()
    model_config.quant_config = _fp4_kv_quant_config()
    model_config.attn_backend = "TRTLLM"

    with pytest.raises(ValueError, match="use_kv_cache_manager_v2=True"):
        get_kv_cache_manager_cls(
            model_config,
            KvCacheConfig(
                dtype="nvfp4",
                tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
                use_kv_cache_manager_v2=False,
            ),
        )


def test_kimi_fp4_mla_rejects_block_reuse_with_kv_pool_rebalance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    model_config = _kimi_model_config()
    model_config.quant_config = _fp4_kv_quant_config()
    model_config.attn_backend = "TRTLLM"

    # The FP4 MLA route keeps the hybrid V2 manager's reuse/rebalance check.
    with pytest.raises(ValueError, match="enable_kv_pool_rebalance"):
        get_kv_cache_manager_cls(
            model_config,
            KvCacheConfig(
                dtype="nvfp4",
                tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
                use_kv_cache_manager_v2=True,
                enable_block_reuse=True,
                enable_kv_pool_rebalance=True,
            ),
        )


@pytest.mark.parametrize(
    "runtime,expected",
    [("PYTHON", KimiK3HybridCacheManager), ("CPP", ValueError)],
    ids=["python_nixl", "cpp"],
)
def test_kimi_fp4_mla_disagg_routes_to_hybrid_v2_manager(
    monkeypatch: pytest.MonkeyPatch, runtime: str, expected
) -> None:
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    model_config = _kimi_model_config()
    model_config.quant_config = _fp4_kv_quant_config()
    model_config.attn_backend = "TRTLLM"

    def select():
        return get_kv_cache_manager_cls(
            model_config,
            KvCacheConfig(
                dtype="nvfp4",
                tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
                use_kv_cache_manager_v2=True,
            ),
            is_disagg=True,
            cache_transceiver_config=CacheTransceiverConfig(
                backend="NIXL", transceiver_runtime=runtime
            ),
        )

    # Only the Python NIXL transceiver moves the composed FP4 MLA and KDA
    # roles; the hybrid V2 manager declares both.
    if expected is ValueError:
        with pytest.raises(ValueError, match="transceiver_runtime='PYTHON'"):
            select()
    else:
        assert select() is expected


def test_kimi_fp4_mla_manager_receives_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _, kwargs = _capture_kimi_v2_manager_ctor(monkeypatch)

    # The FP4 MLA layout reads kv_lora_rank and the per-layer KV dtype policy
    # from the model config.
    assert kwargs["pretrained_config"].kv_lora_rank == 32


@pytest.mark.parametrize("kv_dtype", ["auto", "nvfp4"])
def test_kimi_explicit_v2_manager_enables_owned_kda_replay_policy(
    monkeypatch: pytest.MonkeyPatch,
    kv_dtype: str,
) -> None:
    spec_config = MTPDecodingConfig(max_draft_len=3)
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.kimi_kda._kda_kernels.is_kda_mtp_verify_available",
        lambda: True,
    )

    _, kwargs = _capture_kimi_v2_manager_ctor(monkeypatch, spec_config, kv_dtype=kv_dtype)

    if kv_dtype == "nvfp4":
        # The FP4 MLA layout must not cost KDA its fused-verify replay caches.
        assert kwargs["dtype"] == DataType.NVFP4
        assert kwargs["tokens_per_block"] == FP4_MLA_TOKENS_PER_BLOCK
    assert "speculative_state" not in kwargs
    mgr = object.__new__(KimiK3HybridCacheManager)
    mgr.spec_config = spec_config
    mgr._mamba_ssm_stochastic_rounding = False
    mgr._requested_num_spec = None
    mgr._requested_draft_records = False
    mgr._state_layout = SimpleNamespace(spec_config=spec_config, mamba_pp_layers=())
    mgr._speculative_state = mgr._initialize_spec_state()
    policy = mgr._kda_replay
    assert isinstance(policy, KDAReplayState)
    assert policy.num_speculative_tokens == spec_config.tokens_per_gen_step - 1
    assert kwargs["conv_state_layout"] == "q_k_v"
    assert "kda_replay_num_spec" not in kwargs
    assert "kda_draft_records" not in kwargs


@pytest.mark.parametrize("backbone_asks", [True, False])
def test_kimi_explicit_v2_manager_kda_draft_records_follow_the_model(
    monkeypatch: pytest.MonkeyPatch,
    backbone_asks: bool,
) -> None:
    """The factory asks the V2 manager for per-token KDA verify states only when
    the model's backbone sets ``kda_draft_records``."""
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.kimi_kda._kda_kernels.is_kda_mtp_verify_available",
        lambda: True,
    )
    backbone = SimpleNamespace(kda_draft_records=True) if backbone_asks else SimpleNamespace()
    model_engine = SimpleNamespace(
        model=SimpleNamespace(model=backbone),
        max_cuda_graph_batch_size=4,
        input_processor=MagicMock(spec=BaseMultimodalInputProcessor),
    )

    _, kwargs = _capture_kimi_v2_manager_ctor(
        monkeypatch, MTPDecodingConfig(max_draft_len=3), model_engine
    )

    assert "kda_replay_num_spec" not in kwargs
    assert kwargs.get("kda_draft_records", False) is backbone_asks


@pytest.mark.parametrize("max_draft_len, replay_enabled", [(7, True), (8, False)])
def test_kimi_kda_replay_respects_fused_verify_draft_limit(
    monkeypatch: pytest.MonkeyPatch, max_draft_len: int, replay_enabled: bool
) -> None:
    """Draft lengths beyond the fused KDA verify kernel's limit keep the
    legacy intermediate buffers (sequential FLA verify) instead of
    allocating replay caches the kernel would reject."""
    kda_kernels = "tensorrt_llm._torch.modules.kimi_kda._kda_kernels"
    monkeypatch.setattr(f"{kda_kernels}.is_kda_mtp_verify_available", lambda: True)
    monkeypatch.setattr(f"{kda_kernels}.kda_mtp_max_num_spec", lambda: 7)
    spec_config = MTPDecodingConfig(max_draft_len=max_draft_len)

    _, kwargs = _capture_kimi_v2_manager_ctor(monkeypatch, spec_config)

    assert "kda_replay_num_spec" not in kwargs
    mgr = object.__new__(KimiK3HybridCacheManager)
    mgr.spec_config = spec_config
    mgr._mamba_ssm_stochastic_rounding = False
    mgr._requested_num_spec = None
    mgr._requested_draft_records = False
    mgr._state_layout = SimpleNamespace(spec_config=spec_config, mamba_pp_layers=())
    state = mgr._initialize_spec_state()
    assert isinstance(state, KDAReplayState) is replay_enabled
    if replay_enabled:
        assert state.num_speculative_tokens == max_draft_len


def test_kda_mtp_max_num_spec_fits_the_kernel_chunk() -> None:
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda.kda_mtp_decode import T_PAD
    from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import kda_mtp_max_num_spec

    max_num_spec = kda_mtp_max_num_spec()
    # A verify chunk holds up to max_num_spec replayed drafts plus 1 + max_num_spec new tokens.
    assert max_num_spec == 7
    assert 2 * max_num_spec + 1 <= T_PAD


def test_kimi_explicit_v2_manager_uses_qkv_convolution_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TRTLLM-15216: the factory passes KDA conv-state layout explicitly.

    ``model_type`` is not a substitute: the V2 manager rejects it so a caller
    cannot silently retain the default ``x_b_c`` layout.
    """
    _, kwargs = _capture_kimi_v2_manager_ctor(monkeypatch)
    assert kwargs["conv_state_layout"] == "q_k_v"
    assert "model_type" not in kwargs


def test_kimi_v1_manager_receives_model_identity_for_replay_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V1 resolves Kimi replay and its Q/K/V layout inside the constructor."""
    captured: dict[str, object] = {}

    class RecordingV1Manager(CppMambaHybridCacheManager):
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

    model_config = _kimi_model_config()
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.kimi_kda._kda_kernels.is_kda_mtp_verify_available",
        lambda: True,
    )
    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=RecordingV1Manager,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        kv_cache_config=KvCacheConfig(),
        tokens_per_block=64,
        max_seq_len=2048,
        max_batch_size=4,
        spec_config=MTPDecodingConfig(max_draft_len=3),
        sparse_attention_config=None,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=False,
    )
    kwargs = captured["kwargs"]
    assert "vocab_size" not in kwargs
    assert kwargs["model_type"] == "kimi_linear"
    assert "conv_state_layout" not in kwargs
    assert "kda_replay_num_spec" not in kwargs


def test_v2_manager_rejects_model_type_kwarg() -> None:
    """MambaHybridCacheManagerV2 must fail loudly when handed the V1 managers'
    `model_type` instead of `conv_state_layout` — silently absorbing it into
    **kwargs is how the TRTLLM-15216 wrong-layout bug went unnoticed."""
    with pytest.raises(TypeError, match="conv_state_layout"):
        MambaHybridCacheManagerV2(
            16,  # mamba_d_state
            4,  # mamba_d_conv
            8,  # mamba_num_heads
            1,  # mamba_n_groups
            16,  # mamba_head_dim
            2,  # mamba_num_layers
            [True, True],  # mamba_layer_mask
            torch.float16,
            torch.float16,
            KvCacheConfig(),
            CacheTypeCpp.SELF,
            num_layers=0,
            num_kv_heads=1,
            head_dim=16,
            tokens_per_block=32,
            max_seq_len=64,
            max_batch_size=1,
            mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
            model_type="qwen3_next",
        )


@pytest.mark.parametrize(
    ("use_v2", "enable_block_reuse", "expected"),
    [
        (True, False, Qwen35HybridCacheManager),
        (True, True, Qwen35HybridCacheManager),
        (False, True, CppMambaHybridCacheManager),
        ("auto", True, CppMambaHybridCacheManager),
    ],
)
def test_hybrid_cache_manager_factory_honors_v2_setting(
    monkeypatch, use_v2, enable_block_reuse, expected
):
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=enable_block_reuse,
        mamba_state_config=MambaStateConfig(periodic_snapshot_interval=256),
        use_kv_cache_manager_v2=use_v2,
    )

    assert get_kv_cache_manager_cls(_hybrid_model_config(), kv_cache_config) is expected


def test_qwen3_gdn_replay_supports_cpp_and_v2_managers(monkeypatch):
    """GDN replay uses C++ V1 directly and V2 through state descriptors."""
    captured_cpp = {}
    captured_mixed = {}
    captured_v2 = {}

    class RecordingCppManager(CppMambaHybridCacheManager):
        def __init__(self, *args, **kwargs):
            captured_cpp.update(kwargs)

    class RecordingMixedManager(MixedMambaHybridCacheManager):
        def __init__(self, *args, **kwargs):
            captured_mixed.update(kwargs)

    class RecordingV2Manager(MambaHybridCacheManagerV2):
        def __init__(self, *args, **kwargs):
            captured_v2.update(kwargs)

    pretrained_config = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        vocab_size=151936,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    model_config = SimpleNamespace(
        pretrained_config=pretrained_config,
        quant_config=None,
    )
    # A model whose GDN layer verifies drafts with the fused kernels.
    gdn = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
    torch.nn.Module.__init__(gdn)
    gdn.fused_verify = True
    model_engine = SimpleNamespace(
        model=torch.nn.ModuleList([gdn]),
        attn_runtime_features=SimpleNamespace(chunked_prefill=False),
        max_cuda_graph_batch_size=None,
        input_processor=MagicMock(spec=BaseMultimodalInputProcessor),
    )
    mamba_params = MambaKVCacheParams(
        state_size=8,
        conv_kernel=4,
        num_heads=4,
        n_groups=1,
        head_dim=8,
        mamba_layer_mask=[True, False],
        target_full_attention_layer_mask=[False, True],
        num_mamba_layers=1,
        num_draft_layers=1,
        dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.bfloat16,
    )
    monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "1")
    monkeypatch.setattr("tensorrt_llm._torch.modules.fla.cache_manager.get_sm_version", lambda: 90)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor._util.extract_mamba_kv_cache_params",
        lambda *args, **kwargs: mamba_params,
    )
    info_log = MagicMock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor._util.logger.info",
        info_log,
    )

    common_kwargs = dict(
        model_engine=model_engine,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        tokens_per_block=32,
        max_seq_len=2048,
        # Cover the batch size at which upstream selects partitioned replay.
        max_batch_size=16,
        spec_config=MTPDecodingConfig(max_draft_len=3),
        sparse_attention_config=None,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=False,
    )
    _create_kv_cache_manager(
        kv_cache_manager_cls=RecordingCppManager,
        kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=False),
        **common_kwargs,
    )
    _create_kv_cache_manager(
        kv_cache_manager_cls=RecordingMixedManager,
        kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=False),
        **common_kwargs,
    )
    _create_kv_cache_manager(
        kv_cache_manager_cls=RecordingV2Manager,
        kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True),
        **common_kwargs,
    )

    assert captured_cpp["use_replay_state_update"] is None
    assert captured_cpp["gdn_fused_verify"] is True
    assert "gdn_fused_verify" not in captured_mixed
    assert captured_v2["gdn_fused_verify"] is True
    assert captured_cpp["model_type"] == "qwen3_next"
    assert captured_cpp["max_num_tokens"] == 256
    assert "vocab_size" not in captured_cpp
    assert "vocab_size" not in captured_mixed
    assert captured_mixed["use_replay_state_update"] is None
    assert captured_mixed["model_type"] == "qwen3_next"
    assert captured_mixed["max_num_tokens"] == 256
    assert captured_v2["use_replay_state_update"] is None
    assert "speculative_state" not in captured_v2
    mgr = object.__new__(Qwen35HybridCacheManager)
    mgr.spec_config = common_kwargs["spec_config"]
    mgr.ssm_state_dtype = torch.bfloat16
    mgr._requested_replay = None
    mgr.local_num_mamba_layers = 0
    mgr._state_layout = SimpleNamespace(
        spec_config=mgr.spec_config, mamba_pp_layers=(), n_groups_per_rank=1
    )
    mgr._speculative_state = mgr._initialize_spec_state()
    assert isinstance(mgr._speculative_state, GDNReplayState)
    assert captured_v2["max_num_tokens"] == 256
    assert captured_v2["vocab_size"] == pretrained_config.vocab_size
    assert "model_type" not in captured_v2
    assert captured_v2["conv_state_layout"] == "q_k_v"
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.legacy import (
        _resolve_legacy_replay_options,
    )

    for manager_cls, enabled in ((RecordingCppManager, True), (RecordingMixedManager, False)):
        assert _resolve_legacy_replay_options(
            manager_cls,
            "qwen3_next",
            common_kwargs["spec_config"],
            torch.bfloat16,
            False,
            None,
        ) == ("qwen3_next", enabled, None)
    fallback_logs = [str(call.args[0]) for call in info_log.call_args_list]
    assert any("RecordingMixedManager was selected" in log for log in fallback_logs)
    assert not any("RecordingV2Manager was selected" in log for log in fallback_logs)


@pytest.mark.parametrize(
    "manager_cls",
    [CppMambaHybridCacheManager, MixedMambaHybridCacheManager, MambaHybridCacheManagerV2],
)
@pytest.mark.parametrize("nested_vocab_size", [False, True])
def test_nemotron_factory_passes_vocab_size_only_to_v2(monkeypatch, manager_cls, nested_vocab_size):
    captured = {}

    class RecordingManager(manager_cls):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    config = SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        hybrid_override_pattern="M*",
        vocab_size=131072,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
    )
    if nested_vocab_size:
        config.llm_config = SimpleNamespace(vocab_size=config.vocab_size)
        config.vocab_size = None
    mamba_params = MambaKVCacheParams(
        state_size=8,
        conv_kernel=4,
        num_heads=4,
        n_groups=1,
        head_dim=8,
        mamba_layer_mask=[True, False],
        target_full_attention_layer_mask=[False, True],
        num_mamba_layers=1,
        num_draft_layers=0,
        dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.bfloat16,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor._util.extract_mamba_kv_cache_params",
        lambda *args, **kwargs: mamba_params,
    )
    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=RecordingManager,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        kv_cache_config=KvCacheConfig(),
        tokens_per_block=32,
        max_seq_len=2048,
        max_batch_size=4,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=SimpleNamespace(pretrained_config=config, quant_config=None),
        dtype=torch.bfloat16,
        is_draft=False,
    )

    if manager_cls is MambaHybridCacheManagerV2:
        assert captured["vocab_size"] == 131072
    else:
        assert "vocab_size" not in captured


def test_hybrid_cache_manager_factory_rejects_cpp_preference_with_explicit_v2(
    monkeypatch,
):
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.setenv("TLLM_MAMBA_MANAGER_PREFERENCE", "CPP")

    with pytest.raises(ValueError, match="conflicts with explicit"):
        get_kv_cache_manager_cls(
            _hybrid_model_config(),
            KvCacheConfig(
                enable_block_reuse=True,
                mamba_state_config=MambaStateConfig(periodic_snapshot_interval=256),
                use_kv_cache_manager_v2=True,
            ),
        )


def test_hybrid_cache_manager_factory_v2_preference_does_not_select_v2(
    monkeypatch,
):
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.setenv("TLLM_MAMBA_MANAGER_PREFERENCE", "V2")

    assert (
        get_kv_cache_manager_cls(
            _hybrid_model_config(),
            KvCacheConfig(
                mamba_state_config=MambaStateConfig(periodic_snapshot_interval=256),
                use_kv_cache_manager_v2=False,
            ),
        )
        is CppMambaHybridCacheManager
    )


@pytest.mark.parametrize(
    ("field", "offsets"),
    [
        ("additional_snapshot_offsets_from_start", [128]),
        ("additional_snapshot_offsets_from_end", [0]),
    ],
)
def test_hybrid_cache_manager_factory_requires_v2_for_explicit_snapshots(
    monkeypatch,
    field,
    offsets,
):
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        mamba_state_config=MambaStateConfig(**{field: offsets}),
        use_kv_cache_manager_v2="auto",
    )
    llm_args = TorchLlmArgs(
        model="/tmp/dummy_model",
        kv_cache_config=kv_cache_config,
    )

    assert _resolve_kv_cache_manager_v2_auto(llm_args) is False
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2 is False
    with pytest.raises(ValueError, match="use_kv_cache_manager_v2=True"):
        get_kv_cache_manager_cls(
            _hybrid_model_config(),
            llm_args.kv_cache_config,
        )


@pytest.mark.parametrize("backend", ["NIXL", "DEFAULT"])
def test_hybrid_cache_manager_factory_routes_explicit_v2_disagg(monkeypatch, backend):
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    for env_var in (
        "TRTLLM_USE_NIXL_KVCACHE",
        "TRTLLM_USE_UCX_KVCACHE",
        "TRTLLM_USE_MOONCAKE_KVCACHE",
        "TRTLLM_USE_MPI_KVCACHE",
    ):
        monkeypatch.delenv(env_var, raising=False)
    assert (
        get_kv_cache_manager_cls(
            _hybrid_model_config(),
            KvCacheConfig(
                mamba_state_config=MambaStateConfig(periodic_snapshot_interval=256),
                use_kv_cache_manager_v2=True,
            ),
            is_disagg=True,
            cache_transceiver_config=CacheTransceiverConfig(
                backend=backend, transceiver_runtime="PYTHON"
            ),
        )
        is Qwen35HybridCacheManager
    )


def test_hybrid_cache_manager_factory_rejects_python_v1_disagg_reuse(monkeypatch):
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)

    with pytest.raises(ValueError, match="requires use_kv_cache_manager_v2=True"):
        get_kv_cache_manager_cls(
            _hybrid_model_config(),
            KvCacheConfig(
                enable_block_reuse=True,
                mamba_state_config=MambaStateConfig(periodic_snapshot_interval=256),
                use_kv_cache_manager_v2=False,
            ),
            is_disagg=True,
            cache_transceiver_config=CacheTransceiverConfig(
                backend="NIXL",
                transceiver_runtime="PYTHON",
            ),
        )


@pytest.mark.parametrize(
    ("backend", "runtime", "backend_env"),
    [
        ("DEFAULT", "PYTHON", "TRTLLM_USE_UCX_KVCACHE"),
        ("UCX", "PYTHON", None),
        ("NIXL", "auto", None),
        ("NIXL", None, None),
        ("NIXL", "CPP", None),
        ("UCX", None, None),
    ],
)
def test_hybrid_cache_manager_factory_rejects_unsupported_v2_disagg_route(
    monkeypatch, backend, runtime, backend_env
):
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    for env_var in (
        "TRTLLM_USE_NIXL_KVCACHE",
        "TRTLLM_USE_UCX_KVCACHE",
        "TRTLLM_USE_MOONCAKE_KVCACHE",
        "TRTLLM_USE_MPI_KVCACHE",
    ):
        monkeypatch.delenv(env_var, raising=False)
    if backend_env is not None:
        monkeypatch.setenv(backend_env, "1")

    with pytest.raises(ValueError, match="requires transceiver_runtime='PYTHON'"):
        get_kv_cache_manager_cls(
            _hybrid_model_config(),
            KvCacheConfig(
                mamba_state_config=MambaStateConfig(periodic_snapshot_interval=256),
                use_kv_cache_manager_v2=True,
            ),
            is_disagg=True,
            cache_transceiver_config=CacheTransceiverConfig(
                backend=backend, transceiver_runtime=runtime
            ),
        )


@pytest.mark.parametrize(
    ("env_name", "env_value", "expected_error"),
    [
        (
            "TLLM_MAMBA_MANAGER_PREFERENCE",
            "MIXED",
            "does not support block reuse",
        ),
        (
            "TRTLLM_USE_PY_MAMBA",
            "1",
            "does not support block reuse",
        ),
    ],
)
def test_hybrid_cache_manager_factory_rejects_mixed_override_with_reuse(
    monkeypatch, env_name, env_value, expected_error
):
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(ValueError, match=expected_error):
        get_kv_cache_manager_cls(
            _hybrid_model_config(),
            KvCacheConfig(
                enable_block_reuse=True,
                mamba_state_config=MambaStateConfig(periodic_snapshot_interval=256),
                use_kv_cache_manager_v2=False,
            ),
        )


@pytest.mark.parametrize("use_v2", [False, "auto"])
def test_hybrid_cache_manager_factory_keeps_v1_disagg_route(monkeypatch, use_v2):
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)

    assert (
        get_kv_cache_manager_cls(
            _hybrid_model_config(),
            KvCacheConfig(
                enable_block_reuse=False,
                use_kv_cache_manager_v2=use_v2,
            ),
            is_disagg=True,
        )
        is CppMambaHybridCacheManager
    )
    assert (
        get_kv_cache_manager_cls(
            _hybrid_model_config(),
            KvCacheConfig(
                enable_block_reuse=False,
                use_kv_cache_manager_v2=use_v2,
            ),
            is_disagg=True,
            cache_transceiver_config=CacheTransceiverConfig(
                backend="NIXL", transceiver_runtime="PYTHON"
            ),
        )
        is MixedMambaHybridCacheManager
    )


def test_hybrid_models_prefer_v2_and_python_transceiver(monkeypatch):
    from tensorrt_llm._torch.models.modeling_nemotron_h import NemotronHForCausalLM
    from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import NemotronHMultimodalModel
    from tensorrt_llm._torch.models.modeling_qwen3_5 import Qwen3_5VLModel
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextForCausalLM

    for env_var in (
        "TRTLLM_USE_NIXL_KVCACHE",
        "TRTLLM_USE_UCX_KVCACHE",
        "TRTLLM_USE_MOONCAKE_KVCACHE",
        "TRTLLM_USE_MPI_KVCACHE",
    ):
        monkeypatch.delenv(env_var, raising=False)

    for model_cls in (
        NemotronHForCausalLM,
        Qwen3NextForCausalLM,
        Qwen3_5VLModel,
        NemotronHMultimodalModel,
    ):
        llm_args = TorchLlmArgs(
            model="/tmp/dummy_model",
            kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2="auto"),
            cache_transceiver_config=CacheTransceiverConfig(
                backend="DEFAULT", transceiver_runtime="auto"
            ),
        )
        _resolve_transceiver_runtime_auto(llm_args, model_cls)
        _resolve_kv_cache_manager_v2_auto(llm_args, model_cls)
        assert llm_args.kv_cache_config.use_kv_cache_manager_v2 is True
        assert llm_args.cache_transceiver_config.transceiver_runtime == "PYTHON"


@pytest.mark.parametrize(
    ("replay_env", "manager_setting", "expected_v2"),
    [
        (None, "auto", True),
        ("1", "auto", True),
        ("0", "auto", True),
        (None, True, True),
        (None, False, False),
    ],
)
def test_qwen3_gdn_replay_uses_v2_preference(
    monkeypatch,
    replay_env,
    manager_setting,
    expected_v2,
):
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextForCausalLM

    if replay_env is None:
        monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", replay_env)

    llm_args = TorchLlmArgs(
        model="/tmp/dummy_model",
        kv_cache_config=KvCacheConfig(
            use_kv_cache_manager_v2=manager_setting,
        ),
        speculative_config=MTPDecodingConfig(max_draft_len=3),
    )
    _resolve_kv_cache_manager_v2_auto(
        llm_args,
        Qwen3NextForCausalLM,
    )

    assert llm_args.kv_cache_config.use_kv_cache_manager_v2 is expected_v2
    expected_manager = Qwen35HybridCacheManager if expected_v2 else CppMambaHybridCacheManager
    assert (
        get_kv_cache_manager_cls(
            _hybrid_model_config(),
            llm_args.kv_cache_config,
        )
        is expected_manager
    )


def test_kimi_without_v2_preference_uses_mixed_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kimi K3 uses separate KV and recurrent-state pools for SA decoding."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM

    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)

    llm_args = TorchLlmArgs(
        model="/tmp/dummy_model",
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            tokens_per_block=64,
            use_kv_cache_manager_v2="auto",
        ),
    )
    resolved = _resolve_kv_cache_manager_v2_auto(llm_args, KimiLinearForCausalLM)

    assert resolved is False
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2 is False
    assert llm_args.kv_cache_config.enable_block_reuse is False
    assert llm_args.kv_cache_config.tokens_per_block == 64
    assert (
        get_kv_cache_manager_cls(_kimi_model_config(), llm_args.kv_cache_config)
        is MixedMambaHybridCacheManager
    )


def test_kimi_preferred_transceiver_runtime() -> None:
    """K3 must resolve transceiver_runtime='auto' to the Python transceiver:
    only KvCacheTransceiverV2 can move the KDA recurrent state."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM

    assert KimiLinearForCausalLM.get_preferred_transceiver_runtime() == "PYTHON"


@pytest.mark.parametrize(
    "cache_transceiver_config",
    [
        None,
        CacheTransceiverConfig(backend="NIXL", transceiver_runtime="auto"),
        CacheTransceiverConfig(backend="NIXL", transceiver_runtime="CPP"),
        CacheTransceiverConfig(backend="UCX", transceiver_runtime="CPP"),
        CacheTransceiverConfig(backend="UCX", transceiver_runtime="PYTHON"),
    ],
    ids=["no_config", "auto_unresolved", "explicit_cpp", "ucx_cpp", "ucx_python"],
)
def test_kimi_disagg_rejects_non_python_transceiver_route(
    monkeypatch: pytest.MonkeyPatch, cache_transceiver_config
) -> None:
    """Any K3 disagg route that does not reach the Python NIXL transceiver
    must fail loudly instead of returning a manager the C++ transceiver
    would drive without KDA state transfer (silent wrong results). The
    'auto_unresolved' case covers paths that skip model-default resolution,
    where 'auto' falls back to the C++ runtime."""
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)

    with pytest.raises(ValueError, match="Kimi K3 disaggregated serving requires"):
        get_kv_cache_manager_cls(
            _kimi_model_config(),
            KvCacheConfig(enable_block_reuse=False),
            is_disagg=True,
            cache_transceiver_config=cache_transceiver_config,
        )


def test_kimi_disagg_defaults_use_v2_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kimi's default V2 manager supports the resolved Python NIXL route."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM

    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    llm_args = TorchLlmArgs(
        model="/tmp/dummy_model",
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, tokens_per_block=64),
        cache_transceiver_config=CacheTransceiverConfig(backend="NIXL", transceiver_runtime="auto"),
    )
    _resolve_transceiver_runtime_auto(llm_args, KimiLinearForCausalLM)
    _resolve_kv_cache_manager_v2_auto(llm_args, KimiLinearForCausalLM)

    assert llm_args.kv_cache_config.use_kv_cache_manager_v2 is True
    assert llm_args.cache_transceiver_config.transceiver_runtime == "PYTHON"
    assert (
        get_kv_cache_manager_cls(
            _kimi_model_config(),
            llm_args.kv_cache_config,
            is_disagg=True,
            cache_transceiver_config=llm_args.cache_transceiver_config,
        )
        is KimiK3HybridCacheManager
    )


def test_kimi_disagg_explicit_v1_python_nixl_routes_to_mixed_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)

    assert (
        get_kv_cache_manager_cls(
            _kimi_model_config(),
            KvCacheConfig(
                enable_block_reuse=False,
                use_kv_cache_manager_v2=False,
            ),
            is_disagg=True,
            cache_transceiver_config=CacheTransceiverConfig(
                backend="NIXL", transceiver_runtime="PYTHON"
            ),
        )
        is MixedMambaHybridCacheManager
    )


def test_v2_disagg_slice_skips_state_index_on_mamba_free_pp_rank():
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.local_num_mamba_layers = 0
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._kv_cache_manager = manager
    transceiver._reuse_adapter = SimpleNamespace(tokens_per_block=32)
    transceiver._page_table = SimpleNamespace(layer_groups=[])
    request = SimpleNamespace(
        is_generation_only_request=False,
        prompt_len=0,
        py_request_id=123,
    )

    chunk = transceiver._create_chunk(request)

    # No mamba layer group → no STATE entries in block_ids
    assert all(ids.size == 0 for ids in chunk.block_ids_per_layer_groups)


def test_v2_disagg_gen_init_with_local_mamba_layers_reports_no_local_cached_tokens():
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.local_num_mamba_layers = 1
    # The incoming recurrent state replaces the whole local slot, so nothing survives as a hit.
    assert manager._disagg_transfer_overwrites_whole_cached_prefix()


def test_v2_disagg_gen_init_without_local_mamba_layers_keeps_complete_blocks():
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.local_num_mamba_layers = 0
    assert not manager._disagg_transfer_overwrites_whole_cached_prefix()


def test_v2_disagg_slice_reads_state_index_without_refreshing_batch_mask():
    from tensorrt_llm._torch.disaggregation.resource.page import CacheKind

    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.local_num_mamba_layers = 1
    manager._request_id_to_state_index = {123: 7}
    manager.get_state_indices = MagicMock(
        side_effect=AssertionError("state-index lookup must not refresh the dummy mask")
    )
    # Provide a mamba layer group so _create_chunk places the slot ID
    mamba_lg = SimpleNamespace(kind=CacheKind.STATE)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._kv_cache_manager = manager
    transceiver._reuse_adapter = SimpleNamespace(tokens_per_block=32)
    transceiver._page_table = SimpleNamespace(layer_groups=[mamba_lg])
    request = SimpleNamespace(
        is_generation_only_request=False,
        prompt_len=0,
        py_request_id=123,
    )

    chunk = transceiver._create_chunk(request)

    # Slot index 7 should be in the STATE group's block_ids
    assert chunk.block_ids_per_layer_groups[0][0] == 7
    manager.get_state_indices.assert_not_called()


@pytest.mark.parametrize(
    "max_beam_width, has_connector, expected",
    [
        (2, False, "max_beam_width > 1"),
        # A KV connector alone no longer forces a fallback: it is supported
        # through the pool-layout registration path, so the manager is returned
        # unchanged and nothing is raised.
        (1, True, None),
        # With beam search still incompatible, the connector must not appear in
        # the reason list -- it is not what makes this configuration unsupported.
        (2, True, "max_beam_width > 1"),
    ],
)
def test_v2_hybrid_incompatibility_fails_without_cpp_fallback(
    max_beam_width, has_connector, expected
):
    config = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    model_config = SimpleNamespace(
        pretrained_config=config,
        sparse_attention_config=None,
    )
    creator = object.__new__(KvCacheCreator)
    creator._kv_connector_manager = object() if has_connector else None
    creator._max_beam_width = max_beam_width
    creator._is_disagg = False
    creator._llm_args = SimpleNamespace(sleep_config=None)

    if expected is None:
        assert (
            creator._validate_or_fallback_kv_cache_manager_v2(
                MambaHybridCacheManagerV2, model_config, KvCacheConfig()
            )
            is MambaHybridCacheManagerV2
        )
        return

    with pytest.raises(NotImplementedError, match=expected):
        creator._validate_or_fallback_kv_cache_manager_v2(
            MambaHybridCacheManagerV2, model_config, KvCacheConfig()
        )


def _make_mgr(
    max_batch_size=4, max_draft_len=2, enable_attention_dp=False, use_replay_state_update=False
):
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, enable_attention_dp=enable_attention_dp)
    pool = _get_mamba_hybrid_pool_size(max_batch_size, mapping)
    return PythonMambaCacheManager(
        d_state=8,
        d_conv=4,
        num_heads=4,
        n_groups=1,
        head_dim=8,
        num_layers=2,
        max_batch_size=pool,
        spec_state_size=max_batch_size,
        mapping=mapping,
        dtype=torch.float16,
        ssm_cache_dtype=torch.float16,
        speculative_num_draft_tokens=max_draft_len,
        use_replay_state_update=use_replay_state_update,
    )


@skip_no_cuda
@pytest.mark.parametrize("enable_attention_dp", [False, True])
def test_python_mamba_resource_count_excludes_reserved_dummy_slots(enable_attention_dp):
    max_batch_size = 4
    mgr = _make_mgr(
        max_batch_size=max_batch_size,
        max_draft_len=2,
        enable_attention_dp=enable_attention_dp,
    )

    assert mgr.get_max_resource_count() == max_batch_size
    assert len(mgr.mamba_cache_free_blocks) == max_batch_size


@skip_no_cuda
def test_replay_inactive_without_spec_config():
    mgr = _make_mgr(
        max_batch_size=2,
        max_draft_len=None,
        use_replay_state_update=True,
    )

    assert mgr.use_replay_state_update is False
    assert mgr.get_replay_state_update_metadata() is None


@contextlib.contextmanager
def _no_host_sync():
    """Fail on any host-device synchronization inside the block."""
    torch.cuda.synchronize()
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)


@skip_no_cuda
def test_python_mamba_slot_assignment_without_sync():
    """Requests and dummies joining the v1 manager reset their slots' replay state and
    seeds without synchronizing the stream (the previous decode step is still queued)."""
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1)
    mgr = PythonMambaCacheManager(
        d_state=8,
        d_conv=4,
        num_heads=4,
        n_groups=1,
        head_dim=8,
        num_layers=2,
        max_batch_size=_get_mamba_hybrid_pool_size(4, mapping),
        spec_state_size=4,
        mapping=mapping,
        dtype=torch.float16,
        ssm_cache_dtype=torch.float16,
        speculative_num_draft_tokens=2,
        use_replay_state_update=True,
        mamba_ssm_stochastic_rounding=True,
    )
    assert mgr.use_replay_state_update
    cache = mgr.mamba_cache
    cache.prev_num_accepted_tokens.fill_(5)
    cache.cache_buf_idx.fill_(1)
    seeds = mgr.get_mamba_ssm_rand_seed()
    seeds_before = seeds.clone()

    with _no_host_sync():
        mgr._prepare_mamba_cache_blocks([100, 101])
        # 100 is known: its slot is reset in place; 200 takes a new slot.
        mgr.add_dummy_requests([100, 200])

    slots = [mgr.mamba_cache_index[r] for r in (100, 101, 200)]
    for slot in slots:
        assert int(cache.prev_num_accepted_tokens[slot]) == 0
        assert int(cache.cache_buf_idx[slot]) == 0
    # Fresh assignments rotate their seeds; the dummy never reaches the seed rotation.
    for slot in slots[:2]:
        assert int(seeds[slot]) != int(seeds_before[slot])


@skip_no_cuda
def test_python_mamba_kda_replay_seed_without_sync():
    """The v1 manager's disagg replay seed stages its slots and seeds the replay caches
    without synchronizing the stream."""
    num_layers, num_slots, dim, committed, num_spec = 2, 4, 6, 3, 2

    class _SpecState:
        pass

    cache = _SpecState()
    cache.conv = torch.randn(num_layers, num_slots, 3 * dim, committed, device="cuda")
    for name in ("kda_conv_q", "kda_conv_k", "kda_conv_v"):
        setattr(
            cache,
            name,
            torch.full((num_layers, num_slots, dim, committed + num_spec), 7.0, device="cuda"),
        )
    for name in ("kda_k_cache", "kda_g_cache", "kda_v_cache"):
        setattr(cache, name, torch.full((num_layers, num_slots, num_spec, dim), 7.0, device="cuda"))
    cache.kda_beta_cache = torch.full((num_layers, num_slots, num_spec, 3), 7.0, device="cuda")
    cache.prev_num_accepted_tokens = torch.full((num_slots,), 5, dtype=torch.int32, device="cuda")
    mgr = object.__new__(PythonMambaCacheManager)
    mgr._use_kda_replay_update = True
    mgr.SpeculativeState = _SpecState
    mgr.mamba_cache = cache
    mgr.mamba_cache_index = {101: 1, 202: 3}

    with _no_host_sync():
        mgr.seed_kda_replay_caches_for_disagg_gen([101, 202])

    assert cache.prev_num_accepted_tokens.tolist() == [5, 0, 5, 0]
    for slot in (1, 3):
        assert torch.equal(cache.kda_conv_q[:, slot, :, :committed], cache.conv[:, slot, :dim])
        assert not cache.kda_conv_q[:, slot, :, committed:].any()
        assert not cache.kda_k_cache[:, slot].any()
    assert (cache.kda_conv_q[:, 0] == 7.0).all()


@skip_no_cuda
def test_v2_kda_replay_commit_clears_draft_count_without_sync():
    """Committing a suspended request's pending drafts clears its draft count without
    synchronizing the stream."""
    from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import is_kda_mtp_verify_available

    if not is_kda_mtp_verify_available():
        pytest.skip("needs the fused KDA verify kernel (sm_100 / sm_103 with the CuTe DSL)")
    mgr = _v2_kda_replay_manager_with_large_slots()
    request = None
    try:
        (request,) = mgr.add_dummy_requests(
            [1], token_nums=[8], is_gen=True, max_num_draft_tokens=2
        )
        slot = mgr._request_id_to_state_index[1]
        replay = mgr._kda_replay
        # The first commit compiles its kernel.
        replay.prev_num_accepted_tokens[slot] = 1
        replay.commit_pending_drafts(slot)
        replay.prev_num_accepted_tokens[slot] = 2
        with _no_host_sync():
            replay.commit_pending_drafts(slot)
        assert int(replay.prev_num_accepted_tokens[slot]) == 0
    finally:
        if request is not None:
            mgr.free_resources(request)
        mgr.shutdown()


@skip_no_cuda
def test_padding_slot_not_held_by_parked_real():
    """Padding must not resolve to a slot owned by a parked real
    request outside the current batch."""
    mgr = _make_mgr(max_batch_size=4, max_draft_len=2)
    mgr._prepare_mamba_cache_blocks([100, 101, 102, 103])
    mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])
    # 102 and 103 are parked outside the current batch.
    request_ids = [100, 101, CUDA_GRAPH_DUMMY_REQUEST_ID]
    indices = mgr.get_state_indices(request_ids, [False, False, True])
    real_slots = {mgr.mamba_cache_index[r] for r in [100, 101, 102, 103]}
    assert indices[2] not in real_slots
    assert indices[2] == mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID]


@skip_no_cuda
def test_padding_survives_overlap_scheduler_pressure():
    """Under the overlap scheduler, prior-iter completions linger in
    mamba_cache_index, so N padding entries must not need N free
    slots."""
    mgr = _make_mgr(max_batch_size=4, max_draft_len=0)
    mgr._prepare_mamba_cache_blocks([100, 101, 102, 103])
    mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])
    # 1 real + 3 padding (attention-dp padded_batch_size=4 on this rank).
    request_ids = [100] + [CUDA_GRAPH_DUMMY_REQUEST_ID] * 3
    is_padding = [False] + [True] * 3
    indices = mgr.get_state_indices(request_ids, is_padding)
    dummy_slot = mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID]
    assert indices[0] == mgr.mamba_cache_index[100]
    assert indices[1:] == [dummy_slot] * 3


@skip_no_cuda
def test_all_draft_len_sentinels_share_one_slot():
    """All per-draft-len sentinels must collapse to a single slot, so
    the pool needs only +1 headroom regardless of max_draft_len."""
    max_batch_size, max_draft_len = 4, 3
    mgr = _make_mgr(max_batch_size=max_batch_size, max_draft_len=max_draft_len)
    mgr._prepare_mamba_cache_blocks([100, 101, 102, 103])

    sentinels = [CUDA_GRAPH_DUMMY_REQUEST_ID - k for k in range(max_draft_len + 1)]
    mgr.add_dummy_requests(sentinels)

    shared = mgr.mamba_cache_index[sentinels[0]]
    real_slots = {mgr.mamba_cache_index[r] for r in [100, 101, 102, 103]}
    assert shared not in real_slots
    for s in sentinels:
        assert mgr.mamba_cache_index[s] == shared
    assert mgr.mamba_cache_free_blocks == []


@skip_no_cuda
def test_padding_slot_is_permanent():
    """free_resources drops a sentinel's index entry but the shared
    slot stays reserved for the next batch."""
    mgr = _make_mgr(max_batch_size=4, max_draft_len=2)
    sentinels = [CUDA_GRAPH_DUMMY_REQUEST_ID - k for k in range(3)]
    mgr.add_dummy_requests(sentinels)
    shared = mgr.mamba_cache_index[sentinels[0]]

    def _fake(rid):
        return SimpleNamespace(py_request_id=rid)

    for s in sentinels:
        mgr.free_resources(_fake(s))
        assert s not in mgr.mamba_cache_index
        assert shared not in mgr.mamba_cache_free_blocks

    assert mgr._padding_slot == shared


@skip_no_cuda
def test_replay_update_mamba_states_uses_history_window():
    """Replay path accumulates PNAT until layer kernels write a checkpoint."""
    mgr = _make_mgr(max_batch_size=4, max_draft_len=5, use_replay_state_update=True)
    assert mgr.replay_step_width == 6
    assert mgr.replay_history_size == MIN_REPLAY_HISTORY_SIZE
    assert mgr.mamba_cache.prev_num_accepted_tokens.dtype == torch.int32
    assert mgr.mamba_cache.cache_buf_idx.dtype == torch.int32
    assert mgr.mamba_cache.old_x.shape[3] == MIN_REPLAY_HISTORY_SIZE
    assert mgr.mamba_cache.old_B.shape[3] == MIN_REPLAY_HISTORY_SIZE
    assert mgr.mamba_cache.old_dt.shape[4] == MIN_REPLAY_HISTORY_SIZE
    assert mgr.mamba_cache.old_dA_cumsum.shape[4] == MIN_REPLAY_HISTORY_SIZE

    mgr._prepare_mamba_cache_blocks([100, 101])
    slot_appended = mgr.mamba_cache_index[100]
    slot_checkpointed = mgr.mamba_cache_index[101]

    mgr.mamba_cache.prev_num_accepted_tokens[slot_appended] = 7
    mgr.mamba_cache.prev_num_accepted_tokens[slot_checkpointed] = 13
    mgr.mamba_cache.cache_buf_idx[slot_appended] = 0
    mgr.mamba_cache.cache_buf_idx[slot_checkpointed] = 1
    mgr.mamba_cache.conv.zero_()
    mgr.mamba_cache.intermediate_conv_window.zero_()
    mgr.mamba_cache.intermediate_conv_window[:, 0, 2] = 11.0
    mgr.mamba_cache.intermediate_conv_window[:, 1, 2] = 13.0

    state_indices = torch.tensor(
        [slot_appended, slot_checkpointed], dtype=torch.int32, device="cuda"
    )
    attn = SimpleNamespace(num_seqs=2, num_contexts=0)
    mgr.update_mamba_states(
        attn,
        torch.tensor([3, 3], dtype=torch.int32, device="cuda"),
        state_indices=state_indices,
    )

    assert mgr.mamba_cache.prev_num_accepted_tokens[slot_appended].item() == 10
    assert mgr.mamba_cache.prev_num_accepted_tokens[slot_checkpointed].item() == 3
    assert mgr.mamba_cache.cache_buf_idx[slot_appended].item() == 0
    assert mgr.mamba_cache.cache_buf_idx[slot_checkpointed].item() == 0
    assert torch.all(mgr.mamba_cache.conv[:, slot_appended] == 11.0)
    assert torch.all(mgr.mamba_cache.conv[:, slot_checkpointed] == 13.0)


@skip_no_cuda
def test_replay_update_mamba_states_skips_dummy_slots():
    mgr = _make_mgr(max_batch_size=2, max_draft_len=5, use_replay_state_update=True)
    mgr._prepare_mamba_cache_blocks([100])
    mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])

    real_slot = mgr.mamba_cache_index[100]
    dummy_slot = mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID]
    mgr.mamba_cache.prev_num_accepted_tokens[real_slot] = 13
    mgr.mamba_cache.prev_num_accepted_tokens[dummy_slot] = 13
    mgr.mamba_cache.cache_buf_idx[real_slot] = 1
    mgr.mamba_cache.cache_buf_idx[dummy_slot] = 1

    state_indices = torch.tensor(
        mgr.get_state_indices([100, CUDA_GRAPH_DUMMY_REQUEST_ID], [False, True]),
        dtype=torch.int32,
        device="cuda",
    )
    attn = SimpleNamespace(num_seqs=2, num_contexts=0)
    mgr.update_mamba_states(
        attn,
        torch.tensor([3, 3], dtype=torch.int32, device="cuda"),
        state_indices=state_indices,
    )

    assert mgr.mamba_cache.prev_num_accepted_tokens[real_slot].item() == 3
    assert mgr.mamba_cache.prev_num_accepted_tokens[dummy_slot].item() == 13
    assert mgr.mamba_cache.cache_buf_idx[real_slot].item() == 0
    assert mgr.mamba_cache.cache_buf_idx[dummy_slot].item() == 1


@skip_no_cuda
@pytest.mark.parametrize(
    (
        "position_source_values",
        "position_source_is_token_count",
        "expected_positions",
    ),
    [
        ([3, 2], True, [2, 1]),
        ([1, 3], False, [1, 3]),
    ],
)
def test_promote_mamba_state_fuses_replay_bookkeeping(
    position_source_values,
    position_source_is_token_count,
    expected_positions,
):
    intermediate = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32, device="cuda").reshape(
        2, 2, 4, 3
    )
    dst = torch.zeros(2, 5, 3, dtype=torch.float32, device="cuda")
    src_state_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    accepted_position_source = torch.tensor(
        position_source_values, dtype=torch.int32, device="cuda"
    )
    num_accepted_tokens = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
    dst_state_indices = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    replay_pnat = torch.tensor([0, 13, 0, 7, 0], dtype=torch.int32, device="cuda")
    replay_cache_buf_idx = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32, device="cuda")
    dummy_request_mask = torch.tensor([False, True], dtype=torch.bool, device="cuda")

    _promote_mamba_state_triton(
        dst,
        intermediate,
        src_state_indices,
        accepted_position_source,
        dst_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        position_source_is_token_count=position_source_is_token_count,
        replay_pnat=replay_pnat,
        replay_cache_buf_idx=replay_cache_buf_idx,
        dummy_request_mask=dummy_request_mask,
        replay_step_width=4,
        replay_history_size=MIN_REPLAY_HISTORY_SIZE,
    )

    torch.testing.assert_close(dst[:, 1], intermediate[:, 0, expected_positions[0]])
    torch.testing.assert_close(dst[:, 3], intermediate[:, 1, expected_positions[1]])
    assert replay_pnat.tolist() == [0, 3, 0, 7, 0]
    assert replay_cache_buf_idx.tolist() == [0, 0, 0, 1, 0]


def test_cpp_hybrid_replay_bookkeeping_is_fused_into_conv_promotion(
    monkeypatch,
):
    mgr = object.__new__(CppMambaHybridCacheManager)
    mgr.local_num_mamba_layers = 1
    mgr._use_replay_state_update = True
    mgr._dummy_request_mask_buffer = DeviceCopyBuffer(torch.tensor([False, True]))
    mgr.prev_num_accepted_tokens = torch.tensor([13, 7], dtype=torch.int32)
    mgr.cache_buf_idx = torch.tensor([1, 0], dtype=torch.int32)
    mgr.replay_step_width = 4
    mgr.replay_history_size = MIN_REPLAY_HISTORY_SIZE
    mgr.intermediate_state_indices = torch.arange(2, dtype=torch.int32)
    mgr.all_conv_states = torch.empty(0)
    mgr.intermediate_conv_states = torch.empty(0)
    mgr._commit_gdn_cached_replay_history_layers = MagicMock()

    promote_calls = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.legacy._promote_mamba_state_triton",
        lambda *args, **kwargs: promote_calls.append((args, kwargs)),
    )

    mgr._generation_state_indices = torch.arange(64, dtype=torch.int32)
    if not hasattr(mgr, "_dummy_request_mask"):
        mgr._dummy_request_mask_buffer = DeviceCopyBuffer(torch.zeros(64, dtype=torch.bool))
    mgr.update_mamba_states(
        SimpleNamespace(num_seqs=2, num_contexts=0),
        torch.tensor([3, 2], dtype=torch.int32),
        state_indices=torch.tensor([0, 1], dtype=torch.int32),
    )

    mgr._commit_gdn_cached_replay_history_layers.assert_called_once()
    assert len(promote_calls) == 1
    _, kwargs = promote_calls[0]
    assert kwargs["replay_pnat"] is mgr.prev_num_accepted_tokens
    assert kwargs["replay_cache_buf_idx"] is mgr.cache_buf_idx
    assert kwargs["dummy_request_mask"].tolist() == [False, True]
    assert kwargs["replay_step_width"] == 4
    assert kwargs["replay_history_size"] == MIN_REPLAY_HISTORY_SIZE


@skip_no_cuda
def test_v2_hybrid_fused_replay_promotion_matches_unfused_path():
    """The one-launch promotion leaves the conv states and replay bookkeeping of the
    per-step reference (_advance_replay_state, then per-layer promotion)."""
    from tensorrt_llm._torch.modules.fla import cache_manager as fla_cache_manager

    num_layers, batch_size = 3, 4
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=batch_size,
        num_mamba_layers=num_layers,
        spec_config=MTPDecodingConfig(max_draft_len=3),
        speculative_state=GDNReplayState(4),
        conv_state_layout="q_k_v",
    )
    try:
        state = mgr._speculative_state
        assert isinstance(state, GDNReplayState)
        assert state._stacked_conv is not None
        torch.manual_seed(0)
        state.intermediate_conv.normal_()
        for conv in mgr.all_conv_states:
            conv.normal_()
        slots = torch.tensor([1, 3, 0, 2], dtype=torch.int32, device="cuda")
        state.prev_num_accepted_tokens.copy_(
            torch.randint(0, state.replay_history_size, state.prev_num_accepted_tokens.shape)
        )
        state.cache_buf_idx.copy_(torch.randint(0, 2, state.cache_buf_idx.shape))
        mgr._dummy_request_mask[:batch_size] = torch.tensor([False, False, True, False])
        num_accepted = torch.tensor([1, 4, 2, 3], dtype=torch.int32, device="cuda")

        ref_conv = [conv.clone() for conv in mgr.all_conv_states]
        ref_meta = fla_cache_manager.ReplayStateUpdateMetadata(
            state.prev_num_accepted_tokens.clone(),
            state.cache_buf_idx.clone(),
            state.replay_step_width,
            state.replay_history_size,
        )
        fla_cache_manager._advance_replay_state(
            ref_meta, slots, num_accepted, mgr._dummy_request_mask[:batch_size].clone()
        )
        for layer, dst in enumerate(ref_conv):
            for g in range(batch_size):
                dst[slots[g]] = state.intermediate_conv[layer, g, num_accepted[g] - 1]

        state.commit_all_layers = MagicMock()
        mgr.update_mamba_states(
            SimpleNamespace(num_seqs=batch_size, num_contexts=0),
            num_accepted,
            state_indices=slots,
        )
        torch.cuda.synchronize()
        for conv, ref in zip(mgr.all_conv_states, ref_conv):
            assert torch.equal(conv, ref)
        assert torch.equal(state.prev_num_accepted_tokens, ref_meta.prev_num_accepted_tokens)
        assert torch.equal(state.cache_buf_idx, ref_meta.cache_buf_idx)
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_attention_dp_dummy_has_reserved_slot_with_batch_size_one():
    mgr = _make_mgr(max_batch_size=1, max_draft_len=0, enable_attention_dp=True)
    mgr._prepare_mamba_cache_blocks([100])

    mgr.add_dummy_requests([ATTENTION_DP_DUMMY_REQUEST_ID])

    assert mgr.mamba_cache_free_blocks == []
    assert mgr.mamba_cache_index[100] != mgr._attention_dp_dummy_slot
    assert mgr.mamba_cache_index[ATTENTION_DP_DUMMY_REQUEST_ID] == mgr._attention_dp_dummy_slot

    mgr.free_resources(SimpleNamespace(py_request_id=ATTENTION_DP_DUMMY_REQUEST_ID))
    assert mgr._attention_dp_dummy_slot not in mgr.mamba_cache_free_blocks


@skip_no_cuda
def test_update_mamba_states_mtp_path():
    """MTP forward path: update_mamba_states must scatter using the
    caller-supplied Mamba2Metadata-style state_indices tensor (partition
    order, padded to the captured batch size)."""
    mgr = _make_mgr()
    mgr._prepare_mamba_cache_blocks([100, 101, 102])

    ssm, conv = mgr.mamba_cache.temporal, mgr.mamba_cache.conv
    ssm.zero_()
    conv.zero_()
    mgr.mamba_cache.intermediate_ssm.fill_(7.0)
    mgr.mamba_cache.intermediate_conv_window.fill_(7.0)

    # Simulate mamba_metadata.state_indices — full captured batch of 4
    # (3 reals + 1 padding dummy on slot 0).
    state_indices = torch.tensor(
        [
            mgr.mamba_cache_index[100],
            mgr.mamba_cache_index[101],
            mgr.mamba_cache_index[102],
            0,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    attn = SimpleNamespace(num_seqs=4, num_contexts=0)
    mgr.update_mamba_states(
        attn,
        torch.tensor([1, 1, 1, 1], dtype=torch.int32, device="cuda"),
        state_indices=state_indices,
    )

    for rid in [100, 101, 102]:
        slot = mgr.mamba_cache_index[rid]
        assert torch.all(ssm[:, slot] == 7.0)
        assert torch.all(conv[:, slot] == 7.0)


@skip_no_cuda
def test_update_mamba_states_with_explicit_state_indices():
    mgr = _make_mgr()
    mgr._prepare_mamba_cache_blocks([200, 201, 202])

    ssm, conv = mgr.mamba_cache.temporal, mgr.mamba_cache.conv
    ssm.zero_()
    conv.zero_()
    mgr.mamba_cache.intermediate_ssm.fill_(3.0)
    mgr.mamba_cache.intermediate_conv_window.fill_(3.0)

    # Mimic csi.get_arg("slot_idx", truncate=True): int64, truncated
    # to current_length. One prefill (200), two generation (201, 202).
    state_indices = torch.tensor(
        [
            mgr.mamba_cache_index[200],
            mgr.mamba_cache_index[201],
            mgr.mamba_cache_index[202],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    # num_contexts=1 means only indices [1:] are scattered (the gen slice).
    attn = SimpleNamespace(num_seqs=3, num_contexts=1)
    mgr.update_mamba_states(
        attn,
        torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda"),
        state_indices=state_indices,
    )

    # Only 201 and 202 (the generation slice) should have been written.
    assert torch.all(ssm[:, mgr.mamba_cache_index[200]] == 0.0)
    for rid in [201, 202]:
        slot = mgr.mamba_cache_index[rid]
        assert torch.all(ssm[:, slot] == 3.0)
        assert torch.all(conv[:, slot] == 3.0)


@skip_no_cuda
def test_non_mtp_pytorch_prepare_and_get_state_indices_flow():
    mgr = _make_mgr(max_batch_size=4, max_draft_len=0)
    # Simulate a non-MTP step: mix of context + generation requests,
    # plus a CUDA-graph padding dummy.
    context_ids = [400]
    gen_ids = [401, 402]
    mgr._prepare_mamba_cache_blocks(context_ids + gen_ids)
    mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])

    # All reals have distinct slots.
    reals = set(mgr.mamba_cache_index[r] for r in context_ids + gen_ids)
    assert len(reals) == 3
    # Dummy's slot is distinct from every real.
    assert mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID] not in reals

    # What Mamba2Metadata.prepare would do: resolve indices for the
    # current padded batch (3 reals + 1 padding).
    request_ids = context_ids + gen_ids + [CUDA_GRAPH_DUMMY_REQUEST_ID]
    is_padding = [False, False, False, True]
    indices = mgr.get_state_indices(request_ids, is_padding)
    assert indices == [
        mgr.mamba_cache_index[400],
        mgr.mamba_cache_index[401],
        mgr.mamba_cache_index[402],
        mgr.mamba_cache_index[CUDA_GRAPH_DUMMY_REQUEST_ID],
    ]


def test_v2_hybrid_prepare_expect_snapshot_points():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.enable_block_reuse = True
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=64,
            additional_snapshot_offsets_from_start=[128, 999],
            additional_snapshot_offsets_from_end=[0, 22, 999],
        ),
    )
    requests = [
        SimpleNamespace(prompt_len=150, expect_snapshot_points=[999]),
        SimpleNamespace(prompt_len=128, expect_snapshot_points=[]),
        SimpleNamespace(prompt_len=32, expect_snapshot_points=[]),
    ]

    mgr.prepare_expect_snapshot_points(requests)

    assert [request.expect_snapshot_points for request in requests] == [
        [64, 128, 150],
        [64, 106, 128],
        [10, 32],
    ]


def test_v2_hybrid_prepare_expect_snapshot_points_without_periodic_snapshots():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.enable_block_reuse = True
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=0,
            additional_snapshot_offsets_from_start=[128],
            additional_snapshot_offsets_from_end=[0, 13],
        ),
    )
    request = SimpleNamespace(prompt_len=150, expect_snapshot_points=[])

    mgr.prepare_expect_snapshot_points([request])

    assert request.expect_snapshot_points == [128, 137, 150]


def _fp4_mla_snapshot_manager(**state_config) -> MambaHybridCacheManagerV2:
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.enable_block_reuse = True
    mgr.tokens_per_block = FP4_MLA_TOKENS_PER_BLOCK
    mgr._attention_cache_layout_policy = Fp4MlaV2CacheLayoutPolicy
    mgr._branch_snapshot_points = {}
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        mamba_state_config=MambaStateConfig(**state_config),
    )
    return mgr


def test_v2_hybrid_fp4_mla_snapshot_points_align_to_cache_blocks():
    mgr = _fp4_mla_snapshot_manager(
        periodic_snapshot_interval=100,
        additional_snapshot_offsets_from_end=[0, 13],
    )
    requests = [
        SimpleNamespace(py_request_id=0, prompt_len=350, expect_snapshot_points=[]),
        SimpleNamespace(py_request_id=1, prompt_len=100, expect_snapshot_points=[]),
    ]

    mgr.prepare_expect_snapshot_points(requests)

    # 100 and 87 lie inside the first block and can never be reused; 200,
    # 300, 337, and 350 move down to the block that a reuse would end on.
    assert [request.expect_snapshot_points for request in requests] == [[128, 256], []]


def test_v2_hybrid_without_fp4_mla_keeps_unaligned_snapshot_points():
    mgr = _fp4_mla_snapshot_manager(periodic_snapshot_interval=100)
    mgr._attention_cache_layout_policy = None
    request = SimpleNamespace(prompt_len=350, expect_snapshot_points=[])

    mgr.prepare_expect_snapshot_points([request])

    assert request.expect_snapshot_points == [100, 200, 300]


def test_v2_hybrid_fp4_mla_branch_snapshot_point_aligns_prompt_end():
    mgr = _fp4_mla_snapshot_manager(periodic_snapshot_interval=0, enable_branch_snapshot=True)
    request = SimpleNamespace(
        py_request_id=3,
        prompt_len=300,
        context_current_position=0,
        expect_snapshot_points=[],
    )

    mgr.prepare_expect_snapshot_points([request])

    # The branch snapshot falls back to the prompt end, which FP4 MLA can
    # reuse only up to its last full block.
    assert request.expect_snapshot_points == [256]


@pytest.mark.parametrize("fp4_mla", [False, True])
def test_v2_hybrid_fp4_mla_advances_history_with_pending_snapshot(fp4_mla: bool):
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.enable_block_reuse = True
    mgr.is_draft = False
    mgr.block_reuse_policy = BlockReusePolicy.PER_REQUEST
    mgr._attention_cache_layout_policy = Fp4MlaV2CacheLayoutPolicy if fp4_mla else None
    mgr.conversation_manager = None
    request = SimpleNamespace(
        py_request_id=7,
        is_dummy_request=False,
        context_current_position=128,
        context_remaining_length=172,
        expect_snapshot_points=[256],
        is_last_context_chunk=False,
    )
    # History stays at the committed end while the snapshot is pending.
    kv_cache = SimpleNamespace(
        is_active=True,
        history_length=0,
        num_committed_tokens=0,
        resize=MagicMock(return_value=True),
    )
    mgr.kv_cache_map = {request.py_request_id: kv_cache}
    batch = ScheduledRequests()
    batch.append_context_request(request)

    mgr.update_context_resources(batch)

    # Only the FP4 MLA HP ring needs the completed chunk's history before
    # the pending snapshot at 256 commits.
    if fp4_mla:
        kv_cache.resize.assert_called_once_with(None, 128)
    else:
        kv_cache.resize.assert_not_called()


def test_mamba_snapshot_rule_count_deduplicates_and_filters_unreachable_points():
    config = KvCacheConfig(
        enable_block_reuse=True,
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=0,
            additional_snapshot_offsets_from_start=[64, 65, 64],
            additional_snapshot_offsets_from_end=[0, 32, 4096],
        ),
    )

    assert _mamba_snapshot_rule_counts(config, 128, 32) == (4, 3)


def test_v2_hybrid_snapshot_sizing_scales_with_pp_and_explicit_rules():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.max_batch_size = 4
    mgr.mapping = Mapping(world_size=2, rank=0, tp_size=1, pp_size=2)
    mgr.max_seq_len = 128
    mgr.tokens_per_block = 32
    mgr._num_reserved_dummy_slots = 0
    config = KvCacheConfig(
        enable_block_reuse=True,
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=0,
            additional_snapshot_offsets_from_start=[64],
            additional_snapshot_offsets_from_end=[0, 32],
        ),
    )

    assert mgr._num_ssm_snapshots_for_capacity(512, config) == 24
    assert mgr._num_ssm_states_per_typical_request(128, config) == 4
    assert [desc.capacity for desc in mgr._typical_request_descs(128, config)] == [
        32,
        32,
        32,
        32,
    ]

    periodic_config = KvCacheConfig(
        enable_block_reuse=True,
        mamba_state_config=MambaStateConfig(periodic_snapshot_interval=48),
    )
    assert mgr._num_ssm_states_per_typical_request(47, periodic_config) == 1
    assert mgr._num_ssm_states_per_typical_request(48, periodic_config) == 2


@pytest.mark.parametrize(
    "capacity, interval, from_start, from_end, reuse, branch, expected_states",
    [
        (128, 0, [], [], True, False, 1),
        (128, 256, [], [], True, False, 1),
        (47, 48, [], [], True, False, 1),
        # At the endpoint, the snapshot and live state have identical values but
        # separate storage so decoding can update the live state independently.
        (48, 48, [], [], True, False, 2),
        (49, 48, [], [], True, False, 2),
        (128, 48, [], [], True, False, 3),
        (128, 64, [], [], True, False, 3),
        # Capacity models a typical length, not a hard limit: offset 129 is
        # excluded from the from-start estimate. Each unique from-end offset
        # counts separately, even if it coincides with another modeled point
        # or needs a longer actual prompt to produce a positive position.
        (128, 0, [64, 64, 128, 129], [0, 64, 128], True, False, 6),
        (128, 64, [64, 64, 128, 129], [0, 32, 64, 128], True, False, 7),
        (128, 48, [64, 96], [0, 32, 64], True, False, 7),
        # Identical from-end offsets always coincide, regardless of prompt length.
        (128, 0, [], [0, 64, 64, 128, 129], True, False, 5),
        # max_seq_len is 256: reachable offsets above capacity still count,
        # but offsets at or above the maximum cannot yield a positive position.
        (128, 0, [], [128, 255, 255, 256, 4096], True, False, 3),
        (128, 0, [], [256], True, False, 1),
        (128, 0, [], [4096], True, False, 1),
        (128, 0, [], [256, 4096], True, True, 2),
        (128, 64, [32], [0], False, False, 1),
        # Even a zero length estimate reserves space for configured from-end rules.
        (0, 64, [32], [0], True, False, 2),
        (128, 0, [], [], True, True, 2),
        (128, 256, [], [], True, True, 2),
        (128, 64, [], [], True, True, 3),
        (128, 0, [32], [], True, True, 2),
        (128, 0, [], [0], True, True, 2),
    ],
)
def test_v2_hybrid_typical_state_count(
    capacity: int,
    interval: int,
    from_start: list[int],
    from_end: list[int],
    reuse: bool,
    branch: bool,
    expected_states: int,
) -> None:
    config = KvCacheConfig(
        max_gpu_total_bytes=32 << 20,
        host_cache_size=0,
        avg_seq_len=max(1, capacity),
        enable_block_reuse=reuse,
        block_reuse_config=BlockReuseConfig(enable_branch_snapshot=branch),
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=interval,
            additional_snapshot_offsets_from_start=from_start,
            additional_snapshot_offsets_from_end=from_end,
        ),
    )

    # The conservative estimate can exceed runtime placement for a particular
    # prompt length; expected_states includes one live state plus snapshots.
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.max_seq_len = 256
    assert mgr._num_ssm_states_per_typical_request(capacity, config) == expected_states


def test_cpp_mamba_estimator_handles_disabled_snapshots_without_attention():
    manager = object.__new__(KVCacheManager)
    manager._primary_pool_memory_bytes = 4096
    manager._secondary_pool_memory_bytes = 0
    manager.linear_attention_metadata = SimpleNamespace(
        all_recurrent_states_bytes=64,
        states_snapshot_interval=0,
    )
    manager.max_attention_window_vec = [LinearCacheType.RECURRENT_STATES.value]
    manager.get_cache_bytes_per_token = lambda: 0
    manager.mapping = SimpleNamespace(pp_size=1)
    manager.max_batch_size = 4
    manager.max_seq_len = 128
    manager.tokens_per_block = 32
    manager.spec_config = None

    blocks = manager._calculate_max_num_blocks_for_linear_attention(
        KvCacheConfig(
            max_tokens=512,
            enable_block_reuse=False,
            mamba_state_config=MambaStateConfig(periodic_snapshot_interval=0),
        )
    )

    assert blocks[128] == (16, 0)
    assert blocks[LinearCacheType.RECURRENT_STATES.value] == (5, 0)


def test_hybrid_mtp_layout_honors_explicit_base_partition():
    model_config = _hybrid_cache_sizing_model_config(
        [
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ]
    )
    spec_config = MTPDecodingConfig(max_draft_len=1)
    expected_pp_layers = ([0, 1], [2, 3, 4])

    for rank in range(2):
        mapping = Mapping(
            world_size=2,
            rank=rank,
            tp_size=1,
            pp_size=2,
            pp_partition=[2, 2],
        )
        pp_layers, total_layers = get_pp_layers(
            5,
            mapping,
            spec_config=spec_config,
            layer_mask=[True] * 5,
        )
        assert total_layers == 5
        assert pp_layers == expected_pp_layers[rank]

        _, local_mamba_layers, local_attention_layers = _get_local_mamba_cache_layout(
            model_config,
            mapping,
            spec_config=spec_config,
        )
        assert local_mamba_layers == 1
        assert local_attention_layers == rank + 1

        for manager_cls in (
            CppMambaHybridCacheManager,
            MambaHybridCacheManagerV2,
        ):
            cache_cost = manager_cls.get_cache_size_per_token(
                model_config,
                mapping,
                max_batch_size=1,
                kv_cache_config=KvCacheConfig(enable_block_reuse=False),
                spec_config=spec_config,
            )
            assert cache_cost == (64 * (rank + 1), 2400)


def test_hybrid_separate_mtp_draft_estimator_has_no_mamba_state():
    model_config = _hybrid_cache_sizing_model_config(
        [
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ]
    )
    spec_config = MTPDecodingConfig(max_draft_len=1)

    for rank in range(2):
        mapping = Mapping(world_size=2, rank=rank, tp_size=1, pp_size=2)

        target_cost = MambaHybridCacheManagerV2.get_cache_size_per_token(
            model_config,
            mapping,
            max_batch_size=1,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False),
            spec_config=spec_config,
            use_separate_draft_kv_cache=True,
        )
        assert target_cost == (64, 2400)

        _, local_mamba_layers, local_attention_layers = _get_local_mamba_cache_layout(
            model_config,
            mapping,
            spec_config=spec_config,
            is_draft=True,
        )
        assert local_mamba_layers == 0
        assert local_attention_layers == rank

        draft_cost = MambaHybridCacheManagerV2.get_cache_size_per_token(
            model_config,
            mapping,
            max_batch_size=1,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False),
            num_layers=1,
            spec_config=spec_config,
            is_draft=True,
        )
        assert draft_cost == (64 * rank, 0)


@pytest.mark.parametrize(
    ("spec_config", "enable_attention_dp", "expected_intercept"),
    [
        (None, False, 320),
        (MTPDecodingConfig(max_draft_len=4), True, 384),
        (
            MTPDecodingConfig(
                max_draft_len=4,
                draft_len_schedule={1: 4, 2: 2, 3: 1},
            ),
            True,
            576,
        ),
    ],
)
def test_v2_hybrid_estimator_counts_dummy_states_without_attention_capacity(
    monkeypatch, spec_config, enable_attention_dp, expected_intercept
):
    monkeypatch.setattr(
        KVCacheManager,
        "get_cache_size_per_token",
        lambda *args, **kwargs: 11,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common._get_local_mamba_cache_layout",
        lambda *args, **kwargs: (
            SimpleNamespace(get_states_bytes_per_layer=lambda mapping: 64),
            1,
            1,
        ),
    )
    mapping = Mapping(
        world_size=1,
        tp_size=1,
        pp_size=1,
        enable_attention_dp=enable_attention_dp,
    )

    assert MambaHybridCacheManagerV2.get_cache_size_per_token(
        object(),
        mapping,
        max_batch_size=4,
        kv_cache_config=KvCacheConfig(),
        spec_config=spec_config,
    ) == (11, expected_intercept)


def test_v2_hybrid_attention_bound_is_snapshot_alignment_agnostic():
    model_config = _hybrid_cache_sizing_model_config(["linear_attention", "full_attention"])
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

    def estimate(interval):
        return MambaHybridCacheManagerV2.get_cache_size_per_token(
            model_config,
            mapping,
            max_batch_size=2,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                mamba_state_config=MambaStateConfig(periodic_snapshot_interval=interval),
            ),
            tokens_per_block=32,
            max_seq_len=128,
        )

    aligned = estimate(32)
    unaligned = estimate(48)

    assert aligned[1] == unaligned[1]


def _base_attention_layer_configs(num_layers):
    return [
        AttentionLayerConfig(
            layer_id=LayerId(layer_idx),
            buffers=[BufferConfig(role="key", size=256)],
        )
        for layer_idx in range(num_layers)
    ]


def test_v2_hybrid_fp4_sizing_counts_hp_page_for_snapshot_lineage() -> None:
    model_config = _hybrid_cache_sizing_model_config(["linear_attention", "full_attention"])
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        mamba_state_config=MambaStateConfig(periodic_snapshot_interval=32),
    )

    def estimate(attention_page_bytes_override: int) -> tuple[int, int]:
        return _estimate_mamba_hybrid_cache_cost(
            model_config,
            mapping,
            max_batch_size=2,
            kv_cache_config=kv_cache_config,
            tokens_per_block=32,
            max_seq_len=128,
            num_reserved_dummy_slots=0,
            include_explicit_snapshots=True,
            cap_partial_attention_snapshots=True,
            attention_slope_override=10,
            attention_page_bytes_override=attention_page_bytes_override,
        )

    without_hp_page = estimate(0)
    with_hp_page = estimate(7)

    # A retained partial attention page also pins the sequence's HP ring page.
    assert with_hp_page[0] == without_hp_page[0]
    assert with_hp_page[1] - without_hp_page[1] == 2 * 7


def _fp4_mla_hybrid_manager_stub(
    manager_cls: type[MambaHybridCacheManagerV2] = MambaHybridCacheManagerV2,
) -> MambaHybridCacheManagerV2:
    """A hybrid V2 manager with KDA layers 0 and 2 and MLA layers 1 and 3."""
    mgr = object.__new__(manager_cls)
    mgr.pp_layers = [0, 1, 2, 3]
    mgr.layer_offsets = {layer: layer for layer in mgr.pp_layers}
    mgr._mamba_layer_mask = [True, False, True, False]
    mgr._use_kda_replay_update = False
    # KimiK3HybridCacheManager reads this while building the cache config.
    mgr._kda_replay = None
    mgr.num_local_layers = 4
    mgr.local_num_mamba_layers = 2
    mgr.num_kv_heads_per_layer = [0, 1, 0, 1]
    mgr.head_dim_per_layer = [96] * 4
    mgr.ssm_bytes = 64
    mgr.conv_bytes = 32
    mgr.max_batch_size = 1
    mgr.mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    mgr.max_seq_len = 128
    mgr.max_num_tokens = 128
    mgr.num_extra_kv_tokens = 0
    mgr._kv_reserve_draft_tokens = 0
    mgr.tokens_per_block = FP4_MLA_TOKENS_PER_BLOCK
    mgr.kv_factor = 1
    mgr._num_reserved_dummy_slots = 0
    mgr._minimum_live_gpu_quota = lambda: 0
    mgr.kv_cache_type = CacheTypeCpp.SELFKONLY
    mgr.dtype = DataType.NVFP4
    return mgr


def _configure_fp4_mla_hybrid(mgr, pretrained_config) -> KVCacheManagerConfig:
    mgr.kv_cache_config = Fp4MlaV2CacheLayoutPolicy.configure_manager(
        mgr,
        KvCacheConfig(
            dtype="nvfp4",
            enable_partial_reuse=False,
            tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        ),
        CacheTypeCpp.SELFKONLY,
        dtype=DataType.NVFP4,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        head_dim=96,
        pretrained_config=pretrained_config,
    )
    base_config = KVCacheManagerConfig(
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
        layers=_base_attention_layer_configs(4),
        initial_pool_ratio=[1.0],
    )
    return MambaHybridCacheManagerV2._build_cache_config(mgr, base_config)


def test_v2_hybrid_composes_fp4_mla_layout_with_ssm_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    mgr = _fp4_mla_hybrid_manager_stub()

    config = _configure_fp4_mla_hybrid(mgr, SimpleNamespace(kv_lora_rank=32))

    assert isinstance(config.layers[0], SsmLayerConfig)
    assert isinstance(config.layers[1], AttentionLayerConfig)
    assert isinstance(config.layers[2], SsmLayerConfig)
    assert isinstance(config.layers[3], AttentionLayerConfig)
    # One HP-ring virtual layer per FP4 MLA layer follows the model layers.
    assert len(config.layers) == 6
    assert all(layer.buffers[0].role == Role.MLA_HP_TAIL for layer in config.layers[4:])
    assert mgr._cache_manager_layer_ids == [LayerId(1), LayerId(3)]
    assert mgr._hp_manager_layer_ids == [LayerId(4), LayerId(5)]
    assert mgr._fp4_mla_compact_layer_idx(1) == 0
    assert mgr._fp4_mla_compact_layer_idx(3) == 1
    assert callable(mgr.get_fp4_mla_cache_buffers)
    assert callable(mgr.get_fp4_mla_page_table_spec)
    assert mgr.fp4_mla_hp_pool_size == mgr._fp4_mla_hp_pool_size
    assert mgr.get_buffers(0) is None
    assert mgr.get_buffers(2) is None

    mgr._attention_cache_layout_policy = Fp4MlaV2CacheLayoutPolicy
    mgr.impl = SimpleNamespace(layer_grouping=[[0, 2], [1, 3], [4, 5]])
    assert mgr._get_pool_roles(0) == (MambaRole.SSM_STATE, None)
    assert mgr._get_pool_roles(2) == (Role.MLA_HP_TAIL, None)


def test_v2_hybrid_composes_fp8_fallback_with_fp4_mla_and_ssm_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "3")
    mgr = _fp4_mla_hybrid_manager_stub()
    pretrained_config = SimpleNamespace(
        model_type="kimi_linear",
        num_hidden_layers=4,
        kv_lora_rank=32,
        linear_attn_config={"full_attn_layers": [2, 4]},
    )

    config = _configure_fp4_mla_hybrid(mgr, pretrained_config)

    assert mgr._cache_manager_layer_ids == [LayerId(1)]
    assert mgr._fp8_mla_manager_layer_ids == [LayerId(3)]
    assert mgr._hp_manager_layer_ids == [LayerId(4)]
    assert [buffer.role for buffer in config.layers[3].buffers] == [Role.KEY]
    assert config.layers[3].buffers[0].size == (
        96 * torch.float8_e4m3fn.itemsize * FP4_MLA_TOKENS_PER_BLOCK
    )
    assert mgr.get_kv_cache_dtype(1) == DataType.NVFP4
    assert mgr.get_kv_cache_dtype(3) == DataType.FP8


def _fp4_mla_hybrid_with_fp8_fallback(
    monkeypatch: pytest.MonkeyPatch,
    manager_cls: type[MambaHybridCacheManagerV2] = MambaHybridCacheManagerV2,
):
    """KDA layers 0 and 2, FP4 MLA layer 1, FP8-fallback MLA layer 3."""
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "3")
    mgr = _fp4_mla_hybrid_manager_stub(manager_cls)
    _configure_fp4_mla_hybrid(
        mgr,
        SimpleNamespace(
            model_type="kimi_linear",
            num_hidden_layers=4,
            kv_lora_rank=32,
            linear_attn_config={"full_attn_layers": [2, 4]},
        ),
    )
    # Lifecycles: SSM layers, FP4 MLA latent cache, FP8 fallback, HP ring.
    mgr.impl = SimpleNamespace(layer_grouping=[[0, 2], [1], [3], [4]])
    return mgr


def test_v2_hybrid_fp4_disagg_role_mapping_covers_mla_and_ssm_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fp4_mla_hybrid_with_fp8_fallback(monkeypatch)

    kinds = mgr.get_disagg_role_mapper_kinds()

    # SSM state keeps the INDEXED Role.ALL fallback and convolution state its
    # SECTIONED reshard, while the MLA roles take the FP4 MLA layout per role.
    assert kinds[Role.ALL] == MapperKind.INDEXED
    assert MambaRole.SSM_STATE not in kinds
    assert kinds[MambaRole.CONV_STATE] == MapperKind.SECTIONED
    assert kinds[Role.KEY] == MapperKind.NHD
    assert kinds[Role.KEY_BLOCK_SCALE] == MapperKind.NHD
    assert kinds[Role.MLA_HP_TAIL] == MapperKind.REPLICATED
    # INDEXED stays the whole-manager fallback only, as the page-table
    # builder requires.
    assert [role for role, kind in kinds.items() if kind == MapperKind.INDEXED] == [Role.ALL]


def test_v2_hybrid_without_fp4_mla_keeps_its_disagg_role_mapping() -> None:
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr._use_kda_replay_update = False
    mgr.local_num_mamba_layers = 1

    kinds = mgr.get_disagg_role_mapper_kinds()

    assert kinds[Role.ALL] == MapperKind.INDEXED
    assert Role.KEY not in kinds
    assert Role.MLA_HP_TAIL not in kinds
    assert mgr.get_disagg_ignored_roles() == frozenset()


@pytest.mark.parametrize("kda_replay", [False, True])
def test_v2_hybrid_fp4_disagg_ignores_derived_mla_and_kda_replay_roles(
    monkeypatch: pytest.MonkeyPatch, kda_replay: bool
) -> None:
    mgr = _fp4_mla_hybrid_with_fp8_fallback(monkeypatch, KimiK3HybridCacheManager)
    mgr._state_layout = None
    mgr._kda_replay = (
        SimpleNamespace(disagg_ignored_roles=lambda _: frozenset({DataRole("kda_conv_q")}))
        if kda_replay
        else None
    )

    ignored = mgr.get_disagg_ignored_roles()

    # The generation worker rebuilds V scales and packed V from K and K
    # scales, and seeds its KDA replay buffers from the transferred conv
    # state; neither is transferred.
    assert {Role.MLA_V_SCALE, Role.MLA_V_PACKED} <= ignored
    assert (DataRole("kda_conv_q") in ignored) == kda_replay
    assert ignored.isdisjoint(
        {
            Role.KEY,
            Role.KEY_BLOCK_SCALE,
            Role.MLA_HP_TAIL,
            MambaRole.SSM_STATE,
            MambaRole.CONV_STATE,
        }
    )


def test_v2_hybrid_fp4_disagg_global_layer_ids_cover_every_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fp4_mla_hybrid_with_fp8_fallback(monkeypatch)

    ids = [mgr.get_disagg_global_layer_ids(group) for group in range(4)]

    # SSM layers and MLA latent caches (FP4 and FP8 fallback) map to 2 * L,
    # and the HP ring of FP4 layer 1 to 2 * 1 + 1.
    assert ids == [[0, 4], [2], [6], [3]]
    all_ids = [gid for group in ids for gid in group]
    assert len(set(all_ids)) == len(all_ids)


def test_v2_hybrid_fp4_disagg_global_layer_ids_reject_unknown_virtual_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fp4_mla_hybrid_with_fp8_fallback(monkeypatch)
    mgr.impl = SimpleNamespace(layer_grouping=[[5]])

    with pytest.raises(ValueError, match="Unknown hybrid FP4 MLA internal layer 5"):
        mgr.get_disagg_global_layer_ids(0)


def test_v2_hybrid_virtual_hp_layers_are_not_recurrent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fp4_mla_hybrid_with_fp8_fallback(monkeypatch)

    # The page-table builder asks every layer of a lifecycle, including the
    # HP-ring virtual layers past the model's own, whether it is recurrent.
    assert [mgr._is_local_mamba_layer(layer) for layer in range(5)] == [
        True,
        False,
        True,
        False,
        False,
    ]


def _build_v2_hybrid_fp4_mla(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_disagg: bool = False,
    enable_block_reuse: bool = False,
    periodic_snapshot_interval: int = 0,
    additional_snapshot_offsets_from_end: tuple[int, ...] = (),
) -> MambaHybridCacheManagerV2:
    """A real hybrid V2 manager: KDA layer 0 and FP4 MLA layer 1."""
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    return MambaHybridCacheManagerV2(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=1,
        mamba_layer_mask=[True, False],
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float16,
        kv_cache_config=KvCacheConfig(
            max_tokens=1024,
            dtype="nvfp4",
            enable_block_reuse=enable_block_reuse,
            enable_partial_reuse=False,
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=periodic_snapshot_interval,
                additional_snapshot_offsets_from_end=list(additional_snapshot_offsets_from_end),
            ),
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=96,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_seq_len=512,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        layer_mask=[False, True],
        vocab_size=1024,
        dtype=DataType.NVFP4,
        pretrained_config=SimpleNamespace(
            num_hidden_layers=2, kv_lora_rank=32, qk_rope_head_dim=64
        ),
        is_disagg=is_disagg,
    )


@skip_no_cuda
def test_v2_hybrid_fp4_disagg_page_table_covers_every_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _build_v2_hybrid_fp4_mla(monkeypatch, is_disagg=True)
    try:
        assert mgr._attention_cache_layout_policy is Fp4MlaV2CacheLayoutPolicy
        page_table = build_page_table_from_manager(mgr)

        state_groups = [lg for lg in page_table.layer_groups if isinstance(lg, MambaLayerGroup)]
        paged_groups = [lg for lg in page_table.layer_groups if isinstance(lg, AttentionLayerGroup)]
        assert len(state_groups) == 1
        # The FP4 latent cache and the HP ring are distinct lifecycles.
        assert len(paged_groups) == 2

        state_kinds = {
            frozenset(view.pool_role): view.mapper_kind for view in state_groups[0].pool_views
        }
        assert state_kinds[MAMBA_SSM_ROLE] == MapperKind.INDEXED
        assert state_kinds[MAMBA_CONV_ROLE] == MapperKind.SECTIONED
        assert [ll.global_layer_id for ll in state_groups[0].local_layers] == [0]

        paged_roles = {}
        for group in paged_groups:
            for view in group.pool_views:
                for role in view.pool_role:
                    paged_roles[role] = view.mapper_kind
        # V scales are derived on the generation worker and never transferred.
        assert paged_roles == {
            str(Role.KEY): MapperKind.NHD,
            str(Role.KEY_BLOCK_SCALE): MapperKind.NHD,
            str(Role.MLA_HP_TAIL): MapperKind.REPLICATED,
        }
        paged_ids = sorted(ll.global_layer_id for lg in paged_groups for ll in lg.local_layers)
        assert paged_ids == [2, 3]
    finally:
        mgr.shutdown()


def test_v2_hybrid_fp4_mla_snapshot_realignment_logs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import mamba_cache_manager_v2

    messages = []
    monkeypatch.setattr(mamba_cache_manager_v2.logger, "info", messages.append)
    mgr = SimpleNamespace(
        _attention_cache_layout_policy=object(),
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        _fp4_mla_snapshot_realignment_logged=False,
    )
    align = MambaHybridCacheManagerV2._align_fp4_mla_snapshot_points

    aligned = SimpleNamespace(py_request_id=1, expect_snapshot_points=[128, 256])
    align(mgr, aligned)
    assert aligned.expect_snapshot_points == [128, 256]
    assert messages == []

    moved = SimpleNamespace(py_request_id=2, expect_snapshot_points=[100, 1000])
    align(mgr, moved)
    assert moved.expect_snapshot_points == [896]
    assert len(messages) == 1
    assert "[100, 1000] -> [896]" in messages[0]

    again = SimpleNamespace(py_request_id=3, expect_snapshot_points=[300])
    align(mgr, again)
    assert again.expect_snapshot_points == [256]
    assert len(messages) == 1


@skip_no_cuda
def test_v2_hybrid_fp4_mla_reuses_prefix_at_aligned_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_len = 2 * FP4_MLA_TOKENS_PER_BLOCK
    mgr = _build_v2_hybrid_fp4_mla(
        monkeypatch,
        enable_block_reuse=True,
        # 256 and 156; FP4 MLA moves 156 down to its cache block (128).
        additional_snapshot_offsets_from_end=(0, 100),
    )
    request = _make_v2_conversation_request(1, list(range(prompt_len)), "conv-1")
    probe = _make_v2_conversation_request(2, list(range(prompt_len + 1)), "conv-2")
    try:
        assert mgr.fp4_mla_context_only_cache is False
        _run_v2_hybrid_context(mgr, request)
        assert request.expect_snapshot_points == [FP4_MLA_TOKENS_PER_BLOCK, prompt_len]
        state_index = mgr.get_state_indices([request.py_request_id], [False])[0]
        mgr.free_resources(request)

        mgr.prepare_expect_snapshot_points([probe])
        assert mgr.prepare_context(probe)
        batch = ScheduledRequests()
        batch.append_context_request(probe)
        mgr.prepare_resources(batch)

        # The whole first request is reused: FP4 MLA pages, the HP ring page
        # of its last block, and the recurrent state committed at token 256.
        assert probe.prepopulated_prompt_len == prompt_len
        assert probe.prepopulated_prompt_len % FP4_MLA_TOKENS_PER_BLOCK == 0
        assert mgr.get_state_indices([probe.py_request_id], [False]) == [state_index]
        mgr.free_resources(probe)
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_fp4_mla_reuses_prefix_after_two_chunk_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first chunk resizes the FP4 MLA cache while the only snapshot is pending."""
    prompt_len = 2 * FP4_MLA_TOKENS_PER_BLOCK
    chunk_len = FP4_MLA_TOKENS_PER_BLOCK
    mgr = _build_v2_hybrid_fp4_mla(
        monkeypatch,
        enable_block_reuse=True,
        additional_snapshot_offsets_from_end=(0,),
    )
    request = _make_v2_conversation_request(1, list(range(prompt_len)), "conv-1")
    probe = _make_v2_conversation_request(2, list(range(prompt_len + 1)), "conv-2")
    try:
        assert mgr.fp4_mla_context_only_cache is False
        mgr.prepare_expect_snapshot_points([request])
        assert request.expect_snapshot_points == [prompt_len]
        assert mgr.prepare_context(request)

        batch = ScheduledRequests()
        batch.append_context_request(request)

        assert mgr.resize_context(request, num_tokens=chunk_len)
        mgr.prepare_resources(batch)
        request.context_current_position = chunk_len
        assert request.context_remaining_length == chunk_len
        mgr.update_context_resources(batch)

        assert mgr.resize_context(request, num_tokens=chunk_len)
        mgr.prepare_resources(batch)
        request.context_current_position = prompt_len
        assert request.context_remaining_length == 0
        mgr.update_context_resources(batch)

        state_index = mgr.get_state_indices([request.py_request_id], [False])[0]
        mgr.free_resources(request)

        mgr.prepare_expect_snapshot_points([probe])
        assert mgr.prepare_context(probe)
        probe_batch = ScheduledRequests()
        probe_batch.append_context_request(probe)
        mgr.prepare_resources(probe_batch)

        assert probe.prepopulated_prompt_len == prompt_len
        assert mgr.get_state_indices([probe.py_request_id], [False]) == [state_index]
        mgr.free_resources(probe)
    finally:
        mgr.shutdown()


def test_v2_hybrid_fp4_invalid_iterator_scans_physical_pools() -> None:
    physical = torch.empty((3, 4), dtype=torch.uint8)
    ssm = torch.empty((1, 2), dtype=torch.float32)
    conv = torch.empty((1, 2), dtype=torch.float32)
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr._attention_cache_layout_policy = Fp4MlaV2CacheLayoutPolicy
    mgr._iter_fp4_mla_physical_pool_views = MagicMock(return_value=iter((physical,)))
    mgr.get_fp4_mla_hp_pool = MagicMock()
    mgr._recurrent_buffers = {
        (0, MambaRole.SSM_STATE): ssm,
        (0, MambaRole.CONV_STATE): conv,
    }

    views = list(MambaHybridCacheManagerV2._iter_cache_buffers_for_invalid_check(mgr))

    expected = [physical, ssm, conv]
    assert [layer_id for layer_id, _ in views] == [-1, -1, -1]
    assert all(actual is wanted for (_, actual), wanted in zip(views, expected))
    mgr._iter_fp4_mla_physical_pool_views.assert_called_once_with()
    mgr.get_fp4_mla_hp_pool.assert_not_called()


def _install_default_v2_state(mgr):
    """Populate local layer IDs for tests of V2 config building alone."""
    mgr.mamba_pp_layers = [
        layer_id for layer_id in mgr.pp_layers if mgr._mamba_layer_mask[layer_id]
    ]


def test_v2_hybrid_typical_batch_splits_capacity_across_ssm_states_and_dummies():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.kv_cache_type = CacheTypeCpp.SELF
    mgr.head_dim_per_layer = [64, 64]
    mgr.pp_layers = [0, 1]
    mgr._mamba_layer_mask = [True, False]
    mgr.ssm_bytes = 64
    mgr.conv_bytes = 32
    mgr.max_attention_window_vec = [128, 128]
    mgr.max_batch_size = 2
    mgr.mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    mgr.max_seq_len = 128
    mgr.max_num_tokens = 128
    mgr.tokens_per_block = 32
    mgr.num_local_layers = 2
    mgr.local_num_mamba_layers = 1
    _install_default_v2_state(mgr)
    mgr._num_reserved_dummy_slots = 1
    mgr.dtype = DataType.HALF
    mgr.enable_swa_scratch_reuse = False
    mgr.enable_stats = False
    mgr.num_extra_kv_tokens = 0
    mgr.get_layer_bytes_per_token = lambda **kwargs: 8
    mgr._minimum_live_gpu_quota = lambda: 0
    kv_cache_config = KvCacheConfig(
        enable_partial_reuse=True,
        avg_seq_len=96,
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=48,
            additional_snapshot_offsets_from_start=[32],
            additional_snapshot_offsets_from_end=[0],
        ),
    )
    mgr.kv_cache_config = kv_cache_config
    constraints = [BatchDesc([KVCacheDesc(capacity=64, history_length=0)])]
    base_layers = _base_attention_layer_configs(2)
    base_config = KVCacheManagerConfig(
        tokens_per_block=32,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
        layers=base_layers,
        constraints=constraints,
    )

    config = mgr._build_cache_config(base_config)

    assert isinstance(config.layers[0], SsmLayerConfig)
    assert isinstance(config.layers[1], AttentionLayerConfig)
    assert int(config.layers[1].layer_id) == int(base_layers[1].layer_id)
    assert config.typical_step == BatchDesc(
        (
            [KVCacheDesc(capacity=20, history_length=19)]
            + [KVCacheDesc(capacity=19, history_length=18)] * 4
        )
        * 2
        + [KVCacheDesc(capacity=0, history_length=0)]
    )
    # The caller-provided constraint keeps its dummy-slot padding.
    assert (
        BatchDesc(
            [
                KVCacheDesc(capacity=64, history_length=0),
                KVCacheDesc(capacity=0, history_length=0),
            ]
        )
        in config.constraints
    )
    # An explicit SSM floor constraint is always emitted so the recurrent pool
    # can hold every live + reserved-dummy state slot even when the caller
    # supplies no constraints (e.g. avg_seq_len unset). It is built from
    # zero-capacity requests, so it costs no attention pages.
    required_ssm_slots = mgr._max_resident_sequences() + mgr._num_reserved_dummy_slots
    assert any(
        all(kv.capacity == 0 for kv in batch.kv_caches)
        and len(batch.kv_caches) >= required_ssm_slots
        for batch in config.constraints
    )
    assert sum(kv.capacity for kv in config.typical_step.kv_caches) == 2 * 96
    assert not hasattr(config.typical_step.kv_caches[0], "num_ssm_slots")


@skip_no_cuda
@pytest.mark.parametrize("pool_ratio", [None, [0.25, 0.75]])
@pytest.mark.parametrize(
    "interval, reuse, states_per_request, runtime_snapshots",
    [(0, True, 3, 2), (48, True, 5, 3), (192, True, 3, 2), (48, False, 1, 0)],
)
def test_v2_hybrid_typical_batch_accounts_for_snapshot_retention(
    pool_ratio: list[float] | None,
    interval: int,
    reuse: bool,
    states_per_request: int,
    runtime_snapshots: int,
) -> None:
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=2,
        max_num_tokens=128,
        kv_cache_config=KvCacheConfig(
            max_gpu_total_bytes=32 << 20,
            host_cache_size=0,
            enable_block_reuse=reuse,
            pool_ratio=pool_ratio,
            avg_seq_len=96,
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=interval,
                additional_snapshot_offsets_from_start=[32],
                additional_snapshot_offsets_from_end=[0],
            ),
        ),
    )
    try:
        config = mgr.kv_cache_manager_py_config
        assert isinstance(config.layers[0], SsmLayerConfig)
        assert isinstance(config.layers[1], AttentionLayerConfig)
        assert config.initial_pool_ratio == pool_ratio
        if pool_ratio is None:
            capacity_per_state, remainder = divmod(96, states_per_request)
            capacities = [
                capacity_per_state + int(i < remainder) for i in range(states_per_request)
            ]
            assert config.typical_step == BatchDesc(
                [
                    KVCacheDesc(capacity=capacity, history_length=capacity - 1)
                    for capacity in capacities
                ]
                * 2
                + [KVCacheDesc(capacity=0, history_length=0)]
            )
            assert sum(kv.capacity for kv in config.typical_step.kv_caches) == 2 * 96
            assert (
                BatchDesc(
                    [
                        KVCacheDesc(capacity=128, history_length=0),
                        KVCacheDesc(capacity=0, history_length=0),
                    ]
                )
                in config.constraints
            )
        else:
            assert config.typical_step is None

        # Snapshot retention affects the typical batch, not the feasibility floors.
        assert BatchDesc([KVCacheDesc(capacity=0, history_length=0)] * 3) in config.constraints
        assert len(config.constraints) == (2 if pool_ratio is None else 1)
        assert mgr.all_ssm_states[0].shape[0] >= 3

        request = LlmRequest(
            request_id=1,
            max_new_tokens=1,
            input_tokens=list(range(96)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        mgr.prepare_expect_snapshot_points([request])
        assert len(request.expect_snapshot_points) == runtime_snapshots
    finally:
        mgr.shutdown()


def test_v2_hybrid_warns_when_avg_seq_len_is_missing(monkeypatch):
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.max_seq_len = 4096
    warnings_seen = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2.logger.warning",
        lambda message: warnings_seen.append(message),
    )

    capacity = mgr._get_typical_request_capacity(KvCacheConfig())

    assert capacity == 2048
    assert len(warnings_seen) == 1
    assert "kv_cache_config.avg_seq_len" in warnings_seen[0]
    assert "max_seq_len / 2=2048" in warnings_seen[0]
    assert "workload's average total sequence length" in warnings_seen[0]


@skip_no_cuda
def test_v2_hybrid_fallback_warns_about_snapshot_retention(monkeypatch) -> None:
    warnings_seen = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2.logger.warning",
        lambda message: warnings_seen.append(message),
    )
    mgr = _build_v2_hybrid_with_mamba_layer(
        kv_cache_config=KvCacheConfig(max_gpu_total_bytes=32 << 20, host_cache_size=0),
    )
    try:
        assert len(warnings_seen) == 1
        assert "snapshot retention" in warnings_seen[0]
        assert "prompt-length upper bound" in warnings_seen[0]
    finally:
        mgr.shutdown()


def test_v2_hybrid_rejects_quota_below_live_state_floor():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.layer_specs = None
    mgr._generation_kv_capacity_headroom = 1
    mgr._has_cp_helix = False
    mgr.max_batch_size = 2
    mgr.mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    mgr.local_num_mamba_layers = 1
    mgr._mamba_layer_mask = [True, False]
    mgr.ssm_bytes = 64
    mgr.conv_bytes = 32
    mgr._num_reserved_dummy_slots = 1
    mgr.tokens_per_block = 32
    mgr.num_local_layers = 2
    mgr.pp_layers = [0, 1]
    _install_default_v2_state(mgr)
    mgr.max_attention_window_vec = [128, 128]
    mgr.max_num_tokens = 128
    mgr.enable_swa_scratch_reuse = False
    mgr.get_layer_bytes_per_token = lambda **kwargs: 8
    mgr._attention_cache_bytes_per_token = lambda: 16
    mgr.kv_cache_config = KvCacheConfig(enable_partial_reuse=False)
    minimum_quota = mgr._minimum_live_gpu_quota()

    base_config = KVCacheManagerConfig(
        tokens_per_block=32,
        cache_tiers=[GpuCacheTierConfig(quota=minimum_quota - 1)],
        layers=[],
    )

    with pytest.raises(ValueError, match="too small for live recurrent states"):
        mgr._build_cache_config(base_config)


@skip_no_cuda
@pytest.mark.parametrize(
    "spec_config",
    [
        None,
        MTPDecodingConfig(
            max_draft_len=4,
            acceptance_rate_window_size=8,
            acceptance_rate_threshold=0.5,
        ),
    ],
)
def test_v2_hybrid_snapshot_retention_preserves_live_state_floor(
    spec_config: MTPDecodingConfig | None,
) -> None:
    minimum_quotas = []
    for interval in (0, 32):
        mgr = _build_v2_hybrid_with_mamba_layer(
            max_batch_size=2,
            max_num_tokens=128,
            spec_config=spec_config,
            kv_cache_config=KvCacheConfig(
                max_gpu_total_bytes=32 << 20,
                host_cache_size=0,
                enable_partial_reuse=False,
                mamba_state_config=MambaStateConfig(periodic_snapshot_interval=interval),
            ),
        )
        try:
            minimum_quotas.append(mgr._minimum_live_gpu_quota())
            dummy_slots = 1 if spec_config is None else 2
            assert mgr._num_reserved_dummy_slots == dummy_slots
            assert mgr.all_ssm_states[0].shape[0] >= 2 + dummy_slots
        finally:
            mgr.shutdown()

    assert minimum_quotas[0] == minimum_quotas[1]
    with pytest.raises(ValueError, match="too small for live recurrent states"):
        _build_v2_hybrid_with_mamba_layer(
            max_batch_size=2,
            max_num_tokens=128,
            spec_config=spec_config,
            kv_cache_config=KvCacheConfig(
                max_gpu_total_bytes=minimum_quotas[0] - 1,
                host_cache_size=0,
                enable_partial_reuse=False,
                mamba_state_config=MambaStateConfig(periodic_snapshot_interval=32),
            ),
        )


def test_v2_hybrid_pure_mamba_rank_does_not_reserve_attention_page():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.layer_specs = None
    mgr._generation_kv_capacity_headroom = 1
    mgr._has_cp_helix = False
    mgr.max_batch_size = 2
    mgr.mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    mgr.local_num_mamba_layers = 1
    mgr._mamba_layer_mask = [True]
    mgr.ssm_bytes = 64
    mgr.conv_bytes = 32
    mgr._num_reserved_dummy_slots = 1
    mgr.tokens_per_block = 32
    mgr.num_local_layers = 1
    mgr.pp_layers = [0]
    _install_default_v2_state(mgr)
    mgr.max_attention_window_vec = [128]
    mgr.max_num_tokens = 128
    mgr.enable_swa_scratch_reuse = False
    mgr.get_layer_bytes_per_token = lambda **kwargs: 0
    mgr._attention_cache_bytes_per_token = lambda: 0
    mgr.kv_cache_config = KvCacheConfig(enable_block_reuse=False)

    assert mgr._minimum_live_gpu_quota() == 3 * (64 + 32)


def test_cpp_hybrid_prepare_expect_snapshot_points():
    mgr = object.__new__(CppMambaHybridCacheManager)
    mgr.enable_block_reuse = True
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        mamba_state_config=MambaStateConfig(periodic_snapshot_interval=64),
    )
    mgr.linear_attention_metadata = SimpleNamespace(states_snapshot_interval=64)
    requests = [
        SimpleNamespace(prompt_len=150, expect_snapshot_points=[999]),
        SimpleNamespace(prompt_len=128, expect_snapshot_points=[]),
        SimpleNamespace(prompt_len=32, expect_snapshot_points=[]),
    ]

    mgr.prepare_expect_snapshot_points(requests)

    assert [request.expect_snapshot_points for request in requests] == [
        [64, 128],
        [64, 128],
        [],
    ]


@pytest.mark.parametrize(
    ("allocated_offsets", "context_current_position", "expected_offset"),
    [
        ({9: 25}, 0, 25),
        ({7: 17, 9: 25}, 0, 17),
        ({7: 17, 9: 25}, 256, 25),
    ],
)
def test_cpp_hybrid_state_indices_skip_context_placeholders(
    allocated_offsets, context_current_position, expected_offset
):
    """Capacity-limited chunks use the next real snapshot/final block."""
    null_index = torch.iinfo(torch.int32).max
    block_offsets = torch.full((1, 1, 2, 10), null_index, dtype=torch.int32)
    for logical_index, pool_offset in allocated_offsets.items():
        block_offsets[0, 0, 0, logical_index] = pool_offset

    request = SimpleNamespace(
        py_request_id=0,
        prompt_len=314,
        is_context_finished=False,
        context_current_position=context_current_position,
        context_chunk_size=32,
        prepopulated_prompt_len=0,
        is_dummy=False,
    )
    mgr = object.__new__(CppMambaHybridCacheManager)
    mgr.local_num_mamba_layers = 1
    mgr.requests = [request]
    mgr.tokens_per_block = 32
    mgr.kv_cache_config = SimpleNamespace(enable_block_reuse=True)
    mgr.impl = SimpleNamespace(
        copy_batch_block_offsets=lambda *args: None,
        get_cache_block_ids=lambda *args: [],
    )
    mgr.host_block_offsets = block_offsets
    mgr.recurrent_states_pool_index = 0
    mgr.blocks_per_window = {LinearCacheType.RECURRENT_STATES.value: (1264, 0)}
    mgr._host_state_indices = torch.zeros(1, dtype=torch.int32)
    mgr._state_indices_buffer = DeviceCopyBuffer(torch.zeros(1, dtype=torch.int32))
    mgr._row_indices = torch.arange(1, dtype=torch.long)
    mgr._request_id_to_state_index = {}
    mgr._request_id_to_is_dummy = {}
    mgr._dummy_request_mask_buffer = None

    mgr._setup_state_indices()

    assert mgr._host_state_indices.tolist() == [expected_offset]
    assert mgr.cuda_state_indices.tolist() == [expected_offset]
    assert mgr.get_state_indices([request.py_request_id], [False]) == [expected_offset]


def test_v2_block_reuse_commit_saves_ssm_snapshot_at_snapshot_point():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.enable_block_reuse = True
    mgr.is_draft = False
    mgr.local_num_mamba_layers = 1
    mgr._branch_snapshot_points = {}
    mgr._branch_snapshots_taken_total = 0
    mgr._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]
    mgr._mark_context_position_as_history = MagicMock()

    token_ids = list(range(150))
    request = SimpleNamespace(
        prompt_len=150,
        context_current_position=137,
        context_remaining_length=13,
        expect_snapshot_points=[137],
        is_dummy_request=False,
        is_dummy=False,
        py_request_id=0,
        get_tokens_view=lambda beam_idx: token_ids,
    )
    kv_cache = SimpleNamespace(
        num_committed_tokens=0,
        commit=MagicMock(),
        stop_committing=MagicMock(),
    )

    mgr.try_commit_blocks(request, kv_cache)

    kv_cache.commit.assert_called_once_with(token_ids[:137])
    kv_cache.stop_committing.assert_not_called()
    mgr._mark_context_position_as_history.assert_called_once_with(request, kv_cache)

    # The remaining suffix advances request history but must not publish a
    # second attention/SSM snapshot beyond the configured boundary.
    kv_cache.num_committed_tokens = 137
    request.context_current_position = 150
    request.context_remaining_length = 0
    mgr.try_commit_blocks(request, kv_cache)

    kv_cache.commit.assert_called_once_with(token_ids[:137])
    kv_cache.stop_committing.assert_called_once_with()


def test_v2_hybrid_add_dummy_requests_forwards_encoder_output_lens(mocker):
    mgr = object.__new__(MambaHybridCacheManagerV2)
    base_add_dummy_requests = mocker.patch.object(
        KVCacheManagerV2, "add_dummy_requests", return_value=[]
    )

    mgr.add_dummy_requests([123], encoder_output_lens=[17])

    assert base_add_dummy_requests.call_args.kwargs["encoder_output_lens"] == [17]


@pytest.mark.parametrize(
    "manager_cls,base_cls",
    [
        (MambaHybridCacheManagerV2, KVCacheManagerV2),
        (CppMambaHybridCacheManager, KVCacheManager),
    ],
    ids=["v2", "cpp"],
)
def test_add_dummy_requests_forwards_capture_sampling_params(mocker, manager_cls, base_cls):
    """Regression test: model_engine.py's CUDA graph warmup path always
    passes capture_sampling_params (None on the greedy pass, a real
    SamplingParams on non-greedy passes) to whatever concrete
    kv_cache_manager is active. An override with an explicit signature
    that doesn't accept/forward this kwarg breaks warmup for that
    manager type with a TypeError.
    """
    mgr = object.__new__(manager_cls)
    base_add_dummy_requests = mocker.patch.object(base_cls, "add_dummy_requests", return_value=[])
    sampling_params = SamplingParams(temperature=0.9, top_p=0.95)

    mgr.add_dummy_requests([123], capture_sampling_params=sampling_params)

    assert base_add_dummy_requests.call_args.kwargs["capture_sampling_params"] is sampling_params


@pytest.mark.parametrize(
    "manager_cls",
    [
        MambaHybridCacheManagerV2,
        CppMambaHybridCacheManager,
        MixedMambaHybridCacheManager,
    ],
    ids=["v2", "cpp", "mixed"],
)
def test_hybrid_prepare_expect_snapshot_points_clears_when_reuse_disabled(manager_cls):
    mgr = object.__new__(manager_cls)
    mgr.enable_block_reuse = False
    mgr.kv_cache_config = KvCacheConfig(enable_block_reuse=False)
    request = SimpleNamespace(prompt_len=64, expect_snapshot_points=[64])

    mgr.prepare_expect_snapshot_points([request])

    assert request.expect_snapshot_points == []


def test_cpp_hybrid_prepare_expect_snapshot_points_clears_for_disabled_interval():
    mgr = object.__new__(CppMambaHybridCacheManager)
    mgr.enable_block_reuse = True
    mgr.kv_cache_config = SimpleNamespace(
        enable_block_reuse=True,
        mamba_state_config=SimpleNamespace(periodic_snapshot_interval=0),
    )
    request = SimpleNamespace(prompt_len=150, expect_snapshot_points=[64])

    mgr.prepare_expect_snapshot_points([request])

    assert request.expect_snapshot_points == []


def test_expect_snapshot_points_binding_round_trip():
    request = LlmRequest(
        request_id=1,
        max_new_tokens=1,
        input_tokens=[1, 2, 3],
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )

    assert request.expect_snapshot_points == []
    request.expect_snapshot_points = [64, 128]
    assert request.expect_snapshot_points == [64, 128]


@skip_no_cuda
def test_v2_hybrid_pool_ratio_controls_allocated_memory():
    def allocated_memory(pool_ratio):
        mgr = object.__new__(MambaHybridCacheManagerV2)
        mgr.layer_specs = None
        mgr._generation_kv_capacity_headroom = 1
        mgr._has_cp_helix = False
        mgr.kv_cache_type = CacheTypeCpp.SELF
        mgr.head_dim_per_layer = [64, 64]
        mgr.pp_layers = [0, 1]
        mgr._mamba_layer_mask = [True, False]
        mgr.ssm_bytes = 64
        mgr.conv_bytes = 32
        mgr.max_attention_window_vec = [128, 128]
        mgr.max_batch_size = 2
        mgr.mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
        mgr.max_seq_len = 128
        mgr.max_num_tokens = 128
        mgr.tokens_per_block = 32
        mgr.num_local_layers = 2
        mgr.local_num_mamba_layers = 1
        _install_default_v2_state(mgr)
        mgr._num_reserved_dummy_slots = 1
        mgr.dtype = DataType.HALF
        mgr.enable_swa_scratch_reuse = False
        mgr.enable_stats = False
        mgr.num_extra_kv_tokens = 0
        mgr.get_layer_bytes_per_token = lambda **kwargs: 8

        kv_cache_config = KvCacheConfig(
            pool_ratio=pool_ratio,
            enable_partial_reuse=False,
            mamba_state_config=MambaStateConfig(periodic_snapshot_interval=64),
        )
        mgr.kv_cache_config = kv_cache_config
        base_config = KVCacheManagerConfig(
            tokens_per_block=32,
            cache_tiers=[GpuCacheTierConfig(quota=64 << 20)],
            layers=_base_attention_layer_configs(2),
            initial_pool_ratio=pool_ratio,
        )
        config = mgr._build_cache_config(base_config)
        runtime_manager = RuntimeKVCacheManager(config)
        try:
            statistics = runtime_manager.get_storage_statistics()

            def _slot_sizes(stat):
                # cpp binding exposes `slot_sizes`; the Python backend `slot_size`.
                return stat.slot_sizes if hasattr(stat, "slot_sizes") else stat.slot_size

            allocated_bytes = [
                int(stats.total) * sum(int(size) for size in _slot_sizes(stats))
                for stats in statistics
            ]
            return allocated_bytes, _introspection.current_gpu_ratio(runtime_manager)
        finally:
            runtime_manager.shutdown()

    low_mamba_allocation, low_actual_ratio = allocated_memory([0.25, 0.75])
    high_mamba_allocation, high_actual_ratio = allocated_memory([0.75, 0.25])

    assert low_actual_ratio == pytest.approx([0.25, 0.75])
    assert high_actual_ratio == pytest.approx([0.75, 0.25])
    assert high_mamba_allocation[0] > low_mamba_allocation[0]
    assert high_mamba_allocation[1] < low_mamba_allocation[1]


@skip_no_cuda
def test_v2_hybrid_snapshot_retention_changes_automatic_pool_allocation() -> None:
    def allocated_memory(
        pool_ratio: list[float] | None, interval: int = 64
    ) -> tuple[list[int], list[float]]:
        # State demand must exceed the GPU allocation granularity for the
        # automatic pool-ratio difference to be observable.
        mgr = _build_v2_hybrid_with_mamba_layer(
            max_batch_size=2,
            max_num_tokens=128,
            mamba_d_state=128,
            mamba_head_dim=128,
            mamba_num_heads=32,
            kv_cache_config=KvCacheConfig(
                max_gpu_total_bytes=64 << 20,
                host_cache_size=0,
                pool_ratio=pool_ratio,
                avg_seq_len=96,
                enable_partial_reuse=False,
                mamba_state_config=MambaStateConfig(periodic_snapshot_interval=interval),
            ),
        )
        try:
            assert mgr.kv_cache_manager_py_config.initial_pool_ratio == pool_ratio
            statistics = mgr.impl.get_storage_statistics()
            allocated_bytes = [
                int(stats.total) * sum(int(size) for size in stats.slot_sizes)
                for stats in statistics
            ]
            pool_groups = mgr.impl.get_life_cycle_pool_group_indices(0)
            ratios = _introspection.current_gpu_ratio(mgr.impl)
            return (
                [allocated_bytes[int(group)] for group in pool_groups],
                [ratios[int(group)] for group in pool_groups],
            )
        finally:
            mgr.shutdown()

    low_mamba_allocation, low_actual_ratio = allocated_memory([0.25, 0.75])
    high_mamba_allocation, high_actual_ratio = allocated_memory([0.75, 0.25])

    # Physical page sizes make the realized byte ratios approximate.
    assert low_actual_ratio == pytest.approx([0.25, 0.75], abs=0.01)
    assert high_actual_ratio == pytest.approx([0.75, 0.25], abs=0.01)
    assert high_mamba_allocation[0] > low_mamba_allocation[0]
    assert high_mamba_allocation[1] < low_mamba_allocation[1]

    assert allocated_memory([0.25, 0.75], interval=0) == (
        low_mamba_allocation,
        low_actual_ratio,
    )
    no_periodic_allocation, _ = allocated_memory(None, interval=0)
    periodic_allocation, _ = allocated_memory(None, interval=32)
    assert periodic_allocation[0] > no_periodic_allocation[0]
    assert periodic_allocation[1] < no_periodic_allocation[1]


# ---------------------------------------------------------------------------
# Cpp/V2 Mamba hybrid managers: recurrent-state allocation and reuse
#
# The Cpp pool is sized in
# KVCacheManager._calculate_max_num_blocks_for_linear_attention. It reserves a
# dedicated slot for each padding sentinel kind so dummy requests cannot evict
# live recurrent state. The V2 tests cover unified-pool state views, slot
# bookkeeping, replay, and snapshot reuse.
# ---------------------------------------------------------------------------


def _build_hybrid_with_mamba_layer(
    spec_config=None,
    max_batch_size=4,
    enable_block_reuse=False,
    periodic_snapshot_interval=256,
    is_estimating_kv_cache=False,
    dtype=DataType.HALF,
    mamba_layer_mask=None,
    attention_layer_mask=None,
    mamba_ssm_cache_dtype=torch.float16,
    use_replay_state_update=False,
):
    """Construct a real CppMambaHybridCacheManager with one mamba layer +
    one full-attention layer so the parent KVCacheManager goes through the
    linear-attention pool sizing path."""
    # Layer 0: mamba; Layer 1: full attention. Single rank, no MPI.
    mamba_mask = mamba_layer_mask or [True, False]
    attn_mask = attention_layer_mask or [False, True]
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    # Cap max_tokens to keep the real C++ pool allocation tiny.
    kv_cache_config = KvCacheConfig(
        max_tokens=512,
        enable_block_reuse=enable_block_reuse,
        mamba_state_config=MambaStateConfig(periodic_snapshot_interval=periodic_snapshot_interval),
    )
    return CppMambaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=sum(mamba_mask),
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=sum(attn_mask),
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=max_batch_size,
        mapping=mapping,
        spec_config=spec_config,
        layer_mask=attn_mask,
        is_estimating_kv_cache=is_estimating_kv_cache,
        dtype=dtype,
        use_replay_state_update=use_replay_state_update,
    )


def _build_v2_hybrid_with_mamba_layer(
    max_batch_size=4,
    num_mamba_layers=1,
    num_attention_layers=1,
    num_kv_heads=4,
    mapping=None,
    spec_config=None,
    use_replay_state_update=False,
    enable_block_reuse=False,
    enable_partial_reuse=True,
    block_reuse_policy="all_reusable",
    max_num_turns=1,
    periodic_snapshot_interval=0,
    additional_snapshot_offsets_from_end=None,
    enable_branch_snapshot=False,
    enable_attention_dp=False,
    enable_swa_scratch_reuse=False,
    dtype=DataType.HALF,
    kv_cache_dtype="auto",
    conv_state_layout="x_b_c",
    speculative_state=None,
    mamba_d_conv=4,
    mamba_num_heads=4,
    mamba_n_groups=1,
    mamba_ssm_cache_dtype=torch.float16,
    kda_replay_num_spec=None,
    enable_kv_pool_rebalance=False,
    max_tokens=512,
    max_gpu_total_bytes=0,
    kda_draft_records=False,
    cold_tier_offload_min_priority=None,
    host_cache_size=None,
    is_disagg=False,
    enable_stats=False,
    model_manager_cls=None,
    supports_inforward_snapshot=None,
    kv_cache_config: KvCacheConfig | None = None,
    mamba_d_state: int = 8,
    mamba_head_dim: int = 8,
    max_num_tokens: int = 8192,
) -> MambaHybridCacheManagerV2:
    """Construct a real MambaHybridCacheManagerV2."""
    mamba_mask = [True] * num_mamba_layers + [False] * num_attention_layers
    attn_mask = [False] * num_mamba_layers + [True] * num_attention_layers
    if mapping is None:
        mapping = Mapping(
            world_size=1,
            rank=0,
            tp_size=1,
            pp_size=1,
            enable_attention_dp=enable_attention_dp,
        )
    if kv_cache_config is None:
        kv_cache_config = KvCacheConfig(
            max_tokens=max_tokens,
            max_gpu_total_bytes=max_gpu_total_bytes,
            enable_block_reuse=enable_block_reuse,
            enable_partial_reuse=enable_partial_reuse,
            block_reuse_config=BlockReuseConfig(
                policy=block_reuse_policy,
                max_num_turns=max_num_turns,
                enable_branch_snapshot=enable_branch_snapshot,
            ),
            enable_swa_scratch_reuse=enable_swa_scratch_reuse,
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=periodic_snapshot_interval,
                additional_snapshot_offsets_from_end=list(
                    additional_snapshot_offsets_from_end or []
                ),
            ),
            dtype=kv_cache_dtype,
            enable_kv_pool_rebalance=enable_kv_pool_rebalance,
            cold_tier_offload_min_priority=cold_tier_offload_min_priority,
            host_cache_size=host_cache_size,
        )
    if speculative_state is None and kda_replay_num_spec is not None:
        speculative_state = KDAReplayState(kda_replay_num_spec, draft_records=kda_draft_records)
    model_cls = (
        KimiK3HybridCacheManager
        if isinstance(speculative_state, KDAReplayState) or kda_draft_records
        else Qwen35HybridCacheManager
        if conv_state_layout == "q_k_v"
        else NemotronHybridCacheManager
    )
    if model_manager_cls is not None:
        model_cls = model_manager_cls

    class ModelStateManager(model_cls):
        def _initialize_spec_state(self):
            if getattr(self, "kda_context_only", False):
                return None
            if kda_draft_records and speculative_state is None:
                return KDAIntermediateState()
            if speculative_state is None:
                return super()._initialize_spec_state()
            elif isinstance(speculative_state, KDAReplayState):
                self._kda_replay = speculative_state
                self._kda_replay.validate(self._state_layout)
            else:
                from tensorrt_llm._torch.modules.fla.cache_manager import validate_gdn_layout

                validate_gdn_layout(self, speculative_state)
            return speculative_state

    if supports_inforward_snapshot is not None:
        ModelStateManager._supports_inforward_snapshot = supports_inforward_snapshot
    manager_cls = ModelStateManager
    return manager_cls(
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_num_heads=mamba_num_heads,
        mamba_n_groups=mamba_n_groups,
        mamba_head_dim=mamba_head_dim,
        mamba_num_layers=num_mamba_layers,
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_attention_layers,
        num_kv_heads=num_kv_heads,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_num_tokens=max_num_tokens,
        max_batch_size=max_batch_size,
        mapping=mapping,
        spec_config=spec_config,
        layer_mask=attn_mask,
        vocab_size=1024,
        use_replay_state_update=use_replay_state_update,
        dtype=dtype,
        conv_state_layout=conv_state_layout,
        is_disagg=is_disagg,
        enable_stats=enable_stats,
        **(
            {"kda_draft_records": kda_draft_records}
            if issubclass(model_cls, KimiK3HybridCacheManager)
            else {"index_state_dim": 16}
            if issubclass(model_cls, Glm5NextCacheManager)
            else {}
        ),
    )


def _make_wide_spec_config(max_draft_len=2, tokens_per_gen_step=5):
    """Spec config whose per-step token width is wider than draft depth.

    This mirrors parallel-draft style metadata closely enough for cache-manager
    sizing without constructing the full speculative worker stack.
    """
    return SimpleNamespace(
        max_draft_len=max_draft_len,
        max_total_draft_tokens=tokens_per_gen_step - 1,
        tokens_per_gen_step=tokens_per_gen_step,
        spec_dec_mode=SimpleNamespace(use_one_engine=lambda: False),
    )


def _make_v2_conversation_request(
    request_id: int,
    tokens: list[int],
    conversation_id: str,
    retention: KvCacheRetentionConfig | None = None,
) -> LlmRequest:
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(),
        is_streaming=False,
        kv_cache_retention_config=retention,
    )
    request.py_conversation_params = ConversationParams(conversation_id=conversation_id)
    return request


def _run_v2_hybrid_context(
    manager: MambaHybridCacheManagerV2,
    request: LlmRequest,
    *,
    state_value: int | None = None,
) -> None:
    manager.prepare_expect_snapshot_points([request])
    assert manager.prepare_context(request)
    num_tokens = request.context_remaining_length
    assert manager.resize_context(request, num_tokens=num_tokens)

    batch = ScheduledRequests()
    batch.append_context_request(request)
    manager.prepare_resources(batch)
    if state_value is not None:
        state_index = manager.get_state_indices([request.py_request_id], [False])[0]
        for buffer in (*manager.all_ssm_states, *manager.all_conv_states):
            buffer[state_index].fill_(state_value)
    request.context_current_position = request.prompt_len
    assert request.context_remaining_length == 0
    manager.update_context_resources(batch)


def _build_v2_inforward_snapshot_manager():
    return _build_v2_hybrid_with_mamba_layer(
        max_batch_size=2,
        num_mamba_layers=2,
        enable_block_reuse=True,
        block_reuse_policy="per_request",
        additional_snapshot_offsets_from_end=[2],
        conv_state_layout="q_k_v",
        supports_inforward_snapshot=True,
    )


def _make_v2_prompt_request(request_id: int, tokens: list[int]) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


@skip_no_cuda
def test_v2_inforward_snapshot_commits_point_and_restores_prompt_end_state():
    # The snapshot at P = prompt_len - 2 holds the state the GDN layers left in the live slot;
    # the live slot gets the staged prompt-end state.
    mgr = _build_v2_inforward_snapshot_manager()
    try:
        assert mgr.inforward_snapshot_rows == 2  # max_batch_size
        request = _make_v2_prompt_request(1, list(range(1, 67)))
        mgr.prepare_expect_snapshot_points([request])
        assert request.expect_snapshot_points == [64]
        assert mgr.prepare_context(request)
        assert mgr.resize_context(request, num_tokens=66)
        request.context_chunk_size = 66
        request.py_mamba_inforward_snapshot_point = 64
        batch = ScheduledRequests()
        batch.append_context_request(request)
        mgr.prepare_resources(batch)
        assert mgr.inforward_snapshot_plan == {1: InforwardSnapshotEntry(0, 64, 64, 2)}
        assert request.py_mamba_inforward_snapshot_point is None

        # The forward: live slot = state at P, staging row 0 = state at the prompt end.
        slot = mgr.state_index(1)
        for layer in range(2):
            mgr.get_ssm_states(layer)[slot].fill_(1.0)
            mgr.get_conv_states(layer)[slot].fill_(1.0)
            stage_ssm, stage_conv = mgr.inforward_snapshot_stage_views(layer)
            stage_ssm[0].fill_(2.0)
            stage_conv[0].fill_(2.0)
        mgr.finish_inforward_snapshots(batch)
        assert mgr.inforward_snapshot_plan == {}
        torch.cuda.synchronize()
        request.context_current_position = 66
        mgr.update_context_resources(batch)
        assert mgr.kv_cache_map[1].num_committed_tokens == 64
        for layer in range(2):
            assert torch.all(mgr.get_ssm_states(layer)[slot] == 2.0)
            assert torch.all(mgr.get_conv_states(layer)[slot] == 2.0)

        # A sibling that shares the first 64 tokens resumes from the snapshot at P.
        sibling = _make_v2_prompt_request(2, list(range(1, 65)) + [500, 501, 502, 503])
        mgr.prepare_expect_snapshot_points([sibling])
        assert mgr.prepare_context(sibling)
        assert sibling.context_current_position == 64
        batch = ScheduledRequests()
        batch.append_context_request(sibling)
        mgr.prepare_resources(batch)
        torch.cuda.synchronize()
        sibling_slot = mgr.state_index(2)
        assert sibling_slot != slot
        for layer in range(2):
            assert torch.all(mgr.get_ssm_states(layer)[sibling_slot] == 1.0)
            assert torch.all(mgr.get_conv_states(layer)[sibling_slot] == 1.0)
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_inforward_snapshot_plan_drops_a_mark_in_a_mixed_batch():
    mgr = _build_v2_inforward_snapshot_manager()
    try:
        request = _make_v2_prompt_request(1, list(range(1, 67)))
        request.expect_snapshot_points = [64]
        request.context_chunk_size = 66
        request.py_mamba_inforward_snapshot_point = 64
        batch = SimpleNamespace(context_requests=[request], generation_requests=[object()])
        mgr._plan_inforward_snapshots(batch)
        assert mgr.inforward_snapshot_plan == {}
        assert request.py_mamba_inforward_snapshot_point is None
    finally:
        mgr.shutdown()


def _assert_replay_layer_cache_uses_history_size(layer_cache, history_size):
    assert layer_cache.old_x is not None
    assert layer_cache.old_B is not None
    assert layer_cache.old_dt is not None
    assert layer_cache.old_dA_cumsum is not None
    assert layer_cache.cache_buf_idx is not None
    assert layer_cache.prev_num_accepted_tokens is not None
    assert layer_cache.old_x.dim() == 5
    cache_size = layer_cache.temporal.shape[0]
    assert layer_cache.old_x.shape[0] == cache_size
    assert layer_cache.old_B.shape[0] == cache_size
    assert layer_cache.old_dt.shape[0] == cache_size
    assert layer_cache.old_dA_cumsum.shape[0] == cache_size
    assert layer_cache.cache_buf_idx.shape[0] == cache_size
    assert layer_cache.prev_num_accepted_tokens.shape[0] == cache_size
    assert layer_cache.old_x.shape[1] == 2
    assert layer_cache.old_B.shape[1] == 2
    assert layer_cache.old_dt.shape[1] == 2
    assert layer_cache.old_dA_cumsum.shape[1] == 2
    assert layer_cache.old_x.shape[2] == history_size
    assert layer_cache.old_B.shape[2] == history_size
    assert layer_cache.old_dt.shape[-1] == history_size
    assert layer_cache.old_dA_cumsum.shape[-1] == history_size


@skip_no_cuda
def test_v2_hybrid_allocates_mamba_state_and_dummy_indices():
    mgr = _build_v2_hybrid_with_mamba_layer(max_batch_size=4)
    try:
        assert mgr.local_num_mamba_layers == 1
        assert len(mgr.all_ssm_states) == 1
        assert len(mgr.all_conv_states) == 1
        assert mgr.all_ssm_states[0].shape[1:] == torch.Size([4, 8, 8])
        assert mgr.all_conv_states[0].shape[1:] == torch.Size([48, 3])
        assert mgr.get_max_resource_count() == 4
        assert mgr.blocks_in_primary_pool > 0
        assert isinstance(mgr.check_invalid_values_in_kv_cache(), bool)

        requests = mgr.add_dummy_requests([123], token_nums=[8], is_gen=False)

        assert len(requests) == 1
        indices = mgr.get_state_indices([123], [False])
        assert len(indices) == 1
        assert indices[0] >= 0
        assert mgr._request_id_to_is_dummy[123]
        assert mgr.cuda_state_indices[0].item() == indices[0]
        assert mgr.get_ssm_states(0).data_ptr() == mgr.all_ssm_states[0].data_ptr()
        assert mgr.get_conv_states(0).data_ptr() == mgr.all_conv_states[0].data_ptr()
    finally:
        mgr.shutdown()


@skip_no_cuda
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "nvfp4"])
def test_v2_hybrid_nvfp4_page_table_omits_ssm_block_scales(kv_cache_dtype: str) -> None:
    # "auto" keeps the user config unchanged when the checkpoint selects NVFP4.
    mgr = _build_v2_hybrid_with_mamba_layer(dtype=DataType.NVFP4, kv_cache_dtype=kv_cache_dtype)
    try:
        ssm_pool_id = mgr.impl.get_layer_group_id(LayerId(0))
        attention_pool_id = mgr.impl.get_layer_group_id(LayerId(1))

        assert torch.count_nonzero(mgr.kv_cache_pool_pointers[ssm_pool_id, :, 1]) == 0
        assert torch.count_nonzero(mgr.kv_cache_pool_pointers[attention_pool_id, :, 1]) > 0
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_supports_pure_mamba_pp_rank():
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=2,
        num_attention_layers=0,
    )
    try:
        assert mgr.local_num_mamba_layers == 1
        assert mgr.blocks_in_primary_pool == 0
        assert mgr._attention_cache_bytes_per_token() == 0
        assert len(mgr.all_ssm_states) == 1
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_metadata_supports_attention_only_pp_rank():
    mapping = Mapping(world_size=2, rank=1, tp_size=1, pp_size=2)
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=1,
        mapping=mapping,
    )
    try:
        assert mgr.local_num_mamba_layers == 0
        mgr.add_dummy_requests([123], token_nums=[8], is_gen=False)

        metadata = Mamba2Metadata(max_batch_size=1, chunk_size=8)
        seq_lens = torch.tensor([8], dtype=torch.int32)
        metadata.prepare(
            SimpleNamespace(
                seq_lens=seq_lens,
                seq_lens_cuda=seq_lens.cuda(),
                num_contexts=1,
                num_ctx_tokens=8,
                kv_cache_manager=mgr,
                request_ids=[123],
                kv_cache_params=SimpleNamespace(
                    num_cached_tokens_per_seq=torch.tensor([0], dtype=torch.int32)
                ),
            )
        )

        assert metadata.state_indices[0].item() == 0
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_invalid_check_scans_distinct_attention_pools():
    mgr = _build_v2_hybrid_with_mamba_layer(
        num_attention_layers=2,
        num_kv_heads=[1, 2, 4],
    )
    try:
        mgr.check_invalid_values_in_kv_cache(fill_with_zero=True)
        first_attention_buffer = mgr.get_buffers(1)
        second_attention_buffer = mgr.get_buffers(2)
        assert first_attention_buffer.data_ptr() != second_attention_buffer.data_ptr()

        for buffer in (
            second_attention_buffer,
            mgr.all_ssm_states[0],
            mgr.all_conv_states[0],
        ):
            buffer.flatten()[0] = torch.nan
            assert mgr.check_invalid_values_in_kv_cache()
            assert mgr.check_invalid_values_in_kv_cache(fill_with_zero=True)
            assert not torch.isnan(buffer).any()
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_dummy_indices_keep_cuda_buffer_address():
    max_batch_size = 1
    mgr = _build_v2_hybrid_with_mamba_layer(max_batch_size=max_batch_size, enable_attention_dp=True)
    try:
        request_ids = list(range(100, 100 + max_batch_size))
        mgr.add_dummy_requests(
            request_ids,
            token_nums=[8] * max_batch_size,
            is_gen=False,
        )
        state_indices_ptr = mgr.cuda_state_indices.data_ptr()

        new_requests = mgr.add_dummy_requests(
            [ATTENTION_DP_DUMMY_REQUEST_ID], token_nums=[8], is_gen=False
        )

        assert len(new_requests) == 1
        expected_capacity = max_batch_size + mgr._num_reserved_dummy_slots
        assert mgr.cuda_state_indices.shape[0] == expected_capacity
        assert mgr._host_state_indices.shape[0] == expected_capacity
        assert mgr.cuda_state_indices.data_ptr() == state_indices_ptr
        assert (
            mgr.cuda_state_indices[0].item()
            == mgr.get_state_indices([ATTENTION_DP_DUMMY_REQUEST_ID], [False])[0]
        )
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_reserves_every_persistent_dummy_slot():
    spec_config = MTPDecodingConfig(
        max_draft_len=4,
        draft_len_schedule={1: 4, 2: 2, 3: 1},
    )
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=4,
        spec_config=spec_config,
        enable_attention_dp=True,
    )
    try:
        runtime_draft_lengths = [4, 2, 1, 0]
        cuda_graph_dummy_ids = [
            CUDA_GRAPH_DUMMY_REQUEST_ID - draft_len for draft_len in runtime_draft_lengths
        ]
        request_ids = [101, 102, 103, 104]

        assert mgr._num_reserved_dummy_slots == 5
        # The reserved dummy slots sit on top of the admission pool, whose width
        # depends on the overlap/disagg lease coefficient.
        initial_free_slots = mgr.index_mapper.num_free_slots()
        assert initial_free_slots >= len(request_ids) + 5

        assert (
            mgr.add_dummy_requests(request_ids, token_nums=[1] * len(request_ids), is_gen=False)
            is not None
        )
        for request_id, draft_len in zip(cuda_graph_dummy_ids, runtime_draft_lengths):
            assert (
                mgr.add_dummy_requests(
                    [request_id],
                    is_gen=True,
                    max_num_draft_tokens=draft_len,
                )
                is not None
            )
        assert (
            mgr.add_dummy_requests([ATTENTION_DP_DUMMY_REQUEST_ID], token_nums=[1], is_gen=False)
            is not None
        )

        all_request_ids = request_ids + cuda_graph_dummy_ids + [ATTENTION_DP_DUMMY_REQUEST_ID]
        state_indices = mgr.get_state_indices(all_request_ids, [False] * len(all_request_ids))
        assert len(set(state_indices)) == len(all_request_ids)
        assert mgr.index_mapper.num_free_slots() == initial_free_slots - len(all_request_ids)
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_free_resources_drops_stale_state_index_mapping():
    mgr = _build_v2_hybrid_with_mamba_layer()
    try:
        request = mgr.add_dummy_requests([123], token_nums=[8], is_gen=False)[0]
        request_id = request.py_request_id
        assert request_id in mgr._request_id_to_state_index
        assert request_id in mgr._request_id_to_is_dummy

        # Move state-index preparation to another request before freeing the
        # older one, as happens when an asynchronous transfer finishes late.
        mgr.add_dummy_requests([456], token_nums=[8], is_gen=False)
        mgr.free_resources(request)

        assert request_id not in mgr._request_id_to_state_index
        assert request_id not in mgr._request_id_to_is_dummy
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_uses_upstream_min_snapshot_policy():
    mgr = _build_v2_hybrid_with_mamba_layer(
        enable_block_reuse=True,
        enable_partial_reuse=True,
    )
    try:
        assert mgr.block_reuse_policy is BlockReusePolicy.PER_REQUEST
        assert mgr.kv_cache_config.enable_partial_reuse
        assert mgr.kv_cache_manager_py_config.commit_min_snapshot
    finally:
        mgr.shutdown()


@pytest.mark.parametrize(
    ("rank", "expected_log_count"),
    [
        (0, 1),
        (3, 0),
    ],
)
def test_v2_hybrid_debug_logs_prefix_reuse_only_on_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    expected_log_count: int,
) -> None:
    get_num_reusable_tokens_before_hybrid_pruning = MagicMock(return_value=96)
    get_num_reusable_tokens_before_pruning = MagicMock(return_value=112)
    kv_cache = SimpleNamespace(
        _get_num_reusable_tokens_before_hybrid_pruning=get_num_reusable_tokens_before_hybrid_pruning,
        _get_num_reusable_tokens_before_pruning=get_num_reusable_tokens_before_pruning,
        num_committed_tokens=64,
    )
    create_kv_cache = MagicMock(return_value=kv_cache)
    log_debug = MagicMock()
    monkeypatch.setattr(KVCacheManagerV2, "_create_kv_cache", create_kv_cache)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2.logger.debug",
        log_debug,
    )

    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.mapping = SimpleNamespace(rank=rank)
    mgr.local_num_mamba_layers = 1
    result = mgr._create_kv_cache(
        request_id=123,
        lora_task_id=None,
        input_tokens=list(range(127)),
        expected_prompt_length=127,
    )

    assert result is kv_cache
    assert log_debug.call_count == expected_log_count
    assert get_num_reusable_tokens_before_hybrid_pruning.call_count == expected_log_count
    assert get_num_reusable_tokens_before_pruning.call_count == expected_log_count
    if rank == 0:
        log_debug.assert_called_once_with(
            "[MambaHybridCacheManagerV2] prefix reuse rank=0 request_id=123 "
            "request_total_tokens=128 "
            "longest_attention_match_tokens=96 "
            "content_divergence_tokens=112 "
            "latest_recurrent_snapshot_tokens=64"
        )


@pytest.mark.parametrize(("rank", "expected_log_count"), [(0, 1), (3, 0)])
def test_v2_hybrid_logs_aggregated_recurrent_cache_status_only_on_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    expected_log_count: int,
) -> None:
    first_stats = SimpleNamespace(
        iter_offload_blocks=3,
        iter_offload_bytes=300,
        iter_onboard_blocks=1,
        iter_onboard_bytes=100,
        iter_host_dropped_blocks=2,
        iter_host_dropped_bytes=200,
        primary_used_num_blocks=11,
        primary_free_num_blocks=5,
        primary_evictable_num_blocks=4,
        secondary_used_num_blocks=7,
        secondary_free_num_blocks=9,
    )
    second_stats = SimpleNamespace(
        iter_offload_blocks=2,
        iter_offload_bytes=200,
        iter_onboard_blocks=4,
        iter_onboard_bytes=400,
        iter_host_dropped_blocks=1,
        iter_host_dropped_bytes=100,
        primary_used_num_blocks=13,
        primary_free_num_blocks=6,
        primary_evictable_num_blocks=5,
        secondary_used_num_blocks=8,
        secondary_free_num_blocks=10,
    )
    report = SimpleNamespace(
        by_pool_group={
            6: SimpleNamespace(stats=first_stats),
            7: SimpleNamespace(stats=second_stats),
        },
    )
    monkeypatch.setattr(KVCacheManagerV2, "get_iteration_stats", MagicMock(return_value=report))
    log_debug = MagicMock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2.logger.debug",
        log_debug,
    )

    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.mapping = SimpleNamespace(rank=rank)
    mgr._stats_life_cycle_metadata = MagicMock(
        return_value={
            0: (4, 4096, "attention"),
            1: (6, None, "ssm"),
            2: (7, None, "ssm"),
            3: (6, None, "ssm"),
        }
    )
    mgr._recurrent_evicted_blocks_total = 0
    mgr._recurrent_onboarded_blocks_total = 0
    mgr._recurrent_dropped_blocks_total = 0
    mgr._recurrent_status_logged = False
    mgr._snapshot_pruned_tokens_total = 640
    mgr._page_pruned_tokens_total = 128
    mgr._branch_snapshots_taken_total = 3
    mgr._branch_snapshots_skipped_total = {"no_divergence": 2, "already_reused": 1}

    assert mgr.get_iteration_stats() is report
    assert mgr._recurrent_evicted_blocks_total == 5
    assert mgr._recurrent_onboarded_blocks_total == 5
    assert mgr._recurrent_dropped_blocks_total == 3
    assert log_debug.call_count == expected_log_count
    if rank == 0:
        log_debug.assert_called_once_with(
            "[MambaHybridCacheManagerV2] recurrent cache status "
            "rank=0 pool_group_ids=[6, 7] "
            "evicted_recurrent_blocks=5 evicted_recurrent_bytes=500 "
            "onboarded_recurrent_blocks=5 onboarded_recurrent_bytes=500 "
            "dropped_recurrent_blocks=3 dropped_recurrent_bytes=300 "
            "total_evicted_recurrent_blocks=5 "
            "total_onboarded_recurrent_blocks=5 "
            "total_dropped_recurrent_blocks=3 "
            "gpu_used_recurrent_blocks=24 gpu_free_recurrent_blocks=11 "
            "gpu_evictable_recurrent_blocks=9 "
            "host_used_recurrent_blocks=15 host_free_recurrent_blocks=19 "
            "snapshot_pruned_tokens=640 page_pruned_tokens=128 "
            "branch_snapshots_taken=3 branch_snapshots_skipped=3 "
            "branch_snapshots_skipped_by_reason="
            "{'already_reused': 1, 'no_divergence': 2}"
        )


@skip_no_cuda
def test_v2_hybrid_preserves_per_conversation_and_disables_periodic_snapshots():
    mgr = _build_v2_hybrid_with_mamba_layer(
        enable_block_reuse=True,
        block_reuse_policy="per_conversation",
        periodic_snapshot_interval=64,
        additional_snapshot_offsets_from_end=[0],
    )
    try:
        assert mgr.block_reuse_policy is BlockReusePolicy.PER_CONVERSATION
        assert mgr.conversation_manager is not None
        assert mgr.kv_cache_config.mamba_state_config.periodic_snapshot_interval == 0
        assert mgr.kv_cache_manager_py_config.commit_min_snapshot
        request = SimpleNamespace(prompt_len=150, expect_snapshot_points=[])
        mgr.prepare_expect_snapshot_points([request])
        assert request.expect_snapshot_points == [150]
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_retains_configured_number_of_conversation_turns():
    mgr = _build_v2_hybrid_with_mamba_layer(
        enable_block_reuse=True,
        block_reuse_policy="per_conversation",
        max_num_turns=2,
        additional_snapshot_offsets_from_end=[0],
        cold_tier_offload_min_priority=50,
        host_cache_size=0,
    )
    request_a = _make_v2_conversation_request(
        1,
        list(range(64)),
        "conv-1",
        KvCacheRetentionConfig([KvCacheRetentionConfig.TokenRangeRetentionConfig(0, None, 80)]),
    )
    request_b = _make_v2_conversation_request(2, list(range(100, 164)), "conv-1")
    # Use fresh conversation IDs so probes query the shared prefix cache
    # without altering conv-1's retained-turn accounting.
    # Probe one token past the exact SSM snapshots committed at token 64.
    request_a_probe = _make_v2_conversation_request(3, list(range(65)), "conv-2")
    request_b_probe = _make_v2_conversation_request(4, list(range(100, 165)), "conv-3")
    request_c = _make_v2_conversation_request(5, list(range(200, 264)), "conv-1")
    request_a_after_eviction = _make_v2_conversation_request(6, list(range(65)), "conv-4")
    request_b_after_eviction = _make_v2_conversation_request(7, list(range(100, 165)), "conv-5")
    requests = [
        request_a,
        request_b,
        request_a_probe,
        request_b_probe,
        request_c,
        request_a_after_eviction,
        request_b_after_eviction,
    ]

    def assert_reused_state(request: LlmRequest, value: int) -> None:
        state_index = mgr.get_state_indices([request.py_request_id], [False])[0]
        # Reuse preserves payloads even when cache offload/onboard moves their slots.
        for buffer in (*mgr.all_ssm_states, *mgr.all_conv_states):
            torch.testing.assert_close(
                buffer[state_index], torch.full_like(buffer[state_index], value), rtol=0, atol=0
            )

    try:
        _run_v2_hybrid_context(mgr, request_a, state_value=1)
        cache_a = mgr.kv_cache_map[request_a.py_request_id]
        attention_group = mgr.impl.get_layer_group_id(1)
        priorities = cache_a.get_page_priorities(attention_group)
        assert priorities and set(priorities) == {80}
        assert mgr.impl.init_config.cold_tier_offload_min_priority == 50
        assert not mgr.can_evict
        assert mgr.prefetch_for_context_tokens([request_a]) is True
        mgr.free_resources(request_a)
        _run_v2_hybrid_context(mgr, request_b, state_value=2)
        mgr.free_resources(request_b)

        mgr.prepare_expect_snapshot_points([request_a_probe])
        assert mgr.prepare_context(request_a_probe)
        probe_batch = ScheduledRequests()
        probe_batch.append_context_request(request_a_probe)
        mgr.prepare_resources(probe_batch)
        assert request_a_probe.prepopulated_prompt_len == request_a_probe.prompt_len - 1
        assert_reused_state(request_a_probe, 1)
        mgr.free_resources(request_a_probe)

        mgr.prepare_expect_snapshot_points([request_b_probe])
        assert mgr.prepare_context(request_b_probe)
        probe_batch = ScheduledRequests()
        probe_batch.append_context_request(request_b_probe)
        mgr.prepare_resources(probe_batch)
        assert request_b_probe.prepopulated_prompt_len == request_b_probe.prompt_len - 1
        assert_reused_state(request_b_probe, 2)
        mgr.free_resources(request_b_probe)

        _run_v2_hybrid_context(mgr, request_c, state_value=3)
        mgr.free_resources(request_c)

        mgr.prepare_expect_snapshot_points([request_a_after_eviction])
        assert mgr.prepare_context(request_a_after_eviction)
        assert request_a_after_eviction.prepopulated_prompt_len == 0

        mgr.prepare_expect_snapshot_points([request_b_after_eviction])
        assert mgr.prepare_context(request_b_after_eviction)
        probe_batch = ScheduledRequests()
        probe_batch.append_context_request(request_b_after_eviction)
        mgr.prepare_resources(probe_batch)
        assert (
            request_b_after_eviction.prepopulated_prompt_len
            == request_b_after_eviction.prompt_len - 1
        )
        assert_reused_state(request_b_after_eviction, 2)
    finally:
        for request in requests:
            if request.py_request_id in mgr.kv_cache_map:
                mgr.free_resources(request)
        mgr.shutdown()


@skip_no_cuda
@pytest.mark.parametrize("enable_block_reuse", [False, True])
def test_v2_hybrid_shares_an_inflight_prefix_without_branch_snapshots(enable_block_reuse):
    """The recurrent manager snapshots a prefilling request where a waiting
    request asks, so it needs neither block_reuse_config.enable_branch_snapshot
    nor a reuse policy that commits every chunk. An attention-only SWA cache
    with scratch reuse and branch snapshots off cannot (see
    test_kv_cache_v2_inflight_prefix.py)."""
    mgr = _build_v2_hybrid_with_mamba_layer(
        enable_block_reuse=enable_block_reuse,
        enable_branch_snapshot=False,
        enable_swa_scratch_reuse=True,
    )
    try:
        assert mgr.can_share_inflight_prefix is enable_block_reuse
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_waiter_points_after_history_advanced_commit():
    """Two waiters ask a prefilling owner for snapshots after its history passed
    its committed tokens. commit_min_snapshot accepts a commit only if it starts
    or ends at history_length, so each commit must still meet that."""
    mgr = _build_v2_hybrid_with_mamba_layer(enable_block_reuse=True)
    owner = LlmRequest(
        request_id=1,
        max_new_tokens=1,
        input_tokens=list(range(100)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    probes = [
        LlmRequest(
            request_id=2 + i,
            max_new_tokens=1,
            input_tokens=list(range(shared)) + [1000],
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        for i, shared in enumerate((64, 96))
    ]

    def run_chunk(end: int) -> None:
        mgr.prepare_expect_snapshot_points([owner])
        assert mgr.prepare_context(owner)
        assert mgr.resize_context(owner, num_tokens=end - owner.context_current_position)
        batch = ScheduledRequests()
        batch.append_context_request(owner)
        mgr.prepare_resources(batch)
        owner.context_current_position = end
        mgr.update_context_resources(batch)

    try:
        assert mgr.block_reuse_policy is BlockReusePolicy.PER_REQUEST
        run_chunk(32)
        kv_cache = mgr.kv_cache_map[owner.py_request_id]
        # No snapshot point is pending, so history follows the context.
        assert (kv_cache.num_committed_tokens, kv_cache.history_length) == (0, 32)

        assert mgr.request_prefix_snapshot(owner, 64)
        assert mgr.request_prefix_snapshot(owner, 96)
        run_chunk(64)
        assert (kv_cache.num_committed_tokens, kv_cache.history_length) == (64, 64)
        run_chunk(96)
        assert (kv_cache.num_committed_tokens, kv_cache.history_length) == (96, 96)
        run_chunk(100)
        assert kv_cache.num_committed_tokens == 96

        # Both waiters reuse their whole shared prefix.
        for probe, shared in zip(probes, (64, 96)):
            mgr.prepare_expect_snapshot_points([probe])
            assert mgr.prepare_context(probe)
            batch = ScheduledRequests()
            batch.append_context_request(probe)
            mgr.prepare_resources(batch)
            assert probe.prepopulated_prompt_len == shared
            mgr.free_resources(probe)
    finally:
        for request in (owner, *probes):
            if request.py_request_id in mgr.kv_cache_map:
                mgr.free_resources(request)
        mgr.shutdown()


def _branch_snapshot_manager(
    enable_branch_snapshot=True,
    enable_block_reuse=True,
    tokens_per_block=32,
    local_num_mamba_layers=1,
    periodic_snapshot_interval=0,
    additional_snapshot_offsets_from_end=None,
):
    """Bare V2 hybrid manager carrying only what branch snapshotting reads."""
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.enable_block_reuse = enable_block_reuse
    mgr.tokens_per_block = tokens_per_block
    mgr.local_num_mamba_layers = local_num_mamba_layers
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=enable_block_reuse,
        block_reuse_config=BlockReuseConfig(enable_branch_snapshot=enable_branch_snapshot),
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=periodic_snapshot_interval,
            additional_snapshot_offsets_from_end=list(additional_snapshot_offsets_from_end or []),
        ),
    )
    mgr._branch_snapshot_points = {}
    mgr._snapshot_pruned_tokens_total = 0
    mgr._page_pruned_tokens_total = 0
    mgr._branch_snapshots_taken_total = 0
    mgr._branch_snapshots_skipped_total = {}
    return mgr


def _fake_reuse_match(divergence, hybrid=None, reused=0):
    """Stand-in for a _KVCache exposing only the reuse-depth diagnostics."""
    return SimpleNamespace(
        _get_num_reusable_tokens_before_pruning=lambda: divergence,
        _get_num_reusable_tokens_before_hybrid_pruning=lambda: (
            divergence if hybrid is None else hybrid
        ),
        num_committed_tokens=reused,
    )


def _fake_context_request(
    rid=1,
    prompt_len=256,
    context_current_position=0,
    is_dummy_request=False,
    expect_snapshot_points=None,
    is_disagg_generation_init_state=False,
):
    return SimpleNamespace(
        py_request_id=rid,
        prompt_len=prompt_len,
        context_current_position=context_current_position,
        is_dummy_request=is_dummy_request,
        is_disagg_generation_init_state=is_disagg_generation_init_state,
        expect_snapshot_points=list(expect_snapshot_points or []),
    )


def test_branch_snapshot_uses_divergence_depth_not_page_pruned_depth():
    """The point tracks content divergence, not how far pages happen to survive.

    Attention pages have been evicted back to 64, but the fork is at 192. This
    request re-prefills through 192 either way, so snapshotting there is what
    lets a later sibling reuse the whole shared prefix.
    """
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(prompt_len=256, context_current_position=64)
    kv_cache = _fake_reuse_match(divergence=192, hybrid=64, reused=64)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)

    assert mgr._branch_snapshot_points == {1: 192}
    assert request.expect_snapshot_points == [192, 256]
    assert mgr._branch_snapshots_taken_total == 0
    assert mgr._branch_snapshots_skipped_total == {}


def test_branch_snapshot_point_is_aligned_down_to_tokens_per_block():
    mgr = _branch_snapshot_manager(tokens_per_block=64)
    request = _fake_context_request(prompt_len=512)
    kv_cache = _fake_reuse_match(divergence=200)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=511)

    assert mgr._branch_snapshot_points == {1: 192}
    assert request.expect_snapshot_points == [192, 512]


def test_branch_snapshot_skipped_when_whole_lookup_matched():
    """No fork: the request duplicates a cached prefix end to end."""
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(prompt_len=256, context_current_position=255)
    kv_cache = _fake_reuse_match(divergence=255, reused=255)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)

    assert mgr._branch_snapshot_points == {}
    assert mgr._branch_snapshots_taken_total == 0
    assert mgr._branch_snapshots_skipped_total == {"no_divergence": 1}


def test_branch_snapshot_skipped_when_reuse_already_reached_the_fork():
    """Reuse got there, so a snapshot at that depth already exists."""
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(prompt_len=256, context_current_position=96)
    kv_cache = _fake_reuse_match(divergence=100, hybrid=100, reused=96)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)

    assert mgr._branch_snapshot_points == {}
    assert mgr._branch_snapshots_skipped_total == {"already_reused": 1}


def test_branch_snapshot_uses_cache_depth_before_request_cursor_is_rewound():
    """A shallower retry must apply points hidden by the stale request cursor."""
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(prompt_len=256, context_current_position=192)
    kv_cache = _fake_reuse_match(divergence=128, reused=64)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)

    assert request.context_current_position == 192
    assert request.expect_snapshot_points == [128, 256]
    assert mgr._branch_snapshot_points == {1: 128}
    assert mgr._branch_snapshots_skipped_total == {}


def test_branch_snapshot_reapplies_after_resume_retry_with_stale_cursor():
    """A saved point must survive a failed resume and existing-cache retry."""
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(prompt_len=256, context_current_position=192)
    kv_cache = _fake_reuse_match(divergence=128, reused=64)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)
    request.expect_snapshot_points = []
    mgr._apply_branch_snapshot_point(request)
    assert request.expect_snapshot_points == [256]

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=None)

    assert request.context_current_position == 192
    assert request.expect_snapshot_points == [128, 256]
    assert mgr._branch_snapshot_points == {1: 128}
    assert mgr._branch_snapshots_skipped_total == {"no_fresh_match": 1}


def test_branch_snapshot_skipped_at_or_after_the_prompt_end():
    """The prompt end is snapshotted unconditionally, so this adds nothing."""
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(prompt_len=128)
    kv_cache = _fake_reuse_match(divergence=130)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)

    assert mgr._branch_snapshot_points == {}
    assert mgr._branch_snapshots_skipped_total == {"at_prompt_end": 1}


def test_branch_snapshot_skipped_when_divergence_aligns_away_to_zero():
    mgr = _branch_snapshot_manager(tokens_per_block=32)
    request = _fake_context_request(prompt_len=256)
    kv_cache = _fake_reuse_match(divergence=20)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)

    assert mgr._branch_snapshot_points == {}
    assert mgr._branch_snapshots_skipped_total == {"aligned_to_zero": 1}


def test_branch_snapshot_skipped_without_a_fresh_match():
    """A resumed cache was never matched, so it has no divergence depth."""
    mgr = _branch_snapshot_manager()
    request = _fake_context_request()
    kv_cache = _fake_reuse_match(divergence=192)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=None)

    assert mgr._branch_snapshot_points == {}
    assert mgr._branch_snapshots_skipped_total == {"no_fresh_match": 1}


def test_branch_snapshot_is_inert_when_the_flag_is_off():
    mgr = _branch_snapshot_manager(enable_branch_snapshot=False)
    request = _fake_context_request(prompt_len=256)
    kv_cache = _fake_reuse_match(divergence=192, hybrid=64, reused=64)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)
    mgr.prepare_expect_snapshot_points([request])

    assert mgr._branch_snapshot_points == {}
    assert mgr._branch_snapshots_taken_total == 0
    assert mgr._branch_snapshots_skipped_total == {}
    assert mgr._snapshot_pruned_tokens_total == 0
    assert mgr._page_pruned_tokens_total == 0
    # No prompt-end point either: the flag is what turns that on.
    assert request.expect_snapshot_points == []


def test_branch_snapshot_points_match_on_attention_only_pp_rank():
    mamba_mgr = _branch_snapshot_manager(local_num_mamba_layers=1)
    attention_mgr = _branch_snapshot_manager(local_num_mamba_layers=0)
    mamba_request = _fake_context_request(prompt_len=256)
    attention_request = _fake_context_request(prompt_len=256)
    kv_cache = _fake_reuse_match(divergence=192, hybrid=64, reused=64)

    mamba_mgr._record_branch_snapshot_point(mamba_request, kv_cache, num_lookup_tokens=255)
    attention_mgr._record_branch_snapshot_point(attention_request, kv_cache, num_lookup_tokens=255)

    assert mamba_request.expect_snapshot_points == [192, 256]
    assert attention_request.expect_snapshot_points == [192, 256]
    assert mamba_mgr._branch_snapshot_points == attention_mgr._branch_snapshot_points
    assert mamba_mgr._page_pruned_tokens_total == 192 - 64
    assert attention_mgr._snapshot_pruned_tokens_total == 0
    assert attention_mgr._page_pruned_tokens_total == 0


# A disagg generation request receives its prompt state and never prefills here.
@pytest.mark.parametrize("flag", ["is_dummy_request", "is_disagg_generation_init_state"])
def test_branch_snapshot_skipped_for_requests_without_prefill(flag):
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(prompt_len=256, **{flag: True})
    kv_cache = _fake_reuse_match(divergence=192, hybrid=64, reused=64)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)

    assert mgr._branch_snapshot_points == {}
    assert request.expect_snapshot_points == []
    assert mgr._branch_snapshots_taken_total == 0
    assert mgr._branch_snapshots_skipped_total == {}
    assert mgr._snapshot_pruned_tokens_total == 0


def test_branch_snapshot_point_survives_a_second_scheduler_pass():
    """prepare_expect_snapshot_points overwrites the list on every pass."""
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(prompt_len=256)
    kv_cache = _fake_reuse_match(divergence=192)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)
    assert request.expect_snapshot_points == [192, 256]

    mgr.prepare_expect_snapshot_points([request])
    assert request.expect_snapshot_points == [192, 256]

    mgr.prepare_expect_snapshot_points([request])
    assert request.expect_snapshot_points == [192, 256]


def test_branch_snapshot_drops_the_point_once_prefill_passes_it():
    """Past the fork, the point is no longer a reachable chunk boundary."""
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(prompt_len=256)
    kv_cache = _fake_reuse_match(divergence=192)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)
    request.context_current_position = 192

    mgr.prepare_expect_snapshot_points([request])

    assert request.expect_snapshot_points == [256]
    assert mgr._branch_snapshots_taken_total == 0


def test_branch_snapshot_taken_counts_successful_commit_once():
    mgr = _branch_snapshot_manager()
    mgr.is_draft = False
    mgr._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]
    request = _fake_context_request(prompt_len=256, context_current_position=64)
    request.context_remaining_length = 192
    request.get_tokens_view = lambda beam_idx: list(range(256))
    kv_cache = _fake_reuse_match(divergence=192, hybrid=64, reused=64)

    def commit(tokens):
        kv_cache.num_committed_tokens += len(tokens)

    kv_cache.commit = MagicMock(side_effect=commit)
    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)
    assert mgr._branch_snapshots_taken_total == 0

    request.context_current_position = 192
    request.context_remaining_length = 64
    mgr.try_commit_blocks(request, kv_cache)
    mgr.try_commit_blocks(request, kv_cache)

    kv_cache.commit.assert_called_once_with(list(range(64, 192)))
    assert mgr._branch_snapshots_taken_total == 1


def test_branch_snapshot_adds_prompt_end_without_configured_placement():
    """Keep branch snapshots usable when no baseline placement is configured.

    A branch point below the prompt end must not become the maximum, or the
    feature would stop committing at the branch point.
    """
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(prompt_len=256)
    kv_cache = _fake_reuse_match(divergence=192)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)

    assert max(request.expect_snapshot_points) == request.prompt_len


def test_branch_snapshot_preserves_configured_nonzero_end_offset():
    """Do not replace a same-block recurrent checkpoint before prompt end."""
    mgr = _branch_snapshot_manager(
        tokens_per_block=64,
        additional_snapshot_offsets_from_end=[2],
    )
    request = _fake_context_request(prompt_len=256)
    kv_cache = _fake_reuse_match(divergence=192)

    mgr.prepare_expect_snapshot_points([request])
    assert request.expect_snapshot_points == [254]

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)
    assert request.expect_snapshot_points == [192, 254]

    mgr.prepare_expect_snapshot_points([request])
    assert request.expect_snapshot_points == [192, 254]


def test_branch_snapshot_points_stay_sorted_and_deduplicated():
    mgr = _branch_snapshot_manager(
        periodic_snapshot_interval=64,
        additional_snapshot_offsets_from_end=[0, 64],
    )
    # 192 coincides with both a periodic point and prompt_len - 64.
    request = _fake_context_request(prompt_len=256)
    kv_cache = _fake_reuse_match(divergence=192)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)
    mgr.prepare_expect_snapshot_points([request])

    assert request.expect_snapshot_points == [64, 128, 192, 256]


def test_branch_snapshot_counters_separate_snapshot_loss_from_page_loss():
    """The two losses need different fixes, so they are counted apart."""
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(rid=5, prompt_len=1024)
    kv_cache = _fake_reuse_match(divergence=512, hybrid=320, reused=128)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=1023)

    # Matched in attention but pruned for want of a snapshot.
    assert mgr._snapshot_pruned_tokens_total == 320 - 128
    # Shared content whose attention pages were already evicted.
    assert mgr._page_pruned_tokens_total == 512 - 320


def test_branch_snapshot_counters_are_zero_when_nothing_was_pruned():
    mgr = _branch_snapshot_manager()
    request = _fake_context_request(prompt_len=256)
    kv_cache = _fake_reuse_match(divergence=192, hybrid=192, reused=192)

    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=255)

    assert mgr._snapshot_pruned_tokens_total == 0
    assert mgr._page_pruned_tokens_total == 0


def test_branch_snapshot_counters_accumulate_across_requests():
    mgr = _branch_snapshot_manager()
    for rid in (1, 2, 3):
        request = _fake_context_request(rid=rid, prompt_len=1024)
        mgr._record_branch_snapshot_point(
            request,
            _fake_reuse_match(divergence=512, hybrid=320, reused=128),
            num_lookup_tokens=1023,
        )

    assert mgr._branch_snapshots_taken_total == 0
    assert mgr._snapshot_pruned_tokens_total == 3 * (320 - 128)
    assert mgr._page_pruned_tokens_total == 3 * (512 - 320)
    assert mgr._branch_snapshot_points == {1: 512, 2: 512, 3: 512}


@skip_no_cuda
def test_branch_snapshot_entry_is_released_with_the_request():
    mgr = _build_v2_hybrid_with_mamba_layer(
        enable_block_reuse=True,
        additional_snapshot_offsets_from_end=[0],
        enable_branch_snapshot=True,
    )
    try:
        assert mgr.kv_cache_config.block_reuse_config.enable_branch_snapshot
        request = _fake_context_request(rid=11, prompt_len=128)
        mgr._record_branch_snapshot_point(
            request, _fake_reuse_match(divergence=64), num_lookup_tokens=127
        )
        assert mgr._branch_snapshot_points == {11: 64}

        mgr.free_resources(request)
        assert mgr._branch_snapshot_points == {}
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_branch_snapshot_table_is_cleared_on_shutdown():
    mgr = _build_v2_hybrid_with_mamba_layer(
        enable_block_reuse=True,
        enable_branch_snapshot=True,
    )
    request = _fake_context_request(rid=12, prompt_len=128)
    mgr._record_branch_snapshot_point(
        request, _fake_reuse_match(divergence=64), num_lookup_tokens=127
    )
    assert mgr._branch_snapshot_points == {12: 64}

    mgr.shutdown()
    assert mgr._branch_snapshot_points == {}


@skip_no_cuda
def test_branch_snapshot_coexists_with_per_conversation_and_zero_interval():
    """The flag is orthogonal to periodic snapshots, which stay disabled here."""
    mgr = _build_v2_hybrid_with_mamba_layer(
        enable_block_reuse=True,
        block_reuse_policy="per_conversation",
        periodic_snapshot_interval=64,
        additional_snapshot_offsets_from_end=[0],
        enable_branch_snapshot=True,
    )
    try:
        assert mgr.block_reuse_policy is BlockReusePolicy.PER_CONVERSATION
        assert mgr.kv_cache_config.mamba_state_config.periodic_snapshot_interval == 0
        assert mgr.kv_cache_config.block_reuse_config.enable_branch_snapshot
        assert mgr.tokens_per_block == 32

        request = _fake_context_request(rid=13, prompt_len=150)
        mgr._record_branch_snapshot_point(
            request, _fake_reuse_match(divergence=100), num_lookup_tokens=149
        )
        # 100 aligns down to 96 on a 32-token block.
        assert request.expect_snapshot_points == [96, 150]
    finally:
        mgr.shutdown()


def test_v2_hybrid_saves_conversation_plan_only_after_final_context_chunk():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.enable_block_reuse = True
    mgr.is_draft = False
    mgr.local_num_mamba_layers = 1
    mgr._branch_snapshot_points = {}
    mgr._branch_snapshots_taken_total = 0
    mgr.block_reuse_policy = BlockReusePolicy.PER_CONVERSATION
    events = []
    mgr._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]
    mgr._mark_context_position_as_history = MagicMock()
    mgr.conversation_manager = MagicMock()
    mgr.conversation_manager.save_drop_plan.side_effect = lambda request, kv_cache: events.append(
        "save"
    )

    request = SimpleNamespace(
        py_request_id=7,
        is_dummy_request=False,
        context_current_position=128,
        context_remaining_length=0,
        expect_snapshot_points=[128],
        prompt_len=128,
        is_last_context_chunk=True,
        get_tokens_view=lambda beam_idx: list(range(128)),
    )
    kv_cache = SimpleNamespace(
        is_active=True,
        num_committed_tokens=0,
        resize=MagicMock(return_value=True),
        enable_swa_scratch_reuse=True,
    )

    def commit(tokens):
        events.append("commit")
        kv_cache.num_committed_tokens += len(tokens)

    kv_cache.commit = MagicMock(side_effect=commit)
    kv_cache.stop_committing = MagicMock(side_effect=lambda: events.append("stop"))
    mgr.kv_cache_map = {request.py_request_id: kv_cache}
    batch = ScheduledRequests()
    batch.append_context_request(request)

    mgr.update_context_resources(batch)

    kv_cache.commit.assert_called_once_with(list(range(128)))
    kv_cache.stop_committing.assert_called_once_with()
    mgr.conversation_manager.save_drop_plan.assert_called_once_with(request, kv_cache)
    assert events == ["commit", "stop", "save"]
    assert not kv_cache.enable_swa_scratch_reuse


@skip_no_cuda
def test_v2_hybrid_mamba_state_views_use_logical_slots():
    mgr = _build_v2_hybrid_with_mamba_layer(max_batch_size=4, num_mamba_layers=2)
    try:
        assert len(mgr.all_ssm_states) == 2
        assert len(mgr.all_conv_states) == 2

        ssm_slots = mgr.all_ssm_states[0].shape[0]
        conv_slots = mgr.all_conv_states[0].shape[0]
        assert all(t.shape[0] == ssm_slots for t in mgr.all_ssm_states)
        assert all(t.shape[0] == conv_slots for t in mgr.all_conv_states)
        assert ssm_slots == conv_slots

        local_layer_ids = [mgr.layer_offsets[layer_id] for layer_id in mgr.mamba_pp_layers]
        for local_layer_idx, ssm_state, conv_state in zip(
            local_layer_ids, mgr.all_ssm_states, mgr.all_conv_states
        ):
            layer_id = LayerId(local_layer_idx)
            ssm_scale = mgr.impl.get_page_index_scale(layer_id, MambaRole.SSM_STATE)
            conv_scale = mgr.impl.get_page_index_scale(layer_id, MambaRole.CONV_STATE)
            assert ssm_state.stride(0) == ssm_state[0].numel() * ssm_scale
            assert conv_state.stride(0) == conv_state[0].numel() * conv_scale
            assert (
                ssm_state.shape[0]
                == (
                    mgr.impl.get_page_index_upper_bound(layer_id, MambaRole.SSM_STATE)
                    + ssm_scale
                    - 1
                )
                // ssm_scale
            )
            assert (
                conv_state.shape[0]
                == (
                    mgr.impl.get_page_index_upper_bound(layer_id, MambaRole.CONV_STATE)
                    + conv_scale
                    - 1
                )
                // conv_scale
            )

        mgr.add_dummy_requests([123, 456], token_nums=[8, 8], is_gen=False)
        indices = mgr.get_state_indices([123, 456], [False, False])
        assert all(0 <= index < ssm_slots for index in indices)
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_swa_scratch_keeps_ssm_placeholder_rows():
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=4,
        enable_swa_scratch_reuse=True,
    )
    try:
        request_id = 123
        mgr.add_dummy_requests([request_id], token_nums=[8], is_gen=False)
        block_offsets = torch.zeros(
            mgr.num_attention_op_pools,
            1,
            2,
            mgr.max_blocks_per_seq,
            dtype=torch.int32,
            device="cuda",
        )

        mgr.copy_batch_block_offsets(
            block_offsets,
            [request_id],
            beam_width=1,
            num_contexts=1,
            num_seqs=1,
        )
        torch.cuda.synchronize()

        assert mgr.num_attention_op_pools == mgr.num_local_layers
        assert mgr.kv_cache_pool_mapping.shape[0] == mgr.num_local_layers
        assert mgr.kv_cache_pool_pointers[0, 0].item() != 0
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_disagg_page_table_preserves_lifecycle_indices():
    mgr = _build_v2_hybrid_with_mamba_layer(max_batch_size=4, num_mamba_layers=2)
    try:
        page_table = build_page_table_from_manager(mgr)

        assert len(page_table.layer_groups) == len(mgr.impl.layer_grouping)
        assert isinstance(page_table.layer_groups[0], MambaLayerGroup)
        assert isinstance(page_table.layer_groups[1], AttentionLayerGroup)

        requests = mgr.add_dummy_requests([123], token_nums=[64], is_gen=False)
        assert len(requests) == 1
        attention_blocks = list(
            mgr.kv_cache_map[123].get_aggregated_page_indices(1, valid_only=True)
        )
        assert attention_blocks
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_disagg_page_table_uses_qwen3_next_conv_sections():
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=4,
        conv_state_layout="q_k_v",
    )
    try:
        page_table = build_page_table_from_manager(mgr)
        mamba_group = page_table.layer_groups[0]

        assert isinstance(mamba_group, MambaLayerGroup)
        d_conv_m1 = mgr.conv_state_shape[1]
        conv_elem_size = mgr.all_conv_states[0].element_size()
        conv_view = next(pv for pv in mamba_group.pool_views if pv.pool_role == MAMBA_CONV_ROLE)
        assert conv_view.section_bytes == [
            dim * d_conv_m1 * conv_elem_size for dim in mgr.conv_section_dims
        ]
        assert mgr.conv_section_dims == [8, 8, 32]
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_hybrid_intermediate_states_size_by_tokens_per_gen_step():
    spec_config = _make_wide_spec_config(max_draft_len=2, tokens_per_gen_step=5)
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=4,
        spec_config=spec_config,
    )
    try:
        assert mgr.intermediate_ssm_states.shape[2] == 5
        assert mgr.intermediate_conv_states.shape[2] == 5
        layer_cache = mgr.mamba_layer_cache(0)
        assert layer_cache.intermediate_ssm.data_ptr() == (
            mgr.intermediate_ssm_states[0].data_ptr()
        )
    finally:
        mgr.shutdown()


@skip_no_cuda
@pytest.mark.parametrize("max_tokens,byte_limit", [(512, 0), (None, 16 << 20), (512, 16 << 20)])
def test_v2_kda_replay_policy_allocates_logical_slot_caches(max_tokens, byte_limit):
    spec_config = MTPDecodingConfig(max_draft_len=2)
    policy = KDAReplayState(
        num_speculative_tokens=2,
    )
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=4,
        num_mamba_layers=2,
        spec_config=spec_config,
        conv_state_layout="q_k_v",
        mamba_d_conv=5,
        mamba_num_heads=6,
        mamba_n_groups=6,
        mamba_ssm_cache_dtype=torch.float32,
        speculative_state=policy,
        max_tokens=max_tokens,
        max_gpu_total_bytes=byte_limit,
    )
    try:
        assert mgr._kda_replay is policy
        assert policy.intermediate_indices.shape == (4,)
        assert not hasattr(mgr, "get_replay_state_update_metadata")

        native_bytes = mgr.impl.get_quota(GPU_LEVEL)
        assert mgr._get_external_cache_bytes(mgr.impl) == 0
        if byte_limit:
            assert native_bytes <= byte_limit
        assert mgr.kv_cache_manager_py_config.cache_tiers[0].quota <= native_bytes

        layer_cache = mgr.mamba_layer_cache(0)
        assert not hasattr(layer_cache, "intermediate_ssm")
        assert not hasattr(layer_cache, "intermediate_conv_window")
        cache_size = layer_cache.temporal.shape[0]
        assert layer_cache.kda_conv_q.shape == (cache_size, 48, 6)
        assert layer_cache.kda_conv_k.shape == (cache_size, 48, 6)
        assert layer_cache.kda_conv_v.shape == (cache_size, 48, 6)
        assert layer_cache.kda_conv_q.dtype is torch.bfloat16
        assert layer_cache.kda_conv_q.stride(-2) == 1
        assert layer_cache.kda_k_cache.shape == (cache_size, 2, 48)
        assert layer_cache.kda_g_cache.shape == (cache_size, 2, 48)
        assert layer_cache.kda_v_cache.shape == (cache_size, 2, 48)
        # Six bf16 heads are 12 bytes, so the row is padded to 16 (stride 8)
        # to keep every nested per-draft view 16-byte aligned for CuTe.
        assert layer_cache.kda_beta_cache.shape == (cache_size, 2, 6)
        assert layer_cache.kda_beta_cache.stride(-2) == 8
        for layer_idx in range(2):
            for slot_idx in range(cache_size):
                for draft_idx in range(2):
                    assert (
                        mgr.mamba_layer_cache(layer_idx)
                        .kda_beta_cache[slot_idx, draft_idx]
                        .data_ptr()
                        % _KDA_BETA_CACHE_ALIGNMENT_BYTES
                        == 0
                    )
        assert layer_cache.kda_k_cache.dtype is torch.bfloat16
        assert layer_cache.kda_g_cache.dtype is torch.float32
        assert layer_cache.kda_v_cache.dtype is torch.bfloat16
        assert layer_cache.kda_beta_cache.dtype is torch.bfloat16
        assert layer_cache.prev_num_accepted_tokens.data_ptr() == (
            policy.prev_num_accepted_tokens.data_ptr()
        )
        assert not hasattr(layer_cache, "intermediate_ssm")
        assert not hasattr(layer_cache, "intermediate_conv_window")
    finally:
        mgr.shutdown()


@pytest.mark.parametrize(
    ("spec_config", "conv_state_layout", "num_spec", "match"),
    [
        (None, "q_k_v", 2, "require speculative decoding"),
        (
            MTPDecodingConfig(max_draft_len=2),
            "q_k_v",
            1,
            "must match the draft length",
        ),
        (
            MTPDecodingConfig(max_draft_len=2),
            "x_b_c",
            2,
            "requires conv_state_layout='q_k_v'",
        ),
        (
            MTPDecodingConfig(max_draft_len=2),
            "q_k_v",
            2,
            "require equal",
        ),
    ],
)
def test_v2_kda_replay_policy_validates_configuration(
    spec_config,
    conv_state_layout,
    num_spec,
    match,
):
    policy = KDAReplayState(
        num_speculative_tokens=num_spec,
    )
    with pytest.raises(ValueError, match=match):
        _build_v2_hybrid_with_mamba_layer(
            spec_config=spec_config,
            conv_state_layout=conv_state_layout,
            speculative_state=policy,
        )


@skip_no_cuda
def test_v2_kda_replay_policy_supports_attention_only_pp_rank():
    policy = KDAReplayState(
        num_speculative_tokens=2,
    )
    mapping = Mapping(world_size=2, rank=1, tp_size=1, pp_size=2)
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=1,
        mapping=mapping,
        spec_config=MTPDecodingConfig(max_draft_len=2),
        conv_state_layout="q_k_v",
        mamba_n_groups=4,
        speculative_state=policy,
    )
    try:
        assert mgr.local_num_mamba_layers == 0
        assert mgr._kda_replay is policy
        assert policy.prev_num_accepted_tokens is None
    finally:
        mgr.shutdown()


@skip_no_cuda
@pytest.mark.parametrize("kda_replay", [True, False], ids=["kda_replay", "no_replay"])
def test_v2_kda_replay_refuses_kv_pool_rebalance(kda_replay):
    """The KDA replay caches and prev_num_accepted_tokens hold one entry per SSM slot the pool keeps at construction,
    and a KV pool rebalance can grow that pool past them, so the manager refuses enable_kv_pool_rebalance with them;
    without them the rebalance stays allowed."""
    if not kda_replay:
        _build_v2_hybrid_with_mamba_layer(enable_kv_pool_rebalance=True).shutdown()
        return
    with pytest.raises(ValueError, match="enable_kv_pool_rebalance=False"):
        _build_v2_hybrid_with_mamba_layer(
            spec_config=MTPDecodingConfig(max_draft_len=2),
            conv_state_layout="q_k_v",
            mamba_n_groups=4,
            mamba_ssm_cache_dtype=torch.float32,
            kda_replay_num_spec=2,
            enable_kv_pool_rebalance=True,
        )


def _v2_kda_draft_record_manager(kda_draft_records, enable_block_reuse):
    """A V2 hybrid manager stub for _build_cache_config: one KDA layer with the replay caches and one attention
    layer."""
    mgr = object.__new__(KimiK3HybridCacheManager)
    mgr.layer_specs = None
    mgr._generation_kv_capacity_headroom = 1
    mgr._has_cp_helix = False
    mgr.kv_cache_type = CacheTypeCpp.SELF
    mgr.head_dim_per_layer = [64, 64]
    mgr.pp_layers = [0, 1]
    mgr.mamba_pp_layers = [0]
    mgr._mamba_layer_mask = [True, False]
    # One 2 MiB grain per state slot, so the pool's slot count follows its grains (a Kimi K3 slot holds 27 MB).
    mgr.ssm_bytes = 2 << 20
    mgr.conv_bytes = 64 << 10
    mgr.max_attention_window_vec = [128, 128]
    mgr.max_batch_size = 2
    mgr.mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    mgr.max_seq_len = 128
    mgr.max_num_tokens = 128
    mgr.tokens_per_block = 32
    mgr.num_local_layers = 2
    mgr.local_num_mamba_layers = 1
    mgr._num_reserved_dummy_slots = 1
    mgr.dtype = DataType.HALF
    mgr.enable_swa_scratch_reuse = False
    mgr.enable_stats = False
    mgr.num_extra_kv_tokens = 0
    mgr.get_layer_bytes_per_token = lambda **kwargs: 8
    # Layer 1's 256-byte pages of 32 tokens (_base_attention_layer_configs; layer 0 becomes the SSM layer).
    mgr._attention_cache_bytes_per_token = lambda: 8
    mgr._kda_replay = KDAReplayState(2, draft_records=kda_draft_records)
    # The replay buffers count in each slot's state bytes (_mamba_state_bytes_per_slot); the drafts' records are,
    # per draft, vn, beta * k and the decay, fp32 [H, K] each (24 KiB here).
    mgr.ssm_state_shape = [8, 256, 256]
    mgr.conv_section_dims = [8 * 256] * 3
    mgr.conv_state_shape = [3 * 8 * 256, 3]
    mgr._state_layout = SimpleNamespace(
        mamba_pp_layers=tuple(mgr.mamba_pp_layers),
        ssm_state_shape=tuple(mgr.ssm_state_shape),
        conv_state_shape=tuple(mgr.conv_state_shape),
        conv_section_dims=tuple(mgr.conv_section_dims),
    )
    mgr.kv_cache_config = KvCacheConfig(
        avg_seq_len=64,
        enable_block_reuse=enable_block_reuse,
        enable_partial_reuse=False,
    )
    return mgr


def _ring_gate_manager(
    *,
    draft_len=3,
    head_dim=128,
    d_state=128,
    conv_dtype=torch.bfloat16,
    ssm_dtype=torch.bfloat16,
    conv_state_layout="q_k_v",
    mamba_pp_layers=(0,),
):
    return SimpleNamespace(
        spec_config=MTPDecodingConfig(max_draft_len=draft_len),
        _state_layout=SimpleNamespace(
            mamba_pp_layers=mamba_pp_layers,
            ssm_state_shape=(4, head_dim, d_state) if mamba_pp_layers else (),
            conv_state_dtype=conv_dtype,
            ssm_state_dtype=ssm_dtype,
            conv_state_layout=conv_state_layout,
        ),
    )


@pytest.mark.parametrize(
    ("env", "replay_enabled", "available", "overrides", "expected", "warns"),
    [
        ("1", True, True, {}, True, False),
        ("1", True, True, {"draft_len": 7}, True, False),
        (None, True, True, {}, False, False),
        ("1", False, True, {}, False, True),
        ("1", True, False, {}, False, True),
        ("1", True, True, {"ssm_dtype": torch.float32}, False, True),
        ("1", True, True, {"conv_dtype": torch.float16}, False, True),
        ("1", True, True, {"head_dim": 64}, False, True),
        ("1", True, True, {"draft_len": 2}, False, True),
        ("1", True, True, {"conv_state_layout": "x_b_c"}, False, True),
        ("force", True, True, {"mamba_pp_layers": ()}, False, False),
    ],
    ids=[
        "eligible_T4",
        "eligible_T8",
        "not_opted_in",
        "replay_off",
        "kernel_unavailable",
        "fp32_state",
        "fp16_model",
        "head_dim_64",
        "T3",
        "x_b_c_layout",
        "no_local_gdn_layers",
    ],
)
def test_gdn_flashinfer_ring_replay_gate(
    monkeypatch, env, replay_enabled, available, overrides, expected, warns
):
    if env is None:
        monkeypatch.delenv("TRTLLM_USE_GDN_FLASHINFER_REPLAY", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_USE_GDN_FLASHINFER_REPLAY", env)
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.fla.flashinfer_cached_replay."
        "is_flashinfer_cached_replay_available",
        lambda: available,
    )
    warning_log = MagicMock()
    monkeypatch.setattr("tensorrt_llm._torch.modules.fla.cache_manager.logger.warning", warning_log)

    assert _use_flashinfer_ring_replay(_ring_gate_manager(**overrides), replay_enabled) is expected
    assert warning_log.called is warns


@pytest.mark.parametrize("available", [True, False], ids=["eligible", "kernel_unavailable"])
def test_gdn_flashinfer_ring_replay_gate_force(monkeypatch, available):
    monkeypatch.setenv("TRTLLM_USE_GDN_FLASHINFER_REPLAY", "force")
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.fla.flashinfer_cached_replay."
        "is_flashinfer_cached_replay_available",
        lambda: available,
    )
    if available:
        assert _use_flashinfer_ring_replay(_ring_gate_manager(), True) is True
    else:
        with pytest.raises(RuntimeError, match="force"):
            _use_flashinfer_ring_replay(_ring_gate_manager(), True)


def test_gdn_flashinfer_ring_replay_gate_force_replay_disabled(monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_GDN_FLASHINFER_REPLAY", "force")
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.fla.flashinfer_cached_replay."
        "is_flashinfer_cached_replay_available",
        lambda: True,
    )
    with pytest.raises(RuntimeError, match="requires GDN replay"):
        _use_flashinfer_ring_replay(_ring_gate_manager(), False)


@skip_no_cuda
def test_v2_gdn_ring_replay_allocates_rings_instead_of_double_buffer():
    policy = GDNReplayState(tokens_per_gen_step=4, use_flashinfer_ring=True)
    mgr = _build_v2_hybrid_with_mamba_layer(
        num_mamba_layers=2,
        spec_config=MTPDecodingConfig(max_draft_len=3),
        conv_state_layout="q_k_v",
        speculative_state=policy,
    )
    ring_size = FLASHINFER_REPLAY_RING_SIZE
    try:
        assert mgr.use_gdn_flashinfer_ring_replay
        assert not mgr.use_gdn_cached_replay_all_layer_commit
        assert policy.old_x is None and policy.old_B is None
        assert policy.old_dt is None and policy.old_dA_cumsum is None
        nheads, head_dim, d_state = mgr.ssm_state_shape
        layers, slots = mgr.local_num_mamba_layers, policy.cache_base.shape[0]
        n_groups = mgr._state_layout.n_groups_per_rank
        assert policy.k_ring.shape == (layers, slots, n_groups, ring_size, d_state)
        assert policy.u_ring.shape == (layers, slots, nheads, ring_size, head_dim)
        assert policy.g_ring.shape == (layers, slots, nheads, ring_size)
        assert policy.g_ring.dtype == torch.float32
        for t in (policy.cache_base, policy.k_ring, policy.u_ring, policy.g_ring):
            assert not t.any()

        layer_cache = mgr.mamba_layer_cache(1)
        offset = mgr.mamba_layer_offsets[1]
        assert layer_cache.old_x is None
        assert layer_cache.cache_base is policy.cache_base
        assert layer_cache.k_ring.data_ptr() == policy.k_ring[offset].data_ptr()
        assert layer_cache.u_ring.data_ptr() == policy.u_ring[offset].data_ptr()
        assert layer_cache.g_ring.data_ptr() == policy.g_ring[offset].data_ptr()
    finally:
        mgr.shutdown()
    assert policy.cache_base is None and policy.k_ring is None
    assert policy.u_ring is None and policy.g_ring is None


def test_v2_gdn_ring_replay_update_advances_ring_start():
    policy = GDNReplayState(tokens_per_gen_step=4, use_flashinfer_ring=True)
    policy.prev_num_accepted_tokens = torch.tensor([13, 13, 2, 13, 13], dtype=torch.int32)
    policy.cache_buf_idx = torch.zeros(5, dtype=torch.int32)
    policy.cache_base = torch.tensor([5, 20, 30, 7, 9], dtype=torch.int32)

    # Slot 0 holds a context request, slot 3 a dummy and slot 4 CUDA-graph padding.
    generation_slots = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    policy.update(
        MambaAcceptanceBatch(
            attention_metadata=None,
            num_contexts=1,
            num_generations=4,
            num_accepted_tokens=torch.tensor([3, 2, 3, 3], dtype=torch.int32),
            accepted_positions=torch.tensor([2, 1, 2, 2], dtype=torch.int32),
            source_state_indices=generation_slots,
            destination_state_indices=generation_slots,
            is_dummy_request=torch.tensor([False, False, True, True]),
        )
    )

    # Slot 1 flushed (13 + 4 > 16): its start moves past the 13 folded rows and
    # wraps, (20 + 13) % 32 = 1. Slot 2 only appended. Others are untouched.
    assert policy.cache_base.tolist() == [5, 1, 30, 7, 9]
    assert policy.prev_num_accepted_tokens.tolist() == [13, 3, 4, 13, 13]


def test_v2_gdn_ring_replay_resets_context_slots():
    policy = GDNReplayState(tokens_per_gen_step=4, use_flashinfer_ring=True)
    policy.prev_num_accepted_tokens = torch.tensor([5, 6, 7], dtype=torch.int32)
    policy.cache_buf_idx = torch.ones(3, dtype=torch.int32)
    policy.cache_base = torch.tensor([5, 6, 7], dtype=torch.int32)
    ring_size = FLASHINFER_REPLAY_RING_SIZE
    policy.k_ring = torch.ones(2, 3, 1, ring_size, 4)
    policy.u_ring = torch.ones(2, 3, 2, ring_size, 4)
    policy.g_ring = torch.ones(2, 3, 2, ring_size)

    policy.reset_slots(torch.tensor([2]), [2])

    assert policy.prev_num_accepted_tokens.tolist() == [5, 6, 0]
    assert policy.cache_base.tolist() == [5, 6, 0]
    for ring in (policy.k_ring, policy.u_ring, policy.g_ring):
        assert not ring[:, 2].any()  # the new request's slot, in every layer
        assert ring[:, :2].all()  # other slots untouched


def test_mamba_cache_manager_delegates_kda_replay_capability() -> None:
    mgr = object.__new__(MambaCacheManager)
    mgr._impl = SimpleNamespace(use_kda_replay_update=True)

    assert mgr.use_kda_replay_update


def test_v2_kda_replay_policy_records_acceptance_and_skips_dummy_rows():
    mgr = object.__new__(KimiK3HybridCacheManager)
    mgr.local_num_mamba_layers = 1
    policy = KDAReplayState(
        num_speculative_tokens=2,
    )
    policy.prev_num_accepted_tokens = torch.tensor([7, 8, 9], dtype=torch.int32)
    policy.intermediate_indices = torch.arange(3, dtype=torch.int32)
    mgr._dummy_request_mask_buffer = DeviceCopyBuffer(torch.tensor([False, False, True]))
    mgr._kda_replay = policy
    mgr._speculative_state = policy

    mgr._generation_state_indices = torch.arange(64, dtype=torch.int32)
    if not hasattr(mgr, "_dummy_request_mask"):
        mgr._dummy_request_mask_buffer = DeviceCopyBuffer(torch.zeros(64, dtype=torch.bool))
    mgr.update_mamba_states(
        SimpleNamespace(num_seqs=3, num_contexts=1),
        torch.tensor([1, 3, 2], dtype=torch.int32),
        state_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    assert policy.prev_num_accepted_tokens.tolist() == [7, 2, 9]


def test_legacy_mamba_cache_manager_delegates_kda_replay_capability() -> None:
    mgr = object.__new__(MambaCacheManager)
    mgr._impl = SimpleNamespace(use_kda_replay_update=True)

    assert mgr.use_kda_replay_update


def test_v2_kda_replay_policy_resets_context_slots():
    mgr = object.__new__(KimiK3HybridCacheManager)
    policy = KDAReplayState(
        num_speculative_tokens=2,
    )
    policy.prev_num_accepted_tokens = torch.tensor([5, 6, 7], dtype=torch.int32)
    mgr._kda_replay = policy
    mgr._speculative_state = policy
    mgr._state_indices_buffer = DeviceCopyBuffer(torch.tensor([2, 1], dtype=torch.int32))
    mgr._host_state_indices = torch.tensor([2, 1], dtype=torch.int32)

    mgr._host_state_indices = mgr.cuda_state_indices.clone()
    mgr._reset_context_mamba_slots(num_contexts=1)

    assert policy.prev_num_accepted_tokens.tolist() == [5, 6, 0]


def test_v2_kda_replay_host_drafter_records_active_requests(monkeypatch):
    mgr = object.__new__(KimiK3HybridCacheManager)
    mgr.local_num_mamba_layers = 1
    mgr.spec_config = SimpleNamespace(decoding_type="NGram")
    mgr._request_id_to_state_index = {101: 1, 303: 2}
    mgr._request_id_to_is_dummy = {101: False, 303: True}
    policy = KDAReplayState(
        num_speculative_tokens=2,
    )
    policy.prev_num_accepted_tokens = torch.tensor([7, 8, 9], dtype=torch.int32)
    mgr._kda_replay = policy
    mgr._speculative_state = policy
    policy.acceptance_pending = {101, 202, 404}
    base_update = MagicMock()
    monkeypatch.setattr(KVCacheManagerV2, "update_resources", base_update)
    active = SimpleNamespace(
        py_request_id=101,
        py_draft_tokens=[11, 12],
        py_num_accepted_draft_tokens=2,
    )
    completed = SimpleNamespace(
        py_request_id=202,
        py_draft_tokens=[21, 22],
        py_num_accepted_draft_tokens=1,
    )
    dummy = SimpleNamespace(
        py_request_id=303,
        py_draft_tokens=[31, 32],
        py_num_accepted_draft_tokens=1,
    )
    scheduled_batch = SimpleNamespace(generation_requests=[active, completed, dummy])

    mgr.update_resources(scheduled_batch)

    base_update.assert_called_once_with(scheduled_batch, None, None)
    assert policy.prev_num_accepted_tokens.tolist() == [7, 2, 9]
    # The batch's acceptances are recorded; another batch's are still pending.
    assert policy.acceptance_pending == {404}


_KDA_REPLAY_BUFFERS = (
    "kda_conv_q",
    "kda_conv_k",
    "kda_conv_v",
    "kda_k_cache",
    "kda_g_cache",
    "kda_v_cache",
    "kda_beta_cache",
)


class _SlotCache:
    """A V2 KV cache stand-in that reports one SSM slot and can be suspended."""

    def __init__(self, slot: int) -> None:
        self.slot = slot
        self.is_active = True

    def get_ssm_block_base_index(self, _layer_group_id) -> int:
        return self.slot

    def suspend(self) -> None:
        self.is_active = False


def _v2_kda_replay_stub(num_slots=4, manager_cls=KimiK3HybridCacheManager, draft_records=False):
    """A V2 hybrid manager stub with CPU KDA replay buffers and one layer's V2 conv states ([Q | K | V] sections of 2
    channels, a committed window of 3), for the state-index setup that prepare_resources runs. Committing pending drafts
    is recorded instead of run (it needs the GPU kernels)."""
    mgr = object.__new__(manager_cls)
    mgr.local_num_mamba_layers = 1
    mgr._kda_replay = KDAReplayState(2, draft_records=draft_records)
    mgr._speculative_state = mgr._kda_replay
    mgr.spec_config = SimpleNamespace(decoding_type="MTP")
    mgr._stream = None
    mgr._request_id_to_state_index = {}
    mgr._request_id_to_is_dummy = {}
    mgr._kda_replay.suspended_requests = set()
    mgr._kda_replay.acceptance_pending = set()
    mgr._kda_replay.unseeded_requests = set()
    mgr._host_state_indices = torch.zeros(num_slots, dtype=torch.int32)
    mgr._state_indices_buffer = DeviceCopyBuffer(torch.zeros(num_slots, dtype=torch.int32))
    mgr._dummy_request_mask_buffer = DeviceCopyBuffer(torch.zeros(num_slots, dtype=torch.bool))
    mgr._dummy_request_mask_host = torch.zeros(num_slots, dtype=torch.bool)
    mgr.kv_cache_map = {}
    mgr.ssm_layer_group_id = 0
    mgr._kda_replay.prev_num_accepted_tokens = torch.zeros(num_slots, dtype=torch.int32)
    for name in _KDA_REPLAY_BUFFERS:
        setattr(mgr._kda_replay, name, torch.zeros(1, num_slots, 2, 4))
    if draft_records:
        for name in _KDA_REPLAY_BUFFERS[3:]:
            setattr(mgr._kda_replay, name, None)
        mgr._kda_replay.kda_state_tok = torch.zeros(1, num_slots, 3, 2, 1, 2)
    mgr.conv_state_shape = [6, 3]
    mgr.conv_section_dims = [2, 2, 2]
    mgr.all_conv_states = [torch.zeros(num_slots, 6, 3)]
    mgr._kda_replay._conv_states = tuple(mgr.all_conv_states)
    mgr._kda_replay._committed_window = 3
    mgr._kda_replay._conv_section_dims = (2, 2, 2)
    mgr.committed_slots = []
    mgr._kda_replay.commit_pending_drafts = mgr.committed_slots.append
    mgr._kda_replay.on_cache_stream = lambda stream: contextlib.nullcontext()
    return mgr


def _write_kda_replay_slot(mgr, slot, accepted, value):
    """What a verify step leaves in its request's slot for the next one."""
    mgr._kda_replay.prev_num_accepted_tokens[slot] = accepted
    for buffer in mgr._kda_replay._layer_replay_buffers():
        buffer[:, slot] = value


def _kda_replay_slot(mgr, slot):
    return int(mgr._kda_replay.prev_num_accepted_tokens[slot]), [
        float(buffer[0, slot].flatten()[0]) for buffer in mgr._kda_replay._layer_replay_buffers()
    ]


@pytest.mark.parametrize("resume_slot", [1, 0], ids=["new_slot", "old_slot"])
@pytest.mark.parametrize(
    "manager_cls, draft_records",
    [
        (KimiK3HybridCacheManager, False),
        (KimiK3HybridCacheManager, True),
        (Glm5NextCacheManager, False),
    ],
)
def test_v2_kda_replay_suspended_request_resumes_from_its_v2_states(
    resume_slot, manager_cls, draft_records
):
    """A suspend commits the request's pending drafts and conv window into its slot's V2 states and forgets the slot,
    which can then go to another request. The request resumes in a new slot, or in its old one after that request has
    finished. Either way its replay caches are seeded from the V2 conv state it resumes with, nothing is pending, and
    the other request's replay state is left alone."""
    mgr = _v2_kda_replay_stub(manager_cls=manager_cls, draft_records=draft_records)
    first = SimpleNamespace(py_request_id=1, is_dummy=False)
    second = SimpleNamespace(py_request_id=2, is_dummy=False)
    mgr.kv_cache_map[1] = _SlotCache(0)
    mgr._setup_state_indices([first], num_contexts=0)
    _write_kda_replay_slot(mgr, 0, accepted=2, value=1.0)
    mgr.suspend_request(first)
    assert mgr.committed_slots == [0]
    assert 1 not in mgr._request_id_to_state_index
    assert not mgr.kv_cache_map[1].is_active

    mgr.kv_cache_map[2] = _SlotCache(0)
    mgr._setup_state_indices([second], num_contexts=1)
    mgr._reset_context_mamba_slots(1)
    mgr._setup_state_indices([second], num_contexts=0)
    _write_kda_replay_slot(mgr, 0, accepted=3, value=7.0)
    resumed = [first]
    if resume_slot == 0:
        del mgr.kv_cache_map[2]
        del mgr._request_id_to_state_index[2]
    else:
        resumed.append(second)
    # The V2 conv state the runtime brings back with the request's page.
    window = torch.arange(18.0).view(6, 3)
    mgr.all_conv_states[0][resume_slot] = window
    mgr.kv_cache_map[1].slot = resume_slot
    mgr.kv_cache_map[1].is_active = True
    mgr._setup_state_indices(resumed, num_contexts=0)

    assert int(mgr._kda_replay.prev_num_accepted_tokens[resume_slot]) == 0
    for section, name in enumerate(("kda_conv_q", "kda_conv_k", "kda_conv_v")):
        cache = getattr(mgr._kda_replay, name)[0, resume_slot]
        assert torch.equal(cache[:, :3], window[2 * section : 2 * section + 2])
        assert not cache[:, 3:].any()
    for name in ("kda_k_cache", "kda_g_cache", "kda_v_cache", "kda_beta_cache"):
        buffer = getattr(mgr._kda_replay, name)
        if buffer is not None:
            assert not buffer[0, resume_slot].any()
    if draft_records:
        # A resumed slot has no pending drafts, so its old records remain unread.
        expected_records = 7.0 if resume_slot == 0 else 0.0
        assert torch.all(mgr._kda_replay.kda_state_tok[0, resume_slot] == expected_records)
    if resume_slot != 0:
        assert _kda_replay_slot(mgr, 0) == (3, [7.0] * len(mgr._kda_replay._layer_replay_buffers()))
    assert not mgr._kda_replay.suspended_requests


@pytest.mark.parametrize("recorded_by", ["update_resources", "revert_allocate_generation"])
@pytest.mark.parametrize("manager_cls", [KimiK3HybridCacheManager, Glm5NextCacheManager])
def test_v2_kda_replay_refuses_a_suspend_before_the_host_drafter_acceptance(
    monkeypatch, recorded_by, manager_cls
):
    """With a host drafter, update_resources records the acceptance of a batch's verify, and a suspend needs it to
    commit the request's pending drafts. A suspend between the batch's prepare_resources and its update_resources
    raises instead of committing a stale count. Once update_resources has recorded it, or the batch is reverted without
    running, the suspend commits."""
    mgr = _v2_kda_replay_stub(manager_cls=manager_cls)
    mgr.spec_config = SimpleNamespace(decoding_type="NGram")
    for method in ("prepare_resources", "update_resources", "revert_allocate_generation"):
        monkeypatch.setattr(KVCacheManagerV2, method, MagicMock())
    request = SimpleNamespace(
        py_request_id=1, is_dummy=False, py_draft_tokens=[11, 12], py_num_accepted_draft_tokens=2
    )
    mgr.kv_cache_map[1] = _SlotCache(0)
    batch = SimpleNamespace(context_requests=[], generation_requests=[request])
    mgr.prepare_resources(batch)

    with pytest.raises(RuntimeError, match="acceptance"):
        mgr.suspend_request(request)
    assert not mgr.committed_slots
    assert mgr.kv_cache_map[1].is_active

    if recorded_by == "update_resources":
        mgr.update_resources(batch)
        assert int(mgr._kda_replay.prev_num_accepted_tokens[0]) == 2
    else:
        mgr.revert_allocate_generation(request)
    mgr.suspend_request(request)
    assert mgr.committed_slots == [0]
    assert not mgr.kv_cache_map[1].is_active


@pytest.mark.parametrize("seeded", [False, True], ids=["before_seeding", "after_seeding"])
@pytest.mark.parametrize("manager_cls", [KimiK3HybridCacheManager, Glm5NextCacheManager])
def test_v2_kda_replay_disagg_generation_request_suspends_its_transferred_state(
    monkeypatch, seeded, manager_cls
):
    mgr = _v2_kda_replay_stub(manager_cls=manager_cls)
    monkeypatch.setattr(KVCacheManagerV2, "prepare_resources", MagicMock())
    request = SimpleNamespace(py_request_id=1, is_dummy=False, is_disagg_generation_init_state=True)
    mgr.kv_cache_map[1] = _SlotCache(0)
    _write_kda_replay_slot(mgr, 0, accepted=0, value=5.0)
    mgr.prepare_resources(SimpleNamespace(context_requests=[request], generation_requests=[]))
    request.is_disagg_generation_init_state = False
    window = torch.arange(18.0).view(6, 3)
    mgr.all_conv_states[0][0] = window
    if seeded:
        mgr.seed_kda_replay_caches_for_disagg_gen([1])

    mgr.suspend_request(request)
    assert mgr.committed_slots == ([0] if seeded else [])
    assert 1 not in mgr._request_id_to_state_index
    assert not mgr.kv_cache_map[1].is_active

    mgr.kv_cache_map[1].slot = 2
    mgr.kv_cache_map[1].is_active = True
    mgr.all_conv_states[0][2] = window
    mgr._setup_state_indices([request], num_contexts=0)
    for section, name in enumerate(("kda_conv_q", "kda_conv_k", "kda_conv_v")):
        assert torch.equal(
            getattr(mgr._kda_replay, name)[0, 2][:, :3], window[2 * section : 2 * section + 2]
        )
    assert not mgr._kda_replay.unseeded_requests

    mgr.suspend_request(request)
    assert mgr.committed_slots == ([0, 2] if seeded else [2])


def _v2_kda_replay_manager_with_large_slots():
    """A real V2 hybrid manager at Kimi K3's per-rank KDA shapes (6 heads, K = V = 128, conv width 4) on 6 layers. An
    SSM slot (~2.25 MiB) is larger than the 2 MiB allocation grain, so the SSM pool holds a handful of slots and runs
    out before the attention pages do. max_util_for_resume 1.0 lets resume admit caches until the pool is full."""
    num_mamba_layers = 6
    return KimiK3HybridCacheManager(
        mamba_d_state=128,
        mamba_d_conv=4,
        mamba_num_heads=6,
        mamba_n_groups=6,
        mamba_head_dim=128,
        mamba_num_layers=num_mamba_layers,
        mamba_layer_mask=[True] * num_mamba_layers + [False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(
            max_tokens=4096, enable_block_reuse=False, max_util_for_resume=1.0
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        spec_config=MTPDecodingConfig(max_draft_len=2),
        layer_mask=[False] * num_mamba_layers + [True],
        vocab_size=1024,
        conv_state_layout="q_k_v",
        kda_replay_num_spec=2,
    )


def _take_ssm_slot(mgr, slot, others):
    """Create caches until the V2 runtime gives one of them ``slot``, moving the suspended owner's SSM page to the host
    tier; keep that one and close the rest."""
    holder = None
    for _ in range(mgr._kda_replay.prev_num_accepted_tokens.shape[0] + 1):
        cache = mgr.impl.create_kv_cache()
        assert cache.resume(mgr._stream.cuda_stream)
        others.append(cache)
        assert cache.resize(1)
        if cache.get_ssm_block_base_index(mgr.ssm_layer_group_id) == slot:
            holder = cache
            break
    assert holder is not None, "no cache took the suspended request's SSM slot"
    for cache in others:
        if cache is not holder:
            cache.close()
    others[:] = [holder]


def _fill_kda_slot(mgr, slot, generator):
    """Random V2 states and replay state at ``slot``: a new owner's prefill and verify rounds."""
    for layer_offset, (ssm_states, conv_states) in enumerate(
        zip(mgr.all_ssm_states, mgr.all_conv_states)
    ):
        ssm_states[slot] = torch.randn(ssm_states.shape[1:], generator=generator, device="cuda")
        conv_states[slot] = torch.randn(conv_states.shape[1:], generator=generator, device="cuda")
        for name in _KDA_REPLAY_BUFFERS:
            buffer = mgr._kda_replay.pool_layers[layer_offset][name]
            buffer[slot] = torch.randn(buffer.shape[1:], generator=generator, device="cuda")
    mgr._kda_replay.prev_num_accepted_tokens[slot] = 1


def _kda_verify_round(mgr, slot, layer_weights, round_idx, num_spec):
    """One verify of 1 + num_spec tokens on every KDA layer for the request in ``slot``, as the KDA mixer runs it; the
    inputs depend only on the round. Returns the outputs."""
    tokens = 1 + num_spec
    pending = mgr._kda_replay.prev_num_accepted_tokens[slot : slot + 1].clone()
    slots = torch.tensor([slot], dtype=torch.int32, device="cuda")
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    cu_seqlens[:1].sub_(pending)
    outputs = []
    for layer_offset, weights in enumerate(layer_weights):
        replay = mgr._kda_replay.pool_layers[layer_offset]
        generator = torch.Generator(device="cuda").manual_seed(1000 * round_idx + layer_offset)
        num_heads, head_dim = weights["A_log"].shape[0], 128

        def inputs(*shape):
            return torch.randn(shape, generator=generator, device="cuda").bfloat16()

        outputs.append(
            torch.ops.trtllm.kda_mtp_decode(
                x_q=inputs(1, tokens, num_heads, head_dim),
                x_k=inputs(1, tokens, num_heads, head_dim),
                x_v=inputs(1, tokens, num_heads, head_dim),
                w_q=weights["w_q"],
                w_k=weights["w_k"],
                w_v=weights["w_v"],
                cs_q=replay["kda_conv_q"],
                cs_k=replay["kda_conv_k"],
                cs_v=replay["kda_conv_v"],
                g=inputs(1, tokens, num_heads, head_dim),
                beta=inputs(1, tokens, num_heads),
                A_log=weights["A_log"],
                dt_bias=weights["dt_bias"],
                recurrent_state=mgr.all_ssm_states[layer_offset],
                k_cache=replay["kda_k_cache"],
                g_cache=replay["kda_g_cache"],
                v_cache=replay["kda_v_cache"],
                beta_cache=replay["kda_beta_cache"],
                ssm_state_indices=slots,
                cu_seqlens=cu_seqlens,
                num_spec=num_spec,
                num_accepted_tokens=pending,
                lower_bound=-5.0,
                scale=head_dim**-0.5,
            )
        )
    return torch.stack(outputs)


def _run_kda_verify_rounds(accepted_per_round, suspend_after=None, slot_taken=False):
    """Verify rounds of one request on a V2 manager at Kimi K3's per-rank KDA shapes, each accepting the given number
    of drafts. With ``suspend_after``, the request is suspended after that round's acceptance and resumed, its slot
    taken and overwritten by another cache meanwhile when ``slot_taken``. Returns every round's outputs and the
    request's final SSM states."""
    num_spec = 2
    mgr = _v2_kda_replay_manager_with_large_slots()
    others = []
    request = None
    try:
        (request,) = mgr.add_dummy_requests(
            [1], token_nums=[8], is_gen=True, max_num_draft_tokens=num_spec
        )
        generator = torch.Generator(device="cuda").manual_seed(7)
        num_heads, head_dim, width = 6, 128, 4
        layer_weights = [
            {
                "w_q": torch.randn(num_heads * head_dim, width, generator=generator, device="cuda")
                * 0.5,
                "w_k": torch.randn(num_heads * head_dim, width, generator=generator, device="cuda")
                * 0.5,
                "w_v": torch.randn(num_heads * head_dim, width, generator=generator, device="cuda")
                * 0.5,
                "A_log": torch.randn(num_heads, generator=generator, device="cuda"),
                "dt_bias": torch.randn(num_heads * head_dim, generator=generator, device="cuda"),
            }
            for _ in mgr.all_ssm_states
        ]
        # The state a prefill leaves: the V2 states, and the replay caches seeded from them with nothing pending.
        slot = mgr._request_id_to_state_index[1]
        for ssm_states, conv_states in zip(mgr.all_ssm_states, mgr.all_conv_states):
            ssm_states[slot] = 0.1 * torch.randn(
                ssm_states.shape[1:], generator=generator, device="cuda"
            )
            conv_states[slot] = torch.randn(
                conv_states.shape[1:], generator=generator, device="cuda"
            )
        mgr.seed_kda_replay_caches_for_disagg_gen([1])

        outputs = []
        for round_idx, accepted in enumerate(accepted_per_round):
            slot = mgr._request_id_to_state_index[1]
            outputs.append(_kda_verify_round(mgr, slot, layer_weights, round_idx, num_spec))
            mgr._kda_replay.prev_num_accepted_tokens[slot] = accepted
            if round_idx == suspend_after:
                mgr.suspend_request(request)
                if slot_taken:
                    _take_ssm_slot(mgr, slot, others)
                    _fill_kda_slot(mgr, slot, generator)
                assert mgr.resume_request(request)
                mgr._setup_state_indices([request], num_contexts=0)
                if slot_taken:
                    assert mgr._request_id_to_state_index[1] != slot
        slot = mgr._request_id_to_state_index[1]
        states = torch.stack([ssm_states[slot] for ssm_states in mgr.all_ssm_states])
        torch.cuda.synchronize()
        return outputs, states
    finally:
        for cache in others:
            cache.close()
        if request is not None:
            mgr.free_resources(request)
        mgr.shutdown()


@skip_no_cuda
@pytest.mark.parametrize("slot_taken", [True, False], ids=["slot_taken", "slot_kept"])
def test_v2_kda_replay_suspend_resume_matches_an_unsuspended_run(slot_taken):
    """A request is suspended with two accepted drafts pending and resumes; every later verify gives the outputs and
    leaves the states of the same request never suspended. The verifies up to the suspend match bit for bit. The
    suspend commits the pending drafts as a chunk of their own, and the tensor-core verify computes each chunk in one
    WY-form pass, so the later ones match to the kernel's accuracy rather than bit for bit.

    slot_taken: while the request is suspended, new caches fill the SSM pool, so the V2 runtime moves its SSM page to
    the host tier and gives its slot to one of them, which overwrites the slot's states; the request resumes in
    another slot. slot_kept: nothing else runs while it is suspended."""
    from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import is_kda_mtp_verify_available

    if not is_kda_mtp_verify_available():
        pytest.skip("needs the fused KDA verify kernel (sm_100 / sm_103 with the CuTe DSL)")
    accepted_per_round = [2, 1, 2, 0, 2, 1]
    suspend_after = 2
    expected_outputs, expected_states = _run_kda_verify_rounds(accepted_per_round)
    outputs, states = _run_kda_verify_rounds(
        accepted_per_round, suspend_after=suspend_after, slot_taken=slot_taken
    )

    for round_idx, (actual, expected) in enumerate(zip(outputs, expected_outputs)):
        if round_idx <= suspend_after:
            assert torch.equal(actual, expected), f"verify round {round_idx}"
        else:
            torch.testing.assert_close(
                actual, expected, atol=2e-2, rtol=2e-2, msg=f"verify round {round_idx}"
            )
    torch.testing.assert_close(states, expected_states, atol=1e-3, rtol=1e-3)


def test_v2_kda_replay_policy_seeds_disaggregated_generation_slots():
    policy = KDAReplayState(
        num_speculative_tokens=2,
    )
    policy._conv_section_dims = (2, 2, 2)
    policy.prev_num_accepted_tokens = torch.full((4,), 5, dtype=torch.int32)

    def dim_contiguous_conv_cache() -> torch.Tensor:
        return torch.full((2, 4, 6, 2), 7.0).transpose(-1, -2)

    policy.kda_conv_q = dim_contiguous_conv_cache()
    policy.kda_conv_k = dim_contiguous_conv_cache()
    policy.kda_conv_v = dim_contiguous_conv_cache()
    policy.kda_k_cache = torch.full((2, 4, 2, 2), 7.0, dtype=torch.bfloat16)
    policy.kda_g_cache = torch.full((2, 4, 2, 2), 7.0)
    policy.kda_v_cache = torch.full((2, 4, 2, 2), 7.0, dtype=torch.bfloat16)
    policy.kda_beta_cache = torch.full((2, 4, 2, 1), 7.0, dtype=torch.bfloat16)
    manager = SimpleNamespace(
        conv_state_shape=[6, 4],
        _request_id_to_state_index={101: 1, 202: 3},
        all_conv_states=[
            torch.arange(4 * 6 * 4, dtype=torch.float32).reshape(4, 6, 4),
            torch.arange(4 * 6 * 4, dtype=torch.float32).reshape(4, 6, 4) + 1000,
        ],
    )

    policy._conv_states = tuple(manager.all_conv_states)
    policy._committed_window = manager.conv_state_shape[1]
    policy.seed_transferred_slots(torch.tensor([1, 3]))

    for layer_offset, conv_state in enumerate(manager.all_conv_states):
        for slot in (1, 3):
            for replay_buffer, start in (
                (policy.kda_conv_q, 0),
                (policy.kda_conv_k, 2),
                (policy.kda_conv_v, 4),
            ):
                torch.testing.assert_close(
                    replay_buffer[layer_offset, slot, :, :4],
                    conv_state[slot, start : start + 2, :],
                )
                assert torch.count_nonzero(replay_buffer[layer_offset, slot, :, 4:]) == 0
            assert torch.count_nonzero(policy.kda_k_cache[layer_offset, slot]) == 0
            assert torch.count_nonzero(policy.kda_g_cache[layer_offset, slot]) == 0
            assert torch.count_nonzero(policy.kda_v_cache[layer_offset, slot]) == 0
            assert torch.count_nonzero(policy.kda_beta_cache[layer_offset, slot]) == 0
            assert policy.prev_num_accepted_tokens[slot] == 0
    assert policy.prev_num_accepted_tokens[0] == 5
    assert policy.prev_num_accepted_tokens[2] == 5


def test_v2_kda_draft_records_seed_disaggregated_generation_slots():
    """Without the per-draft replay caches, seeding a transferred request's slot sets its conv caches and clears its
    draft count; its drafts' records stay as they are, since with no pending drafts none of them is read."""
    mgr = object.__new__(KimiK3HybridCacheManager)
    mgr._kda_replay = KDAReplayState(2, draft_records=True)
    mgr._speculative_state = mgr._kda_replay
    mgr.local_num_mamba_layers = 2
    mgr.conv_state_shape = [6, 4]
    mgr.conv_section_dims = [2, 2, 2]
    mgr._request_id_to_state_index = {101: 1, 202: 3}
    mgr._kda_replay.prev_num_accepted_tokens = torch.full((4,), 5, dtype=torch.int32)
    mgr.all_conv_states = [
        torch.arange(4 * 6 * 4, dtype=torch.float32).reshape(4, 6, 4),
        torch.arange(4 * 6 * 4, dtype=torch.float32).reshape(4, 6, 4) + 1000,
    ]

    def dim_contiguous_conv_cache() -> torch.Tensor:
        return torch.full((2, 4, 6, 2), 7.0).transpose(-1, -2)

    mgr._kda_replay.kda_conv_q = dim_contiguous_conv_cache()
    mgr._kda_replay.kda_conv_k = dim_contiguous_conv_cache()
    mgr._kda_replay.kda_conv_v = dim_contiguous_conv_cache()
    mgr._kda_replay.kda_k_cache = None
    mgr._kda_replay.kda_g_cache = None
    mgr._kda_replay.kda_v_cache = None
    mgr._kda_replay.kda_beta_cache = None
    mgr._kda_replay.kda_state_tok = torch.full((2, 4, 3, 2, 1, 2), 7.0)

    mgr._kda_replay._conv_states = tuple(mgr.all_conv_states)
    mgr._kda_replay._conv_section_dims = tuple(mgr.conv_section_dims)
    mgr._kda_replay._committed_window = mgr.conv_state_shape[1]
    mgr.on_state_transfer_complete([101, 202])

    for layer_offset, conv_state in enumerate(mgr.all_conv_states):
        for slot in (1, 3):
            for replay_buffer, start in (
                (mgr._kda_replay.kda_conv_q, 0),
                (mgr._kda_replay.kda_conv_k, 2),
                (mgr._kda_replay.kda_conv_v, 4),
            ):
                torch.testing.assert_close(
                    replay_buffer[layer_offset, slot, :, :4],
                    conv_state[slot, start : start + 2, :],
                )
                assert torch.count_nonzero(replay_buffer[layer_offset, slot, :, 4:]) == 0
            assert mgr._kda_replay.prev_num_accepted_tokens[slot] == 0
    assert mgr._kda_replay.prev_num_accepted_tokens.tolist() == [5, 0, 5, 0]
    assert (mgr._kda_replay.kda_state_tok == 7.0).all()


def test_kda_replay_path_stays_on_with_the_drafts_records():
    """``has_kda_replay_caches`` follows the replay conv caches: it holds with the per-draft replay caches and with
    the drafts' records in their place, and not without the replay path."""
    t = torch.zeros(1)
    conv_caches = dict(kda_conv_q=t, kda_conv_k=t, kda_conv_v=t)
    with_caches = KDAReplayLayerCache(
        conv=t,
        temporal=t,
        kda_k_cache=t,
        kda_g_cache=t,
        kda_v_cache=t,
        kda_beta_cache=t,
        **conv_caches,
    )
    with_records = KDAReplayLayerCache(conv=t, temporal=t, kda_state_tok=t, **conv_caches)
    assert with_caches.has_kda_replay_caches
    assert with_records.has_kda_replay_caches
    assert not KDAReplayLayerCache(conv=t, temporal=t).has_kda_replay_caches
    assert not PythonMambaCacheManager.State(conv=t, temporal=t).has_kda_replay_caches


def test_v2_kda_replay_seeds_bf16_conv_state_from_disagg_transfer():
    """The conv half of the KDA state crosses ctx->gen in bf16.

    ``extract_mamba_kv_cache_params`` forces the KDA delta-rule (ssm) pool to
    fp32 but leaves the short-convolution pool at the model dtype, so what the
    context rank serializes and the generation rank restores for the conv slot
    is bf16, the same dtype as the replay conv caches the generation rank
    rebuilds from it (k / v / beta replay caches are bf16 too, the gate cache
    fp32). Seeding must copy the restored rows bit for bit, keep the logical
    shape, leave the restored bf16 pool itself untouched so the decode path
    reuses exactly what was transferred, and clear only the draft scratch of
    the seeded slots.
    """
    torch.manual_seed(0)
    mgr = object.__new__(KimiK3HybridCacheManager)
    policy = KDAReplayState(num_speculative_tokens=2)
    mgr._kda_replay = policy
    policy._committed_window = 4
    policy._conv_section_dims = (2, 2, 2)
    mgr._request_id_to_state_index = {101: 1, 202: 3}
    policy.prev_num_accepted_tokens = torch.full((4,), 5, dtype=torch.int32)
    # The restored (transferred) conv pool, in the dtype the transceiver moves.
    mgr.all_conv_states = [(torch.randn(4, 6, 4) * 0.1).to(torch.bfloat16) for _ in range(2)]
    policy._conv_states = tuple(mgr.all_conv_states)
    restored = [conv.clone() for conv in mgr.all_conv_states]

    # Same dtypes as _allocate_pool_replay_buffers: bf16 conv / k / v / beta
    # replay caches, fp32 gate cache.
    def dim_contiguous_conv_cache() -> torch.Tensor:
        return torch.full((2, 4, 6, 2), 7.0, dtype=torch.bfloat16).transpose(-1, -2)

    policy.kda_conv_q = dim_contiguous_conv_cache()
    policy.kda_conv_k = dim_contiguous_conv_cache()
    policy.kda_conv_v = dim_contiguous_conv_cache()
    policy.kda_k_cache = torch.full((2, 4, 2, 2), 7.0, dtype=torch.bfloat16)
    policy.kda_g_cache = torch.full((2, 4, 2, 2), 7.0)
    policy.kda_v_cache = torch.full((2, 4, 2, 2), 7.0, dtype=torch.bfloat16)
    policy.kda_beta_cache = torch.full((2, 4, 2, 1), 7.0, dtype=torch.bfloat16)

    mgr.on_state_transfer_complete([101, 202])

    for layer_offset, conv_state in enumerate(mgr.all_conv_states):
        assert conv_state.dtype is torch.bfloat16
        for slot in (1, 3):
            for replay_buffer, start in (
                (policy.kda_conv_q, 0),
                (policy.kda_conv_k, 2),
                (policy.kda_conv_v, 4),
            ):
                # bf16 replay cache, logical shape unchanged by the seeding.
                assert replay_buffer.dtype is torch.bfloat16
                seeded = replay_buffer[layer_offset, slot, :, :4]
                assert seeded.shape == (2, 4)
                # Same dtype on both sides: the restored values are copied bit
                # for bit.
                torch.testing.assert_close(
                    seeded,
                    conv_state[slot, start : start + 2, :],
                    rtol=0,
                    atol=0,
                )
                assert torch.count_nonzero(replay_buffer[layer_offset, slot, :, 4:]) == 0
            assert torch.count_nonzero(policy.kda_k_cache[layer_offset, slot]) == 0
            assert torch.count_nonzero(policy.kda_g_cache[layer_offset, slot]) == 0
            assert torch.count_nonzero(policy.kda_v_cache[layer_offset, slot]) == 0
            assert torch.count_nonzero(policy.kda_beta_cache[layer_offset, slot]) == 0
            assert policy.prev_num_accepted_tokens[slot] == 0
        # Slots no request restored keep their pre-seed contents.
        for slot in (0, 2):
            assert (policy.kda_conv_q[layer_offset, slot] == 7.0).all()
            assert (policy.kda_k_cache[layer_offset, slot] == 7.0).all()
            assert (policy.kda_g_cache[layer_offset, slot] == 7.0).all()
        # The restored bf16 pool is read-only here: plain decode (speculation
        # off) reuses exactly the state the transfer wrote.
        torch.testing.assert_close(conv_state, restored[layer_offset], rtol=0, atol=0)
    assert policy.prev_num_accepted_tokens[0] == 5
    assert policy.prev_num_accepted_tokens[2] == 5


@skip_no_cuda
def test_v2_hybrid_static_dynamic_tree_capacity():
    spec_config = MTPDecodingConfig(
        max_draft_len=6,
        max_total_draft_tokens=31,
        use_dynamic_tree=True,
        dynamic_tree_max_topK=10,
    )
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=4,
        spec_config=spec_config,
    )
    try:
        assert mgr.intermediate_ssm_states.shape[2] == 32
        assert mgr.intermediate_conv_states.shape[2] == 32
        assert mgr._kv_reserve_draft_tokens == 31
        assert mgr._num_reserved_dummy_slots == 1
        assert mgr.get_replay_state_update_metadata() is None
    finally:
        mgr.shutdown()


@skip_no_cuda
@pytest.mark.parametrize(
    "builder",
    [_build_hybrid_with_mamba_layer, _build_v2_hybrid_with_mamba_layer],
    ids=["cpp", "v2"],
)
def test_hybrid_replay_buffers_size_by_tokens_per_gen_step(builder):
    spec_config = _make_wide_spec_config(max_draft_len=2, tokens_per_gen_step=5)
    mgr = builder(
        max_batch_size=4,
        spec_config=spec_config,
        use_replay_state_update=True,
    )
    try:
        replay_metadata = mgr.get_replay_state_update_metadata()
        assert replay_metadata is not None
        assert replay_metadata.replay_step_width == spec_config.tokens_per_gen_step
        assert replay_metadata.replay_history_size == max(
            MIN_REPLAY_HISTORY_SIZE, spec_config.tokens_per_gen_step
        )
        layer_cache = mgr.mamba_layer_cache(0)
        _assert_replay_layer_cache_uses_history_size(
            layer_cache, replay_metadata.replay_history_size
        )
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_gdn_replay_builds_affine_state_layout():
    policy = GDNReplayState(tokens_per_gen_step=4)
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=16,
        num_mamba_layers=3,
        spec_config=MTPDecodingConfig(max_draft_len=3),
        conv_state_layout="q_k_v",
        speculative_state=policy,
    )
    try:
        assert mgr.use_gdn_cached_replay_all_layer_commit
        assert policy._state_descriptors is None
        state_strides = policy._state_strides
        assert state_strides is not None
        states = mgr.all_ssm_states
        first_state = states[0]
        expected_layer_stride = (
            states[1].data_ptr() - first_state.data_ptr()
        ) // first_state.element_size()
        assert state_strides == (
            mgr.local_num_mamba_layers,
            expected_layer_stride,
            first_state.stride(0),
        )
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_gdn_replay_all_layer_commit_matches_contiguous_layout():
    from tensorrt_llm._torch.modules.fla.cached_replay import (
        commit_gdn_cached_replay_history_layers,
    )

    batch_size = 16
    policy = GDNReplayState(tokens_per_gen_step=4)
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=batch_size,
        num_mamba_layers=3,
        spec_config=MTPDecodingConfig(max_draft_len=3),
        conv_state_layout="q_k_v",
        speculative_state=policy,
    )
    try:
        torch.manual_seed(1234)
        for state in mgr.all_ssm_states:
            state.normal_()
        policy.old_x.normal_()
        policy.old_B.normal_()
        policy.old_dt.uniform_(-0.2, -0.01)

        positions = torch.arange(batch_size, device="cuda", dtype=torch.int32)
        replay_work_items = torch.stack(
            (
                positions,
                positions,
                torch.full_like(positions, policy.replay_history_size),
                torch.zeros_like(positions),
            ),
            dim=1,
        )
        n_writes = torch.tensor([batch_size], device="cuda", dtype=torch.int32)
        expected = torch.stack(mgr.all_ssm_states)
        commit_gdn_cached_replay_history_layers(
            ssm_states=expected,
            old_u=policy.old_x,
            old_k=policy.old_B,
            old_G=policy.old_dt,
            replay_work_items=replay_work_items,
            n_writes=n_writes,
            history_size=policy.replay_history_size,
        )

        policy.commit_all_layers(
            SimpleNamespace(
                mamba_metadata=SimpleNamespace(
                    replay_num_decodes=batch_size,
                    replay_work_items=replay_work_items,
                    replay_n_writes=n_writes,
                )
            ),
            batch_size,
        )

        torch.testing.assert_close(torch.stack(mgr.all_ssm_states), expected, rtol=0, atol=0)
    finally:
        mgr.shutdown()


def test_v2_gdn_replay_commits_before_advancing_bookkeeping(monkeypatch):
    mgr = object.__new__(Qwen35HybridCacheManager)
    batch_size = 16
    mgr.local_num_mamba_layers = 1
    policy = GDNReplayState(tokens_per_gen_step=4)
    policy.prev_num_accepted_tokens = torch.zeros(batch_size, dtype=torch.int32)
    policy.cache_buf_idx = torch.zeros(batch_size, dtype=torch.int32)
    policy.intermediate_indices = torch.arange(batch_size, dtype=torch.int32)
    mgr._dummy_request_mask_buffer = DeviceCopyBuffer(torch.zeros(batch_size, dtype=torch.bool))
    policy.intermediate_conv = torch.empty((1, batch_size, 1))
    policy.old_x = torch.empty(0)
    policy.old_B = torch.empty(0)
    policy.old_dt = torch.empty(0)
    policy._state_strides = (1, 0, 0)
    mgr._speculative_state = policy
    mgr.all_ssm_states = [torch.empty(0)]
    mgr.all_conv_states = [torch.empty(0)]
    policy._ssm_states = tuple(mgr.all_ssm_states)
    policy._conv_states = tuple(mgr.all_conv_states)

    events = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.fla.cache_manager.commit_gdn_cached_replay_history_layers",
        lambda **_kwargs: events.append("commit"),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.fla.cache_manager._advance_replay_state",
        lambda *_args, **_kwargs: events.append("advance"),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager._promote_mamba_state_triton",
        lambda *_args, **_kwargs: None,
    )

    mgr._generation_state_indices = torch.arange(64, dtype=torch.int32)
    if not hasattr(mgr, "_dummy_request_mask"):
        mgr._dummy_request_mask_buffer = DeviceCopyBuffer(torch.zeros(64, dtype=torch.bool))
    mgr.update_mamba_states(
        SimpleNamespace(
            num_seqs=batch_size,
            num_contexts=0,
            mamba_metadata=SimpleNamespace(
                replay_num_decodes=batch_size,
                replay_work_items=torch.empty((batch_size, 4), dtype=torch.int32),
                replay_n_writes=torch.empty(1, dtype=torch.int32),
            ),
        ),
        torch.full((batch_size,), 3, dtype=torch.int32),
        state_indices=torch.arange(batch_size, dtype=torch.int32),
    )

    assert events == ["commit", "advance"]


def test_v2_hybrid_replay_update_skips_dummy_and_padding_rows(monkeypatch):
    mgr = object.__new__(NemotronHybridCacheManager)
    mgr.local_num_mamba_layers = 1
    mgr._request_id_to_state_index = {
        100: 0,
        101: 1,
        102: 2,
        103: 3,
    }
    mgr._request_id_to_is_dummy = {
        100: False,
        101: False,
        102: True,
        103: False,
    }
    policy = ReplayHistory(tokens_per_gen_step=5)
    mgr._dummy_request_mask_buffer = DeviceCopyBuffer(torch.zeros(4, dtype=torch.bool))
    mgr._dummy_request_mask_host = torch.zeros(4, dtype=torch.bool)
    policy.prev_num_accepted_tokens = torch.full((4,), 13, dtype=torch.int32)
    policy.cache_buf_idx = torch.ones(4, dtype=torch.int32)
    policy.intermediate_indices = torch.arange(4, dtype=torch.int32)
    policy.intermediate_conv = torch.empty((1, 4, 1))
    mgr._speculative_state = policy
    mgr.all_ssm_states = []
    mgr.all_conv_states = [torch.empty(0)]
    policy._ssm_states = tuple(mgr.all_ssm_states)
    policy._conv_states = tuple(mgr.all_conv_states)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager._promote_mamba_state_triton",
        lambda *args, **kwargs: None,
    )

    request_ids = [100, 101, 102, 103]
    state_indices = torch.tensor(
        mgr.get_state_indices(request_ids, [False, False, False, True]),
        dtype=torch.int32,
    )
    assert mgr._dummy_request_mask.tolist() == [False, False, True, True]

    mgr._generation_state_indices = torch.arange(64, dtype=torch.int32)
    if not hasattr(mgr, "_dummy_request_mask"):
        mgr._dummy_request_mask_buffer = DeviceCopyBuffer(torch.zeros(64, dtype=torch.bool))
    mgr.update_mamba_states(
        SimpleNamespace(num_seqs=4, num_contexts=1),
        torch.tensor([1, 3, 3, 3], dtype=torch.int32),
        state_indices=state_indices,
    )

    assert policy.prev_num_accepted_tokens.tolist() == [13, 3, 13, 13]
    assert policy.cache_buf_idx.tolist() == [1, 0, 1, 1]


def test_v2_hybrid_dynamic_tree_promotes_accepted_leaf_state(monkeypatch):
    mgr = object.__new__(NemotronHybridCacheManager)
    mgr.local_num_mamba_layers = 1
    policy = Mamba2State()
    policy.intermediate_indices = torch.arange(2, dtype=torch.int32)
    policy.intermediate_ssm = torch.empty((1, 2, 8))
    policy.intermediate_conv = torch.empty((1, 2, 8))
    mgr._speculative_state = policy
    mgr.all_ssm_states = [torch.empty(0)]
    mgr.all_conv_states = [torch.empty(0)]
    policy._ssm_states = tuple(mgr.all_ssm_states)
    policy._conv_states = tuple(mgr.all_conv_states)

    promoted_positions = []

    def capture_promoted_position(_dst, _src, _src_indices, positions, _dst_indices):
        promoted_positions.append(positions.clone())

    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager._promote_mamba_state_triton",
        capture_promoted_position,
    )

    mgr._generation_state_indices = torch.arange(64, dtype=torch.int32)
    if not hasattr(mgr, "_dummy_request_mask"):
        mgr._dummy_request_mask_buffer = DeviceCopyBuffer(torch.zeros(64, dtype=torch.bool))
    mgr.update_mamba_states(
        SimpleNamespace(num_seqs=3, num_contexts=1),
        torch.tensor([1, 2, 3], dtype=torch.int32),
        state_indices=torch.tensor([9, 10, 11], dtype=torch.int32),
        accepted_leaf_positions=torch.tensor([4, 7], dtype=torch.int64),
    )

    assert len(promoted_positions) == 2
    assert all(
        torch.equal(positions, torch.tensor([4, 7], dtype=torch.int32))
        for positions in promoted_positions
    )


@skip_no_cuda
@pytest.mark.parametrize(
    "mamba_ssm_cache_dtype",
    [torch.float16, torch.float32, torch.bfloat16],
)
def test_cpp_hybrid_passes_per_window_pool_dtypes_for_nvfp4_kv_cache(
    mamba_ssm_cache_dtype,
):
    mgr = _build_hybrid_with_mamba_layer(
        dtype=DataType.NVFP4,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
    )
    recurrent_pool_dtype = torch_dtype_to_binding(mamba_ssm_cache_dtype)

    expected_dtypes = [
        (LinearCacheType.RECURRENT_STATES.value, recurrent_pool_dtype),
        (128, DataType.NVFP4),
    ]
    assert [
        (config.window_size, config.dtype) for config in mgr.pool_configurations
    ] == expected_dtypes
    assert [
        (config.window_size, config.dtype) for config in mgr.impl.pool_configurations
    ] == expected_dtypes
    assert mgr._layer_to_pool_idx == {0: 0, 1: 1}
    assert mgr.recurrent_states_pool_index == 0
    assert mgr.impl.get_recurrent_states_pool().dtype == mamba_ssm_cache_dtype

    compact_scale_pointers = mgr.impl.get_block_scale_pool_pointers()
    assert mgr.impl.get_block_pool_pointers().shape == (2, 2)
    assert compact_scale_pointers.shape == (1, 2)
    assert mgr.kv_cache_pool_pointers.shape == (2, 2, 2)
    assert torch.count_nonzero(mgr.kv_cache_pool_pointers[0, :, 1]) == 0
    assert torch.equal(mgr.kv_cache_pool_pointers[1, :, 1], compact_scale_pointers[0])


@skip_no_cuda
def test_cpp_hybrid_merges_compact_scale_rows_with_unmanaged_layers():
    mgr = _build_hybrid_with_mamba_layer(
        dtype=DataType.NVFP4,
        mamba_layer_mask=[True, False, True, False],
        attention_layer_mask=[False, False, False, True],
    )

    assert mgr.pp_layers == [0, 2, 3]
    recurrent_window = LinearCacheType.RECURRENT_STATES.value
    assert mgr.max_attention_window_vec == [
        recurrent_window,
        recurrent_window,
        mgr.max_seq_len,
    ]
    assert mgr.kv_cache_pool_mapping[:, 0].tolist() == [0, 0, 1]
    compact_scale_pointers = mgr.impl.get_block_scale_pool_pointers()
    assert compact_scale_pointers.shape == (1, 2)
    assert mgr.kv_cache_pool_pointers.shape == (2, 2, 2)
    assert torch.count_nonzero(mgr.kv_cache_pool_pointers[0, :, 1]) == 0
    assert torch.equal(mgr.kv_cache_pool_pointers[1, :, 1], compact_scale_pointers[0])


@skip_no_cuda
def test_cpp_hybrid_recurrent_pool_reserves_cuda_graph_padding_slot():
    """Without spec decoding, the recurrent-state snapshot pool must
    have at least max_batch_size + 1 slots — one extra for the
    CUDA-graph padding sentinel (CUDA_GRAPH_DUMMY_REQUEST_ID). Without
    it, the padding sentinel evicts live recurrent state under load."""
    max_batch_size = 4
    mgr = _build_hybrid_with_mamba_layer(spec_config=None, max_batch_size=max_batch_size)
    recurrent_primary, _ = mgr.blocks_per_window[LinearCacheType.RECURRENT_STATES.value]
    assert recurrent_primary >= max_batch_size + 1, (
        f"recurrent-state pool has {recurrent_primary} slots, "
        f"need >= max_batch_size + 1 = {max_batch_size + 1} to host the "
        f"CUDA-graph padding sentinel without evicting live state"
    )


@skip_no_cuda
def test_cpp_hybrid_recurrent_pool_reserves_draft_len_sentinel_slots():
    """With spec decoding, CUDAGraphRunner._get_padded_batch issues a
    distinct dummy request id for each runtime_draft_len in
    [0, max_draft_len], so the recurrent-state snapshot pool must reserve
    one slot per draft length on top of the CUDA-graph padding slot."""
    max_batch_size, max_draft_len = 4, 2
    spec_config = MTPDecodingConfig(max_draft_len=max_draft_len)
    mgr = _build_hybrid_with_mamba_layer(spec_config=spec_config, max_batch_size=max_batch_size)
    recurrent_primary, _ = mgr.blocks_per_window[LinearCacheType.RECURRENT_STATES.value]
    expected_min = max_batch_size + 1 + max_draft_len
    assert recurrent_primary >= expected_min, (
        f"recurrent-state pool has {recurrent_primary} slots, "
        f"need >= max_batch_size + 1 + max_draft_len = {expected_min} so "
        f"per-draft-len sentinels don't collide with live state"
    )


def _build_hybrid_with_mamba_layer_pp(
    spec_config=None, max_batch_size=4, enable_block_reuse=False, pp_size=2
):
    """Same as ``_build_hybrid_with_mamba_layer`` but with ``pp_size`` >= 1.

    Uses ``world_size = pp_size`` and ``rank = 0`` so the real C++ KVCacheManager
    still goes through its single-process path while the Python pool-sizing
    code sees ``mapping.pp_size > 1``. Constructs ``pp_size * 2`` total layers
    (alternating mamba/attn) so that each PP slice has both a mamba and an
    attention layer — otherwise some ranks would hit a slope=0 edge case in
    the affine memory model when block reuse is disabled.
    """
    pairs = pp_size  # one (mamba, attn) pair per PP stage so every rank has both kinds
    mamba_mask = [True, False] * pairs
    attn_mask = [False, True] * pairs
    mamba_num_layers = sum(mamba_mask)
    num_layers = sum(attn_mask)
    mapping = Mapping(world_size=pp_size, rank=0, tp_size=1, pp_size=pp_size)
    kv_cache_config = KvCacheConfig(max_tokens=512, enable_block_reuse=enable_block_reuse)
    return CppMambaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=mamba_num_layers,
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float16,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_layers,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=max_batch_size,
        mapping=mapping,
        spec_config=spec_config,
        layer_mask=attn_mask,
    )


# Skip when running under pytest --run-ray (sets TLLM_DISABLE_MPI=1). In that
# mode ``Mapping`` resolves to ``DeviceMeshTopology``, whose ``pp_rank``
# requires torch.distributed initialisation that isn't available in this
# single-process unit test. The pp-sharding behaviour exercised here is
# orthogonal to the Ray orchestrator.
_skip_under_ray = pytest.mark.skipif(
    os.environ.get("TLLM_DISABLE_MPI") == "1",
    reason="pp_size>1 helper builds Mapping with world_size>1 which needs "
    "torch.distributed under TLLM_DISABLE_MPI=1 (Ray) sessions",
)


@_skip_under_ray
@skip_no_cuda
@pytest.mark.parametrize("pp_size", [2, 4])
def test_cpp_hybrid_recurrent_pool_scales_with_pp_size(pp_size):
    """With pipeline parallelism, multiple microbatches are in-flight on the
    same rank concurrently, each holding up to ``max_batch_size`` sequences'
    Mamba state. The recurrent-state pool must therefore size for
    ``max_batch_size * pp_size`` live slots (plus the CUDA-graph padding
    sentinel). Without this scaling, the first inference batch under PP>1
    trips ``No free block found`` once requests beyond the first microbatch
    enter the pool (cf. TestNemotronV3Super::test_nvfp4_parallelism[TP4_PP2]).
    """
    max_batch_size = 4
    mgr = _build_hybrid_with_mamba_layer_pp(
        spec_config=None, max_batch_size=max_batch_size, pp_size=pp_size
    )
    recurrent_primary, _ = mgr.blocks_per_window[LinearCacheType.RECURRENT_STATES.value]
    expected_min = max_batch_size * pp_size + 1
    assert recurrent_primary >= expected_min, (
        f"recurrent-state pool has {recurrent_primary} slots with pp_size={pp_size}, "
        f"need >= max_batch_size * pp_size + 1 = {expected_min} so concurrent "
        f"in-flight microbatches don't exhaust live-state slots"
    )


@skip_no_cuda
def test_cpp_hybrid_recurrent_pool_floor_with_block_reuse():
    """With block reuse enabled, the block-reuse branch must not drop the
    live-state + CUDA-graph-padding floor.

    With max_batch_size=4, periodic_snapshot_interval=256, max_tokens=512:
      naive: max_snapshots = 512 // 256 = 2  (drops live-state floor!)
      fixed: max_snapshots = max(2, 4 + 1) = 5
    """
    max_batch_size = 4
    mgr = _build_hybrid_with_mamba_layer(
        spec_config=None,
        max_batch_size=max_batch_size,
        enable_block_reuse=True,
        periodic_snapshot_interval=256,
    )
    recurrent_primary, _ = mgr.blocks_per_window[LinearCacheType.RECURRENT_STATES.value]
    assert recurrent_primary >= max_batch_size + 1, (
        f"recurrent-state pool has {recurrent_primary} slots with block reuse enabled, "
        f"need >= max_batch_size + 1 = {max_batch_size + 1} to prevent the padding "
        f"sentinel from evicting live recurrent state"
    )


@skip_no_cuda
def test_cpp_hybrid_dry_run_recurrent_pool_additive_with_block_reuse():
    """Dry-run path (is_estimating_kv_cache=True) under block reuse must
    keep the live-state floor *plus* room for snapshots, not collapse to
    max(snapshots, live). With max_batch_size=4, interval=256, max_tokens=512:
      old:  max_snapshots = max(512//256, 4)         = 4   (no headroom for snapshots)
      new:  max_snapshots = 4 + 512//256             = 6   (live + snapshots)
    """
    max_batch_size = 4
    mgr = _build_hybrid_with_mamba_layer(
        spec_config=None,
        max_batch_size=max_batch_size,
        enable_block_reuse=True,
        periodic_snapshot_interval=256,
        is_estimating_kv_cache=True,
    )
    recurrent_primary, _ = mgr.blocks_per_window[LinearCacheType.RECURRENT_STATES.value]
    # 4 live state slots + 2 reuse snapshots = 6.
    expected_min = max_batch_size + (512 // 256)
    assert recurrent_primary >= expected_min, (
        f"dry-run recurrent-state pool has {recurrent_primary} slots, "
        f"need >= live_state + reuse_snapshots = {expected_min}; the old "
        f"max(reuse, live) formula dropped reuse headroom"
    )


# ---------------------------------------------------------------------------
# CppMambaHybridCacheManager: rank with zero local mamba layers
#
# Regression test for the early-exit path added when a rank ends up with no
# mamba layers (e.g. under PP sharding when all mamba layers fall on other
# ranks). On that path, the constructor must:
#   - call the real parent KVCacheManager with the union layer_mask and
#     num_layers=num_layers (not mamba_num_layers + num_layers),
#   - skip allocating any mamba-only state, and
#   - leave self.requests = [] so Mamba-only hooks and metadata preparation
#     can no-op without touching uninitialized state.
#
# We exercise the same Python branch with world_size=1 (so the real C++
# KVCacheManager init doesn't need MPI) and a layer mask that contains zero
# mamba layers.
# ---------------------------------------------------------------------------


def _build_zero_mamba_hybrid(
    enable_block_reuse=False,
    periodic_snapshot_interval=256,
):
    """Construct a real CppMambaHybridCacheManager whose this-rank slice has
    no mamba layers. world_size=1 / pp_size=1 keeps the real parent
    KVCacheManager off the MPI path."""
    # [other, other, full_attn, full_attn]
    mamba_mask = [False, False, False, False]
    attn_mask = [False, False, True, True]
    mamba_num_layers = sum(mamba_mask)  # 0
    num_layers = sum(attn_mask)  # 4

    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    # Cap KV pool size so the real C++ allocator only takes a tiny slice of
    # GPU memory; we don't actually use the cache.
    kv_cache_config = KvCacheConfig(
        max_tokens=128,
        enable_block_reuse=enable_block_reuse,
        mamba_state_config=MambaStateConfig(periodic_snapshot_interval=periodic_snapshot_interval),
    )

    mgr = CppMambaHybridCacheManager(
        # mamba cache parameters — values are unused on the early-exit path
        # but must be type-valid.
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=mamba_num_layers,
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float16,
        # kv cache parameters
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_layers,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_num_tokens=96,
        max_batch_size=2,
        mapping=mapping,
        spec_config=None,
        layer_mask=attn_mask,
    )
    return mgr


@skip_no_cuda
def test_cpp_hybrid_zero_local_mamba_layers():
    """End-to-end: real parent KVCacheManager + real early-exit. Verifies
    early-exit invariants on the manager state and that Mamba hooks and
    metadata preparation do not touch uninitialized Mamba-only state."""
    mgr = _build_zero_mamba_hybrid(
        enable_block_reuse=True,
        periodic_snapshot_interval=64,
    )

    # Early-exit indicators.
    assert mgr.local_num_mamba_layers == 0
    assert mgr.mamba_pp_layers == []
    assert mgr.requests == []
    assert mgr.pp_layers == [2, 3]

    # Parent KVCacheManager was really initialized. self.impl is the C++
    # KVCacheManagerCpp object; blocks_per_window is set up by it.
    assert hasattr(mgr, "impl")
    assert hasattr(mgr, "blocks_per_window")
    # Parent saw num_layers = num_layers (4), not mamba_num_layers + num_layers.
    # On the early-exit branch, num_layers is forwarded as-is.
    assert mgr.num_layers == 4
    assert mgr.num_local_layers == 2
    assert mgr.max_num_tokens == 96
    assert all(
        config.window_size != LinearCacheType.RECURRENT_STATES.value
        for config in mgr.pool_configurations
    )
    assert all(
        config.window_size != LinearCacheType.RECURRENT_STATES.value
        for config in mgr.impl.pool_configurations
    )

    # No mamba-only state was allocated.
    for attr in (
        "ssm_state_shape",
        "conv_state_shape",
        "mamba_layer_offsets",
        "cuda_state_indices",
        "host_block_offsets",
        "recurrent_states_pool_index",
    ):
        assert not hasattr(mgr, attr), f"{attr} must not be set on the zero-mamba early-exit path"
    # Parent must not have been told to treat this as linear attention.
    assert mgr.is_linear_attention is False

    # The scheduler hook is still advertised on ranks without local Mamba
    # layers, so its inputs must be initialized with the same interval as
    # Mamba-owning PP ranks to keep their scheduling decisions aligned.
    assert mgr.kv_cache_config.enable_block_reuse is True
    assert mgr.linear_attention_metadata.states_snapshot_interval == 64
    request = SimpleNamespace(prompt_len=150, expect_snapshot_points=[])
    mgr.prepare_expect_snapshot_points([request])
    assert request.expect_snapshot_points == [64, 128]
    # The shared interval must not make this attention-only rank consult its
    # nonexistent recurrent-state pool and report zero KV capacity.
    attention_capacity = KVCacheManager.get_num_available_tokens(mgr, 128)
    assert attention_capacity > 0
    assert mgr.get_num_available_tokens(128) == attention_capacity

    # Guards on the three mamba-only methods must turn them into no-ops
    # instead of crashing on the missing state above.
    empty_batch = ScheduledRequests()
    mgr.prepare_resources(empty_batch)  # super() runs, then guard returns
    mgr.update_mamba_states(attn_metadata=None, num_accepted_tokens=None, state_indices=None)
    mgr._setup_state_indices()

    metadata = Mamba2Metadata(max_batch_size=1, chunk_size=8)
    seq_lens = torch.tensor([8], dtype=torch.int32)
    metadata.prepare(
        SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cuda=seq_lens.cuda(),
            num_contexts=1,
            num_ctx_tokens=8,
            kv_cache_manager=mgr,
            request_ids=[123],
            kv_cache_params=SimpleNamespace(
                num_cached_tokens_per_seq=torch.tensor([0], dtype=torch.int32)
            ),
        )
    )
    assert metadata.state_indices[0].item() == 0


@skip_no_cuda
def test_v2_stack_state_views_detects_affine_layout():
    pool = torch.zeros(3, 4, 8, device="cuda")
    layers = [pool[i] for i in range(3)]

    stacked = _stack_state_views(layers)

    assert stacked.shape == (3, 4, 8)
    stacked[2, 1, 3] = 5.0
    assert layers[2][1, 3] == 5.0


def test_v2_stack_state_views_rejects_non_affine_layout():
    pool = torch.zeros(4, 4, 8)
    stack = _stack_state_views

    assert stack([pool[0], pool[1], pool[3]]) is None
    assert stack([pool[0], torch.zeros(5, 8)]) is None
    assert stack([]) is None


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=skip_no_cuda)])
def test_v2_kda_admission_slot_updates_without_sync(device):
    """Disagg GEN admission resets, relocates and seeds KDA replay slots; on CUDA none of it may
    synchronize the stream (the previous step is still queued)."""
    num_slots, num_layers, num_spec, dim, window = 8, 2, 2, 2, 3
    mgr = object.__new__(KimiK3HybridCacheManager)
    policy = KDAReplayState(num_spec)
    mgr._kda_replay = policy
    mgr._speculative_state = policy
    mgr.local_num_mamba_layers = num_layers
    policy.prev_num_accepted_tokens = torch.arange(
        1, num_slots + 1, dtype=torch.int32, device=device
    )
    # Stale device indices: the reset must take its slots from _host_state_indices.
    mgr._state_indices_buffer = DeviceCopyBuffer(
        torch.full((num_slots,), 7, dtype=torch.int32, device=device)
    )
    mgr._host_state_indices = torch.tensor([5, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    mgr._request_id_to_state_index = {10: 4, 11: 2}
    mgr.conv_state_shape = [3 * dim, window]
    mgr.conv_section_dims = [dim] * 3
    mgr.all_conv_states = [
        torch.randn(num_slots, 3 * dim, window, device=device) for _ in range(num_layers)
    ]
    policy.kda_conv_q, policy.kda_conv_k, policy.kda_conv_v = (
        torch.randn(num_layers, num_slots, dim, window + num_spec, device=device) for _ in range(3)
    )
    policy.kda_k_cache, policy.kda_g_cache, policy.kda_v_cache, policy.kda_beta_cache = (
        torch.randn(num_layers, num_slots, num_spec, dim, device=device) for _ in range(4)
    )

    policy._conv_states = tuple(mgr.all_conv_states)
    policy._conv_section_dims = tuple(mgr.conv_section_dims)
    policy._committed_window = window

    if device == "cuda":
        torch.cuda.synchronize()
        previous_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
    try:
        mgr._reset_context_mamba_slots(2)
        policy.reset_slots(torch.tensor([3]), [3])
        policy.relocate_slots([1], [6])
        mgr.seed_kda_replay_caches_for_disagg_gen([10, 11])
    finally:
        if device == "cuda":
            torch.cuda.set_sync_debug_mode(previous_mode)

    # Reset 5, 1 and 3; slot 6 takes slot 1's (reset) count; seeding resets 4 and 2; 7 untouched.
    assert policy.prev_num_accepted_tokens.tolist() == [1, 0, 0, 0, 0, 0, 0, 8]


@skip_no_cuda
@pytest.mark.parametrize("kda_replay_num_spec", [2, None])
def test_v2_kda_draft_records_replace_the_replay_caches(kda_replay_num_spec):
    """On the KDA replay path, ``kda_draft_records`` allocates the fp32 records of every draft of each slot (vn,
    beta * k, decay), per layer, in place of the per-draft replay caches; the conv caches and the replay path stay.
    Without the replay path it allocates nothing."""
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=4,
        num_mamba_layers=2,
        spec_config=MTPDecodingConfig(max_draft_len=2),
        conv_state_layout="q_k_v",
        mamba_d_conv=5,
        mamba_num_heads=6,
        mamba_n_groups=6,
        mamba_ssm_cache_dtype=torch.float32,
        kda_replay_num_spec=kda_replay_num_spec,
        kda_draft_records=True,
    )
    try:
        if kda_replay_num_spec is None:
            assert getattr(mgr, "kda_state_tok", None) is None
            assert not mgr.keeps_kda_draft_records
            return
        assert mgr.keeps_kda_draft_records
        assert set(mgr._kda_replay_layout()) == {
            "kda_conv_q",
            "kda_conv_k",
            "kda_conv_v",
            "kda_state_tok",
        }
        # A KVCM state buffer of the SSM layers, kept out of disaggregated transfers like the other replay buffers.
        assert DataRole("kda_state_tok") in mgr.get_disagg_ignored_roles()
        assert len(mgr.kda_state_tok) == 2
        for layer_idx in range(2):
            layer_cache = mgr.mamba_layer_cache(layer_idx)
            cache_size = layer_cache.temporal.shape[0]
            assert layer_cache.has_kda_replay_caches
            assert layer_cache.kda_conv_q.shape == (cache_size, 48, 6)
            assert layer_cache.kda_k_cache is None
            assert layer_cache.kda_g_cache is None
            assert layer_cache.kda_v_cache is None
            assert layer_cache.kda_beta_cache is None
            states = layer_cache.kda_state_tok
            num_heads, _, head_dim = layer_cache.temporal.shape[1:]
            assert states.shape == (cache_size, 3, 2, num_heads, head_dim)
            assert states.dtype is torch.float32
            assert states.data_ptr() == mgr.kda_state_tok[layer_idx].data_ptr()
            # The records' replay addresses a slot by its stride and reads its records contiguously.
            assert states[0].is_contiguous()
    finally:
        mgr.shutdown()
    # Released with the other replay buffers.
    assert mgr.kda_state_tok is None


@skip_no_cuda
def test_v2_kda_draft_records_count_in_the_per_slot_budget():
    """The capacity math sees the drafts' records in place of the per-draft replay caches: per draft and layer, a slot
    costs vn, beta * k and the decay, fp32 [H, K] each, instead of the k / g / v / beta entries."""
    kwargs = dict(
        num_mamba_layers=2,
        spec_config=MTPDecodingConfig(max_draft_len=2),
        conv_state_layout="q_k_v",
        mamba_d_conv=5,
        mamba_num_heads=6,
        mamba_n_groups=6,
        mamba_ssm_cache_dtype=torch.float32,
        kda_replay_num_spec=2,
    )
    plain = _build_v2_hybrid_with_mamba_layer(**kwargs)
    try:
        plain_bytes = plain._mamba_state_bytes_per_slot()
    finally:
        plain.shutdown()
    with_states = _build_v2_hybrid_with_mamba_layer(kda_draft_records=True, **kwargs)
    try:
        num_heads, _, head_dim = with_states.ssm_state_shape
        per_token = 2 * 2 * 3 * num_heads * head_dim * 4
        replay_caches = 2 * sum(
            dtype.itemsize * math.prod(shape)
            for name, (dtype, shape) in _kda_replay_layout(2, num_heads, head_dim, 4).items()
            if name in _KDA_REPLAY_BUFFERS[3:]
        )
        assert with_states._mamba_state_bytes_per_slot() == plain_bytes - replay_caches + per_token
    finally:
        with_states.shutdown()


@skip_no_cuda
def test_v2_state_index_setup_skips_unchanged_uploads(monkeypatch):
    """A step whose state indices and dummy flags did not change enqueues no upload of either: the device buffers
    keep values the host never sent. A changed step uploads both again."""
    # The test writes the buffers behind _setup_state_indices to see the skip, which the check of skipped copies
    # reports.
    monkeypatch.setattr("tensorrt_llm._device_copy._verify_skipped_device_copies", False)

    class StateCache:
        def __init__(self, slot: int) -> None:
            self.slot = slot

        def get_ssm_block_base_index(self, _layer_group_id) -> int:
            return self.slot

        def close(self) -> None:
            pass

    mgr = _build_v2_hybrid_with_mamba_layer(
        spec_config=MTPDecodingConfig(max_draft_len=2),
        conv_state_layout="q_k_v",
        mamba_n_groups=4,
        mamba_ssm_cache_dtype=torch.float32,
        kda_replay_num_spec=2,
    )
    live = SimpleNamespace(py_request_id=101, is_dummy=False)
    padding = SimpleNamespace(py_request_id=202, is_dummy=True)
    mgr.kv_cache_map.update({101: StateCache(0), 202: StateCache(1)})
    try:
        mgr._setup_state_indices([live, padding])
        torch.cuda.synchronize()
        assert mgr.cuda_state_indices[:2].tolist() == [0, 1]
        assert mgr._dummy_request_mask[:2].tolist() == [False, True]

        mgr.cuda_state_indices.fill_(-1)
        mgr._dummy_request_mask.fill_(True)
        mgr._setup_state_indices([live, padding])
        torch.cuda.synchronize()
        assert mgr.cuda_state_indices[:2].tolist() == [-1, -1]
        assert mgr._dummy_request_mask[:2].tolist() == [True, True]

        mgr.kv_cache_map[202].slot = 2
        padding.is_dummy = False
        mgr._setup_state_indices([live, padding])
        torch.cuda.synchronize()
        assert mgr.cuda_state_indices[:2].tolist() == [0, 2]
        assert mgr._dummy_request_mask[:2].tolist() == [False, False]
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_kda_replay_allocates_logical_slot_caches():
    spec_config = MTPDecodingConfig(max_draft_len=2)
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=4,
        num_mamba_layers=2,
        spec_config=spec_config,
        conv_state_layout="q_k_v",
        mamba_d_conv=5,
        mamba_num_heads=6,
        mamba_n_groups=6,
        mamba_ssm_cache_dtype=torch.float32,
        kda_replay_num_spec=2,
    )
    try:
        assert mgr.intermediate_ssm_states is None
        assert mgr.intermediate_conv_states is None
        assert mgr.intermediate_state_indices.shape == (4,)
        assert not hasattr(mgr, "get_replay_state_update_metadata")

        layer_cache = mgr.mamba_layer_cache(0)
        cache_size = layer_cache.temporal.shape[0]
        assert layer_cache.kda_conv_q.shape == (cache_size, 48, 6)
        assert layer_cache.kda_conv_k.shape == (cache_size, 48, 6)
        assert layer_cache.kda_conv_v.shape == (cache_size, 48, 6)
        assert layer_cache.kda_conv_q.dtype is torch.bfloat16
        assert layer_cache.kda_conv_q.stride(-2) == 1
        assert layer_cache.kda_k_cache.shape == (cache_size, 2, 48)
        assert layer_cache.kda_g_cache.shape == (cache_size, 2, 48)
        assert layer_cache.kda_v_cache.shape == (cache_size, 2, 48)
        # Six bf16 heads are 12 bytes, so the row is padded to 16 (stride 8)
        # to keep every nested per-draft view 16-byte aligned for CuTe.
        assert layer_cache.kda_beta_cache.shape == (cache_size, 2, 6)
        assert layer_cache.kda_beta_cache.stride(-2) == 8
        for layer_idx in range(2):
            for slot_idx in range(cache_size):
                for draft_idx in range(2):
                    assert (
                        mgr.mamba_layer_cache(layer_idx)
                        .kda_beta_cache[slot_idx, draft_idx]
                        .data_ptr()
                        % _KDA_BETA_CACHE_ALIGNMENT_BYTES
                        == 0
                    )
        assert layer_cache.kda_k_cache.dtype is torch.bfloat16
        assert layer_cache.kda_g_cache.dtype is torch.float32
        assert layer_cache.kda_v_cache.dtype is torch.bfloat16
        assert layer_cache.kda_beta_cache.dtype is torch.bfloat16
        assert layer_cache.prev_num_accepted_tokens.dtype == torch.int32
        assert layer_cache.prev_num_accepted_tokens.shape == (cache_size,)
        assert layer_cache.prev_num_accepted_tokens is (
            mgr.mamba_layer_cache(1).prev_num_accepted_tokens
        )
        assert not hasattr(layer_cache, "intermediate_ssm")
        assert not hasattr(layer_cache, "intermediate_conv_window")
        assert layer_cache.kda_state_tok is None
        assert not mgr.keeps_kda_draft_records
        assert layer_cache.has_kda_replay_caches
    finally:
        mgr.shutdown()


def _kda_draft_record_build_config(quota, initial_pool_ratio=None):
    return KVCacheManagerConfig(
        tokens_per_block=32,
        cache_tiers=[GpuCacheTierConfig(quota=quota)],
        layers=_base_attention_layer_configs(2),
        initial_pool_ratio=initial_pool_ratio,
    )


@pytest.mark.parametrize("pool_sizing", ["typical_step", "block_reuse", "pool_ratio"])
def test_v2_kda_draft_records_size_with_the_ssm_pool(pool_sizing):
    """The drafts' records are a buffer of the SSM layer in each recurrent-state slot, in place of the k / g / v /
    beta caches. However the SSM pool is sized (by the typical step, with block reuse, or by a pool_ratio), every slot
    holds them: the build leaves the whole GPU quota to the pools, and the quotas count the records per state slot."""
    enable_block_reuse = pool_sizing == "block_reuse"
    mgr = _v2_kda_draft_record_manager(
        kda_draft_records=True, enable_block_reuse=enable_block_reuse
    )
    plain = _v2_kda_draft_record_manager(
        kda_draft_records=False, enable_block_reuse=enable_block_reuse
    )
    records = 3 * 2 * 8 * 256 * 4
    replay_caches = sum(
        dtype.itemsize * math.prod(shape)
        for name, (dtype, shape) in _kda_replay_layout(2, 8, 256, 3).items()
        if name in _KDA_REPLAY_BUFFERS[3:]
    )
    slot_delta = records - replay_caches
    assert mgr._mamba_state_bytes_per_slot() - plain._mamba_state_bytes_per_slot() == slot_delta
    live_slots = mgr._max_resident_sequences() + mgr._num_reserved_dummy_slots
    assert (
        mgr._minimum_live_gpu_quota() - plain._minimum_live_gpu_quota() == live_slots * slot_delta
    )

    quota = 128 << 20
    pool_ratio = [0.5, 0.5] if pool_sizing == "pool_ratio" else None
    built = mgr._build_cache_config(_kda_draft_record_build_config(quota, pool_ratio))
    assert built.cache_tiers[0].quota == quota
    ssm_buffers = {str(buffer.role): buffer.size for buffer in built.layers[0].buffers}
    assert ssm_buffers["kda_state_tok"] == records
    assert not set(ssm_buffers) & set(_KDA_REPLAY_BUFFERS[3:])
    with pytest.raises(ValueError, match="too small for live recurrent states"):
        mgr._build_cache_config(_kda_draft_record_build_config(mgr._minimum_live_gpu_quota() - 1))


@skip_no_cuda
@pytest.mark.parametrize("enable_block_reuse", [False, True])
def test_v2_kda_draft_records_leave_the_quota_to_the_pools(enable_block_reuse):
    """With the drafts' records inside the SSM pool, the runtime pools take the whole GPU quota (rounded up to whole
    2 MiB grains) with or without block reuse: nothing is set aside beside them."""
    mgr = _v2_kda_draft_record_manager(
        kda_draft_records=True, enable_block_reuse=enable_block_reuse
    )
    runtime_manager = RuntimeKVCacheManager(
        mgr._build_cache_config(_kda_draft_record_build_config(128 << 20))
    )
    try:
        pool_bytes = runtime_manager.get_quota(GPU_LEVEL)
    finally:
        runtime_manager.shutdown()
    assert 128 << 20 <= pool_bytes < (128 << 20) + (2 << 20)


@skip_no_cuda
def test_v2_kda_replay_commit_without_the_replay_caches():
    """With the verify kernels' draft records (``kda_state_tok``) in place of the per-draft replay caches, the
    suspend-time commit replays a slot's records onto its V2 SSM state and stores its conv window, bit for bit as the
    records' replay and the conv store do on their own, and clears only that slot's draft count."""
    from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (
        replay_kda_token_records,
        store_kda_replay_conv_window,
    )

    num_layers, num_slots, num_spec, num_heads, head_dim, width = 2, 4, 2, 6, 128, 4
    section_dim = num_heads * head_dim
    generator = torch.Generator(device="cuda").manual_seed(13)

    def randn(*shape):
        return 0.1 * torch.randn(*shape, generator=generator, device="cuda")

    mgr = object.__new__(KimiK3HybridCacheManager)
    mgr._kda_replay = KDAReplayState(num_spec, draft_records=True)
    mgr._stream = torch.cuda.Stream()
    mgr.all_ssm_states = [
        randn(num_slots, num_heads, head_dim, head_dim) for _ in range(num_layers)
    ]
    mgr.all_conv_states = [randn(num_slots, 3 * section_dim, width - 1) for _ in range(num_layers)]
    mgr._kda_replay.pool_layers = [
        {
            **{
                name: randn(num_slots, width - 1 + num_spec, section_dim).transpose(-1, -2)
                for name in ("kda_conv_q", "kda_conv_k", "kda_conv_v")
            },
            "kda_state_tok": randn(num_slots, 3, num_spec, num_heads, head_dim),
        }
        for _ in range(num_layers)
    ]
    mgr._kda_replay.prev_num_accepted_tokens = torch.tensor(
        [0, 2, 1, 0], dtype=torch.int32, device="cuda"
    )
    assert mgr.keeps_kda_draft_records

    mgr._kda_replay._ssm_states = tuple(mgr.all_ssm_states)
    mgr._kda_replay._conv_states = tuple(mgr.all_conv_states)
    slot = 1
    slots = torch.tensor([slot], dtype=torch.int32, device="cuda")
    num_accepted = mgr._kda_replay.prev_num_accepted_tokens[slots.long()].clone()
    expected_ssm, expected_conv = [], []
    for layer_offset, replay in enumerate(mgr._kda_replay.pool_layers):
        ssm = mgr.all_ssm_states[layer_offset].clone()
        conv = mgr.all_conv_states[layer_offset].clone()
        replay_kda_token_records(ssm, replay["kda_state_tok"], slots, num_accepted)
        store_kda_replay_conv_window(
            conv,
            replay["kda_conv_q"],
            replay["kda_conv_k"],
            replay["kda_conv_v"],
            slots,
            num_accepted,
        )
        expected_ssm.append(ssm)
        expected_conv.append(conv)

    mgr._kda_replay.commit_pending_drafts(slot)
    torch.cuda.synchronize()

    assert mgr._kda_replay.prev_num_accepted_tokens.tolist() == [0, 0, 1, 0]
    for actual, expected in zip(mgr.all_ssm_states, expected_ssm):
        assert torch.equal(actual, expected)
    for actual, expected in zip(mgr.all_conv_states, expected_conv):
        assert torch.equal(actual, expected)


def _build_kda_replay_manager(**kwargs):
    """A real V2 hybrid manager with two KDA layers (6 heads, K = V = 8) and replay caches for two drafts."""
    return _build_v2_hybrid_with_mamba_layer(
        num_mamba_layers=2,
        spec_config=MTPDecodingConfig(max_draft_len=2),
        conv_state_layout="q_k_v",
        mamba_num_heads=6,
        mamba_n_groups=6,
        mamba_ssm_cache_dtype=torch.float32,
        kda_replay_num_spec=2,
        **kwargs,
    )


@skip_no_cuda
@pytest.mark.parametrize("manager_cls", [KimiK3HybridCacheManager, Glm5NextCacheManager])
def test_v2_kda_replay_buffers_are_kvcm_state_buffers(manager_cls):
    """Replay caches and the accepted-draft count are registered as SSM-layer buffers, counted in the per-slot state
    bytes, and kept out of disaggregated transfers."""
    mgr = _build_kda_replay_manager(model_manager_cls=manager_cls)
    try:
        layout = _kda_replay_layout(2, 6, 8, 3)
        replay_roles = {str(DataRole(name)) for name in layout}
        for layer in mgr.mamba_pp_layers:
            local_layer_id = mgr.layer_offsets[layer]
            roles = {
                str(buffer_id.role)
                for pool_group in mgr.impl.pool_group_descs
                for variant in pool_group.slot_desc.variants
                for coalesced in variant.coalesced_buffers
                for buffer_id in coalesced.buffer_ids
                if int(buffer_id.layer_id) == local_layer_id
            }
            assert replay_roles <= roles
            assert ("prev_num_accepted_tokens" in roles) == (layer == mgr.mamba_pp_layers[0])
        layer_bytes = sum(dtype.itemsize * math.prod(shape) for dtype, shape in layout.values())
        assert mgr._mamba_state_bytes_per_slot() == (
            2 * (mgr.ssm_bytes + mgr.conv_bytes) + 2 * layer_bytes + 16
        )
        ignored = {str(role) for role in mgr.get_disagg_ignored_roles()}
        assert replay_roles | {"prev_num_accepted_tokens"} <= ignored
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_v2_kda_replay_acceptance_and_cuda_graph():
    mgr = _build_kda_replay_manager()
    try:
        requests = [_make_v2_conversation_request(i, list(range(8)), str(i)) for i in (101, 102)]
        for req in requests:
            _run_v2_hybrid_context(mgr, req)
        slots = torch.tensor(mgr.get_state_indices([101, 102]), device="cuda")
        # Mark the second row as padding through the public metadata interface.
        mgr.get_state_indices([101, 102], [False, True])
        accepted = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
        metadata = SimpleNamespace(num_seqs=2, num_contexts=0)
        for layer_id in range(2):
            counts = mgr.mamba_layer_cache(layer_id).prev_num_accepted_tokens
            assert counts[slots].tolist() == [0, 0]
        counts[slots[1]] = 1
        # Warm up on a side stream before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mgr.update_mamba_states(metadata, accepted, state_indices=slots)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mgr.update_mamba_states(metadata, accepted, state_indices=slots)
        accepted[0] = 2
        graph.replay()
        for layer_id in range(2):
            assert mgr.mamba_layer_cache(layer_id).prev_num_accepted_tokens[slots].tolist() == [
                1,
                1,
            ]
    finally:
        mgr.shutdown()


@skip_no_cuda
@pytest.mark.parametrize("manager_cls", [KimiK3HybridCacheManager, Glm5NextCacheManager])
@pytest.mark.parametrize(
    "role,is_disagg", [("context", True), ("generation", True), (None, True), ("context", False)]
)
def test_v2_kda_worker_role_cache(monkeypatch, role, is_disagg, manager_cls):
    """Only a disaggregated context server drops the KDA replay and draft-verify state."""
    if role is None:
        monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    mgr = _build_kda_replay_manager(is_disagg=is_disagg, model_manager_cls=manager_cls)
    try:
        req = _make_v2_conversation_request(123, list(range(8)), "role")
        _run_v2_hybrid_context(mgr, req)
        cache = mgr.mamba_layer_cache(0)
        ctx_only = is_disagg and role == "context"
        assert mgr.kda_context_only == ctx_only
        assert mgr.use_kda_replay_update != ctx_only
        assert isinstance(cache, KDAReplayLayerCache) != ctx_only
        assert cache.conv.shape[1:] == (144, 3)
        assert cache.temporal.shape[1:] == (6, 8, 8)
        if ctx_only:
            assert mgr.intermediate_ssm_states is None
            assert mgr.intermediate_conv_states is None
            assert not mgr.get_disagg_ignored_roles() & {
                DataRole(name) for name in _KDA_REPLAY_FIELDS
            }
            mgr.update_mamba_states(
                SimpleNamespace(num_seqs=1, num_contexts=1),
                torch.ones(1, device="cuda", dtype=torch.int32),
            )
        else:
            slot = mgr.get_state_indices([123])[0]
            assert cache.prev_num_accepted_tokens[slot].item() == 0
    finally:
        mgr.shutdown()


@skip_no_cuda
@pytest.mark.parametrize("kda_draft_records", [False, True], ids=["replay_caches", "draft_records"])
@pytest.mark.parametrize("snapshot", [False, True])
def test_v2_kda_replay_survives_cache_lifecycle(snapshot, kda_draft_records):
    """KVCM moves the replay caches, or the drafts' records in their place, with the recurrent state through
    suspend, offload to host and snapshot reuse."""
    mgr = _build_kda_replay_manager(
        host_cache_size=16 << 20,
        enable_stats=True,
        kda_draft_records=kda_draft_records,
        enable_block_reuse=True,
    )
    replay_fields = (*mgr._kda_replay_layout(), "prev_num_accepted_tokens")
    assert ("kda_state_tok" in replay_fields) == kda_draft_records
    assert mgr.reuse_match_backoff == 0
    kv = None
    reused = None
    pressure = []
    try:
        stream = torch.cuda.current_stream().cuda_stream
        kv = mgr.impl.create_kv_cache()
        assert kv.resume(stream)
        assert kv.resize(32)
        group = mgr.impl.get_layer_group_id(LayerId(0))
        slot = kv.get_ssm_block_base_index(group)
        expected = []
        for layer_id in range(2):
            cache = mgr.mamba_layer_cache(layer_id)
            fields = {}
            for offset, name in enumerate((*replay_fields, "conv", "temporal")):
                tensor = getattr(cache, name)
                # Accepted counts describe the request and are shared across layers.
                value = offset + 1 if name == "prev_num_accepted_tokens" else offset + layer_id + 1
                tensor[slot].fill_(value)
                fields[name] = tensor[slot].clone()
            expected.append(fields)
        if snapshot:
            kv.commit(list(range(32)))
            kv.close()
            kv = None
        else:
            kv.suspend()
        # Fill the recurrent pool with newer suspended requests, forcing
        # the original state to host while respecting KVCM's quota floor.
        pool = next(
            pool
            for pool in mgr.impl.pool_group_descs
            if any(variant.layer_group_id == group for variant in pool.slot_desc.variants)
        )
        for _ in range(pool.num_slots + 1):
            other = mgr.impl.create_kv_cache()
            pressure.append(other)
            assert other.resume(stream)
            other.suspend()
        for other in pressure:
            other.commit_pending_stats()
            other.close()
        pressure.clear()
        stats = mgr.impl.get_and_reset_iteration_stats()
        assert stats[group].iter_offload_blocks > 0
        if snapshot:
            reused = mgr.impl.create_kv_cache(input_tokens=list(range(32)))
            assert reused.num_committed_tokens == 32
            active = reused
        else:
            active = kv
        assert active.resume(stream)
        assert active.resize(33)
        new_slot = active.get_ssm_block_base_index(group)
        for layer_id, fields in enumerate(expected):
            cache = mgr.mamba_layer_cache(layer_id)
            for name, value in fields.items():
                torch.testing.assert_close(getattr(cache, name)[new_slot], value, rtol=0, atol=0)
        torch.cuda.synchronize()
    finally:
        for other in pressure:
            other.close()
        if reused is not None:
            reused.close()
        if kv is not None:
            kv.close()
        mgr.shutdown()


@skip_no_cuda
def test_v2_kda_asymmetric_context_generation_transfer(monkeypatch):
    """A context server without replay caches and a generation server with them describe the same transferable
    recurrent state, and the generation server seeds its replay caches from the transferred conv state."""
    from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import (
        MambaPolicy,
        mamba_receiver_payload_bytes,
    )
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    ctx = _build_kda_replay_manager(is_disagg=True)
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "generation")
    gen = _build_kda_replay_manager(is_disagg=True)
    try:
        assert ctx.kda_context_only and not gen.kda_context_only
        for mgr in (ctx, gen):
            _run_v2_hybrid_context(mgr, _make_v2_conversation_request(101, [1, 2], "transfer"))
        src_slot = ctx.get_state_indices([101])[0]
        dst_slot = gen.get_state_indices([101])[0]
        src_table = build_page_table_from_manager(ctx)
        dst_table = build_page_table_from_manager(gen)
        src_rank = RankInfo.from_kv_cache_manager("ctx", ctx, device_id=0)
        dst_rank = RankInfo.from_kv_cache_manager("gen", gen, device_id=0)
        MambaPolicy.validate_peer_compatible(src_rank, dst_rank, src_table, dst_table)
        src_group = next(g for g in src_table.layer_groups if isinstance(g, MambaLayerGroup))
        dst_group = next(g for g in dst_table.layer_groups if isinstance(g, MambaLayerGroup))
        for group in (src_group, dst_group):
            assert {pv.pool_role for pv in group.pool_views} == {MAMBA_CONV_ROLE, MAMBA_SSM_ROLE}
        assert mamba_receiver_payload_bytes(src_table, dst_table, dst_slot) == sum(
            gen.mamba_layer_cache(i).conv[dst_slot].numel() * 2
            + gen.mamba_layer_cache(i).temporal[dst_slot].numel() * 4
            for i in range(2)
        )

        # What the transfer writes into the generation slot.
        for layer_id in range(2):
            source = ctx.mamba_layer_cache(layer_id)
            target = gen.mamba_layer_cache(layer_id)
            source.conv[src_slot].fill_(layer_id + 1)
            source.temporal[src_slot].fill_(layer_id + 2)
            target.conv[dst_slot].copy_(source.conv[src_slot])
            target.temporal[dst_slot].copy_(source.temporal[src_slot])
        gen.seed_kda_replay_caches_for_disagg_gen([101])
        for layer_id in range(2):
            source = ctx.mamba_layer_cache(layer_id)
            target = gen.mamba_layer_cache(layer_id)
            assert target.prev_num_accepted_tokens[dst_slot] == 0
            for section, name in enumerate(_KDA_REPLAY_BUFFERS[:3]):
                torch.testing.assert_close(
                    getattr(target, name)[dst_slot, :, :3],
                    source.conv[src_slot, section * 48 : (section + 1) * 48].to(torch.bfloat16),
                )
    finally:
        ctx.shutdown()
        gen.shutdown()


def _kda_cache_budget(monkeypatch, available, model, mapping, **kwargs):
    from tensorrt_llm._torch.modules.kimi_kda import _kda_kernels

    monkeypatch.setattr(_kda_kernels, "is_kda_mtp_verify_available", lambda: available)
    monkeypatch.setattr(_kda_kernels, "kda_mtp_max_num_spec", lambda: 7)
    return KimiK3HybridCacheManager.get_cache_size_per_token(model, mapping, **kwargs)


@pytest.mark.parametrize(
    "role,is_disagg,available",
    [
        ("context", True, True),
        ("generation", True, True),
        (None, True, True),
        ("context", False, True),
        ("generation", True, False),
    ],
)
@pytest.mark.parametrize("interval,replay_slope", [(0, 0), (32, 95), (33, 92), (48, 63), (256, 0)])
def test_kda_cache_budget_accounts_for_replay_role(
    monkeypatch, role, is_disagg, available, interval, replay_slope
):
    if role is None:
        monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    model = _kimi_model_config()
    spec_config = MTPDecodingConfig(max_draft_len=2)
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    kwargs = dict(
        max_batch_size=2,
        max_seq_len=128,
        tokens_per_block=32,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,
            mamba_state_config=MambaStateConfig(periodic_snapshot_interval=interval),
        ),
        spec_config=spec_config,
        is_disagg=is_disagg,
    )
    baseline = _kda_cache_budget(monkeypatch, False, model, mapping, **kwargs)
    result = _kda_cache_budget(monkeypatch, available, model, mapping, **kwargs)
    if available and not (is_disagg and role == "context"):
        # Two layers: three BF16 [5,32] conv windows, two BF16 and one
        # FP32 [2,32] histories and BF16 [2,8] padded beta per layer;
        # one shared count per slot, padded to 16 bytes.
        per_layer = 3 * 5 * 32 * 2 + 2 * 32 * 8 + 2 * 8 * 2
        # At interval 33, round the combined recurrent + replay snapshot
        # cost once: ceil(6224 / 33) - ceil(3200 / 33) = 92 bytes/token.
        assert result[0] - baseline[0] == replay_slope
        live_and_dummy_slots = 2 + _get_num_cuda_graph_padding_dummy_slots(spec_config, 2)
        assert result[1] - baseline[1] == live_and_dummy_slots * (2 * per_layer + 16)
    else:
        assert result == baseline


@pytest.mark.parametrize("enable_block_reuse", [False, True])
@pytest.mark.parametrize("enable_attention_dp", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("pp_partition,stage_kda_layers", [(None, [2, 0]), ([2, 2], [1, 1])])
def test_kda_cache_budget_counts_live_dummy_and_fixed_snapshot_slots(
    monkeypatch, enable_block_reuse, enable_attention_dp, rank, pp_partition, stage_kda_layers
):
    model = _kimi_model_config()
    mapping = Mapping(
        world_size=2,
        rank=rank,
        tp_size=1,
        pp_size=2,
        enable_attention_dp=enable_attention_dp,
        pp_partition=pp_partition,
    )
    spec_config = MTPDecodingConfig(max_draft_len=2, draft_len_schedule={1: 2, 2: 1})
    kwargs = dict(
        max_batch_size=4,
        max_seq_len=128,
        tokens_per_block=32,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=enable_block_reuse,
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=0,
                additional_snapshot_offsets_from_start=[32, 64, 64, 256],
                additional_snapshot_offsets_from_end=[0, 32, 128],
            ),
        ),
        spec_config=spec_config,
        is_disagg=False,
    )
    baseline = _kda_cache_budget(monkeypatch, False, model, mapping, **kwargs)
    result = _kda_cache_budget(monkeypatch, True, model, mapping, **kwargs)
    # Without an explicit partition, splitting the four target layers plus
    # one MTP layer puts both KDA layers on rank 0. Partition [2,2] places one
    # KDA layer on each rank, with the MTP layer appended to rank 1.
    # Every live slot (max_batch_size * pp_size), every CUDA-graph padding
    # dummy plus one more for attention DP, and with reuse four reachable,
    # distinct snapshot rules per resident sequence carry replay caches.
    local_kda_layers = stage_kda_layers[rank]
    replay_bytes_per_slot = 1504 * local_kda_layers + (16 if local_kda_layers else 0)
    slots = (
        8
        + _get_num_cuda_graph_padding_dummy_slots(spec_config, 4)
        + int(enable_attention_dp)
        + (8 * 4 if enable_block_reuse else 0)
    )
    assert result == (baseline[0], baseline[1] + slots * replay_bytes_per_slot)


def test_v2_kda_replay_seeds_disaggregated_generation_slots():
    mgr = object.__new__(KimiK3HybridCacheManager)
    mgr._kda_replay = KDAReplayState(2)
    mgr.local_num_mamba_layers = 2
    mgr.conv_state_shape = [6, 4]
    mgr.conv_section_dims = [2, 2, 2]
    mgr._request_id_to_state_index = {101: 1, 202: 3}
    mgr._kda_replay.prev_num_accepted_tokens = torch.full((4,), 5, dtype=torch.int32)
    mgr.all_conv_states = [
        torch.arange(4 * 6 * 4, dtype=torch.float32).reshape(4, 6, 4),
        torch.arange(4 * 6 * 4, dtype=torch.float32).reshape(4, 6, 4) + 1000,
    ]

    def dim_contiguous_conv_cache() -> torch.Tensor:
        return torch.full((2, 4, 6, 2), 7.0).transpose(-1, -2)

    replay = {}
    replay["kda_conv_q"] = dim_contiguous_conv_cache()
    replay["kda_conv_k"] = dim_contiguous_conv_cache()
    replay["kda_conv_v"] = dim_contiguous_conv_cache()
    replay["kda_k_cache"] = torch.full((2, 4, 2, 2), 7.0)
    replay["kda_g_cache"] = torch.full((2, 4, 2, 2), 7.0)
    replay["kda_v_cache"] = torch.full((2, 4, 2, 2), 7.0)
    replay["kda_beta_cache"] = torch.full((2, 4, 2, 1), 7.0)

    mgr._kda_replay.pool_layers = [
        {name: buffer[layer] for name, buffer in replay.items()} for layer in range(2)
    ]

    mgr._kda_replay._conv_states = tuple(mgr.all_conv_states)
    mgr._kda_replay._conv_section_dims = tuple(mgr.conv_section_dims)
    mgr._kda_replay._committed_window = mgr.conv_state_shape[1]
    mgr.seed_kda_replay_caches_for_disagg_gen([101, 202])

    for layer_offset, conv_state in enumerate(mgr.all_conv_states):
        for slot in (1, 3):
            for replay_buffer, start in (
                (replay["kda_conv_q"], 0),
                (replay["kda_conv_k"], 2),
                (replay["kda_conv_v"], 4),
            ):
                torch.testing.assert_close(
                    replay_buffer[layer_offset, slot, :, :4],
                    conv_state[slot, start : start + 2, :],
                )
                assert torch.count_nonzero(replay_buffer[layer_offset, slot, :, 4:]) == 0
            assert torch.count_nonzero(replay["kda_k_cache"][layer_offset, slot]) == 0
            assert torch.count_nonzero(replay["kda_g_cache"][layer_offset, slot]) == 0
            assert torch.count_nonzero(replay["kda_v_cache"][layer_offset, slot]) == 0
            assert torch.count_nonzero(replay["kda_beta_cache"][layer_offset, slot]) == 0
            assert mgr._kda_replay.prev_num_accepted_tokens[slot] == 0
    assert mgr._kda_replay.prev_num_accepted_tokens[0] == 5
    assert mgr._kda_replay.prev_num_accepted_tokens[2] == 5


_KDA_REPLAY_FIELDS = (*_KDA_REPLAY_BUFFERS, "prev_num_accepted_tokens")


def _kda_replay_seed_state(stacked: bool):
    """Two KDA layers whose state and replay buffers are per-layer views of
    one allocation each, like the KVCM pool views."""
    generator = torch.Generator().manual_seed(0)
    state = KDAReplayState(2)
    state.prev_num_accepted_tokens = torch.full((4,), 5, dtype=torch.int32, device="cuda")
    conv = torch.randn(2, 4, 6, 4, generator=generator).to("cuda", torch.bfloat16)
    state._conv_states = tuple(conv[layer] for layer in range(2))
    state._conv_section_dims = (2, 2, 2)
    state._committed_window = 4
    replay = {
        name: torch.randn(2, 4, 6, 2, generator=generator)
        .to("cuda", torch.bfloat16)
        .transpose(-1, -2)
        for name in ("kda_conv_q", "kda_conv_k", "kda_conv_v")
    }
    for name, dtype in (
        ("kda_k_cache", torch.bfloat16),
        ("kda_g_cache", torch.float32),
        ("kda_v_cache", torch.bfloat16),
    ):
        replay[name] = torch.randn(2, 4, 2, 2, generator=generator).to("cuda", dtype)
    replay["kda_beta_cache"] = torch.randn(2, 4, 2, 4, generator=generator).to(
        "cuda", torch.bfloat16
    )[..., :2]
    state.pool_layers = [
        {name: buffer[layer] for name, buffer in replay.items()} for layer in range(2)
    ]
    if stacked:
        state._stack_seed_views()
        assert state._stacked_seed_replay is not None
    return state, replay


@skip_no_cuda
def test_v2_kda_replay_seed_of_all_layers_at_once_is_bit_identical():
    per_layer, per_layer_replay = _kda_replay_seed_state(stacked=False)
    stacked, stacked_replay = _kda_replay_seed_state(stacked=True)
    slots = torch.tensor([1, 3], device="cuda")

    per_layer.seed_transferred_slots(slots)
    stacked.seed_transferred_slots(slots)

    for name, buffer in per_layer_replay.items():
        assert torch.equal(stacked_replay[name], buffer), name
    assert torch.equal(stacked.prev_num_accepted_tokens, per_layer.prev_num_accepted_tokens)
    # The seeded slots hold the committed windows, the others are untouched.
    assert torch.equal(
        stacked_replay["kda_conv_k"][1, 3, :, :4], stacked._conv_states[1][3, 2:4, :]
    )
    assert stacked.prev_num_accepted_tokens.tolist() == [5, 0, 5, 0]
