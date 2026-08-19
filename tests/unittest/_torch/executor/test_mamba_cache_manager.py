# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Python, Cpp, and V2 Mamba cache managers."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup, MambaLayerGroup
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
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
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import BlockReusePolicy, KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import (
    ATTENTION_DP_DUMMY_REQUEST_ID,
    LlmRequest,
    SamplingConfig,
)
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    MIN_REPLAY_HISTORY_SIZE,
    CppMambaHybridCacheManager,
    MambaHybridCacheManagerV2,
    MambaRole,
    MixedMambaHybridCacheManager,
    PythonMambaCacheManager,
    ReplayStateUpdateMetadata,
    _advance_replay_state,
    _get_local_mamba_cache_layout,
    _get_mamba_hybrid_pool_size,
    _get_num_cuda_graph_padding_dummy_slots,
    _mamba_snapshot_rule_counts,
    _promote_mamba_state_triton,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    KVCacheManager,
    get_pp_layers,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings.internal.batch_manager import LinearCacheType
from tensorrt_llm.conversation_params import ConversationParams
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
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    GpuCacheTierConfig,
    KVCacheDesc,
    KVCacheManagerConfig,
    LayerId,
    SsmLayerConfig,
    _introspection,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager as RuntimeKVCacheManager

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
        get_num_mamba_layers=lambda: 1,
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
    assert params.conv_kernel == 5
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


def _kimi_model_config() -> SimpleNamespace:
    config = SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
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
        quant_config=None,
        sparse_attention_config=None,
        get_num_mamba_layers=lambda: 2,
    )


def _capture_kimi_v2_manager_ctor(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[tuple, dict]:
    """Route a Kimi config through _create_kv_cache_manager with an explicit
    V2 manager and capture the constructor arguments."""
    captured: dict[str, object] = {}

    class RecordingV2Manager(MambaHybridCacheManagerV2):
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

    model_config = _kimi_model_config()
    kv_cache_config = KvCacheConfig(use_kv_cache_manager_v2=True)
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)

    assert get_kv_cache_manager_cls(model_config, kv_cache_config) is MambaHybridCacheManagerV2

    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=RecordingV2Manager,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        kv_cache_config=kv_cache_config,
        tokens_per_block=64,
        max_seq_len=2048,
        max_batch_size=4,
        spec_config=None,
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
        5,
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
    assert "kda_replay_num_spec" not in kwargs


def test_kimi_explicit_v2_manager_uses_qkv_convolution_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TRTLLM-15216: MambaHybridCacheManagerV2 takes the KDA conv-state
    sectioning by `conv_state_layout`, not by `model_type`. Passing
    `model_type` instead is silently swallowed by **kwargs and leaves the
    default 'x_b_c' layout, i.e. a wrong KDA conv state with no error."""
    _, kwargs = _capture_kimi_v2_manager_ctor(monkeypatch)
    assert kwargs["conv_state_layout"] == "q_k_v"
    assert "model_type" not in kwargs


def test_kimi_v1_manager_still_selects_qwen3_next_model_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The V1 managers have no `conv_state_layout` parameter; they must keep
    getting `model_type='qwen3_next'` (TRTLLM-15216 regression guard)."""
    captured: dict[str, object] = {}

    class RecordingV1Manager(CppMambaHybridCacheManager):
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

    model_config = _kimi_model_config()
    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=RecordingV1Manager,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        kv_cache_config=KvCacheConfig(),
        tokens_per_block=64,
        max_seq_len=2048,
        max_batch_size=4,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=False,
    )
    kwargs = captured["kwargs"]
    assert kwargs["model_type"] == "qwen3_next"
    assert "conv_state_layout" not in kwargs


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
        (True, False, MambaHybridCacheManagerV2),
        (True, True, MambaHybridCacheManagerV2),
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
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor._util.get_sm_version", lambda: 90)
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
        model_engine=None,
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

    assert captured_cpp["use_replay_state_update"] is True
    assert captured_cpp["model_type"] == "qwen3_next"
    assert captured_cpp["max_num_tokens"] == 256
    assert captured_mixed["use_replay_state_update"] is False
    assert captured_mixed["model_type"] == "qwen3_next"
    assert captured_mixed["max_num_tokens"] == 256
    assert captured_v2["use_replay_state_update"] is True
    assert captured_v2["max_num_tokens"] == 256
    assert "model_type" not in captured_v2
    assert captured_v2["conv_state_layout"] == "q_k_v"
    fallback_logs = [str(call.args[0]) for call in info_log.call_args_list]
    assert any("RecordingMixedManager was selected" in log for log in fallback_logs)
    assert not any("RecordingV2Manager was selected" in log for log in fallback_logs)


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
        is MambaHybridCacheManagerV2
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
    from tensorrt_llm._torch.models.modeling_qwen3_5 import Qwen3_5VLModel
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextForCausalLM

    for env_var in (
        "TRTLLM_USE_NIXL_KVCACHE",
        "TRTLLM_USE_UCX_KVCACHE",
        "TRTLLM_USE_MOONCAKE_KVCACHE",
        "TRTLLM_USE_MPI_KVCACHE",
    ):
        monkeypatch.delenv(env_var, raising=False)

    for model_cls in (NemotronHForCausalLM, Qwen3NextForCausalLM, Qwen3_5VLModel):
        llm_args = TorchLlmArgs(
            model="/tmp/dummy_model",
            cache_transceiver_config=CacheTransceiverConfig(backend="DEFAULT"),
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
    expected_manager = MambaHybridCacheManagerV2 if expected_v2 else CppMambaHybridCacheManager
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
        CacheTransceiverConfig(backend="NIXL"),  # runtime left at 'auto'
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
    'auto_unresolved' case covers paths that skip model-default resolution
    (e.g. AutoDeploy), where 'auto' falls back to the C++ runtime."""
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)

    with pytest.raises(ValueError, match="Kimi K3 disaggregated serving requires"):
        get_kv_cache_manager_cls(
            _kimi_model_config(),
            KvCacheConfig(enable_block_reuse=False),
            is_disagg=True,
            cache_transceiver_config=cache_transceiver_config,
        )


def test_kimi_disagg_python_nixl_routes_to_mixed_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)

    assert (
        get_kv_cache_manager_cls(
            _kimi_model_config(),
            KvCacheConfig(enable_block_reuse=False),
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

    kv_slice = transceiver._create_kv_slice(request)

    assert kv_slice.mamba_state_index is None


def test_v2_disagg_slice_reads_state_index_without_refreshing_batch_mask():
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.local_num_mamba_layers = 1
    manager._request_id_to_state_index = {123: 7}
    manager.get_state_indices = MagicMock(
        side_effect=AssertionError("state-index lookup must not refresh the dummy mask")
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._kv_cache_manager = manager
    transceiver._reuse_adapter = SimpleNamespace(tokens_per_block=32)
    transceiver._page_table = SimpleNamespace(layer_groups=[])
    request = SimpleNamespace(
        is_generation_only_request=False,
        prompt_len=0,
        py_request_id=123,
    )

    kv_slice = transceiver._create_kv_slice(request)

    assert kv_slice.mamba_state_index == 7
    manager.get_state_indices.assert_not_called()


@pytest.mark.parametrize(
    "max_beam_width, has_connector, expected",
    [
        (2, False, "max_beam_width > 1"),
        (1, True, "kv_connector_manager"),
        (2, True, "kv_connector_manager, max_beam_width > 1"),
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
    mgr._dummy_request_mask = torch.tensor([False, True])
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
        "tensorrt_llm._torch.pyexecutor.mamba_cache_manager._promote_mamba_state_triton",
        lambda *args, **kwargs: promote_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.mamba_cache_manager._advance_replay_state",
        lambda *args, **kwargs: pytest.fail(
            "Cpp replay bookkeeping must not advance before conv promotion"
        ),
    )

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
def test_update_mamba_states_autodeploy_path():
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
    assert mgr._num_ssm_states_per_typical_request(48, periodic_config) == 1


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
        "tensorrt_llm._torch.pyexecutor.mamba_cache_manager._get_local_mamba_cache_layout",
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
        [KVCacheDesc(capacity=32, history_length=31)] * 6
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


def test_v2_hybrid_warns_when_avg_seq_len_is_missing(monkeypatch):
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.max_seq_len = 4096
    warnings_seen = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.mamba_cache_manager.logger.warning",
        lambda message: warnings_seen.append(message),
    )

    capacity = mgr._get_typical_request_capacity(KvCacheConfig())

    assert capacity == 2048
    assert len(warnings_seen) == 1
    assert "kv_cache_config.avg_seq_len" in warnings_seen[0]
    assert "max_seq_len / 2=2048" in warnings_seen[0]
    assert "workload's average total sequence length" in warnings_seen[0]


def test_v2_hybrid_rejects_quota_below_live_state_floor():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr._has_cp_helix = False
    mgr.max_batch_size = 2
    mgr.mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    mgr.local_num_mamba_layers = 1
    mgr.ssm_bytes = 64
    mgr.conv_bytes = 32
    mgr._num_reserved_dummy_slots = 1
    mgr.tokens_per_block = 32
    mgr.num_local_layers = 2
    mgr.pp_layers = [0, 1]
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


def test_v2_hybrid_pure_mamba_rank_does_not_reserve_attention_page():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr._has_cp_helix = False
    mgr.max_batch_size = 2
    mgr.mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    mgr.local_num_mamba_layers = 1
    mgr.ssm_bytes = 64
    mgr.conv_bytes = 32
    mgr._num_reserved_dummy_slots = 1
    mgr.tokens_per_block = 32
    mgr.num_local_layers = 1
    mgr.pp_layers = [0]
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
    mgr.cuda_state_indices = torch.zeros(1, dtype=torch.int32)
    mgr._row_indices = torch.arange(1, dtype=torch.long)
    mgr._request_id_to_state_index = {}
    mgr._request_id_to_is_dummy = {}
    mgr._dummy_request_mask = None

    mgr._setup_state_indices()

    assert mgr._host_state_indices.tolist() == [expected_offset]
    assert mgr.cuda_state_indices.tolist() == [expected_offset]
    assert mgr.get_state_indices([request.py_request_id], [False]) == [expected_offset]


def test_v2_block_reuse_commit_saves_ssm_snapshot_at_snapshot_point():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.enable_block_reuse = True
    mgr.is_draft = False
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
        get_tokens=lambda beam_idx: token_ids,
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
            statistics = _introspection.storage_statistics(runtime_manager)

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
    enable_attention_dp=False,
    enable_swa_scratch_reuse=False,
    dtype=DataType.HALF,
    conv_state_layout="x_b_c",
):
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
    kv_cache_config = KvCacheConfig(
        max_tokens=512,
        enable_block_reuse=enable_block_reuse,
        enable_partial_reuse=enable_partial_reuse,
        block_reuse_config=BlockReuseConfig(
            policy=block_reuse_policy,
            max_num_turns=max_num_turns,
        ),
        enable_swa_scratch_reuse=enable_swa_scratch_reuse,
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=periodic_snapshot_interval,
            additional_snapshot_offsets_from_end=list(additional_snapshot_offsets_from_end or []),
        ),
        dtype="nvfp4" if dtype == DataType.NVFP4 else "auto",
    )
    return MambaHybridCacheManagerV2(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=num_mamba_layers,
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float16,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_attention_layers,
        num_kv_heads=num_kv_heads,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=max_batch_size,
        mapping=mapping,
        spec_config=spec_config,
        layer_mask=attn_mask,
        vocab_size=1024,
        use_replay_state_update=use_replay_state_update,
        dtype=dtype,
        conv_state_layout=conv_state_layout,
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
) -> LlmRequest:
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    request.py_conversation_params = ConversationParams(conversation_id=conversation_id)
    return request


def _run_v2_hybrid_context(
    manager: MambaHybridCacheManagerV2,
    request: LlmRequest,
) -> None:
    manager.prepare_expect_snapshot_points([request])
    assert manager.prepare_context(request)
    num_tokens = request.context_remaining_length
    assert manager.resize_context(request, num_tokens=num_tokens)

    batch = ScheduledRequests()
    batch.append_context_request(request)
    manager.prepare_resources(batch)
    request.context_current_position = request.prompt_len
    assert request.context_remaining_length == 0
    manager.update_context_resources(batch)


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
def test_v2_hybrid_nvfp4_page_table_omits_ssm_block_scales():
    mgr = _build_v2_hybrid_with_mamba_layer(dtype=DataType.NVFP4)
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
        assert mgr.index_mapper.num_free_slots() == len(request_ids) + 5

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
        assert mgr.index_mapper.num_free_slots() == 0
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
        pytest.param(
            0,
            1,
            marks=pytest.mark.xfail(
                reason="MambaHybridCacheManagerV2 on this branch does not "
                "override _create_kv_cache with the rank-0 prefix-reuse "
                "debug log yet. Runtime-side logging is a follow-up "
                "(TRTLLM-14813).",
                strict=True,
            ),
        ),
        (3, 0),
    ],
)
def test_v2_hybrid_debug_logs_prefix_reuse_only_on_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    expected_log_count: int,
) -> None:
    get_num_tokens_before_hybrid_pruning = MagicMock(return_value=96)
    kv_cache = SimpleNamespace(
        _get_num_tokens_before_hybrid_pruning=get_num_tokens_before_hybrid_pruning,
        num_committed_tokens=64,
    )
    create_kv_cache = MagicMock(return_value=kv_cache)
    log_debug = MagicMock()
    monkeypatch.setattr(KVCacheManagerV2, "_create_kv_cache", create_kv_cache)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.mamba_cache_manager.logger.debug", log_debug
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
    assert get_num_tokens_before_hybrid_pruning.call_count == expected_log_count
    if rank == 0:
        log_debug.assert_called_once_with(
            "[MambaHybridCacheManagerV2] prefix reuse rank=0 request_id=123 "
            "request_total_tokens=128 "
            "longest_attention_match_tokens=96 "
            "latest_recurrent_snapshot_tokens=64"
        )


@pytest.mark.xfail(
    reason="MambaHybridCacheManagerV2 on this branch does not override "
    "get_iteration_stats with the recurrent-pool aggregation counters "
    "(_recurrent_evicted_blocks_total et al.) or the rank-0 status log. "
    "Runtime-side accounting is a follow-up (TRTLLM-14813).",
    strict=True,
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
    log_info = MagicMock()
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.mamba_cache_manager.logger.info", log_info)

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

    assert mgr.get_iteration_stats() is report
    assert mgr._recurrent_evicted_blocks_total == 5
    assert mgr._recurrent_onboarded_blocks_total == 5
    assert mgr._recurrent_dropped_blocks_total == 3
    assert log_info.call_count == expected_log_count
    if rank == 0:
        log_info.assert_called_once_with(
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
            "host_used_recurrent_blocks=15 host_free_recurrent_blocks=19"
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
    )
    request_a = _make_v2_conversation_request(1, list(range(64)), "conv-1")
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

    try:
        _run_v2_hybrid_context(mgr, request_a)
        request_a_state_index = mgr.get_state_indices([request_a.py_request_id], [False])[0]
        mgr.free_resources(request_a)
        _run_v2_hybrid_context(mgr, request_b)
        request_b_state_index = mgr.get_state_indices([request_b.py_request_id], [False])[0]
        mgr.free_resources(request_b)

        mgr.prepare_expect_snapshot_points([request_a_probe])
        assert mgr.prepare_context(request_a_probe)
        probe_batch = ScheduledRequests()
        probe_batch.append_context_request(request_a_probe)
        mgr.prepare_resources(probe_batch)
        assert request_a_probe.prepopulated_prompt_len == request_a_probe.prompt_len - 1
        assert mgr.get_state_indices([request_a_probe.py_request_id], [False]) == [
            request_a_state_index
        ]
        mgr.free_resources(request_a_probe)

        mgr.prepare_expect_snapshot_points([request_b_probe])
        assert mgr.prepare_context(request_b_probe)
        probe_batch = ScheduledRequests()
        probe_batch.append_context_request(request_b_probe)
        mgr.prepare_resources(probe_batch)
        assert request_b_probe.prepopulated_prompt_len == request_b_probe.prompt_len - 1
        assert mgr.get_state_indices([request_b_probe.py_request_id], [False]) == [
            request_b_state_index
        ]
        mgr.free_resources(request_b_probe)

        _run_v2_hybrid_context(mgr, request_c)
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
        assert mgr.get_state_indices([request_b_after_eviction.py_request_id], [False]) == [
            request_b_state_index
        ]
    finally:
        for request in requests:
            if request.py_request_id in mgr.kv_cache_map:
                mgr.free_resources(request)
        mgr.shutdown()


def test_v2_hybrid_saves_conversation_plan_only_after_final_context_chunk():
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.enable_block_reuse = True
    mgr.is_draft = False
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
        get_tokens=lambda beam_idx: list(range(128)),
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
        assert mamba_group.conv_section_bytes == [
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
        assert not mgr.use_replay_state_update
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
        assert mgr.use_replay_state_update is True
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
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=16,
        num_mamba_layers=3,
        spec_config=MTPDecodingConfig(max_draft_len=3),
        use_replay_state_update=True,
        conv_state_layout="q_k_v",
    )
    try:
        assert mgr.use_gdn_cached_replay_all_layer_commit
        assert mgr._gdn_cached_replay_state_descriptors is None
        state_strides = mgr._gdn_cached_replay_state_strides
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
    mgr = _build_v2_hybrid_with_mamba_layer(
        max_batch_size=batch_size,
        num_mamba_layers=3,
        spec_config=MTPDecodingConfig(max_draft_len=3),
        use_replay_state_update=True,
        conv_state_layout="q_k_v",
    )
    try:
        torch.manual_seed(1234)
        for state in mgr.all_ssm_states:
            state.normal_()
        mgr.old_x.normal_()
        mgr.old_B.normal_()
        mgr.old_dt.uniform_(-0.2, -0.01)

        positions = torch.arange(batch_size, device="cuda", dtype=torch.int32)
        replay_work_items = torch.stack(
            (
                positions,
                positions,
                torch.full_like(positions, mgr.replay_history_size),
                torch.zeros_like(positions),
            ),
            dim=1,
        )
        n_writes = torch.tensor([batch_size], device="cuda", dtype=torch.int32)
        expected = torch.stack(mgr.all_ssm_states)
        commit_gdn_cached_replay_history_layers(
            ssm_states=expected,
            old_u=mgr.old_x,
            old_k=mgr.old_B,
            old_G=mgr.old_dt,
            replay_work_items=replay_work_items,
            n_writes=n_writes,
            history_size=mgr.replay_history_size,
        )

        mgr._commit_gdn_cached_replay_history_layers(
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
    mgr = object.__new__(MambaHybridCacheManagerV2)
    batch_size = 16
    mgr.local_num_mamba_layers = 1
    mgr._use_replay_state_update = True
    mgr._use_gdn_cached_replay_all_layer_commit = True
    mgr.replay_step_width = 4
    mgr.replay_history_size = MIN_REPLAY_HISTORY_SIZE
    mgr.prev_num_accepted_tokens = torch.zeros(batch_size, dtype=torch.int32)
    mgr.cache_buf_idx = torch.zeros(batch_size, dtype=torch.int32)
    mgr.intermediate_state_indices = torch.arange(batch_size, dtype=torch.int32)
    mgr._dummy_request_mask = torch.zeros(batch_size, dtype=torch.bool)
    mgr.all_conv_states = [torch.empty(0)]
    mgr.intermediate_conv_states = torch.empty(0)

    events = []
    mgr._commit_gdn_cached_replay_history_layers = lambda *_args, **_kwargs: events.append("commit")
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.mamba_cache_manager._advance_replay_state",
        lambda *_args, **_kwargs: events.append("advance"),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.mamba_cache_manager._promote_mamba_state_triton",
        lambda *_args, **_kwargs: None,
    )

    mgr.update_mamba_states(
        SimpleNamespace(num_seqs=batch_size, num_contexts=0),
        torch.full((batch_size,), 3, dtype=torch.int32),
        state_indices=torch.arange(batch_size, dtype=torch.int32),
    )

    assert events == ["commit", "advance"]


def test_v2_hybrid_replay_update_skips_dummy_and_padding_rows(monkeypatch):
    mgr = object.__new__(MambaHybridCacheManagerV2)
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
    mgr._dummy_request_mask = torch.zeros(4, dtype=torch.bool)
    mgr._dummy_request_mask_host = torch.zeros(4, dtype=torch.bool)
    mgr._use_replay_state_update = True
    mgr.replay_step_width = 5
    mgr.replay_history_size = 16
    mgr.prev_num_accepted_tokens = torch.full((4,), 13, dtype=torch.int32)
    mgr.cache_buf_idx = torch.ones(4, dtype=torch.int32)
    mgr.intermediate_state_indices = torch.arange(4, dtype=torch.int32)
    mgr.all_ssm_states = []
    mgr.all_conv_states = [torch.empty(0)]
    mgr.intermediate_conv_states = torch.empty(0)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.mamba_cache_manager._promote_mamba_state_triton",
        lambda *args, **kwargs: None,
    )

    request_ids = [100, 101, 102, 103]
    state_indices = torch.tensor(
        mgr.get_state_indices(request_ids, [False, False, False, True]),
        dtype=torch.int32,
    )
    assert mgr._dummy_request_mask.tolist() == [False, False, True, True]

    mgr.update_mamba_states(
        SimpleNamespace(num_seqs=4, num_contexts=1),
        torch.tensor([1, 3, 3, 3], dtype=torch.int32),
        state_indices=state_indices,
    )

    assert mgr.prev_num_accepted_tokens.tolist() == [13, 3, 13, 13]
    assert mgr.cache_buf_idx.tolist() == [1, 0, 1, 1]


def test_v2_hybrid_dynamic_tree_promotes_accepted_leaf_state(monkeypatch):
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.local_num_mamba_layers = 1
    mgr._use_replay_state_update = False
    mgr.intermediate_state_indices = torch.arange(2, dtype=torch.int32)
    mgr.all_ssm_states = [torch.empty(0)]
    mgr.all_conv_states = [torch.empty(0)]
    mgr.intermediate_ssm_states = torch.empty((1, 2, 8))
    mgr.intermediate_conv_states = torch.empty((1, 2, 8))

    promoted_positions = []

    def capture_promoted_position(_dst, _src, _src_indices, positions, _dst_indices):
        promoted_positions.append(positions.clone())

    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.mamba_cache_manager._promote_mamba_state_triton",
        capture_promoted_position,
    )

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
