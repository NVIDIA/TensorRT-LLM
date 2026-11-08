# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Unit tests for speculative modeling classes."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch import model_config as model_config_module
from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.model_config import ModelConfig, MoESelection
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
from tensorrt_llm._torch.models.modeling_speculative import (
    Eagle3ForCausalLM,
    MTPForCausalLM,
    SpecDecOneEngineForCausalLM,
    _build_mtp_one_model_draft,
    _copy_model_config_with_moe_selection,
    _draft_moe_selection,
    _get_mtp_moe_quant_config,
    _replacement_mtp_model_config,
    _set_draft_kv_cache_quant_algo,
    external_drafter_config_kwargs,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig, MoePerLayerDynamicLoadBalancerConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo, QuantMode


@pytest.mark.cpu_only
def test_draft_kv_cache_quant_algo_override_updates_all_layers() -> None:
    target = SimpleNamespace(
        quant_config=SimpleNamespace(kv_cache_quant_algo=QuantAlgo.NVFP4),
        extra_attrs={"draft_kv_cache_quant_algo_override": QuantAlgo.FP8},
    )
    layer_quant_0 = SimpleNamespace(kv_cache_quant_algo=QuantAlgo.NVFP4)
    layer_quant_1 = SimpleNamespace(kv_cache_quant_algo=QuantAlgo.NVFP4)
    draft = SimpleNamespace(
        quant_config=SimpleNamespace(kv_cache_quant_algo=QuantAlgo.NVFP4),
        quant_config_dict={
            "model.layers.0": layer_quant_0,
            "model.layers.1": layer_quant_1,
        },
    )

    _set_draft_kv_cache_quant_algo(draft, target)

    assert draft.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert layer_quant_0.kv_cache_quant_algo == QuantAlgo.FP8
    assert layer_quant_1.kv_cache_quant_algo == QuantAlgo.FP8
    assert target.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4


@pytest.mark.cpu_only
@pytest.mark.parametrize("target_algo", [None, QuantAlgo.FP8, QuantAlgo.NVFP4])
def test_draft_kv_cache_quant_algo_inheritance_preserves_layer_settings(
    target_algo: QuantAlgo | None,
) -> None:
    """Without an override, inherit only the global setting as before."""
    target = SimpleNamespace(
        quant_config=SimpleNamespace(kv_cache_quant_algo=target_algo),
        extra_attrs={},
    )
    layer_algos = (None, QuantAlgo.FP8, QuantAlgo.NVFP4)
    layer_configs = {
        f"model.layers.{i}": SimpleNamespace(kv_cache_quant_algo=algo)
        for i, algo in enumerate(layer_algos)
    }
    draft = SimpleNamespace(
        quant_config=SimpleNamespace(kv_cache_quant_algo=QuantAlgo.FP8),
        quant_config_dict=layer_configs,
    )

    _set_draft_kv_cache_quant_algo(draft, target)

    assert draft.quant_config.kv_cache_quant_algo == target_algo
    assert draft.quant_config_dict is layer_configs
    assert tuple(config.kv_cache_quant_algo for config in layer_configs.values()) == layer_algos
    assert target.quant_config.kv_cache_quant_algo == target_algo


@pytest.mark.cpu_only
@pytest.mark.parametrize("override", [None, QuantAlgo.FP8])
def test_draft_kv_cache_quant_algo_without_layer_settings(override: QuantAlgo | None) -> None:
    """Global inheritance and overrides also support uniform checkpoints."""
    target = SimpleNamespace(
        quant_config=SimpleNamespace(kv_cache_quant_algo=QuantAlgo.NVFP4),
        extra_attrs={"draft_kv_cache_quant_algo_override": override},
    )
    draft = SimpleNamespace(
        quant_config=SimpleNamespace(kv_cache_quant_algo=None),
        quant_config_dict=None,
    )

    _set_draft_kv_cache_quant_algo(draft, target)

    expected_algo = QuantAlgo.NVFP4 if override is None else override
    assert draft.quant_config.kv_cache_quant_algo == expected_algo
    assert draft.quant_config_dict is None
    assert target.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4


class _FakeDraftModel(nn.Module):
    """Minimal stand-in for Eagle3DraftModel with attributes used by apply_eagle3_fc."""

    def __init__(self, hidden_size, num_capture_layers, dtype, use_fc_norm, use_norm_before_fc):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self._norm_before_fc = use_norm_before_fc

        in_features = hidden_size * num_capture_layers

        if use_fc_norm:
            self.fc_norm = nn.ModuleList(
                [
                    RMSNorm(
                        hidden_size=hidden_size,
                        eps=1e-5,
                        dtype=dtype,
                    )
                    for _ in range(num_capture_layers)
                ]
            ).cuda()
        else:
            self.fc_norm = None

        if use_norm_before_fc:
            self.input_norm = RMSNorm(
                hidden_size=in_features,
                eps=1e-5,
                dtype=dtype,
            ).cuda()
        else:
            self.input_norm = None

        self.fc = nn.Linear(in_features, hidden_size, bias=False, dtype=dtype).cuda()


class _FakeEagle3Wrapper:
    """Minimal wrapper that has a .model attribute for calling the real apply_eagle3_fc."""

    def __init__(self, model):
        self.model = model


@pytest.mark.parametrize("num_capture_layers", [2, 3], ids=["2_layers", "3_layers"])
def test_apply_eagle3_fc_with_fc_norm(num_capture_layers):
    """Test that apply_eagle3_fc correctly chunks, normalizes per-chunk, and projects."""
    torch.manual_seed(42)

    hidden_size = 64
    dtype = torch.float32
    batch_size = 4
    device = "cuda"

    model = _FakeDraftModel(
        hidden_size=hidden_size,
        num_capture_layers=num_capture_layers,
        dtype=dtype,
        use_fc_norm=True,
        use_norm_before_fc=True,  # Should be ignored when fc_norm is set
    )

    wrapper = _FakeEagle3Wrapper(model)

    # Input: concatenated hidden states from multiple capture layers
    in_features = hidden_size * num_capture_layers
    hidden_states = torch.randn(batch_size, in_features, dtype=dtype, device=device)

    # Call the REAL production method as an unbound method
    result = Eagle3ForCausalLM.apply_eagle3_fc(wrapper, hidden_states)

    # Assert output shape is (batch_size, hidden_size)
    assert result.shape == (batch_size, hidden_size), (
        f"Expected shape ({batch_size}, {hidden_size}), got {result.shape}"
    )

    # Verify the fc_norm path was taken by manually computing expected result
    chunks = hidden_states.chunk(num_capture_layers, dim=-1)
    assert len(chunks) == num_capture_layers, (
        f"Expected {num_capture_layers} chunks, got {len(chunks)}"
    )

    # Each chunk should have size hidden_size
    for i, chunk in enumerate(chunks):
        assert chunk.shape == (batch_size, hidden_size), f"Chunk {i} shape mismatch: {chunk.shape}"

    # Manually apply per-chunk norm and concat
    normed_chunks = []
    for norm, chunk in zip(model.fc_norm, chunks):
        normed_chunks.append(norm(chunk))
    normed_concat = torch.cat(normed_chunks, dim=-1)

    # Apply fc
    expected = model.fc(normed_concat)

    # Results should match exactly (same computation path)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)

    # Verify fc_norm takes priority over _norm_before_fc:
    # If we disable fc_norm and enable _norm_before_fc, result should differ
    torch.manual_seed(42)
    model_no_fc_norm = _FakeDraftModel(
        hidden_size=hidden_size,
        num_capture_layers=num_capture_layers,
        dtype=dtype,
        use_fc_norm=False,
        use_norm_before_fc=True,
    )
    # Share same fc weights so only the normalization differs
    model_no_fc_norm.fc.weight.data.copy_(model.fc.weight.data)

    wrapper_no_fc_norm = _FakeEagle3Wrapper(model_no_fc_norm)
    result_norm_before_fc = Eagle3ForCausalLM.apply_eagle3_fc(wrapper_no_fc_norm, hidden_states)

    # The two paths should produce different results (different normalization)
    # Per-chunk norm != whole-tensor norm for non-trivial inputs
    assert not torch.allclose(result, result_norm_before_fc, atol=1e-5), (
        "fc_norm path and _norm_before_fc path produced identical results; "
        "fc_norm should apply per-chunk normalization which differs from "
        "whole-tensor normalization"
    )


# ---------------------------------------------------------------------------
# SpecDecOneEngineForCausalLM: optional hidden_size / vocab_size
# ---------------------------------------------------------------------------

_BASE_CLS = "tensorrt_llm._torch.models.modeling_utils.DecoderModelForCausalLM"


def _init_specdec_with_mocked_base(model_config, **kwargs):
    """Instantiate SpecDecOneEngineForCausalLM with the base class stubbed out.

    DecoderModelForCausalLM is built on the PostInitCaller metaclass, which
    invokes __post_init__/__pp_init__ right after __init__ returns. Those
    hooks must be stubbed too: the mocked __init__ never sets the attributes
    (model_config, lm_head, ...) they rely on.

    Returns the kwargs captured by the mocked base __init__.
    """
    with (
        patch(f"{_BASE_CLS}.__init__", return_value=None) as mock_init,
        patch(f"{_BASE_CLS}.__post_init__"),
        patch(f"{_BASE_CLS}.__pp_init__"),
    ):
        SpecDecOneEngineForCausalLM(MagicMock(), model_config, **kwargs)
    _, captured_kwargs = mock_init.call_args
    return captured_kwargs


def test_specdec_one_engine_reads_from_pretrained_config() -> None:
    """Default path: hidden_size/vocab_size come from pretrained_config."""
    hidden_size = 4096
    vocab_size = 32000
    model_config = ModelConfig(
        pretrained_config=PretrainedConfig(hidden_size=hidden_size, vocab_size=vocab_size)
    )

    kwargs = _init_specdec_with_mocked_base(model_config)
    assert kwargs["hidden_size"] == hidden_size
    assert kwargs["vocab_size"] == vocab_size


def test_specdec_one_engine_accepts_explicit_sizes() -> None:
    """Composite configs (e.g. VL wrappers) can pass sizes explicitly."""
    hidden_size = 8192
    vocab_size = 128256
    # Bare PretrainedConfig lacks hidden_size/vocab_size; the caller
    # supplies them instead.
    model_config = ModelConfig(pretrained_config=PretrainedConfig())

    kwargs = _init_specdec_with_mocked_base(
        model_config, hidden_size=hidden_size, vocab_size=vocab_size
    )
    assert kwargs["hidden_size"] == hidden_size
    assert kwargs["vocab_size"] == vocab_size


def test_specdec_one_engine_explicit_overrides_pretrained_config() -> None:
    """Explicit args take precedence over pretrained_config when both present."""
    hidden_size = 2048
    vocab_size = 64000
    model_config = ModelConfig(
        pretrained_config=PretrainedConfig(hidden_size=4096, vocab_size=32000)
    )

    kwargs = _init_specdec_with_mocked_base(
        model_config, hidden_size=hidden_size, vocab_size=vocab_size
    )
    assert kwargs["hidden_size"] == hidden_size
    assert kwargs["vocab_size"] == vocab_size


def _fake_dflash_attention(rope_params, source):
    if source == "rotary_emb":
        return SimpleNamespace(
            rotary_emb=SimpleNamespace(
                rope_params=rope_params,
                head_dim=128,
                is_neox=True,
            ),
            pos_embd_params=None,
        )
    return SimpleNamespace(
        rotary_emb=None,
        pos_embd_params=SimpleNamespace(rope=rope_params, is_neox=True),
        head_dim=128,
    )


def _fake_dflash_wrapper(rope_params, source):
    layers = [
        SimpleNamespace(self_attn=_fake_dflash_attention(params, source)) for params in rope_params
    ]
    wrapper = DFlashForCausalLM.__new__(DFlashForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = SimpleNamespace(layers=layers)
    wrapper.config = SimpleNamespace(layer_types=["sliding_attention", "full_attention"])
    return wrapper


@pytest.mark.parametrize("source", ["rotary_emb", "pos_embd_params"])
def test_dflash_allows_mixed_layer_types_with_uniform_rope(source):
    rope_params = RopeParams(dim=128, theta=1_000_000.0, max_positions=4096)
    wrapper = _fake_dflash_wrapper([rope_params, rope_params], source)

    DFlashForCausalLM._validate_uniform_rope(wrapper)


@pytest.mark.parametrize("source", ["rotary_emb", "pos_embd_params"])
def test_dflash_rejects_different_effective_rope(source):
    wrapper = _fake_dflash_wrapper(
        [
            RopeParams(dim=128, theta=1_000_000.0, max_positions=4096),
            RopeParams(dim=128, theta=10_000_000.0, max_positions=4096),
        ],
        source,
    )

    with pytest.raises(
        ValueError,
        match=r"layers \[1\] have a different effective RoPE configuration",
    ):
        DFlashForCausalLM._validate_uniform_rope(wrapper)


def _fake_dflash_mask_wrapper(config, is_dflash2=False, sliding_layers_causal=False):
    wrapper = DFlashForCausalLM.__new__(DFlashForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.config = config
    wrapper._is_dflash2 = is_dflash2
    wrapper._sliding_layers_causal = sliding_layers_causal
    return wrapper


def test_dflash_attention_mask_args():
    wrapper = _fake_dflash_mask_wrapper(
        SimpleNamespace(
            num_hidden_layers=4,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=4096,
            use_sliding_window=True,
        )
    )

    assert wrapper._get_attention_mask_args(0) == (True, (4095, 0))
    assert wrapper._get_attention_mask_args(1) == (False, (-1, -1))
    assert wrapper._get_attention_mask_args(2) == (True, (4095, 0))

    with patch("tensorrt_llm._torch.models.modeling_dflash.logger.warning") as warning:
        wrapper._warn_inferred_attention_windows()
    warning.assert_not_called()

    disabled_wrapper = _fake_dflash_mask_wrapper(
        SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=4096,
            use_sliding_window=False,
        )
    )

    assert disabled_wrapper._get_attention_mask_args(0) == (False, (-1, -1))

    missing_window_wrapper = _fake_dflash_mask_wrapper(
        SimpleNamespace(
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
            use_sliding_window=True,
        )
    )

    with pytest.raises(
        ValueError,
        match="use_sliding_window=True requires a positive integer sliding_window",
    ):
        missing_window_wrapper._get_attention_mask_args(0)

    laguna_wrapper = _fake_dflash_mask_wrapper(
        SimpleNamespace(
            model_type="laguna",
            architectures=["DFlashLagunaForCausalLM"],
            num_hidden_layers=5,
            layer_types=["sliding_attention"] * 5,
            sliding_window=512,
        ),
        sliding_layers_causal=True,
    )

    for layer_idx in range(5):
        assert laguna_wrapper._get_attention_mask_args(layer_idx) == (True, (511, 0))

    with patch("tensorrt_llm._torch.models.modeling_dflash.logger.warning") as warning:
        laguna_wrapper._warn_inferred_attention_windows()
    warning.assert_called_once_with(
        "DFlash inferred pooled-context sliding-window attention from checkpoint "
        "config for draft layers [0, 1, 2, 3, 4]: window=512. Context attention "
        "is truncated to 512 tokens for these layers; if the drafter expects full "
        "context, acceptance rate may drop. Set use_sliding_window explicitly to "
        "confirm or disable windowing."
    )


def _fake_dflash_buffer_wrapper():
    wrapper = DFlashForCausalLM.__new__(DFlashForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper._dflash_trtllm_gen_ops = SimpleNamespace(
        get_workspace_size=MagicMock(side_effect=lambda **kwargs: kwargs["max_num_requests"] * 16),
        get_multi_ctas_kv_counter_size=MagicMock(
            side_effect=lambda _num_heads, max_batch_size, _sm_count: max_batch_size * 8
        ),
    )
    wrapper.validate_block_attention_windows = MagicMock(return_value=["causal decode"])
    wrapper._dflash_trtllm_gen_workspace = None
    wrapper._dflash_trtllm_gen_counters = None
    wrapper.register_buffer("_dflash_batch_indices", None, persistent=False)
    wrapper.register_buffer("_dflash_block_offsets", None, persistent=False)
    wrapper._dflash_trtllm_gen_device = None
    wrapper._dflash_trtllm_gen_sm_count = None
    return wrapper


def _prepare_dflash_buffers(wrapper, max_batch_size):
    wrapper._prepare_dflash_trtllm_gen_buffers(
        dtype=torch.float16,
        device=torch.device("cpu"),
        max_batch_size=max_batch_size,
        block_size=4,
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
    )


def test_dflash_trtllm_gen_buffers_reuse_and_grow():
    wrapper = _fake_dflash_buffer_wrapper()
    device_properties = SimpleNamespace(multi_processor_count=148)

    with (
        patch("torch.cuda.get_device_properties", return_value=device_properties) as get_props,
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
    ):
        _prepare_dflash_buffers(wrapper, 2)
        workspace = wrapper._dflash_trtllm_gen_workspace
        counters = wrapper._dflash_trtllm_gen_counters

        _prepare_dflash_buffers(wrapper, 2)
        assert wrapper._dflash_trtllm_gen_workspace is workspace
        assert wrapper._dflash_trtllm_gen_counters is counters

        _prepare_dflash_buffers(wrapper, 4)
        assert wrapper._dflash_trtllm_gen_workspace.numel() >= 64
        assert wrapper._dflash_trtllm_gen_counters.numel() >= 32
        get_props.assert_called_once_with(torch.device("cpu"))


def test_dflash_trtllm_gen_buffers_reject_capture_time_allocation():
    wrapper = _fake_dflash_buffer_wrapper()
    device_properties = SimpleNamespace(multi_processor_count=148)

    with (
        patch("torch.cuda.get_device_properties", return_value=device_properties),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
    ):
        _prepare_dflash_buffers(wrapper, 2)

    with patch("torch.cuda.is_current_stream_capturing", return_value=True):
        with pytest.raises(RuntimeError, match="workspace.*before CUDA graph capture"):
            _prepare_dflash_buffers(wrapper, 4)

        wrapper._dflash_trtllm_gen_counters = torch.empty(16, dtype=torch.uint8, device="meta")
        with pytest.raises(RuntimeError, match="counter buffer.*before CUDA graph capture"):
            _prepare_dflash_buffers(wrapper, 2)


# ---------------------------------------------------------------------------
# One-engine draft MoE backend selection
# ---------------------------------------------------------------------------


def _draft_backend_test_model_config(moe_backend: str = "CUTLASS") -> ModelConfig:
    return ModelConfig(
        pretrained_config=PretrainedConfig(
            architectures=["DraftBackendTestForCausalLM"],
            hidden_size=64,
            vocab_size=128,
            num_hidden_layers=2,
        ),
        moe_backend=moe_backend,
    )


def _external_spec_config(
    moe_backend: str | None, moe_tactic: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        spec_dec_mode=SpeculativeDecodingMode.PARD,
        moe_backend=moe_backend,
        moe_tactic=moe_tactic,
    )


def test_external_draft_moe_backend_none_inherits_target() -> None:
    """None preserves the existing target-backend inheritance behavior."""
    model_config = _draft_backend_test_model_config("CUTLASS")

    kwargs = external_drafter_config_kwargs(model_config, _external_spec_config(None))

    assert kwargs["moe_backend"] == "CUTLASS"
    assert kwargs["moe_tactic"] is None


@pytest.mark.cpu_only
def test_external_draft_of_a_pinned_target_resolves_auto() -> None:
    """The target's tactic names the target's implementations, not the draft's."""
    model_config = _draft_backend_test_model_config("CUTEDSL")
    model_config.moe_tactic = "cutedsl.grouped_gemm.nvfp4"

    kwargs = external_drafter_config_kwargs(model_config, _external_spec_config(None))

    assert kwargs["moe_backend"] == "AUTO"
    assert kwargs["moe_tactic"] is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("target_tactic", [None, "cutedsl.grouped_gemm.nvfp4"])
def test_external_draft_moe_tactic_reaches_draft_loader(target_tactic: str | None) -> None:
    model_config = _draft_backend_test_model_config("CUTEDSL")
    model_config.moe_tactic = target_tactic

    kwargs = external_drafter_config_kwargs(
        model_config, _external_spec_config(None, moe_tactic="cutlass")
    )

    assert kwargs["moe_backend"] == "AUTO"
    assert kwargs["moe_tactic"] == "cutlass"


def test_external_draft_moe_backend_auto_reaches_draft_loader() -> None:
    """AUTO remains unresolved until the draft checkpoint quant config is read."""
    model_config = _draft_backend_test_model_config("TRTLLM")

    kwargs = external_drafter_config_kwargs(model_config, _external_spec_config("AUTO"))

    assert kwargs["moe_backend"] == "AUTO"


def test_loaded_draft_moe_backend_uses_isolated_model_config() -> None:
    """Resolving a loaded draft config does not modify another config."""
    target_config = _draft_backend_test_model_config("CUTLASS")
    with patch.object(ModelConfig, "resolve_moe_backend", return_value="TRTLLM") as resolve_backend:
        draft_config = _copy_model_config_with_moe_selection(
            target_config, MoESelection(backend="AUTO")
        )

    assert draft_config is not target_config
    assert draft_config.moe_backend == "TRTLLM"
    assert target_config.moe_backend == "CUTLASS"
    resolve_backend.assert_called_once_with(
        "AUTO", "DraftBackendTestForCausalLM", quant_config=target_config.quant_config
    )


@pytest.mark.cpu_only
def test_draft_moe_backend_drops_the_targets_pinned_tactic() -> None:
    """A draft that names its own backend selects by it, not by the target's pin."""
    target_config = _draft_backend_test_model_config("CUTLASS")
    target_config.moe_tactic = "grouped_gemm.nvfp4"
    with patch.object(ModelConfig, "resolve_moe_backend", return_value="TRTLLM"):
        draft_config = _copy_model_config_with_moe_selection(
            target_config, MoESelection(backend="TRTLLM")
        )

    assert draft_config.moe_backend == "TRTLLM"
    assert draft_config.moe_tactic is None
    assert target_config.moe_tactic == "grouped_gemm.nvfp4"


def test_internal_mtp_without_override_reuses_target_model_config() -> None:
    target_config = _draft_backend_test_model_config("CUTEDSL")
    target_config.spec_config = SimpleNamespace(moe_backend=None)
    target_model = SimpleNamespace(aux_stream_dict={}, preload_weight_modules=[])

    with patch(
        "tensorrt_llm._torch.models.modeling_speculative.MTPForCausalLM",
        return_value=sentinel.draft_model,
    ) as mtp_cls:
        draft_model = _build_mtp_one_model_draft(
            target_config, None, sentinel.lm_head, target_model
        )

    assert draft_model is sentinel.draft_model
    assert mtp_cls.call_args.args[0] is target_config
    assert target_model.preload_weight_modules == []


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "model_type,replacement,head_algo",
    [
        ("nemotron_h", True, QuantAlgo.FP8),
        ("nemotron_h_puzzle", True, QuantAlgo.W4A16_NVFP4),
        ("nemotron_h", True, QuantAlgo.NVFP4),
        ("nemotron_h", False, QuantAlgo.FP8),
        ("qwen3_next", True, QuantAlgo.FP8),
    ],
)
def test_replacement_owns_quantized_mtp_head_across_architectures(
    monkeypatch, model_type, replacement, head_algo
):
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_nemotron_h.NemotronHMTP", lambda *args: nn.Identity()
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_qwen3_next.Qwen3NextMTP", lambda *args: nn.Identity()
    )
    monkeypatch.setattr("tensorrt_llm._torch.modules.linear.get_sm_version", lambda: 100)
    config = ModelConfig(
        pretrained_config=SimpleNamespace(
            model_type=model_type,
            num_nextn_predict_layers=1,
            hidden_size=16,
            vocab_size=32,
            torch_dtype=torch.bfloat16,
            tie_word_embeddings=False,
        ),
        spec_config=SimpleNamespace(
            uses_replacement_heads=replacement,
            max_draft_len=1,
            spec_dec_mode=SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL,
        ),
        quant_config_dict={
            "lm_head": QuantConfig(quant_algo=head_algo, group_size=16),
        },
    )
    with torch.device("cpu"):
        target_head = nn.Linear(16, 32, bias=False)
        target = SimpleNamespace(aux_stream_dict={}, embed_tokens=nn.Embedding(32, 16))
        draft = MTPForCausalLM(config, 2, target_head, target)
    owns_head = replacement
    assert draft.owns_lm_head == owns_head
    assert (draft.lm_head is target_head) == (not owns_head)
    assert draft.embed_tokens is target.embed_tokens
    if owns_head:
        packed = head_algo in (QuantAlgo.W4A16_NVFP4, QuantAlgo.NVFP4)
        assert draft.lm_head.weight.shape == (32, 8 if packed else 16)
        assert draft.lm_head.weight.dtype == (torch.uint8 if packed else torch.float8_e4m3fn)


@pytest.mark.cpu_only
def test_mtp_drafter_builds_the_target_models_own_layer():
    """A target model that names ``mtp_layer_cls`` gets that layer, whatever its model type."""
    built = []

    class OwnedMTPLayer(nn.Module):
        def __init__(self, model_config, layer_idx, aux_stream_dict):
            super().__init__()
            built.append(layer_idx)

    config = ModelConfig(
        pretrained_config=SimpleNamespace(
            model_type="deepseek_v3",
            num_nextn_predict_layers=1,
            hidden_size=16,
            vocab_size=32,
            torch_dtype=torch.bfloat16,
            tie_word_embeddings=False,
        ),
        spec_config=SimpleNamespace(
            uses_replacement_heads=False,
            max_draft_len=3,
            spec_dec_mode=SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL,
        ),
    )
    with torch.device("cpu"):
        target_head = nn.Linear(16, 32, bias=False)
        target = SimpleNamespace(
            aux_stream_dict={}, embed_tokens=nn.Embedding(32, 16), mtp_layer_cls=OwnedMTPLayer
        )
        draft = MTPForCausalLM(config, 2, target_head, target)
    assert [type(layer) for layer in draft.mtp_layers] == [OwnedMTPLayer]
    assert built == [2]
    assert draft.lm_head is target_head


@pytest.mark.parametrize("requested_backend", ["TRTLLM", "AUTO"])
@pytest.mark.parametrize(
    "quant_config_key",
    ["model.layers.2.mlp.experts", "mtp.layers.0.mlp.experts"],
)
def test_internal_mtp_moe_backend_uses_isolated_layer_config(
    requested_backend: str,
    quant_config_key: str,
) -> None:
    target_config = _draft_backend_test_model_config("CUTEDSL")
    target_config.spec_config = SimpleNamespace(moe_backend=requested_backend)
    mtp_quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    target_config.quant_config_dict = {
        quant_config_key: mtp_quant_config,
    }
    target_config._frozen = True
    target_model = SimpleNamespace(aux_stream_dict={}, preload_weight_modules=[])

    with (
        patch.object(ModelConfig, "resolve_moe_backend", return_value="TRTLLM") as resolve_backend,
        patch(
            "tensorrt_llm._torch.models.modeling_speculative.MTPForCausalLM",
            return_value=sentinel.draft_model,
        ) as mtp_cls,
    ):
        draft_model = _build_mtp_one_model_draft(
            target_config, None, sentinel.lm_head, target_model
        )

    mtp_model_config = mtp_cls.call_args.args[0]
    assert draft_model is sentinel.draft_model
    assert mtp_model_config is not target_config
    assert mtp_model_config.moe_backend == "TRTLLM"
    assert mtp_model_config.quant_config_dict is target_config.quant_config_dict
    assert mtp_model_config.extra_attrs is target_config.extra_attrs
    assert target_config.moe_backend == "CUTEDSL"
    assert target_config._frozen
    assert target_model.preload_weight_modules == ["experts", "routing_method", "all_reduce"]
    resolve_backend.assert_called_once_with(
        requested_backend,
        "DraftBackendTestForCausalLM",
        quant_config=mtp_quant_config,
    )


def test_internal_mtp_auto_resolves_from_layer_quantization() -> None:
    target_config = _draft_backend_test_model_config("CUTEDSL")
    target_config.spec_config = SimpleNamespace(moe_backend="AUTO")
    target_config.quant_config = QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION)
    target_config.quant_config_dict = {
        "model.layers.2.mlp.experts": QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
    }
    target_config._frozen = True
    target_model = SimpleNamespace(aux_stream_dict={}, preload_weight_modules=[])

    with (
        patch("tensorrt_llm._torch.model_config.is_sm_100f", return_value=True),
        patch(
            "tensorrt_llm._torch.models.modeling_speculative.MTPForCausalLM",
            return_value=sentinel.draft_model,
        ) as mtp_cls,
    ):
        _build_mtp_one_model_draft(target_config, None, sentinel.lm_head, target_model)

    mtp_model_config = mtp_cls.call_args.args[0]
    assert mtp_model_config.moe_backend == "TRTLLM"
    assert target_config.moe_backend == "CUTEDSL"


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "target_backend,draft_backend", [("CUTLASS", "VANILLA"), ("VANILLA", "CUTLASS")]
)
def test_internal_mtp_rejects_nemotron_backend_mismatch(
    target_backend: str, draft_backend: str
) -> None:
    target_config = _draft_backend_test_model_config(target_backend)
    target_config.pretrained_config.model_type = "nemotron_h"
    target_config.spec_config = SimpleNamespace(moe_backend=draft_backend)
    target_model = SimpleNamespace(aux_stream_dict={}, preload_weight_modules=[])

    with pytest.raises(ValueError, match="must run VANILLA exactly when the target does"):
        _build_mtp_one_model_draft(target_config, None, sentinel.lm_head, target_model)


@pytest.mark.parametrize("folded", [False, True])
def test_dflash_workspace_counts_folded_queries_once(folded: bool) -> None:
    wrapper = _fake_dflash_buffer_wrapper()
    wrapper.validate_block_attention_windows.return_value = (
        ["non-causal folded decode"] if folded else ["causal decode"]
    )
    with (
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=148),
        ),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
    ):
        _prepare_dflash_buffers(wrapper, 2048)
    calls = wrapper._dflash_trtllm_gen_ops.get_workspace_size.call_args_list
    assert len(calls) == 1 + int(folded)
    assert [(call.kwargs["num_tokens"], call.kwargs["num_heads"]) for call in calls] == (
        [(2048 * 4, 8), (2048, 8 * 4)] if folded else [(2048 * 4, 8)]
    )
    assert all(call.kwargs["num_tokens"] == call.kwargs["num_gen_tokens"] for call in calls)
    counter_calls = wrapper._dflash_trtllm_gen_ops.get_multi_ctas_kv_counter_size.call_args_list
    assert [call.args for call in counter_calls] == (
        [(8 * 4, 2048, 148)] if folded else [(8, 2048, 148)]
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_external_draft_inherits_block_scaling_backend(enabled: bool) -> None:
    model_config = _draft_backend_test_model_config()
    model_config.use_cute_dsl_blockscaling_mm = enabled
    kwargs = external_drafter_config_kwargs(model_config, _external_spec_config(None))
    assert kwargs["use_cute_dsl_blockscaling_mm"] is enabled


@pytest.mark.parametrize(
    "block,heads,kv_heads,chunk", [(16, 12, 1, 10), (9, 32, 2, 8), (21, 6, 2, 21)]
)
def test_dflash_workspace_sizes_the_largest_folded_chunk(
    block: int, heads: int, kv_heads: int, chunk: int
) -> None:
    wrapper = _fake_dflash_buffer_wrapper()
    wrapper.validate_block_attention_windows.return_value = ["non-causal folded decode"]
    with (
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=148),
        ),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
    ):
        wrapper._prepare_dflash_trtllm_gen_buffers(
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            max_batch_size=3,
            block_size=block,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=128,
        )
    calls = wrapper._dflash_trtllm_gen_ops.get_workspace_size.call_args_list
    assert [(call.kwargs["num_tokens"], call.kwargs["num_heads"]) for call in calls] == [
        (3 * block, heads),
        (3, heads * chunk),
    ]
    counter_calls = wrapper._dflash_trtllm_gen_ops.get_multi_ctas_kv_counter_size.call_args_list
    assert [call.args for call in counter_calls] == [(heads * chunk, 3, 148)]


@pytest.mark.cpu_only
def test_internal_nemotron_mtp_tactic_of_a_vanilla_target_is_rejected() -> None:
    """No tactic names VANILLA, so a pinned draft would load the target's VANILLA layout wrongly."""
    target_config = _draft_backend_test_model_config("VANILLA")
    target_config.pretrained_config.model_type = "nemotron_h"
    target_config.spec_config = SimpleNamespace(moe_backend=None, moe_tactic="cutlass")

    with (
        patch.object(ModelConfig, "pinned_moe_backend", return_value="CUTLASS"),
        pytest.raises(ValueError, match="must run VANILLA exactly when the target does"),
    ):
        _build_mtp_draft_config(target_config)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "target_backend,draft_backend", [("MARLIN", "CUTLASS"), ("CUTLASS", "TRTLLM")]
)
def test_internal_nemotron_mtp_may_select_another_non_vanilla_backend(
    target_backend: str, draft_backend: str
) -> None:
    """The target's mapper lays out every backend but VANILLA alike."""
    target_config = _draft_backend_test_model_config(target_backend)
    target_config.pretrained_config.model_type = "nemotron_h"
    target_config.spec_config = SimpleNamespace(moe_backend=draft_backend, moe_tactic=None)

    with patch.object(ModelConfig, "resolve_moe_backend", return_value=draft_backend):
        mtp_model_config = _build_mtp_draft_config(target_config)

    assert mtp_model_config.moe_backend == draft_backend
    assert target_config.moe_backend == target_backend


@pytest.mark.cpu_only
def test_internal_nemotron_mtp_tactic_may_name_another_family() -> None:
    """A MARLIN-pinned NVFP4 target can pin CUTLASS for its unquantized MTP layers."""
    target_config = _draft_backend_test_model_config("MARLIN")
    target_config.pretrained_config.model_type = "nemotron_h"
    target_config.moe_tactic = "marlin"
    target_config.spec_config = SimpleNamespace(
        moe_backend=None, moe_tactic="trtllm.cutlass.grouped_gemm.none"
    )

    with patch.object(ModelConfig, "pinned_moe_backend", return_value="CUTLASS"):
        mtp_model_config = _build_mtp_draft_config(target_config)

    assert mtp_model_config.moe_backend == "CUTLASS"
    assert mtp_model_config.moe_tactic == "trtllm.cutlass.grouped_gemm.none"
    assert target_config.moe_tactic == "marlin"


def _build_mtp_draft_config(
    target_config: ModelConfig, draft_config: ModelConfig | None = None
) -> ModelConfig:
    target_model = SimpleNamespace(aux_stream_dict={}, preload_weight_modules=[])
    with patch(
        "tensorrt_llm._torch.models.modeling_speculative.MTPForCausalLM",
        return_value=sentinel.draft_model,
    ) as mtp_cls:
        _build_mtp_one_model_draft(target_config, draft_config, sentinel.lm_head, target_model)
    return mtp_cls.call_args.args[0]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "global_algo, exclude_modules, entries, unquantized",
    [
        pytest.param(QuantAlgo.NVFP4, ["model.layers.2*"], {}, True, id="excluded_layer"),
        pytest.param(
            QuantAlgo.NVFP4, ["model.layers.2.mlp.experts"], {}, True, id="excluded_experts"
        ),
        # Exclusion wins over a per-layer entry, as in moe_experts_quant_config.
        pytest.param(
            QuantAlgo.NVFP4,
            ["model.layers.2*"],
            {"model.layers.2.mlp.experts": QuantAlgo.NVFP4},
            True,
            id="excluded_despite_an_entry",
        ),
        # MIXED_PRECISION lists every quantized module.
        pytest.param(QuantAlgo.MIXED_PRECISION, None, {}, True, id="mixed_precision_unlisted"),
        pytest.param(
            QuantAlgo.MIXED_PRECISION,
            None,
            {"model.layers.2.mlp.experts": QuantAlgo.FP8_BLOCK_SCALES},
            False,
            id="mixed_precision_listed",
        ),
        pytest.param(QuantAlgo.NVFP4, None, {}, False, id="global_format"),
    ],
)
def test_the_mtp_experts_format_follows_what_the_checkpoint_quantizes(
    global_algo, exclude_modules, entries, unquantized
) -> None:
    config = _draft_backend_test_model_config()
    config.quant_config = QuantConfig(
        quant_algo=global_algo, kv_cache_quant_algo=QuantAlgo.FP8, exclude_modules=exclude_modules
    )
    config.quant_config_dict = {
        name: QuantConfig(quant_algo=algo) for name, algo in entries.items()
    } or None

    quant_config = _get_mtp_moe_quant_config(config, 2)

    if unquantized:
        assert quant_config.quant_algo is None
        assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    elif entries:
        assert quant_config is config.quant_config_dict["model.layers.2.mlp.experts"]
    else:
        assert quant_config is config.quant_config


@pytest.mark.cpu_only
def test_a_draft_tactic_resolves_against_the_excluded_mtp_layers_format() -> None:
    """DeepSeek-R1 NVFP4 excludes its MTP layer through exclude_modules, so it runs BF16.

    ``fused_moe`` names TRTLLM-Gen and Marlin implementations of NVFP4 but only the TRTLLM-Gen
    one of BF16, so the family read from the MTP layer's real format is TRTLLM.
    """
    target_config = _draft_backend_test_model_config("TRTLLM")
    target_config.pretrained_config.num_nextn_predict_layers = 1
    target_config.quant_config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        kv_cache_quant_algo=QuantAlgo.FP8,
        exclude_modules=["model.layers.2*"],
    )
    target_config.spec_config = SimpleNamespace(moe_backend=None, moe_tactic="fused_moe")

    mtp_model_config = _build_mtp_draft_config(target_config)

    assert mtp_model_config.moe_backend == "TRTLLM"
    assert mtp_model_config.moe_tactic == "fused_moe"


@pytest.mark.cpu_only
def test_internal_mtp_of_a_pinned_target_resolves_auto_for_its_layers() -> None:
    """An NVFP4 tactic must not reach the unquantized MTP layer of an NVFP4 checkpoint."""
    target_config = _draft_backend_test_model_config("CUTEDSL")
    target_config.moe_tactic = "cutedsl.grouped_gemm.nvfp4"
    target_config.spec_config = SimpleNamespace(moe_backend=None, moe_tactic=None)
    mtp_quant_config = QuantConfig()
    target_config.quant_config_dict = {"model.layers.2.mlp.experts": mtp_quant_config}

    with patch.object(ModelConfig, "resolve_moe_backend", return_value="CUTLASS") as resolve:
        mtp_model_config = _build_mtp_draft_config(target_config)

    assert mtp_model_config is not target_config
    assert mtp_model_config.moe_backend == "CUTLASS"
    assert mtp_model_config.moe_tactic is None
    assert target_config.moe_tactic == "cutedsl.grouped_gemm.nvfp4"
    resolve.assert_called_once_with(
        "AUTO", "DraftBackendTestForCausalLM", quant_config=mtp_quant_config
    )


@pytest.mark.cpu_only
def test_internal_mtp_moe_tactic_pins_its_layers() -> None:
    target_config = _draft_backend_test_model_config("CUTEDSL")
    target_config.spec_config = SimpleNamespace(moe_backend=None, moe_tactic="cutlass")
    mtp_quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    target_config.quant_config_dict = {"model.layers.2.mlp.experts": mtp_quant_config}

    with patch.object(ModelConfig, "pinned_moe_backend", return_value="CUTLASS") as pinned:
        mtp_model_config = _build_mtp_draft_config(target_config)

    assert mtp_model_config.moe_backend == "CUTLASS"
    assert mtp_model_config.moe_tactic == "cutlass"
    assert target_config.moe_tactic is None
    # The family is read from the MTP layer's format alone.
    pinned.assert_called_once_with(
        "cutlass", mtp_quant_config, None, architecture="DraftBackendTestForCausalLM"
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("model_type", ["nemotron_h", "nemotron_h_puzzle"])
def test_embedded_nemotron_mtp_of_a_pinned_target_resolves_auto(model_type: str) -> None:
    """AUTO is never VANILLA, so the target's mapper layout still fits the MTP experts."""
    target_config = _draft_backend_test_model_config("MARLIN")
    target_config.pretrained_config.model_type = model_type
    target_config.moe_tactic = "marlin"
    target_config.spec_config = SimpleNamespace(moe_backend=None, moe_tactic=None)
    # The NVFP4 checkpoint leaves its MTP experts in BF16.
    target_config.quant_config_dict = {"model.layers.2.mlp.experts": QuantConfig()}

    mtp_model_config = _build_mtp_draft_config(target_config)

    assert mtp_model_config.moe_backend == "CUTLASS"
    assert mtp_model_config.moe_tactic is None
    assert target_config.moe_backend == "MARLIN"
    assert target_config.moe_tactic == "marlin"


def _replacement_spec_config(moe_tactic: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(moe_backend=None, moe_tactic=moe_tactic, uses_replacement_heads=True)


@pytest.mark.cpu_only
def test_replacement_mtp_heads_of_a_pinned_target_resolve_auto() -> None:
    """The heads load through their own mapper, so they select for their own weights."""
    target_config = _draft_backend_test_model_config("CUTLASS")
    target_config.moe_tactic = "cutlass.grouped_gemm.nvfp4"
    target_config.spec_config = _replacement_spec_config()
    replacement_config = _draft_backend_test_model_config("CUTLASS")
    replacement_config.moe_tactic = target_config.moe_tactic
    head_experts = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    replacement_config.quant_config_dict = {"model.layers.2.mlp.experts": head_experts}

    with patch.object(ModelConfig, "resolve_moe_backend", return_value="TRTLLM") as resolve:
        mtp_model_config = _build_mtp_draft_config(target_config, replacement_config)

    assert mtp_model_config is not replacement_config
    assert mtp_model_config.moe_backend == "TRTLLM"
    assert mtp_model_config.moe_tactic is None
    assert replacement_config.moe_tactic == "cutlass.grouped_gemm.nvfp4"
    resolve.assert_called_once_with(
        "AUTO", "DraftBackendTestForCausalLM", quant_config=head_experts
    )


@pytest.mark.cpu_only
def test_replacement_mtp_heads_honor_the_draft_tactic() -> None:
    target_config = _draft_backend_test_model_config("CUTEDSL")
    target_config.spec_config = _replacement_spec_config(moe_tactic="cutlass")
    replacement_config = _draft_backend_test_model_config("CUTEDSL")
    head_experts = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    replacement_config.quant_config_dict = {"model.layers.2.mlp.experts": head_experts}

    with patch.object(ModelConfig, "pinned_moe_backend", return_value="CUTLASS") as pinned:
        mtp_model_config = _build_mtp_draft_config(target_config, replacement_config)

    assert mtp_model_config.moe_backend == "CUTLASS"
    assert mtp_model_config.moe_tactic == "cutlass"
    assert replacement_config.moe_backend == "CUTEDSL"
    # The family is read from the replacement's own MTP experts.
    pinned.assert_called_once_with(
        "cutlass", head_experts, None, architecture="DraftBackendTestForCausalLM"
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "target_backend, target_tactic, spec_tactic, moe_kwargs",
    [
        pytest.param(
            "CUTLASS", None, None, {"moe_backend": "CUTLASS", "moe_tactic": None}, id="inherit"
        ),
        # AUTO is hinted by architecture, as from_pretrained hints it.
        pytest.param(
            "MARLIN", "marlin", None, {"moe_backend": "CUTLASS", "moe_tactic": None}, id="auto"
        ),
        pytest.param(
            "CUTLASS",
            None,
            "cutlass",
            {"moe_backend": "CUTLASS", "moe_tactic": "cutlass"},
            id="tactic",
        ),
    ],
)
def test_replacement_quantization_is_parsed_for_the_draft_selection(
    tmp_path, target_backend, target_tactic, spec_tactic, moe_kwargs
) -> None:
    """Parsing picks some formats by the MoE request, which is the heads' own."""
    (tmp_path / "hf_quant_config.json").write_text("{}")
    target_config = ModelConfig(
        pretrained_config=PretrainedConfig(
            architectures=["NemotronHForCausalLM"], num_hidden_layers=2, num_nextn_predict_layers=1
        ),
        spec_config=SimpleNamespace(
            speculative_model=tmp_path, moe_backend=None, moe_tactic=spec_tactic
        ),
        moe_backend=target_backend,
    )
    target_config.moe_tactic = target_tactic

    with patch.object(
        ModelConfig, "load_modelopt_quant_config", return_value=(QuantConfig(), None)
    ) as load:
        _replacement_mtp_model_config(target_config)

    load.assert_called_once_with(
        str(tmp_path / "hf_quant_config.json"),
        str(tmp_path),
        architecture="NemotronHForCausalLM",
        **moe_kwargs,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "target_tactic, spec_backend, spec_tactic, load_balancer, expected",
    [
        pytest.param(None, None, None, None, None, id="inherit"),
        pytest.param("cutlass", None, None, None, MoESelection(backend="AUTO"), id="pinned_target"),
        pytest.param(
            "mega_moe",
            None,
            None,
            "per_layer_dynamic",
            MoESelection(backend="MEGAMOE_CUTEDSL"),
            id="pinned_target_per_layer_dynamic",
        ),
        pytest.param(
            None,
            "AUTO",
            None,
            "per_layer_dynamic",
            MoESelection(backend="MEGAMOE_CUTEDSL"),
            id="explicit_auto_per_layer_dynamic",
        ),
        pytest.param(
            None,
            "AUTO",
            None,
            "standard",
            MoESelection(backend="AUTO"),
            id="explicit_auto_standard",
        ),
        pytest.param(
            None,
            "TRTLLM",
            None,
            "per_layer_dynamic",
            MoESelection(backend="TRTLLM"),
            id="explicit_backend",
        ),
        pytest.param(
            "cutlass",
            None,
            "deepgemm",
            "per_layer_dynamic",
            MoESelection(tactic="deepgemm"),
            id="draft_tactic",
        ),
    ],
)
def test_draft_moe_selection_default_rule(
    target_tactic, spec_backend, spec_tactic, load_balancer, expected
) -> None:
    target_config = _draft_backend_test_model_config("CUTLASS")
    target_config.moe_tactic = target_tactic
    spec_config = SimpleNamespace(moe_backend=spec_backend, moe_tactic=spec_tactic)
    load_balancer_config = {
        None: None,
        "standard": MoeLoadBalancerConfig(num_slots=8),
        "per_layer_dynamic": MoePerLayerDynamicLoadBalancerConfig(num_slots=72),
    }[load_balancer]

    assert (
        _draft_moe_selection(target_config, spec_config, load_balancer=load_balancer_config)
        == expected
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "target_tactic, spec_backend",
    [("trtllm.cutedsl.mega_moe.nvfp4", None), (None, "AUTO")],
    ids=["pinned_target", "explicit_auto"],
)
def test_internal_mtp_auto_under_per_layer_dynamic_eplb_runs_megamoe_cutedsl(
    target_tactic: str | None, spec_backend: str | None
) -> None:
    """The MTP layers register into the target's balancer, which only MegaMoE CuteDSL serves."""
    target_config = _draft_backend_test_model_config("MEGAMOE_CUTEDSL")
    target_config.moe_tactic = target_tactic
    target_config.moe_load_balancer = MoePerLayerDynamicLoadBalancerConfig(num_slots=72)
    target_config.spec_config = SimpleNamespace(moe_backend=spec_backend, moe_tactic=None)

    mtp_model_config = _build_mtp_draft_config(target_config)

    assert mtp_model_config.moe_backend == "MEGAMOE_CUTEDSL"
    assert mtp_model_config.moe_tactic is None


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "target_tactic, spec_tactic, stored_algo, run_algo, backend",
    [
        pytest.param(
            None,
            "trtllm.trtllm_gen.fused_moe.w4a16_mxfp4",
            QuantAlgo.W4A8_MXFP4_MXFP8,
            QuantAlgo.W4A16_MXFP4,
            "TRTLLM",
            id="draft_tactic",
        ),
        # The target's tactic picked its format; AUTO takes the SM's first choice.
        pytest.param(
            "trtllm_gen.w4a8_mxfp4_fp8",
            None,
            QuantAlgo.W4A8_MXFP4_FP8,
            QuantAlgo.W4A8_MXFP4_MXFP8,
            "CUTLASS",
            id="pinned_target",
        ),
    ],
)
def test_internal_mtp_mxfp4_experts_run_as_the_draft_selection_picks(
    monkeypatch, target_tactic, spec_tactic, stored_algo, run_algo, backend
) -> None:
    """MXFP4 storage does not fix the run format, which checkpoint parsing chose for the target."""
    monkeypatch.setattr(model_config_module, "get_sm_version", lambda: 100)
    monkeypatch.delenv("OVERRIDE_QUANT_ALGO", raising=False)
    target_config = _draft_backend_test_model_config("TRTLLM")
    target_config.pretrained_config.num_nextn_predict_layers = 1
    target_config.moe_tactic = target_tactic
    target_config.spec_config = SimpleNamespace(moe_backend=None, moe_tactic=spec_tactic)
    # As DeepSeek-V4 parsing writes them: one entry shared by the target and MTP layers.
    experts = QuantConfig(quant_algo=stored_algo, group_size=32)
    # The target's layers read its quant modes first, which caches them.
    assert experts.layer_quant_mode == QuantMode.from_quant_algo(stored_algo)
    assert experts.quant_mode.objs == [QuantMode.from_quant_algo(stored_algo)]
    target_quant_config_dict = {f"model.layers.{i}.mlp.experts": experts for i in range(3)}
    target_config.quant_config_dict = dict(target_quant_config_dict)

    mtp_model_config = _build_mtp_draft_config(target_config)

    mtp_experts = mtp_model_config.quant_config_dict["model.layers.2.mlp.experts"]
    assert mtp_experts.quant_algo == run_algo
    assert mtp_experts.group_size == 32
    assert mtp_experts.layer_quant_mode == QuantMode.from_quant_algo(run_algo)
    assert mtp_experts.quant_mode.objs == [QuantMode.from_quant_algo(run_algo)]
    assert experts.layer_quant_mode == QuantMode.from_quant_algo(stored_algo)
    assert mtp_model_config.moe_backend == backend
    assert mtp_model_config.moe_tactic == spec_tactic
    assert mtp_model_config.quant_config_dict["model.layers.0.mlp.experts"] is experts
    assert target_config.quant_config_dict == target_quant_config_dict
    assert experts.quant_algo == stored_algo
