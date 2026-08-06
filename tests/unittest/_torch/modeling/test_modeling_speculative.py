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
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention_backend.interface import RopeParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
from tensorrt_llm._torch.models.modeling_speculative import (
    Eagle3ForCausalLM,
    SpecDecOneEngineForCausalLM,
    _copy_model_config_with_moe_backend,
    external_drafter_config_kwargs,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode


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


def _fake_dflash_mask_wrapper(config, sliding_layers_causal=False):
    wrapper = DFlashForCausalLM.__new__(DFlashForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.config = config
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


def _external_spec_config(moe_backend: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        spec_dec_mode=SpeculativeDecodingMode.PARD,
        moe_backend=moe_backend,
    )


def test_external_draft_moe_backend_none_inherits_target() -> None:
    """None preserves the existing target-backend inheritance behavior."""
    model_config = _draft_backend_test_model_config("CUTLASS")

    kwargs = external_drafter_config_kwargs(model_config, _external_spec_config(None))

    assert kwargs["moe_backend"] == "CUTLASS"


def test_external_draft_moe_backend_auto_reaches_draft_loader() -> None:
    """AUTO remains unresolved until the draft checkpoint quant config is read."""
    model_config = _draft_backend_test_model_config("TRTLLM")

    kwargs = external_drafter_config_kwargs(model_config, _external_spec_config("AUTO"))

    assert kwargs["moe_backend"] == "AUTO"


def test_loaded_draft_moe_backend_uses_isolated_model_config() -> None:
    """Resolving a loaded draft config does not modify another config."""
    target_config = _draft_backend_test_model_config("CUTLASS")
    with patch.object(ModelConfig, "resolve_moe_backend", return_value="TRTLLM") as resolve_backend:
        draft_config = _copy_model_config_with_moe_backend(target_config, "AUTO")

    assert draft_config is not target_config
    assert draft_config.moe_backend == "TRTLLM"
    assert target_config.moe_backend == "CUTLASS"
    resolve_backend.assert_called_once_with(
        "AUTO", "DraftBackendTestForCausalLM", quant_config=target_config.quant_config
    )
