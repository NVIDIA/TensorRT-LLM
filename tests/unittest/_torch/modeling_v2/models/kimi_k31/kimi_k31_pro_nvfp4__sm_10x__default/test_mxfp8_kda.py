# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native MXFP8 checkpoint loading and KDA state updates through the target loader."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default import (
    weights,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.modeling import (  # noqa: E501
    _kda_forward_core,
    _project,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaLayerCache
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _assert_format_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = (actual.float() - expected.float()).norm()
    relative = error / expected.float().norm().clamp_min(1e-6)
    assert relative < 0.03, relative.item()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-2)


class SeparateMXFP8Projection(nn.Module):
    """Independent projection using decoded MXFP8 operands and a BF16 GEMM."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        codes, scales = torch.ops.trtllm.mxfp8_quantize(flat, False)
        scales = scales.reshape(-1, shape[-1] // 32)[: flat.shape[0]]
        decoded = codes[: flat.shape[0]].float() * torch.exp2(
            scales.float() - 127
        ).repeat_interleave(32, dim=1)
        output = torch.nn.functional.linear(decoded.to(x.dtype), self.weight)
        return output.reshape(*shape[:-1], self.weight.shape[0])


def _loaded_pair(heads: int):
    cfg = SimpleNamespace(
        hidden_size=512,
        rms_norm_eps=1e-5,
        kda_fp32_gate_params=True,
        linear_attn_config=dict(
            num_heads=heads,
            head_dim=128,
            short_conv_kernel_size=4,
            use_full_rank_gate=True,
            gate_lower_bound=-5.0,
        ),
    )
    config = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(),
        quant_config=QuantConfig(),
        quant_config_dict={
            name: QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)
            for name in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "f_a_proj", "b_proj")
        },
    )
    bank = nn.ParameterDict()
    with torch.device("cuda"):
        actual = weights.declare_kda(bank, cfg, 0, config, Mapping())
        reference = KimiKDALinearAttention(cfg, 0)
    # Compare checkpoint fusion with the same recurrence algorithm: auto can
    # select the Small-BH kernel for BF16 pools, while this target uses BT16.
    reference._dispatch.prefill_mode = "bt16"
    checkpoint = {}
    torch.manual_seed(10)
    quantized = {"q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "f_a_proj", "b_proj"}
    for name, parameter in reference.named_parameters():
        value = torch.randn(parameter.shape, device="cuda", dtype=torch.bfloat16) * 0.02
        if name in ("A_log", "dt_bias"):
            value = torch.zeros_like(parameter, dtype=torch.float32)
        if name.rsplit(".", 1)[0] in quantized:
            codes, scales = torch.ops.trtllm.mxfp8_quantize(value, False)
            codes = codes[: value.shape[0]]
            scales = scales.reshape(-1, value.shape[1] // 32)[: value.shape[0]]
            checkpoint[name] = codes
            checkpoint[name.removesuffix("weight") + "weight_scale"] = scales
            dequantized = codes.float() * torch.exp2(scales.float() - 127).repeat_interleave(
                32, dim=1
            )
            parameter.copy_(dequantized)
        else:
            checkpoint[name] = value
            parameter.copy_(value)
    for name in quantized:
        projection = getattr(reference, name)
        setattr(reference, name, SeparateMXFP8Projection(projection.weight.detach()))
    weights.load_kda(
        bank,
        actual,
        {f"model.layers.0.self_attn.{name}": value for name, value in checkpoint.items()},
    )

    class Forward:
        # Tests exercise the complete loaded attention operation and its externally owned caches.
        A_log = actual.weight_view.a_log
        dt_bias = actual.weight_view.dt_bias

        def __call__(self, x, metadata):
            core = _kda_forward_core(x, actual.weight_view, 0, metadata)
            return _project(core.flatten(1), actual.weight_view.output)

        def forward_prefill(self, x, lengths, mm, count, conv, state, slots):
            mm.query_start_loc_long = lengths
            cache = MambaLayerCache(conv=conv, temporal=state)
            metadata = SimpleNamespace(
                mamba_metadata=mm,
                num_contexts=count,
                num_ctx_tokens=x.shape[0],
                num_tokens=x.shape[0],
                seq_lens=lengths[1:] - lengths[:-1],
                kv_cache_manager=SimpleNamespace(mamba_layer_cache=lambda _: cache),
            )
            return _kda_forward_core(x, actual.weight_view, 0, metadata)

        def forward_verify_fused(self, x, steps, cache, state, slots):
            cache.conv = torch.empty(0, device=x.device)
            cache.temporal = state
            cache.has_kda_replay_caches = True
            metadata = SimpleNamespace(
                mamba_metadata=SimpleNamespace(state_indices=slots),
                num_contexts=0,
                num_ctx_tokens=0,
                num_tokens=x.shape[0],
                seq_lens=torch.full((slots.numel(),), steps, device=x.device),
                kv_cache_manager=SimpleNamespace(mamba_layer_cache=lambda _: cache),
            )
            return _kda_forward_core(x, actual.weight_view, 0, metadata)

    return Forward(), reference


@torch.inference_mode()
@pytest.mark.parametrize("heads", [3, 6, 12])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_decode_matches_separate_quantized_projections(heads, state_dtype):
    actual, reference = _loaded_pair(heads)
    slots = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    conv = torch.zeros(5, heads * 128 * 3, 3, dtype=torch.bfloat16, device="cuda")
    state = torch.zeros(5, heads, 128, 128, dtype=state_dtype, device="cuda")
    actual_cache = MambaLayerCache(conv=conv.clone(), temporal=state.clone())
    reference_cache = MambaLayerCache(conv=conv.clone(), temporal=state.clone())

    def metadata(cache):
        return SimpleNamespace(
            mamba_metadata=SimpleNamespace(
                state_indices=slots,
                query_start_loc_long=torch.zeros(1, device="cuda", dtype=torch.long),
                _arange_buffer=torch.arange(3, device="cuda", dtype=torch.int32),
            ),
            num_contexts=0,
            num_ctx_tokens=0,
            num_tokens=2,
            seq_lens=torch.ones(2, device="cuda", dtype=torch.int32),
            kv_cache_manager=SimpleNamespace(mamba_layer_cache=lambda _: cache),
        )

    for _ in range(3):
        x = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16) * 0.05
        output = actual(x, metadata(actual_cache))
        expected = reference(x, metadata(reference_cache))
        _assert_format_close(output, expected)
        _assert_format_close(actual_cache.temporal, reference_cache.temporal)
        torch.testing.assert_close(actual_cache.conv, reference_cache.conv, atol=1e-3, rtol=3e-2)
        assert not actual_cache.temporal[[0, 2, 4]].count_nonzero()
    assert actual.A_log.dtype == actual.dt_bias.dtype == torch.float32


@torch.inference_mode()
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_prefill_and_draft_replay(state_dtype):
    actual, reference = _loaded_pair(6)
    reference._build_mtp_conv_weights()
    heads, dim, pool, drafts = 6, 768, 5, 4
    slots = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    conv = torch.zeros(pool, 3 * dim, 3, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(pool, heads, 128, 128, device="cuda", dtype=state_dtype)
    actual_conv, reference_conv = conv.clone(), conv.clone()
    actual_state, reference_state = state.clone(), state.clone()
    lengths = torch.tensor([0, 17, 50], dtype=torch.int64, device="cuda")
    metadata = SimpleNamespace(
        has_initial_states=torch.zeros(2, device="cuda", dtype=torch.bool),
        state_indices=slots,
        query_start_loc=lengths.int(),
        kda_max_seqlen=33,
    )
    x = torch.randn(50, 512, device="cuda", dtype=torch.bfloat16) * 0.05
    output = actual.forward_prefill(x, lengths, metadata, 2, actual_conv, actual_state, slots)
    expected = reference.forward_prefill(
        x, lengths, metadata, 2, reference_conv, reference_state, slots
    )
    _assert_format_close(output, expected)
    _assert_format_close(actual_state, reference_state)

    def replay_cache(conv_pool):
        cache = SimpleNamespace(
            kda_k_cache=torch.zeros(pool, drafts, dim, device="cuda", dtype=torch.bfloat16),
            kda_g_cache=torch.zeros(pool, drafts, dim, device="cuda", dtype=torch.float32),
            kda_v_cache=torch.zeros(pool, drafts, dim, device="cuda", dtype=torch.bfloat16),
            kda_beta_cache=torch.zeros(pool, drafts, heads, device="cuda", dtype=torch.bfloat16),
            prev_num_accepted_tokens=torch.zeros(pool, device="cuda", dtype=torch.int32),
        )
        for index, name in enumerate(("q", "k", "v")):
            window = torch.zeros(
                pool, drafts + 3, dim, device="cuda", dtype=torch.bfloat16
            ).transpose(-1, -2)
            window[..., :3].copy_(conv_pool[:, index * dim : (index + 1) * dim])
            setattr(cache, f"kda_conv_{name}", window)
        return cache

    actual_cache = replay_cache(actual_conv)
    reference_cache = replay_cache(reference_conv)
    for _ in range(2):
        x = torch.randn(2 * (drafts + 1), 512, device="cuda", dtype=torch.bfloat16) * 0.05
        output = actual.forward_verify_fused(x, drafts + 1, actual_cache, actual_state, slots)
        expected = reference.forward_verify_fused(
            x, drafts + 1, reference_cache, reference_state, slots
        )
        _assert_format_close(output, expected)
        _assert_format_close(actual_state, reference_state)
        assert not actual_state[[0, 2, 4]].count_nonzero()
        actual_cache.prev_num_accepted_tokens[slots.long()] = torch.tensor(
            [0, drafts], device="cuda", dtype=torch.int32
        )
        reference_cache.prev_num_accepted_tokens.copy_(actual_cache.prev_num_accepted_tokens)
