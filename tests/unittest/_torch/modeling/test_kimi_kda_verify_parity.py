# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA runtime projection and speculative-verification parity.

The verify path must produce, for every step t, exactly the state and output
that t sequential single-token _forward_decode calls would produce — the two
paths call the same FLA kernels with the same [B, 1] shapes, so agreement is
expected to near-bitwise tolerance. A real mismatch here means the verify
implementation (state threading, conv stepping, intermediate writes) is
wrong; e2e text divergence on truncated models alone does NOT indicate a bug
(batched MoE reduction order flips argmax on noise logits).

The fused prefill/decode projection paths must also match the
separate-projection reference for output and cache updates.

Needs 1 GPU + fla-core; runs with random weights (no checkpoint).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.models.modeling_kimi_linear import KimiKDARuntime
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream


class _Cfg:
    hidden_size = 256
    rms_norm_eps = 1e-6
    linear_attn_config = {
        "num_heads": 4,
        "head_dim": 64,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
        "gate_lower_bound": None,
    }


class _K3Cfg:
    hidden_size = 768
    rms_norm_eps = 1e-6
    linear_attn_config = {
        "num_heads": 6,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }


class _LayerCache:
    def __init__(self, slots, dim3, w, h, v, k, t_max, device):
        self.conv = torch.zeros(slots, dim3, w, dtype=torch.bfloat16, device=device)
        self.temporal = torch.zeros(slots, h, v, k, dtype=torch.float32, device=device)
        self.intermediate_conv_window = torch.zeros(
            slots, t_max, dim3, w, dtype=torch.bfloat16, device=device
        )
        self.intermediate_ssm = torch.zeros(
            slots, t_max, h, v, k, dtype=torch.float32, device=device
        )


def _decode_metadata(layer_cache: SimpleNamespace, slot_indices: torch.Tensor) -> SimpleNamespace:
    batch = slot_indices.shape[0]
    mamba_metadata = SimpleNamespace(
        state_indices=slot_indices.to(torch.int32),
        state_indices_long=slot_indices,
        query_start_loc_long=torch.zeros(1, device=slot_indices.device, dtype=torch.long),
        _arange_buffer=torch.arange(batch + 1, device=slot_indices.device, dtype=torch.int32),
    )
    cache_manager = SimpleNamespace(mamba_layer_cache=lambda _: layer_cache)
    return SimpleNamespace(
        mamba_metadata=mamba_metadata,
        num_contexts=0,
        num_ctx_tokens=0,
        seq_lens=torch.ones(batch, device=slot_indices.device, dtype=torch.int32),
        kv_cache_manager=cache_manager,
    )


def _prefill_metadata(
    layer_cache: SimpleNamespace, slot_indices: torch.Tensor, cu_seqlens: torch.Tensor
) -> SimpleNamespace:
    batch = slot_indices.shape[0]
    mamba_metadata = SimpleNamespace(
        state_indices=slot_indices.to(torch.int32),
        state_indices_long=slot_indices,
        query_start_loc_long=cu_seqlens,
        use_initial_states=False,
    )
    cache_manager = SimpleNamespace(mamba_layer_cache=lambda _: layer_cache)
    return SimpleNamespace(
        mamba_metadata=mamba_metadata,
        num_contexts=batch,
        num_ctx_tokens=int(cu_seqlens[-1]),
        seq_lens=cu_seqlens[1:] - cu_seqlens[:-1],
        kv_cache_manager=cache_manager,
    )


@torch.no_grad()
def test_kda_fused_prefill_matches_separate_projections():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")

    torch.manual_seed(0)
    device = "cuda"
    cfg = _K3Cfg()
    lin = cfg.linear_attn_config
    h = lin["num_heads"]
    head_dim = lin["head_dim"]
    d = h * head_dim
    w = lin["short_conv_kernel_size"]

    runtime = KimiKDARuntime(cfg, layer_idx=0).to(device)
    if runtime.mixer.decode_kernel_path != "optimized":
        pytest.skip("needs an SM100/SM103 GPU")
    for param in runtime.parameters():
        if param.is_floating_point():
            torch.nn.init.normal_(param, std=0.02)

    reference = KimiKDARuntime(cfg, layer_idx=0).to(device)
    reference.load_state_dict(runtime.state_dict())
    runtime.finalize_decode_weights()
    assert runtime._qkvg_proj_weight is not None
    assert runtime._bfa_proj_weight is not None

    slots = 4
    slot_indices = torch.tensor([2, 0], device=device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, 3, 5], device=device, dtype=torch.long)
    hidden_states = torch.randn(5, cfg.hidden_size, device=device, dtype=torch.bfloat16) * 0.05
    conv_seed = torch.randn(slots, 3 * d, w, device=device, dtype=torch.bfloat16) * 0.02
    state_seed = (
        torch.randn(slots, h, head_dim, head_dim, device=device, dtype=torch.float32) * 0.01
    )

    expected_cache = SimpleNamespace(conv=conv_seed.clone(), temporal=state_seed.clone())
    expected = reference(hidden_states, _prefill_metadata(expected_cache, slot_indices, cu_seqlens))

    actual_cache = SimpleNamespace(conv=conv_seed.clone(), temporal=state_seed.clone())
    actual = runtime(hidden_states, _prefill_metadata(actual_cache, slot_indices, cu_seqlens))

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache.conv, expected_cache.conv, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache.temporal, expected_cache.temporal, rtol=2e-2, atol=2e-2)


@torch.no_grad()
def test_kda_qkvg_multistream_decode_matches_separate_projections():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")

    torch.manual_seed(0)
    device = "cuda"
    cfg = _K3Cfg()
    lin = cfg.linear_attn_config
    h = lin["num_heads"]
    head_dim = lin["head_dim"]
    d = h * head_dim
    w = lin["short_conv_kernel_size"]

    runtime = KimiKDARuntime(cfg, layer_idx=0, aux_stream=torch.cuda.Stream()).to(device)
    if runtime.mixer.decode_kernel_path != "optimized":
        pytest.skip("needs an SM100/SM103 GPU")
    for param in runtime.parameters():
        if param.is_floating_point():
            torch.nn.init.normal_(param, std=0.02)

    reference = KimiKDARuntime(cfg, layer_idx=0).to(device)
    reference.load_state_dict(runtime.state_dict())
    runtime.finalize_decode_weights()
    assert runtime._qkvg_proj_weight is not None
    assert runtime._bfa_proj_weight is not None

    batch = 3
    slots = batch + 2
    slot_indices = torch.tensor([2, 0, 4], device=device, dtype=torch.long)
    hidden_states = torch.randn(batch, cfg.hidden_size, device=device, dtype=torch.bfloat16) * 0.05
    conv_seed = torch.randn(slots, 3 * d, w, device=device, dtype=torch.bfloat16) * 0.02
    state_seed = (
        torch.randn(slots, h, head_dim, head_dim, device=device, dtype=torch.float32) * 0.01
    )

    expected_cache = SimpleNamespace(conv=conv_seed.clone(), temporal=state_seed.clone())
    expected = reference(hidden_states, _decode_metadata(expected_cache, slot_indices))

    actual_cache = SimpleNamespace(conv=conv_seed.clone(), temporal=state_seed.clone())
    with with_multi_stream(True):
        actual = runtime(hidden_states, _decode_metadata(actual_cache, slot_indices))

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache.conv, expected_cache.conv, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache.temporal, expected_cache.temporal, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("t_steps", [2, 3])
def test_kda_verify_matches_sequential_decode(batch, t_steps):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    torch.manual_seed(0)
    device = "cuda"
    cfg = _Cfg()
    lin = cfg.linear_attn_config
    h = lin["num_heads"]
    dim = h * lin["head_dim"]
    w = lin["short_conv_kernel_size"]

    runtime = KimiKDARuntime(cfg, layer_idx=0).to(device)
    # dt_bias is torch.empty at construction and only filled by
    # load_weights(); with random weights it holds heap garbage, and a
    # NaN/Inf bit pattern poisons both paths identically (nvbug 6599150).
    torch.nn.init.normal_(runtime.mixer.dt_bias, std=0.1)
    slots = batch + 2  # non-trivial slot mapping
    cache = _LayerCache(slots, 3 * dim, w, h, lin["head_dim"], lin["head_dim"], t_steps, device)
    slot_indices = torch.arange(2, 2 + batch, device=device, dtype=torch.long)

    # Random-but-fixed starting state and inputs.
    torch.nn.init.normal_(cache.conv[2 : 2 + batch], std=0.02)
    torch.nn.init.normal_(cache.temporal[2 : 2 + batch], std=0.02)
    x = torch.randn(batch, t_steps, cfg.hidden_size, dtype=torch.bfloat16, device=device) * 0.1

    # --- Reference: t sequential in-place decodes on a cloned pool. ---
    ref_conv = cache.conv.clone()
    ref_ssm = cache.temporal.clone()
    ref_outs, ref_conv_steps, ref_ssm_steps = [], [], []
    for t in range(t_steps):
        out = runtime._forward_decode(x[:, t], ref_conv, ref_ssm, slot_indices)
        ref_outs.append(out)
        ref_conv_steps.append(ref_conv.index_select(0, slot_indices).clone())
        ref_ssm_steps.append(ref_ssm.index_select(0, slot_indices).clone())

    # --- Verify path: one call, intermediates into the scratch buffers. ---
    pristine_conv = cache.conv.clone()
    pristine_ssm = cache.temporal.clone()
    out_verify = runtime._forward_verify(
        x.reshape(batch * t_steps, cfg.hidden_size),
        t_steps,
        cache,
        cache.conv,
        cache.temporal,
        slot_indices,
    )

    # Live pools must be untouched by verification.
    torch.testing.assert_close(cache.conv, pristine_conv, rtol=0, atol=0)
    torch.testing.assert_close(cache.temporal, pristine_ssm, rtol=0, atol=0)

    out_verify = out_verify.reshape(batch, t_steps, cfg.hidden_size)
    for t in range(t_steps):
        torch.testing.assert_close(out_verify[:, t], ref_outs[t], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            cache.intermediate_conv_window[:batch, t], ref_conv_steps[t], rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            cache.intermediate_ssm[:batch, t], ref_ssm_steps[t], rtol=2e-2, atol=2e-2
        )
