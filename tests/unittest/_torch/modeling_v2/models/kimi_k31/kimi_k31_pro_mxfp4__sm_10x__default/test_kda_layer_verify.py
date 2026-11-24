# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA decode with fused projection weights matches the engine KDA layer's unfused decode.

After finalize_decode_weights the target runs its fused-projection decode path; the engine's KDA layer with the same
weights, not finalized, runs its unfused fallback. Both must produce the same output and the same updates of the
selected conv / recurrent state slots, and leave every other slot untouched.

Needs 1 GPU + fla-core; runs with random weights (no checkpoint).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E501
    K31KdaLayer,
)
from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import is_kda_optimized_supported
from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import (
    KimiKDALinearAttention as EngineKimiKDALinearAttention,
)
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream


class _K31Cfg:
    hidden_size = 768
    rms_norm_eps = 1e-6
    kda_fp32_gate_params = True
    linear_attn_config = {
        "num_heads": 6,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }


def _fp32_gate_params(layer) -> None:
    """Give ``A_log`` / ``dt_bias`` FP32 storage, as Kimi K3.1 keeps them (the engine layer rounds them to BF16)."""
    for name in ("A_log", "dt_bias"):
        param = getattr(layer, name)
        param.data = param.data.float()


def _decode_metadata(layer_cache: SimpleNamespace, slot_indices: torch.Tensor) -> SimpleNamespace:
    batch = slot_indices.shape[0]
    mamba_metadata = SimpleNamespace(
        state_indices=slot_indices.to(torch.int32),
        query_start_loc_long=torch.zeros(1, device=slot_indices.device, dtype=torch.long),
        _arange_buffer=torch.arange(batch + 1, device=slot_indices.device, dtype=torch.int32),
        generation_state_indices=None,
    )
    cache_manager = SimpleNamespace(mamba_layer_cache=lambda _: layer_cache)
    return SimpleNamespace(
        mamba_metadata=mamba_metadata,
        num_contexts=0,
        num_ctx_tokens=0,
        num_tokens=batch,
        seq_lens=torch.ones(batch, device=slot_indices.device, dtype=torch.int32),
        kv_cache_manager=cache_manager,
    )


@torch.no_grad()
@pytest.mark.parametrize("num_heads", [6, 8])
def test_kda_decode_matches_reference(num_heads):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")

    torch.manual_seed(0)
    device = "cuda"
    cfg = _K31Cfg()
    cfg.linear_attn_config = {**cfg.linear_attn_config, "num_heads": num_heads}
    lin = cfg.linear_attn_config
    h = lin["num_heads"]
    head_dim = lin["head_dim"]
    d = h * head_dim
    w = lin["short_conv_kernel_size"]

    if not is_kda_optimized_supported():
        pytest.skip("needs an SM100/SM103 GPU")
    runtime = K31KdaLayer(cfg, layer_idx=0, aux_stream=torch.cuda.Stream()).to(device)
    for param in runtime.parameters():
        if param.is_floating_point():
            torch.nn.init.normal_(param, std=0.02)

    reference = EngineKimiKDALinearAttention(cfg, layer_idx=0).to(device)
    _fp32_gate_params(reference)
    reference.load_state_dict(runtime.state_dict())
    runtime.finalize_decode_weights()
    assert runtime._qkvg_proj_weight is not None
    assert runtime._bfa_proj_weight is not None

    batch = 3
    slots = batch + 2
    slot_indices = torch.tensor([2, 0, 4], device=device, dtype=torch.long)
    hidden_states = torch.randn(batch, cfg.hidden_size, device=device, dtype=torch.bfloat16) * 0.05
    conv_seed = torch.randn(slots, 3 * d, w - 1, device=device, dtype=torch.bfloat16) * 0.02
    state_seed = (
        torch.randn(slots, h, head_dim, head_dim, device=device, dtype=torch.float32) * 0.01
    )

    expected_cache = SimpleNamespace(
        conv=conv_seed.clone(), temporal=state_seed.clone(), has_kda_replay_caches=False
    )
    expected = reference(hidden_states, _decode_metadata(expected_cache, slot_indices))

    actual_cache = SimpleNamespace(
        conv=conv_seed.clone(), temporal=state_seed.clone(), has_kda_replay_caches=False
    )
    with with_multi_stream(True):
        actual = runtime(hidden_states, _decode_metadata(actual_cache, slot_indices))

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache.conv, expected_cache.conv, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_cache.temporal, expected_cache.temporal, rtol=2e-4, atol=2e-4)
    untouched = torch.ones(slots, dtype=torch.bool, device=device)
    untouched[slot_indices] = False
    assert torch.equal(actual_cache.conv[untouched], expected_cache.conv[untouched])
    assert torch.equal(actual_cache.temporal[untouched], expected_cache.temporal[untouched])
