# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA speculative-verification parity: _forward_verify vs sequential decode.

The verify path must produce, for every step t, exactly the state and output
that t sequential single-token _forward_decode calls would produce — the two
paths call the same FLA kernels with the same [B, 1] shapes, so agreement is
expected to near-bitwise tolerance. A real mismatch here means the verify
implementation (state threading, conv stepping, intermediate writes) is
wrong; e2e text divergence on truncated models alone does NOT indicate a bug
(batched MoE reduction order flips argmax on noise logits).

Needs 1 GPU + fla-core; runs with random weights (no checkpoint).
"""

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.models.modeling_kimi_linear import KimiKDARuntime


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
