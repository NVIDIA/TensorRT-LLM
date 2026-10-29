# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 KDA MTP verify: the compile classes do not follow the batch size.

``trtllm::kda_mtp_decode_fp8`` (``quantize_output=True``, the call K3 makes with FP8-resident
projections) returns an MXFP8 scale buffer: a flat uint8 tensor of ``ceil(rows / 128) * 128 * H *
4`` bytes, which grows with the batch size. Its extent used to be a static dimension of the
compile key, so the kernel was compiled once per 128-row class of the batch (about a second each,
on every rank; a class that no CUDA-graph warmup visited compiled at its first use). The kernels
read only the buffer's pointer, so the compile key must follow the layout classes (``split_v``,
compact) alone, as the BF16-output call's does.

The test runs both calls over batch sizes that cross the 128-row classes of the scale buffer and
the layout classes, with one pool as in serving, and compares the numbers of compiled variants.
Only the compile cache is observed (the inputs are zeros): the parity tests check the results.
"""

import pytest
import torch

try:
    import cuda.bindings.driver  # noqa: F401
    import cutlass  # noqa: F401

    from tensorrt_llm._torch.custom_ops import (
        cute_dsl_kimi_k3_kda_mtp_compact_policy as compact_policy,
    )
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_kda_mtp_ops as kda_ops
except ImportError as e:
    kda_ops = None
    compact_policy = None
    _DEP_ERR = str(e)
else:
    _DEP_ERR = None


def _skip_reason():
    if kda_ops is None:
        return f"deps: {_DEP_ERR}"
    if not torch.cuda.is_available():
        return "needs a GPU"
    prop = torch.cuda.get_device_properties(0)
    if prop.major * 10 + prop.minor not in (100, 103, 107):
        return "needs sm100/sm103/sm107"
    return None


requires_device = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")

K = V = 128
W = 4
POOL = (
    160  # slots per layer: constant across batch sizes, as the serving pools are sized by max slots
)
# Batch sizes crossing the 128-row classes of the scale buffer (rows = N * (1 + num_spec)) and the
# layout classes (split_v at H * N = 64, compact above the SM count).
BATCH_SIZES = (1, 2, 5, 6, 17, 18, 33, 48, 64, 96, 100, 128)
# At most: the cluster-pair, the one-CTA-per-tile and the compact layout.
MAX_LAYOUT_CLASSES = 3


def _pools(H, num_spec):
    """Per-layer state pools in the layout of the serving pools (layer 1 of a two-layer buffer)."""
    d, S = H * K, W - 1 + num_spec
    padded_heads = (H + 7) // 8 * 8

    def pool(shape, dtype):
        return torch.zeros((2, POOL, *shape), dtype=dtype, device="cuda")[1]

    return dict(
        w_q=torch.zeros(d, W, device="cuda"),
        w_k=torch.zeros(d, W, device="cuda"),
        w_v=torch.zeros(d, W, device="cuda"),
        A_log=torch.zeros(H, device="cuda"),
        dt_bias=torch.zeros(d, device="cuda"),
        cs_q=pool((S, d), torch.bfloat16).transpose(-1, -2),
        cs_k=pool((S, d), torch.bfloat16).transpose(-1, -2),
        cs_v=pool((S, d), torch.bfloat16).transpose(-1, -2),
        recurrent_state=pool((H, V, K), torch.bfloat16),
        k_cache=pool((num_spec, d), torch.bfloat16),
        g_cache=pool((num_spec, d), torch.float32),
        v_cache=pool((num_spec, d), torch.bfloat16),
        beta_cache=pool((num_spec, padded_heads), torch.bfloat16)[..., :H],
        num_accepted_tokens=torch.zeros(POOL, dtype=torch.int32, device="cuda"),
        onorm_weight=torch.ones(V, device="cuda"),
    )


def _call(pools, N, H, num_spec):
    """One verify call of N requests in the K3 packed token layout (sections of a fused projection)."""
    d, T = H * K, N * (1 + num_spec)
    qkvg = torch.zeros(T, 4 * d, dtype=torch.bfloat16, device="cuda")
    bfa = torch.zeros(T, K + H, dtype=torch.bfloat16, device="cuda")
    return dict(
        x_q=qkvg[:, 0 * d : 1 * d].view(1, T, H, K),
        x_k=qkvg[:, 1 * d : 2 * d].view(1, T, H, K),
        x_v=qkvg[:, 2 * d : 3 * d].view(1, T, H, V),
        onorm_g=qkvg[:, 3 * d : 4 * d].view(1, T, H, V),
        g=torch.zeros(T, d, dtype=torch.bfloat16, device="cuda").view(1, T, H, K),
        beta=bfa[:, K : K + H].view(1, T, H),
        ssm_state_indices=torch.arange(N, dtype=torch.int32, device="cuda"),
        cu_seqlens=None,
        num_spec=num_spec,
        lower_bound=-5.0,
        scale=K**-0.5,
        packed_token_layout=True,
        onorm_eps=1e-5,
        **pools,
    )


def _compiled_variants(monkeypatch, H, num_spec, quantize_output):
    """Number of variants the op compiles over BATCH_SIZES (the entries of a fresh compile cache)."""
    cache = {}
    monkeypatch.setattr(kda_ops, "_compiled_cache", cache)
    pools = _pools(H, num_spec)
    for N in BATCH_SIZES:
        kda_ops.kda_mtp_decode_impl(
            **_call(pools, N, H, num_spec), fuse_output_norm=True, quantize_output=quantize_output
        )
    torch.cuda.synchronize()
    return len(cache)


@requires_device
@pytest.mark.parametrize("compact", [False, True], ids=["intree", "compact"])
@pytest.mark.parametrize("H,num_spec", [(12, 2), (96, 2), (96, 7)], ids=lambda v: str(v))
def test_fp8_verify_compiles_per_layout_class_not_per_batch_size(H, num_spec, compact, monkeypatch):
    monkeypatch.setenv("TRTLLM_KDA_MTP_BF16_MATH", "0")
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT", "1" if compact else "0")
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_QUANTIZED", "1")
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_TWO_CTAS", "1")
    monkeypatch.setattr(compact_policy, "_kda_compact_policy_cache", {})
    bf16_out = _compiled_variants(monkeypatch, H, num_spec, quantize_output=False)
    fp8_out = _compiled_variants(monkeypatch, H, num_spec, quantize_output=True)
    assert fp8_out == bf16_out, (
        f"the FP8-output call compiled {fp8_out} variants over batch sizes {BATCH_SIZES} against "
        f"{bf16_out} for the BF16-output call: the scale buffer's extent is in the compile key"
    )
    assert fp8_out <= MAX_LAYOUT_CLASSES, fp8_out
