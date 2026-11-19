# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs of the KDA verify entries (`kda_mtp_verify`, `rubin_kda_mtp_verify`) and of the
`kda_mtp_decode_v2` kernel in the target's runtime layouts.

Not a test file: the entry and kernel tests import it by name. The builders take a device, so the
CPU reference test uses them too.

The q / k / v / output-gate projections are column sections of one fused per-token projection
(row stride 4 * H * 128), beta columns of the padded [f_a | b] projection, the conv caches
dim-contiguous bf16 `[pool, H * 128, 3 + num_spec]`, the replay caches plausible pending drafts
(L2-normalized k, gates in (lower_bound, 0), SiLU'd v, sigmoid'd beta), the beta cache rows padded
to 16 bytes, the state V-first, the requests on a random subset of the pool's slots with accepted
counts cycling over 0..num_spec. With `stacked` the projections are column views of the
specialized targets' one `[q | k | v | g | f_a | b | 0]` GEMM output.
"""

from __future__ import annotations

import torch

__extra_import_path__ = ["."]
from _kda_ops_util import clone_written, stacked_projection  # noqa: E402 — path declared above

K = V = 128
W = 4
LOWER_BOUND = -5.0
ONORM_EPS = 1e-5
#: The tensors a verify call writes in place.
MUTATED = ("recurrent_state", "cs_q", "cs_k", "cs_v", "k_cache", "g_cache", "v_cache", "beta_cache")


def verify_args(spec: dict, seed: int, device="cuda") -> dict:
    """Keyword arguments of `rubin_kda_mtp_verify` for a cell spec (N, H, num_spec, pool,
    state_dtype, out "bf16" | "mxfp8", stacked and state_layer_view).

    `state_layer_view`: the state is layer 1 of a 3-layer pool `[pool, 3, H, V, K]` (slot stride
    3 * H * V * K), like the runtime's per-layer view of the SSM cache.
    """
    N, H, M = spec["N"], spec["H"], spec["num_spec"]
    f32, bf16 = torch.float32, torch.bfloat16
    R, dim = N * (M + 1), H * K
    pool = spec.get("pool") or N
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=f32)

    if spec.get("stacked"):
        proj = stacked_projection(R, H, gen, device).unsqueeze(0)
        qkvg, fab = proj[..., : 4 * dim], proj[..., 4 * dim :]
    else:
        qkvg = randn(1, R, 4 * dim).to(bf16)
        fab = randn(1, R, (K + H + 7) // 8 * 8).to(bf16)
    padded = (H + 7) // 8 * 8
    beta_cache = torch.zeros(pool, M, padded, dtype=bf16, device=device)[..., :H]
    beta_cache.copy_(torch.sigmoid(randn(pool, M, H)))
    v_raw = 2.0 * randn(pool, M, dim)
    slots = torch.randperm(pool, generator=gen, device=device)[:N].to(torch.int32)
    accepted = torch.zeros(pool, dtype=torch.int32, device=device)
    accepted[slots.long()] = (torch.arange(N, device=device) % (M + 1)).to(torch.int32)
    if spec.get("state_layer_view"):
        state = (randn(pool, 3, H, V, K) * 0.5)[:, 1]
    else:
        state = randn(pool, H, V, K) * 0.5
    state = state.to(getattr(torch, spec.get("state_dtype", "float32")))
    args = dict(
        x_q=qkvg[..., :dim].view(1, R, H, K),
        x_k=qkvg[..., dim : 2 * dim].view(1, R, H, K),
        x_v=qkvg[..., 2 * dim : 3 * dim].view(1, R, H, V),
        w_q=(randn(dim, W) * 0.5).contiguous(),
        w_k=(randn(dim, W) * 0.5).contiguous(),
        w_v=(randn(dim, W) * 0.5).contiguous(),
        cs_q=randn(pool, W - 1 + M, dim).to(bf16).transpose(1, 2),
        cs_k=randn(pool, W - 1 + M, dim).to(bf16).transpose(1, 2),
        cs_v=randn(pool, W - 1 + M, dim).to(bf16).transpose(1, 2),
        g=randn(1, R, H, K).to(bf16),
        beta=fab[..., K : K + H],
        A_log=randn(H) * 0.5,
        dt_bias=randn(dim) * 0.5,
        recurrent_state=state,
        k_cache=torch.nn.functional.normalize(randn(pool, M, H, K), dim=-1)
        .reshape(pool, M, dim)
        .to(bf16),
        g_cache=LOWER_BOUND * torch.sigmoid(randn(pool, M, dim)),
        v_cache=(v_raw * torch.sigmoid(v_raw)).to(bf16),
        beta_cache=beta_cache,
        ssm_state_indices=slots,
        num_spec=M,
        num_accepted_tokens=accepted,
        lower_bound=LOWER_BOUND,
        scale=K**-0.5,
        onorm_g=qkvg[..., 3 * dim :].view(1, R, H, V),
        onorm_weight=torch.linspace(0.75, 1.25, V, dtype=f32, device=device),
        onorm_eps=ONORM_EPS,
        quantize_output=spec.get("out", "bf16") == "mxfp8",
    )
    return args


def kda_verify_args(
    N: int,
    H: int,
    num_spec: int,
    pool: int | None = None,
    seed: int = 0,
    state_layer_view: bool = False,
    device="cuda",
) -> dict:
    """Keyword arguments of the family's `kda_mtp_verify` entry and of its kernel
    `kda_mtp_decode_v2.k31_kda_mtp_verify`: `verify_args` without the output quantization flag."""
    spec = dict(N=N, H=H, num_spec=num_spec, pool=pool, state_layer_view=state_layer_view)
    args = verify_args(spec, seed, device)
    del args["quantize_output"]
    return args


def clone_mutated(args: dict) -> dict:
    """A copy of `args` whose in-place outputs are same-stride copies."""
    return clone_written(args, MUTATED)


# The verify kernels run the recurrence in chunk form with 3xTF32 tensor-core products and the
# references one fp32 token step at a time: the two agree to fp32 rounding of a few accumulation
# orders on the state (GB300, all kda_mtp_verify cells: rel RMS <= 3e-6, max abs <= 3e-5), which
# the bf16 rounding of the core and of the normed output turns into one-ulp flips of single
# elements (max abs 1.6e-2 = one ulp in [2, 4), rel RMS <= 1.1e-3 over the cells). The output
# gates leave 2x on the RMS and one more ulp per element.

#: The band of the BF16 output, elementwise.
BF16_ATOL, BF16_RTOL = 3.2e-2, 8e-3


def rel_rms(out: torch.Tensor, ref: torch.Tensor) -> float:
    """The RMS of `out - ref` relative to the RMS of `ref`."""
    diff = (out.float() - ref.float()).pow(2).mean().sqrt()
    return (diff / ref.float().pow(2).mean().sqrt().clamp_min(1e-12)).item()


def check_bf16_output(out: torch.Tensor, ref: torch.Tensor) -> None:
    rel = rel_rms(out, ref)
    assert rel < 2.5e-3, f"normed core relative RMS {rel:.3e}"
    torch.testing.assert_close(out.float(), ref.float(), atol=BF16_ATOL, rtol=BF16_RTOL)


def check_writes(op_args: dict, ref_args: dict, untouched: list, before: torch.Tensor) -> None:
    """Every tensor a verify call writes in place against the reference's. `before` holds the
    state of the `untouched` slots, which the batch does not own, before the call."""
    slots = op_args["ssm_state_indices"].long()
    got, want = (
        op_args["recurrent_state"][slots].float(),
        ref_args["recurrent_state"][slots].float(),
    )
    if op_args["recurrent_state"].dtype == torch.float32:
        rel = rel_rms(got, want)
        assert rel < 3e-5, f"committed state relative RMS {rel:.3e}"
        torch.testing.assert_close(got, want, atol=2e-3, rtol=1e-3)
    else:  # one bf16 rounding of a state that agrees to fp32 rounding
        torch.testing.assert_close(got, want, atol=1e-2, rtol=8e-3)
    assert torch.equal(op_args["recurrent_state"][untouched], before)
    for name in ("cs_q", "cs_k", "cs_v"):  # raw inputs are copied: exact
        assert torch.equal(op_args[name], ref_args[name]), name
    # Cached k / v / beta are bf16 roundings of values that agree to fp32 rounding (one ulp at
    # most); the gate is fp32 (fast exp / sigmoid on the device) of the same bf16 g.
    for name in ("k_cache", "v_cache", "beta_cache"):
        torch.testing.assert_close(
            op_args[name].float(), ref_args[name].float(), atol=1e-2, rtol=8e-3
        )
    torch.testing.assert_close(op_args["g_cache"], ref_args["g_cache"], atol=2e-5, rtol=1e-5)
