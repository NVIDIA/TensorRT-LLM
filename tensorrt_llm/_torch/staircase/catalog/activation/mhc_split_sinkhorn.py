# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-Connection coefficient split into pre / post / comb, via the trtllm mhc op.

`comb` is doubly stochastic at the **certified `sinkhorn_repeat = 20`**, not at
every count: the column divide is by `cs + eps`, so at one pass a column whose
four entries are all near zero lands at `4/5`. The contract carries the
measurement under "Accepted silently and NOT guarded".
"""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

#: The kernel hard-codes `HC_MULT = 4` (`mhcKernels.cu`), so every buffer width
#: below is fixed, not derived from an argument. A caller with a different
#: hyper-connection multiplier needs a different kernel, not a different call.
HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT  # 24 = pre(4) + post(4) + comb(4x4)


def mhc_split_sinkhorn(
    y_acc: torch.Tensor,
    r_acc: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    k: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one projection into `(pre, post, comb)` in one call; comb Sinkhorn-normalized.

    `comb` is doubly stochastic at the certified `sinkhorn_repeat = 20`; see the
    module docstring and the contract for what a smaller count gives instead.
    """
    # `M` is DERIVED, never passed in. The op takes it as an argument and writes
    # `M` rows into whatever buffers it is given: driven here, `M = 2 * rows`
    # against correctly sized outputs was accepted and wrote past their end. A
    # wrapper that allocates the outputs itself and derives `M` from `y_acc`
    # cannot express that, which is better than guarding it.
    assert y_acc.dim() == 2 and y_acc.shape[1] == MIX_HC, (
        f"y_acc must be [M, {MIX_HC}] -- the kernel's HC_MULT is a compile-time 4 -- "
        f"but is {tuple(y_acc.shape)}"
    )
    m = y_acc.shape[0]
    # EIGHT guard branches, in five asserts, and every one of them is PURE TENSOR
    # METADATA -- shape, stride, length. That restriction is the catalog's, not a
    # style preference: a wrapper assert may only cover a metadata violation the
    # op was observed to accept silently.
    #
    # Each branch is earned by a case in
    # `tests/unittest/_torch/staircase/activation/test_staircase_mhc_split_sinkhorn.py`
    # that FIRST drives the raw op with the malformed argument and asserts it is
    # accepted, and only then asserts this wrapper rejects it. Each malformed
    # argument differs from the correct one in METADATA ONLY -- same logical
    # values, with any storage the kernel then reads past them poisoned by the
    # test -- so the observed damage is attributable to the shape or the stride
    # rather than to the values.
    #
    # WHAT IS DELIBERATELY NOT GUARDED, and why:
    #
    #   * `k` and `sinkhorn_repeat` are COMPUTATION VALUES, not metadata: `k` is
    #     the divisor of the square sum and `sinkhorn_repeat` is a loop bound.
    #     Their invalid domains are silent -- `k <= 0` gives NaN or a collapsed
    #     `rstd`, and `sinkhorn_repeat <= 1` is clamped to the 1-pass result --
    #     but silence alone does not authorize a wrapper assert, so those are
    #     documented in the contract's Preconditions and measured in the test
    #     rather than rejected here. An earlier revision asserted both; that was
    #     a wrapper-contract violation, not a safety feature.
    #   * `hc_scale`, `hc_base` and `r_acc` LONGER than required are accepted and
    #     bit-equal to correct -- the kernel indexes them and ignores the tail --
    #     so `numel() == 3` / `== 24` / `shape == (M,)` were rejecting valid
    #     calls. They are minimum-length checks now.
    #   * a non-fp32 `y_acc`, which the op rejects itself and loudly
    #     (`expected scalar type Float but found BFloat16`).
    assert y_acc.is_contiguous(), (
        "y_acc must be contiguous; a column slice of a wider buffer is accepted and read "
        "with the wrong stride"
    )
    assert r_acc.numel() >= m and r_acc.is_contiguous(), (
        f"r_acc must be a contiguous square-sum buffer of at least {m} elements, one per "
        f"token, but holds {r_acc.numel()}; a non-contiguous one is accepted and read with "
        f"the wrong stride, and a short one is read past its end -- measured with a "
        f"controlled tail, every token past the end took the tail's value with no diagnostic"
    )
    assert hc_scale.numel() >= 3 and hc_scale.is_contiguous(), (
        f"hc_scale must be contiguous and hold at least 3 elements (pre/post/comb), got "
        f"{hc_scale.numel()}; a non-contiguous one is accepted and read with the wrong stride"
    )
    assert hc_base.numel() >= MIX_HC and hc_base.is_contiguous(), (
        f"hc_base must be contiguous and hold at least {MIX_HC} elements, got "
        f"{hc_base.numel()}; a non-contiguous one is accepted and read with the wrong stride"
    )
    pre = torch.empty(m, HC_MULT, device=y_acc.device, dtype=torch.float32)
    post = torch.empty(m, HC_MULT, device=y_acc.device, dtype=torch.float32)
    comb = torch.empty(m, HC_MULT, HC_MULT, device=y_acc.device, dtype=torch.float32)
    torch.ops.trtllm.mhc_split_sinkhorn(
        y_acc,
        r_acc,
        hc_scale,
        hc_base,
        pre,
        post,
        comb,
        m,
        k,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return pre, post, comb
