# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Alignment regression test for the GDN standard-decode path: FlashInfer
``gated_delta_rule`` requires every tensor argument's data pointer to be
32-byte aligned, and ``_flashinfer_gdn_decode`` must realign int32 index
slices before dispatch (``.int()`` is a no-op on an already-int32 tensor and
keeps the misaligned pointer), matching ``_flashinfer_gdn_verify``.
"""

import pytest
import torch


def _fi_decode_available() -> bool:
    if not torch.cuda.is_available():
        return False
    from tensorrt_llm._utils import is_flashinfer_gdn_decode_supported_arch

    if not is_flashinfer_gdn_decode_supported_arch():
        return False
    try:
        from flashinfer.gdn_kernels import gdn_decode_bf16_state as gdn_decode
    except (ImportError, RuntimeError):
        # Mirror the guard in fused_sigmoid_gating_recurrent.py (and
        # FlashInfer's own gdn_kernels/__init__.py): a missing build raises
        # ImportError, a CuTe/CUTLASS mismatch raises RuntimeError; in both
        # cases production falls back to Triton, so the FI path is
        # unavailable and the test must skip.
        return False
    return hasattr(gdn_decode, "gated_delta_rule")


SKIP_UNSUPPORTED = pytest.mark.skipif(
    not _fi_decode_available(),
    reason="Requires SM90/SM100/SM103 and a FlashInfer build with "
    "gdn_decode_bf16_state.gated_delta_rule",
)


@SKIP_UNSUPPORTED
def test_fi_decode_misaligned_index_slice() -> None:
    """Index slices with a non-32B-aligned storage offset must be realigned.

    A caller splitting a mixed batch passes the decode half
    ``state_indices[num_prefills:]`` — an int32 view whose 4*num_prefills-byte
    storage offset violates the FI kernel's 32-byte alignment assert
    (``Misaligned Tensor data on argument`` at runtime) whenever
    ``num_prefills % 8 != 0``. ``_flashinfer_gdn_decode`` must copy such views
    before dispatch, exactly like ``_flashinfer_gdn_verify`` already does.
    """
    from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
        _flashinfer_gdn_decode,
    )

    torch.manual_seed(0)
    dev = "cuda"
    N, H, HV, K, V = 3, 4, 8, 128, 128
    # Standard decode: one token per sequence, packed varlen layout.
    q = (torch.randn(1, N, H, K, device=dev) * 0.1).to(torch.bfloat16)
    k = (torch.randn(1, N, H, K, device=dev) * 0.1).to(torch.bfloat16)
    v = (torch.randn(1, N, HV, V, device=dev) * 0.1).to(torch.bfloat16)
    a = torch.randn(1, N, HV, device=dev) * 0.1
    b = torch.randn(1, N, HV, device=dev) * 0.1
    A_log = torch.empty(HV, device=dev).uniform_(1.0, 16.0).log()
    dt_bias = torch.randn(HV, device=dev) * 0.1
    state_pool = (torch.randn(N + 1, HV, V, K, device=dev) * 0.1).to(torch.bfloat16)
    cu_seqlens = torch.arange(N + 1, device=dev, dtype=torch.long)

    # int32 slice with a 4-byte storage offset (mimics the decode half
    # state_indices[num_prefills:] with num_prefills == 1).
    idx_buf = torch.arange(N + 1, device=dev, dtype=torch.int32)
    idx_misaligned = idx_buf[1:]
    assert idx_misaligned.data_ptr() % 32 != 0

    common_kwargs = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )

    # The decode kernel updates the state pool in place, so give each call
    # its own copy and compare the final pools as well as the outputs.
    pool_mis = state_pool.clone()
    out_mis = _flashinfer_gdn_decode(
        initial_state_source=pool_mis,
        initial_state_indices=idx_misaligned,
        **common_kwargs,
    )

    pool_ref = state_pool.clone()
    out_ref = _flashinfer_gdn_decode(
        initial_state_source=pool_ref,
        initial_state_indices=idx_misaligned.clone(),
        **common_kwargs,
    )

    torch.testing.assert_close(out_mis.float(), out_ref.float())
    torch.testing.assert_close(pool_mis.float(), pool_ref.float())


@SKIP_UNSUPPORTED
@pytest.mark.parametrize("N", [1, 3])
def test_fi_decode_misaligned_ab_slices(N: int) -> None:
    """Non-32B-aligned ``a``/``b`` column slices must be realigned.

    ``gdn_mixer`` splits the fused ``in_proj_ba`` output into
    ``b = ba[:, :num_v_heads_per_tp]`` and ``a = ba[:, num_v_heads_per_tp:]``,
    so ``a`` starts ``num_v_heads_per_tp`` elements in. The FI kernel asserts
    32-byte alignment, so any TP split whose per-rank v-head count is not a
    multiple of 16 (e.g. Qwen3.6-35B-A3B TEP4: 32 / 4 = 8)
    aborts unless ``_flashinfer_gdn_decode`` clones the slice first.
    ``.contiguous()`` is not a fix: at decode the token dim is 1, so the
    offset view already reports as contiguous.

    Two fused buffers exercise each guard: the ordinary split misaligns ``a``;
    shifting the fused buffer by 16 bytes misaligns ``b`` instead.
    """
    from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
        _flashinfer_gdn_decode,
    )

    torch.manual_seed(0)
    dev = "cuda"
    H, HV, K, V = 4, 8, 128, 128
    q = (torch.randn(1, N, H, K, device=dev) * 0.1).to(torch.bfloat16)
    k = (torch.randn(1, N, H, K, device=dev) * 0.1).to(torch.bfloat16)
    v = (torch.randn(1, N, HV, V, device=dev) * 0.1).to(torch.bfloat16)
    A_log = torch.empty(HV, device=dev).uniform_(1.0, 16.0).log()
    dt_bias = torch.randn(HV, device=dev) * 0.1
    state_pool = (torch.randn(N, HV, V, K, device=dev) * 0.1).to(torch.bfloat16)
    idx = torch.arange(N, device=dev, dtype=torch.int32)
    cu_seqlens = torch.arange(N + 1, device=dev, dtype=torch.long)

    # bf16 gating inputs, as the model passes them: HV=8 elements -> 16-byte offset.
    ba = (torch.randn(N, 2 * HV, device=dev) * 0.1).to(torch.bfloat16)
    b_aligned = ba[:, :HV].view(1, N, HV)
    a_misaligned = ba[:, HV:].view(1, N, HV)
    assert b_aligned.data_ptr() % 32 == 0
    assert a_misaligned.data_ptr() % 32 == 16

    shifted_storage = (torch.randn(N * 2 * HV + HV, device=dev) * 0.1).to(torch.bfloat16)
    shifted_ba = shifted_storage[HV:].view(N, 2 * HV)
    b_misaligned = shifted_ba[:, :HV].view(1, N, HV)
    a_aligned = shifted_ba[:, HV:].view(1, N, HV)
    assert b_misaligned.data_ptr() % 32 == 16
    assert a_aligned.data_ptr() % 32 == 0
    if N == 1:
        # Real decode shape: a size-1 token dim makes the offset view report as
        # contiguous, so ``.contiguous()`` would keep the misaligned pointer.
        assert a_misaligned.is_contiguous()
        assert b_misaligned.is_contiguous()

    def _run(a_in, b_in):
        pool = state_pool.clone()
        out = _flashinfer_gdn_decode(
            A_log=A_log,
            a=a_in,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b_in,
            initial_state_source=pool,
            initial_state_indices=idx,
            scale=K**-0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        return out, pool

    for a, b in [(a_misaligned, b_aligned), (a_aligned, b_misaligned)]:
        out_mis, pool_mis = _run(a, b)
        out_ref, pool_ref = _run(
            a.clone(memory_format=torch.contiguous_format),
            b.clone(memory_format=torch.contiguous_format),
        )
        torch.testing.assert_close(out_mis.float(), out_ref.float())
        torch.testing.assert_close(pool_mis.float(), pool_ref.float())
