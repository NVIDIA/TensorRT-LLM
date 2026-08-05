# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity test for the GDN MTP *verify* path: FlashInfer ``gated_delta_rule_mtp``
vs the Triton ``fused_recurrent_gated_delta_rule_update`` reference.

The verify path (speculative decoding) runs the gated delta rule over
``draft_token_num`` tokens per sequence and must write the SSM state *after each
draft token* into an ``intermediate_states_buffer`` (so the cache manager can
later select the state at the accepted position), without updating the live
state pool (``disable_state_update=True``).

This test asserts the FlashInfer MTP kernel produces the same attention output
AND the same per-step intermediate states as the Triton kernel, so the verify
branch in ``gdn_mixer`` can dispatch to FlashInfer with a Triton fallback.
"""

import pytest
import torch


def _fi_mtp_available() -> bool:
    if not torch.cuda.is_available():
        return False
    from tensorrt_llm._utils import is_flashinfer_gdn_supported_arch

    if not is_flashinfer_gdn_supported_arch():
        return False
    try:
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp  # noqa: F401
    except Exception:
        return False
    return True


skip_unsupported = pytest.mark.skipif(
    not _fi_mtp_available(),
    reason="Requires SM90/SM100/SM103 and a FlashInfer build with "
    "gdn_decode_bf16_state.gated_delta_rule_mtp",
)


@skip_unsupported
@pytest.mark.parametrize("draft_token_num", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("num_decodes", [1, 3])
@pytest.mark.parametrize("H,HV", [(4, 8), (2, 2)])
def test_fi_mtp_verify_matches_triton(draft_token_num, num_decodes, H, HV):
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

    from tensorrt_llm._torch.modules.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update,
    )
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import fused_gdn_gating

    torch.manual_seed(0)
    dev = "cuda"
    N, T, K, V = num_decodes, draft_token_num, 128, 128
    scale = K**-0.5

    q = (torch.randn(N, T, H, K, device=dev) * 0.1).to(torch.bfloat16)
    k = (torch.randn(N, T, H, K, device=dev) * 0.1).to(torch.bfloat16)
    v = (torch.randn(N, T, HV, V, device=dev) * 0.1).to(torch.bfloat16)
    a = torch.randn(N, T, HV, device=dev) * 0.1
    b = torch.randn(N, T, HV, device=dev) * 0.1
    A_log = torch.empty(HV, device=dev).uniform_(1.0, 16.0).log()
    dt_bias = torch.randn(HV, device=dev) * 0.1
    state_pool = (torch.randn(N, HV, V, K, device=dev) * 0.1).to(torch.bfloat16)
    idx = torch.arange(N, device=dev, dtype=torch.int32)

    # --- Triton reference (gdn_mixer is_target_verify branch) ---
    g = fused_gdn_gating(A_log, a.view(N * T, HV), dt_bias).view(N, T, HV)
    beta = b.sigmoid()
    buf_tri = torch.zeros(N, T, HV, V, K, device=dev, dtype=torch.bfloat16)
    out_tri = fused_recurrent_gated_delta_rule_update(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state_source=state_pool.clone(),
        initial_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,
        intermediate_states_buffer=buf_tri,
        cache_steps=T,
    )

    # --- FlashInfer MTP verify ---
    buf_fi = torch.zeros(N, T, HV, V, K, device=dev, dtype=torch.bfloat16)
    out_fi = q.new_empty(N, T, HV, V)
    gated_delta_rule_mtp(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state_pool.clone(),
        initial_state_indices=idx,
        intermediate_states_buffer=buf_fi,
        disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
        output=out_fi,
    )

    # bf16 recurrent accumulation: use bf16-appropriate tolerance.
    torch.testing.assert_close(out_fi.float(), out_tri.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(buf_fi.float(), buf_tri.float(), rtol=2e-2, atol=2e-2)


@skip_unsupported
def test_fi_mtp_verify_buffer_is_batch_scoped():
    """FI requires ``intermediate_states_buffer`` to be batch-scoped (dim0 == B).

    Unlike the Triton kernel (which indexes a pool-scoped buffer by
    ``initial_state_indices``), FI writes batch row ``i`` to buffer row ``i`` and
    rejects a pool-sized buffer. gdn_mixer therefore passes the
    ``[:num_decodes]`` prefix slice. This guards that contract so a future FI
    bump that silently accepts a larger buffer (writing the wrong rows) is
    caught.
    """
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

    dev = "cuda"
    N, T, H, HV, K, V = 2, 4, 4, 8, 128, 128
    pool = 8  # pool-scoped buffer larger than the batch
    torch.manual_seed(0)
    q = (torch.randn(N, T, H, K, device=dev) * 0.1).to(torch.bfloat16)
    k = (torch.randn(N, T, H, K, device=dev) * 0.1).to(torch.bfloat16)
    v = (torch.randn(N, T, HV, V, device=dev) * 0.1).to(torch.bfloat16)
    a = torch.randn(N, T, HV, device=dev) * 0.1
    b = torch.randn(N, T, HV, device=dev) * 0.1
    A_log = torch.empty(HV, device=dev).uniform_(1.0, 16.0).log()
    dt_bias = torch.randn(HV, device=dev) * 0.1
    state_pool = (torch.randn(N, HV, V, K, device=dev) * 0.1).to(torch.bfloat16)
    idx = torch.arange(N, device=dev, dtype=torch.int32)
    buf_pool = torch.zeros(pool, T, HV, V, K, device=dev, dtype=torch.bfloat16)
    out = q.new_empty(N, T, HV, V)

    with pytest.raises(AssertionError):
        gated_delta_rule_mtp(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state_pool,
            initial_state_indices=idx,
            intermediate_states_buffer=buf_pool,  # dim0=pool != B=N
            disable_state_update=True,
            use_qk_l2norm_in_kernel=True,
            scale=K**-0.5,
            output=out,
        )


@skip_unsupported
def test_fi_mtp_verify_misaligned_index_slice():
    """Index slices with a non-32B-aligned storage offset must be realigned.

    In the mixed prefill+decode verify path, gdn_mixer passes
    ``state_indices_d = cache_indices[num_prefills:]`` — an int32 view whose
    4*num_prefills-byte storage offset violates the FI kernel's 32-byte
    alignment assert (``Misaligned Tensor data on argument`` at runtime).
    ``_flashinfer_gdn_verify`` must copy such views before dispatch.
    """
    from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
        _flashinfer_gdn_verify,
    )

    torch.manual_seed(0)
    dev = "cuda"
    N, T, H, HV, K, V = 2, 4, 4, 8, 128, 128
    q = (torch.randn(N, T, H, K, device=dev) * 0.1).to(torch.bfloat16)
    k = (torch.randn(N, T, H, K, device=dev) * 0.1).to(torch.bfloat16)
    v = (torch.randn(N, T, HV, V, device=dev) * 0.1).to(torch.bfloat16)
    a = torch.randn(N, T, HV, device=dev) * 0.1
    b = torch.randn(N, T, HV, device=dev) * 0.1
    A_log = torch.empty(HV, device=dev).uniform_(1.0, 16.0).log()
    dt_bias = torch.randn(HV, device=dev) * 0.1
    state_pool = (torch.randn(N, HV, V, K, device=dev) * 0.1).to(torch.bfloat16)
    buf = torch.zeros(N, T, HV, V, K, device=dev, dtype=torch.bfloat16)

    # int32 slice with a 4-byte storage offset (mimics cache_indices[1:]).
    idx_buf = torch.arange(N + 1, device=dev, dtype=torch.int32) - 1
    idx_misaligned = idx_buf[1:]
    assert idx_misaligned.data_ptr() % 32 != 0

    out_mis = _flashinfer_gdn_verify(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state_pool,
        initial_state_indices=idx_misaligned,
        intermediate_states_buffer=buf,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
    )

    out_ref = _flashinfer_gdn_verify(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state_pool,
        initial_state_indices=idx_misaligned.clone(),
        intermediate_states_buffer=buf.clone(),
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(out_mis.float(), out_ref.float())


@skip_unsupported
def test_fi_verify_gate_env_killswitch(monkeypatch):
    """The dispatch gate honors the disable env vars and shape constraints."""
    import tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent as fsg

    pool = torch.zeros(4, 8, 128, 128, device="cuda", dtype=torch.bfloat16)
    assert fsg._can_use_flashinfer_gdn_verify(pool, 128, 128, 4)

    monkeypatch.setenv("TRTLLM_FLA_DISABLE_FLASHINFER_GDN_VERIFY", "1")
    assert not fsg._can_use_flashinfer_gdn_verify(pool, 128, 128, 4)
    monkeypatch.delenv("TRTLLM_FLA_DISABLE_FLASHINFER_GDN_VERIFY")

    monkeypatch.setenv("TRTLLM_FLA_DISABLE_FLASHINFER_GDN", "1")
    assert not fsg._can_use_flashinfer_gdn_verify(pool, 128, 128, 4)
    monkeypatch.delenv("TRTLLM_FLA_DISABLE_FLASHINFER_GDN")

    # Shape/dtype constraints
    assert not fsg._can_use_flashinfer_gdn_verify(pool.float(), 128, 128, 4)
    assert not fsg._can_use_flashinfer_gdn_verify(pool, 64, 128, 4)
    assert not fsg._can_use_flashinfer_gdn_verify(pool, 128, 128, 0)
    assert not fsg._can_use_flashinfer_gdn_verify(pool, 128, 128, fsg._FI_GDN_MAX_MTP_T + 1)
    assert not fsg._can_use_flashinfer_gdn_verify(None, 128, 128, 4)
