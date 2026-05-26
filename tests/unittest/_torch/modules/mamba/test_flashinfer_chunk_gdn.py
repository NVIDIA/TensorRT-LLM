# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-level tests for the FlashInfer GDN prefill adapter wrapper.

Compares ``tensorrt_llm._torch.modules.fla.flashinfer_chunk.chunk_gated_delta_rule``
against the vendored Triton ``tensorrt_llm._torch.modules.fla.chunk.chunk_gated_delta_rule``
across the call shapes used by ``Qwen3NextGatedDeltaNet.forward_extend``.
"""

import pytest
import torch

# Skip rules ---------------------------------------------------------------


def _supported_arch() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    # SM90 (Hopper) or SM100 (Blackwell)
    return major in (9, 10)


skip_unsupported = pytest.mark.skipif(
    not _supported_arch(),
    reason="FlashInfer GDN prefill requires SM90 (Hopper) or SM100 (Blackwell)",
)


# Input factory ------------------------------------------------------------


@torch.no_grad()
def _make_inputs(
    seq_lens: list[int],
    num_q_heads: int = 4,
    num_v_heads: int = 16,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 0,
):
    """Build (q, k, v, g, beta, cu_seqlens) packed in TRT-LLM ``[1, T, H, D]`` layout.

    Mirrors what ``Qwen3NextGatedDeltaNet.forward_extend`` passes after the QKV
    split. ``g`` / ``beta`` are produced post-``fused_gdn_gating`` (fp32).
    """
    torch.manual_seed(seed)
    total_t = sum(seq_lens)
    q = torch.randn(1, total_t, num_q_heads, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(1, total_t, num_q_heads, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(1, total_t, num_v_heads, head_dim, dtype=dtype, device=device) * 0.1
    # g is the "log-forget" gate; emulate post-`fused_gdn_gating` (negative, fp32).
    g = -torch.rand(1, total_t, num_v_heads, dtype=torch.float32, device=device) * 0.05
    beta = torch.rand(1, total_t, num_v_heads, dtype=torch.float32, device=device)
    cu = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
        dtype=torch.int64,
        device=device,
    )
    return q, k, v, g, beta, cu


def _zero_initial_state(num_seqs, num_heads, head_dim, device, dtype=torch.float32):
    return torch.zeros(num_seqs, num_heads, head_dim, head_dim, dtype=dtype, device=device)


# Pure-Python import smoke (no GPU required) ------------------------------


def test_wrapper_module_importable():
    """Smoke import of the wrapper. Pure Python; does not require CUDA."""
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import (  # noqa: F401
        chunk_gated_delta_rule,
    )


# Parity tests against the Triton reference -------------------------------


@skip_unsupported
def test_basic_single_seq_no_l2norm_matches_triton():
    """Single-seq, no initial state, no L2 norm, no output_final_state."""
    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as triton_cgdr
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule as fi_cgdr

    seq_lens = [4096]
    q, k, v, g, beta, cu = _make_inputs(seq_lens)
    init = _zero_initial_state(len(seq_lens), v.shape[2], v.shape[3], q.device)

    out_triton, _ = triton_cgdr(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init,
        initial_state_indices=None,
        inplace_indexed_state_update=False,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )

    out_fi, final_fi = fi_cgdr(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init,
        initial_state_indices=None,
        inplace_indexed_state_update=False,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )

    assert final_fi is None
    assert out_fi.shape == out_triton.shape
    torch.testing.assert_close(out_fi, out_triton, atol=2e-2, rtol=2e-2)


@skip_unsupported
def test_basic_single_seq_with_l2norm_matches_triton():
    """Single-seq, no initial state, with L2 norm (Qwen3.5 production setting)."""
    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as triton_cgdr
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule as fi_cgdr

    seq_lens = [8192]
    q, k, v, g, beta, cu = _make_inputs(seq_lens)
    init = _zero_initial_state(len(seq_lens), v.shape[2], v.shape[3], q.device)

    out_triton, _ = triton_cgdr(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init,
        initial_state_indices=None,
        inplace_indexed_state_update=False,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    out_fi, _ = fi_cgdr(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init,
        initial_state_indices=None,
        inplace_indexed_state_update=False,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(out_fi, out_triton, atol=2e-2, rtol=2e-2)


@skip_unsupported
@pytest.mark.parametrize(
    "seq_lens",
    [
        [4096, 4096],
        [4096, 8192, 4096],
        [1024, 16384],
    ],
)
def test_varlen_with_l2norm_matches_triton(seq_lens):
    """Varlen batches — production prefill packs multiple requests."""
    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as triton_cgdr
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule as fi_cgdr

    q, k, v, g, beta, cu = _make_inputs(seq_lens)
    init = _zero_initial_state(len(seq_lens), v.shape[2], v.shape[3], q.device)

    out_triton, _ = triton_cgdr(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init,
        initial_state_indices=None,
        inplace_indexed_state_update=False,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    out_fi, _ = fi_cgdr(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init,
        initial_state_indices=None,
        inplace_indexed_state_update=False,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(out_fi, out_triton, atol=2e-2, rtol=2e-2)


@skip_unsupported
def test_packed_initial_state_with_output_final_state_matches_triton():
    """target_verify prefill path: caller pre-gathers ssm_states[state_indices_p] and
    writes the returned final state back manually (output_final_state=True)."""
    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as triton_cgdr
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule as fi_cgdr

    seq_lens = [4096, 8192]
    q, k, v, g, beta, cu = _make_inputs(seq_lens)
    num_seqs = len(seq_lens)
    init = (torch.randn(num_seqs, v.shape[2], v.shape[3], v.shape[3], device=q.device) * 0.01).to(
        torch.float32
    )

    out_triton, final_triton = triton_cgdr(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init.clone(),
        initial_state_indices=None,
        inplace_indexed_state_update=False,
        output_final_state=True,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    out_fi, final_fi = fi_cgdr(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init.clone(),
        initial_state_indices=None,
        inplace_indexed_state_update=False,
        output_final_state=True,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    assert final_fi is not None
    assert final_triton is not None
    torch.testing.assert_close(out_fi, out_triton, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        final_fi.to(torch.float32),
        final_triton.to(torch.float32),
        atol=5e-2,
        rtol=5e-2,
    )


@skip_unsupported
def test_indexed_gather_inplace_scatter_matches_triton():
    """Non-spec prefill path: caller passes the full SSM pool plus cache_indices,
    kernel does inplace gather/scatter (inplace_indexed_state_update=True, output_final_state=False)."""
    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as triton_cgdr
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule as fi_cgdr

    seq_lens = [4096, 8192]
    q, k, v, g, beta, cu = _make_inputs(seq_lens)
    num_v_heads, head_dim = v.shape[2], v.shape[3]

    # Simulate a 16-slot SSM pool; sequences live at slots [3, 7].
    pool_slots = 16
    cache_indices = torch.tensor([3, 7], dtype=torch.int32, device=q.device)
    pool_init = (
        torch.randn(pool_slots, num_v_heads, head_dim, head_dim, device=q.device) * 0.01
    ).to(torch.float32)

    pool_triton = pool_init.clone()
    out_triton, _ = triton_cgdr(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=pool_triton,
        initial_state_indices=cache_indices,
        inplace_indexed_state_update=True,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    pool_fi = pool_init.clone()
    out_fi, final_fi = fi_cgdr(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=pool_fi,
        initial_state_indices=cache_indices,
        inplace_indexed_state_update=True,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    assert final_fi is None  # caller asks for None when inplace=True
    torch.testing.assert_close(out_fi, out_triton, atol=2e-2, rtol=2e-2)

    # The two written slots must match within tolerance; the others must be untouched.
    torch.testing.assert_close(
        pool_fi[cache_indices].to(torch.float32),
        pool_triton[cache_indices].to(torch.float32),
        atol=5e-2,
        rtol=5e-2,
    )
    untouched = [i for i in range(pool_slots) if i not in cache_indices.tolist()]
    torch.testing.assert_close(pool_fi[untouched], pool_init[untouched], atol=0.0, rtol=0.0)


# Env-flag routing test (no GPU required) ---------------------------------


def test_gdn_mixer_default_uses_flashinfer_wrapper(monkeypatch):
    """Default (no env): gdn_mixer imports the FlashInfer wrapper.
    Opt-out (``TLLM_USE_FLASHINFER_GDN_PREFILL=0``) restores the Triton path.

    Independent of GPU availability; only checks Python import wiring.
    """
    import importlib

    # Default — env unset, expect FlashInfer wrapper.
    monkeypatch.delenv("TLLM_USE_FLASHINFER_GDN_PREFILL", raising=False)
    import tensorrt_llm._torch.modules.mamba.gdn_mixer as gdn_mixer

    importlib.reload(gdn_mixer)
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import (
        chunk_gated_delta_rule as wrapper_fn,
    )

    assert gdn_mixer.chunk_gated_delta_rule is wrapper_fn

    # Opt out — env=0 restores the Triton path.
    monkeypatch.setenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "0")
    importlib.reload(gdn_mixer)
    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as triton_fn

    assert gdn_mixer.chunk_gated_delta_rule is triton_fn

    # Reset to default for subsequent tests in the same process.
    monkeypatch.delenv("TLLM_USE_FLASHINFER_GDN_PREFILL", raising=False)
    importlib.reload(gdn_mixer)
