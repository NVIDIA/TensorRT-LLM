# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-level tests for the FlashInfer GDN prefill adapter wrapper.

Compares ``tensorrt_llm._torch.modules.fla.flashinfer_chunk.chunk_gated_delta_rule``
against the vendored Triton ``tensorrt_llm._torch.modules.fla.chunk.chunk_gated_delta_rule``
across the call shapes used by ``Qwen3NextGatedDeltaNet.forward_extend``.
"""

import pytest
import torch

from tensorrt_llm._utils import is_sm_100f

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

# Reuse the wrapper's own predicate so the gate cannot drift from the dispatch
# condition in ``flashinfer_chunk.chunk_gated_delta_rule``. CPU-safe:
# ``get_sm_version()`` returns -1 when no CUDA device is present.
skip_unless_indexed_pool_io = pytest.mark.skipif(
    not is_sm_100f(),
    reason="FlashInfer implements indexed state-pool I/O only on SM100/SM103",
)


# Arch-gating predicate (GPU-free) -----------------------------------------


@pytest.mark.parametrize(
    "sm_version, expected",
    [
        (90, True),  # Hopper
        (100, True),  # datacenter Blackwell (B200)
        (103, True),  # datacenter Blackwell (B300/GB200)
        (120, False),  # consumer Blackwell (RTX 5090 / PRO 6000) -> Triton
        (121, False),  # other consumer Blackwell -> Triton
        (89, False),  # Ada
        (80, False),  # Ampere
    ],
)
def test_is_flashinfer_gdn_supported_arch(sm_version, expected):
    """FlashInfer ships GDN prefill/decode kernels only for SM90/SM100/SM103;
    every other arch (notably SM120) must fall back to Triton. Pure predicate,
    no GPU required."""
    from tensorrt_llm._utils import is_flashinfer_gdn_supported_arch

    assert is_flashinfer_gdn_supported_arch(sm_version) is expected


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
@pytest.mark.parametrize(
    "pool_dtype",
    [torch.float32, torch.bfloat16],
    ids=["fp32_pool", "bf16_pool"],
)
def test_indexed_gather_inplace_scatter_matches_triton(monkeypatch, pool_dtype):
    """Non-spec prefill path via the **gather/scatter** branch: caller passes the
    full SSM pool plus cache_indices, the wrapper gathers into FlashInfer's packed
    layout and scatters the result back (inplace_indexed_state_update=True,
    output_final_state=False).

    ``is_sm_100f`` is forced False to pin the wrapper to that branch. Without the
    pin this test would take the indexed-pool fast path on SM100/SM103 -- which
    ``test_indexed_state_pool_fast_path_bf16_matches_triton`` already covers --
    and since l0_b200/l0_b300 are the only test-db entries for this directory,
    gather/scatter (still the production path on SM90/SM120) would have no CI
    coverage at all.

    The pin also forces ``state_dtype`` to fp32, so the ``bf16_pool`` case
    exercises the up-cast/down-cast that SM90/SM120 require (bf16 -> fp32 on
    gather, fp32 -> bf16 on scatter). With ``fp32_pool`` no cast happens and the
    two helpers degenerate to a plain indexed gather/scatter, which is why both
    dtypes are covered.
    """
    from tensorrt_llm._torch.modules.fla import flashinfer_chunk
    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as triton_cgdr

    fi_cgdr = flashinfer_chunk.chunk_gated_delta_rule
    monkeypatch.setattr(flashinfer_chunk, "is_sm_100f", lambda *a, **kw: False)

    seq_lens = [4096, 8192]
    q, k, v, g, beta, cu = _make_inputs(seq_lens)
    num_v_heads, head_dim = v.shape[2], v.shape[3]

    # Simulate a 16-slot SSM pool; sequences live at slots [3, 7].
    pool_slots = 16
    cache_indices = torch.tensor([3, 7], dtype=torch.int32, device=q.device)
    pool_init = (
        torch.randn(pool_slots, num_v_heads, head_dim, head_dim, device=q.device) * 0.01
    ).to(pool_dtype)

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


@skip_unless_indexed_pool_io
@pytest.mark.parametrize(
    "cache_indices_list",
    [
        [11, 2, 14, 5],  # scattered and out of ascending order
        [15, 0],  # both pool edges, descending
    ],
    ids=["scattered_unsorted", "edges_descending"],
)
def test_indexed_state_pool_fast_path_bf16_matches_triton(monkeypatch, cache_indices_list):
    """SM100/SM103 fast path: the pool and its slot indices go straight to the
    kernel (no gather/scatter passes, see ``flashinfer_chunk.py`` "Fast path").

    Covers the production prefill configuration the fast path was written for --
    a **bf16** state pool (SM100/SM103 keep the recurrent state fp32 in TMEM, so
    the pool needs no fp32 round-trip) addressed by **non-contiguous, unsorted**
    ``cache_indices``. Since the kernel now writes the pool itself, the slot
    addressing is its responsibility rather than the wrapper's scatter, so this
    asserts all three: the output, the updated slots, and that every other slot
    is left bit-identical.
    """
    from tensorrt_llm._torch.modules.fla import flashinfer_chunk
    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as triton_cgdr

    fi_cgdr = flashinfer_chunk.chunk_gated_delta_rule

    seq_lens = [4096, 8192, 1024, 2048][: len(cache_indices_list)]
    q, k, v, g, beta, cu = _make_inputs(seq_lens)
    num_v_heads, head_dim = v.shape[2], v.shape[3]

    pool_slots = 16
    cache_indices = torch.tensor(cache_indices_list, dtype=torch.int32, device=q.device)
    # bf16 pool, matching the SM100/SM103 production dtype.
    pool_init = (
        torch.randn(pool_slots, num_v_heads, head_dim, head_dim, device=q.device) * 0.01
    ).to(torch.bfloat16)

    common = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state_indices=cache_indices,
        inplace_indexed_state_update=True,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    pool_triton = pool_init.clone()
    out_triton, _ = triton_cgdr(initial_state=pool_triton, **common)

    # Prove the *direct* indexed-pool dispatch, not just a matching result: the
    # value assertions below would also pass through the gather/scatter
    # fallback, so a regression that silently disables the fast path would go
    # unnoticed. Trip on the fallback's first step instead.
    def _no_gather(*args, **kwargs):
        raise AssertionError(
            "indexed-pool fast path must not gather: gather_cast_vk_to_fp32_vk was called"
        )

    monkeypatch.setattr(flashinfer_chunk, "gather_cast_vk_to_fp32_vk", _no_gather)

    pool_fi = pool_init.clone()
    out_fi, final_fi = fi_cgdr(initial_state=pool_fi, **common)

    assert final_fi is None  # inplace=True returns no separate final state
    assert pool_fi.dtype == torch.bfloat16  # fast path must not silently up-cast
    torch.testing.assert_close(out_fi, out_triton, atol=2e-2, rtol=2e-2)

    # Written slots: compare per index so a mis-addressed write (e.g. writing in
    # cu_seqlens order instead of slot order) fails here rather than averaging out.
    for seq_id, slot in enumerate(cache_indices_list):
        torch.testing.assert_close(
            pool_fi[slot].to(torch.float32),
            pool_triton[slot].to(torch.float32),
            atol=5e-2,
            rtol=5e-2,
            msg=lambda s, seq_id=seq_id, slot=slot: f"seq {seq_id} -> pool slot {slot}: {s}",
        )

    # Every other slot must be bit-identical to the pre-call pool: the kernel
    # writes the pool directly now, so an off-by-one or a full-pool store would
    # corrupt live state belonging to other requests.
    untouched = [i for i in range(pool_slots) if i not in cache_indices_list]
    assert torch.equal(pool_fi[untouched], pool_init[untouched]), (
        f"fast path modified untouched pool slots {untouched}"
    )


# Env-flag routing test (no GPU required) ---------------------------------


def test_gdn_mixer_resolve_chunk_gated_delta_rule(monkeypatch):
    """gdn_mixer resolves its prefill kernel lazily (``_resolve_chunk_gated_delta_rule``):
    the FlashInfer wrapper when the env opt-in is set (default) *and* the arch is
    supported (SM90/SM100/SM103), otherwise the vendored Triton kernel (env
    opt-out, or an unsupported arch such as SM120).

    The arch predicate is monkeypatched so the routing is checked independent of
    the actual GPU; only dispatch wiring is exercised (no kernel launch).
    """
    import tensorrt_llm._torch.modules.mamba.gdn_mixer as gdn_mixer
    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as triton_fn
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import (
        chunk_gated_delta_rule as flashinfer_fn,
    )

    def resolve(env, arch_supported):
        if env is None:
            monkeypatch.delenv("TLLM_USE_FLASHINFER_GDN_PREFILL", raising=False)
        else:
            monkeypatch.setenv("TLLM_USE_FLASHINFER_GDN_PREFILL", env)
        monkeypatch.setattr(gdn_mixer, "is_flashinfer_gdn_supported_arch", lambda: arch_supported)
        gdn_mixer._resolve_chunk_gated_delta_rule.cache_clear()
        return gdn_mixer._resolve_chunk_gated_delta_rule()

    # Default env + supported arch -> FlashInfer wrapper.
    assert resolve(None, True) is flashinfer_fn
    # Explicit opt-in + supported arch -> FlashInfer wrapper.
    assert resolve("1", True) is flashinfer_fn
    # Opt-out env -> Triton even on a supported arch.
    assert resolve("0", True) is triton_fn
    # Unsupported arch (e.g. SM120) -> Triton even with the default opt-in.
    assert resolve(None, False) is triton_fn

    # Clear the cached resolution so later tests re-resolve against the real
    # arch/env (monkeypatch restores the env var and predicate on teardown).
    gdn_mixer._resolve_chunk_gated_delta_rule.cache_clear()
