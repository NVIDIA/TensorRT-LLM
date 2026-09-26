# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-level tests for the FlashInfer GDN prefill adapter wrapper.

Compares ``tensorrt_llm._torch.modules.fla.flashinfer_chunk.chunk_gated_delta_rule``
against the vendored Triton ``tensorrt_llm._torch.modules.fla.chunk.chunk_gated_delta_rule``
across the call shapes used by ``Qwen3NextGatedDeltaNet.forward_extend``.
"""

import pytest
import torch

from tensorrt_llm._utils import is_flashinfer_gdn_prefill_supported_arch, is_sm_100f

# Skip rules ---------------------------------------------------------------


def _supported_arch() -> bool:
    if not torch.cuda.is_available():
        return False
    return is_flashinfer_gdn_prefill_supported_arch()


skip_unsupported = pytest.mark.skipif(
    not _supported_arch(),
    reason="FlashInfer GDN prefill requires SM90 (Hopper) or SM100/SM103/SM120/SM121 (Blackwell)",
)


# Arch-gating predicate (GPU-free) -----------------------------------------


@pytest.mark.parametrize(
    "sm_version, expected",
    [
        (90, True),  # Hopper
        (100, True),  # datacenter Blackwell (B200)
        (103, True),  # datacenter Blackwell (B300/GB200)
        (120, True),  # consumer Blackwell (RTX 5090 / PRO 6000)
        (121, True),  # consumer Blackwell (GB10 / DGX Spark)
        (89, False),  # Ada
        (80, False),  # Ampere
    ],
)
def test_is_flashinfer_gdn_prefill_supported_arch(sm_version, expected):
    """Pure predicate, no GPU required."""
    from tensorrt_llm._utils import is_flashinfer_gdn_prefill_supported_arch

    assert is_flashinfer_gdn_prefill_supported_arch(sm_version) is expected


@pytest.mark.parametrize(
    "sm_version, expected",
    [
        (90, True),  # Hopper
        (100, True),  # datacenter Blackwell (B200)
        (103, True),  # datacenter Blackwell (B300/GB200)
        (120, False),  # consumer Blackwell: prefill kernel exists, decode does not
        (121, False),  # consumer Blackwell: prefill kernel exists, decode does not
        (89, False),  # Ada
        (80, False),  # Ampere
    ],
)
def test_is_flashinfer_gdn_decode_supported_arch(sm_version, expected):
    """The bf16-state decode / MTP-verify kernels are SM90/SM100/SM103 only --
    narrower than the prefill set, which also covers SM120/SM121. Pure predicate,
    no GPU required."""
    from tensorrt_llm._utils import is_flashinfer_gdn_decode_supported_arch

    assert is_flashinfer_gdn_decode_supported_arch(sm_version) is expected


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
    """Create a zero-filled recurrent state pool."""
    return torch.zeros(num_seqs, num_heads, head_dim, head_dim, dtype=dtype, device=device)


# Pure-Python import smoke (no GPU required) ------------------------------


def test_wrapper_module_importable():
    """Smoke import of the wrapper. Pure Python; does not require CUDA."""
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import (  # noqa: F401
        chunk_gated_delta_rule,
    )


@skip_unsupported
@pytest.mark.parametrize("num_seqs", [1, 8, 32, 512])
@pytest.mark.parametrize("inplace", [False, True])
def test_preallocated_state_workspace(num_seqs, inplace):
    """Reuse maximum-capacity state storage without changing adapter results."""
    from unittest.mock import patch

    import flashinfer

    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule
    from tensorrt_llm._utils import is_sm_100f

    q, k, v, g, beta, cu = _make_inputs([4] * num_seqs)
    initial = _zero_initial_state(num_seqs, v.shape[2], v.shape[3], q.device, q.dtype)
    indices = torch.arange(num_seqs, device=q.device, dtype=torch.int32) if inplace else None
    args = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu,
        initial_state_indices=indices,
        inplace_indexed_state_update=inplace,
        output_final_state=not inplace,
    )
    expected_pool = initial.clone()
    expected, expected_state = chunk_gated_delta_rule(initial_state=expected_pool, **args)
    state_in = torch.empty(
        (max(32, num_seqs), *initial.shape[1:]),
        device=q.device,
        dtype=q.dtype if is_sm_100f() else torch.float32,
    )
    workspace = (state_in, torch.empty_like(state_in))
    for buffer in workspace:
        buffer.fill_(float("nan"))
    with patch.object(
        flashinfer, "chunk_gated_delta_rule", wraps=flashinfer.chunk_gated_delta_rule
    ) as call:
        actual, actual_state = chunk_gated_delta_rule(
            initial_state=initial, state_workspace=workspace, **args
        )
    assert call.call_args.kwargs["initial_state"].data_ptr() == workspace[0].data_ptr()
    assert call.call_args.kwargs["output_state"].data_ptr() == workspace[1].data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(initial, expected_pool, rtol=0, atol=0)
    if not inplace:
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
        # Returned state must not alias scratch reused by another layer.
        assert all(
            actual_state.untyped_storage().data_ptr() != buffer.untyped_storage().data_ptr()
            for buffer in workspace
        )
    assert all(torch.isnan(buffer[num_seqs:]).all() for buffer in workspace)


@pytest.mark.parametrize("buffer_index", [0, 1])
@pytest.mark.parametrize("invalid", ["capacity", "shape", "dtype", "device"])
def test_state_workspace_rejects_invalid_buffer(monkeypatch, buffer_index, invalid):
    """Reject either invalid state buffer before launching a GPU kernel."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    import tensorrt_llm._torch.modules.fla.flashinfer_chunk as adapter

    kernel = Mock(side_effect=RuntimeError("unexpected kernel launch"))
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(chunk_gated_delta_rule=kernel))
    monkeypatch.setattr(adapter, "is_sm_100f", lambda: False)
    monkeypatch.setattr(adapter, "gather_cast_vk_to_fp32_vk", kernel)
    q, k, v, g, beta, cu = _make_inputs(
        [1, 1], num_q_heads=2, num_v_heads=2, head_dim=4, device="cpu"
    )
    state = torch.empty(2, 2, 4, 4, dtype=torch.bfloat16)
    shape = (
        (1, 2, 4, 4)
        if invalid == "capacity"
        else (2, 2, 4, 3)
        if invalid == "shape"
        else state.shape
    )
    buffers = [torch.empty(state.shape), torch.empty(state.shape)]
    buffers[buffer_index] = torch.empty(
        shape,
        dtype=torch.bfloat16 if invalid == "dtype" else torch.float32,
        device="meta" if invalid == "device" else "cpu",
    )
    with pytest.raises(AssertionError):
        adapter.chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu,
            initial_state=state,
            state_workspace=tuple(buffers),
        )
    kernel.assert_not_called()


@pytest.mark.parametrize("with_workspace", [False, True])
def test_state_workspace_dispatch(monkeypatch, with_workspace):
    """Do not pass the FlashInfer-only option to the Triton fallback."""
    from unittest.mock import Mock

    import tensorrt_llm._torch.modules.mamba.gdn_mixer as mixer

    impl = Mock()
    monkeypatch.setattr(mixer, "_resolve_chunk_gated_delta_rule", lambda: impl)
    workspace = (torch.empty(1), torch.empty(1)) if with_workspace else None
    mixer.chunk_gated_delta_rule(state_workspace=workspace)
    assert impl.call_args.kwargs == ({"state_workspace": workspace} if with_workspace else {})


@pytest.mark.parametrize("metadata_backend", ["base", "trtllm"])
@pytest.mark.parametrize("native_state", [False, True])
@pytest.mark.parametrize(
    "max_sequences,max_tokens,capacity", [(512, 2048, 512), (512, 16, 16), (None, 2048, 512)]
)
def test_state_workspace_warmup_and_reuse(
    monkeypatch, native_state, max_sequences, max_tokens, capacity, metadata_backend
):
    """Warmup reserves the maximum, shared by layers, not the current batch."""
    from types import SimpleNamespace

    import tensorrt_llm._torch.modules.mamba.gdn_mixer as mixer
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    monkeypatch.setattr(mixer, "_use_flashinfer_gdn_prefill", lambda: True)
    monkeypatch.setattr(mixer, "is_sm_100f", lambda: native_state)
    config = SimpleNamespace(extra_attrs={})
    layer = SimpleNamespace(model_config=config)
    # Keep the real metadata contract; bypass only TRTLLM's unrelated CUDA buffers.
    monkeypatch.setattr(
        TrtllmAttentionMetadata, "_post_init_with_buffers", lambda self, buffers: None
    )
    metadata_cls = AttentionMetadata if metadata_backend == "base" else TrtllmAttentionMetadata
    metadata = metadata_cls(
        max_num_sequences=max_sequences,
        max_num_requests=512,
        max_num_tokens=max_tokens,
    )
    assert not hasattr(metadata, "is_warmup")
    state = torch.empty(1, 2, 4, 4, dtype=torch.bfloat16)
    get_workspace = mixer.Qwen3NextGatedDeltaNet._get_prefill_state_workspace
    workspace = get_workspace(layer, metadata, state)
    if native_state:
        assert workspace is None
        assert config.extra_attrs == {}
    else:
        assert isinstance(workspace, tuple) and len(workspace) == 2
        assert all(buffer.shape == (capacity, 2, 4, 4) for buffer in workspace)
        assert all(buffer.dtype == torch.float32 for buffer in workspace)
        assert (
            workspace[0].untyped_storage().data_ptr() != workspace[1].untyped_storage().data_ptr()
        )
    other_layer = SimpleNamespace(model_config=config)
    with monkeypatch.context() as context:
        context.setattr(
            torch, "empty", lambda *args, **kwargs: pytest.fail("unexpected allocation")
        )
        context.setattr(
            torch, "empty_like", lambda *args, **kwargs: pytest.fail("unexpected allocation")
        )
        assert get_workspace(layer, metadata, state) is workspace
        assert get_workspace(other_layer, metadata, state) is workspace
    monkeypatch.setattr(mixer, "_use_flashinfer_gdn_prefill", lambda: False)
    assert get_workspace(layer, metadata, state) is None


def test_state_workspace_warmup_growth(monkeypatch):
    """Grow both cached buffers from 16 to 512 and reuse them during inference."""
    from types import SimpleNamespace
    from unittest.mock import patch

    import tensorrt_llm._torch.modules.mamba.gdn_mixer as mixer

    monkeypatch.setattr(mixer, "_use_flashinfer_gdn_prefill", lambda: True)
    monkeypatch.setattr(mixer, "is_sm_100f", lambda: False)
    layer = SimpleNamespace(model_config=SimpleNamespace(extra_attrs={}))
    metadata = SimpleNamespace(max_num_sequences=16, max_num_requests=512, max_num_tokens=2048)
    state = torch.empty(1, 2, 4, 4, dtype=torch.bfloat16)
    get_workspace = mixer.Qwen3NextGatedDeltaNet._get_prefill_state_workspace
    small = get_workspace(layer, metadata, state)
    assert all(buffer.shape[0] == 16 for buffer in small)
    metadata.max_num_sequences = 512
    large = get_workspace(layer, metadata, state)
    assert all(buffer.shape == (512, 2, 4, 4) for buffer in large)
    assert large[0].untyped_storage().data_ptr() != large[1].untyped_storage().data_ptr()
    assert all(before.data_ptr() != after.data_ptr() for before, after in zip(small, large))
    with (
        patch.object(torch, "empty", side_effect=RuntimeError("unexpected allocation")),
        patch.object(torch, "empty_like", side_effect=RuntimeError("unexpected allocation")),
    ):
        assert get_workspace(layer, metadata, state) is large


# Parity tests against the Triton reference -------------------------------


@skip_unsupported
@pytest.mark.skipif(is_sm_100f(), reason="Mixer-managed state workspace is only used on Hopper")
def test_state_workspace_maximum_batch_and_cuda_graph():
    """A one-sequence warmup must provision state scratch for a full mixed batch."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet

    layer = SimpleNamespace(model_config=SimpleNamespace(extra_attrs={}))
    metadata = SimpleNamespace(max_num_sequences=512, max_num_requests=512, max_num_tokens=2048)
    pool = torch.zeros(512, 16, 128, 128, dtype=torch.bfloat16, device="cuda")
    get_workspace = Qwen3NextGatedDeltaNet._get_prefill_state_workspace
    workspace = get_workspace(layer, metadata, pool)
    assert workspace is not None
    peaks = []
    for num_seqs in (1, 512):
        q, k, v, g, beta, cu = _make_inputs([2048 // num_seqs] * num_seqs)
        indices = torch.arange(num_seqs, dtype=torch.int32, device="cuda")
        args = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu,
            initial_state=pool,
            initial_state_indices=indices,
            inplace_indexed_state_update=True,
            state_workspace=workspace,
        )
        out, _ = chunk_gated_delta_rule(**args)
        torch.cuda.synchronize()
        del out
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        out, _ = chunk_gated_delta_rule(**args)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - before)
        assert get_workspace(layer, metadata, pool) is workspace
        del out
    # Two dynamically sized H100 state buffers would add ~1 GiB here.
    assert peaks[1] - peaks[0] < 128 * 2**20

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            graph_out, _ = chunk_gated_delta_rule(**args)
    torch.cuda.current_stream().wait_stream(stream)
    pool.zero_()
    expected, _ = chunk_gated_delta_rule(**args)
    expected = expected.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, _ = chunk_gated_delta_rule(**args)
    for _ in range(3):
        # Simulate another layer overwriting the shared scratch between replays.
        for buffer in workspace:
            buffer.fill_(float("nan"))
        pool.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_out, expected, rtol=0, atol=0)


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


def test_gdn_mixer_resolve_chunk_gated_delta_rule(monkeypatch):
    """gdn_mixer resolves its prefill kernel lazily (``_resolve_chunk_gated_delta_rule``):
    the FlashInfer wrapper when the env opt-in is set (default) *and* the arch is
    supported (SM90/SM100/SM103/SM120/SM121), otherwise the vendored Triton kernel
    (env opt-out, or an unsupported arch such as SM89).

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
        monkeypatch.setattr(
            gdn_mixer, "is_flashinfer_gdn_prefill_supported_arch", lambda: arch_supported
        )
        gdn_mixer._resolve_chunk_gated_delta_rule.cache_clear()
        return gdn_mixer._resolve_chunk_gated_delta_rule()

    # Default env + supported arch -> FlashInfer wrapper.
    assert resolve(None, True) is flashinfer_fn
    # Explicit opt-in + supported arch -> FlashInfer wrapper.
    assert resolve("1", True) is flashinfer_fn
    # Opt-out env -> Triton even on a supported arch.
    assert resolve("0", True) is triton_fn
    # Unsupported arch (e.g. SM89) -> Triton even with the default opt-in.
    assert resolve(None, False) is triton_fn

    # Clear the cached resolution so later tests re-resolve against the real
    # arch/env (monkeypatch restores the env var and predicate on teardown).
    gdn_mixer._resolve_chunk_gated_delta_rule.cache_clear()
