# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production module/core behavior for FlashInfer prefill and BF16 state ownership."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fla")

from tests.unittest._torch.modules.kimi_kda.kda_module_test_utils import (
    compare,
    make_attention,
    make_batch,
    use_fla_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="FlashInfer KDA requires SM100/SM103/SM107",
)


def assert_parity(actual: torch.Tensor, expected: torch.Tensor) -> None:
    metrics = compare(actual, expected)
    assert metrics["passed"], metrics


@pytest.mark.parametrize("heads", [6, 12, 24, 48, 96])
@pytest.mark.parametrize("lengths", [[0, 15, 16, 17], [63, 64, 65], [150]])
@pytest.mark.parametrize("distribution", ["normal", "uniform", "sparse"])
@torch.inference_mode()
def test_indexed_prefill_matches_reference(
    heads: int, lengths: list[int], distribution: str
) -> None:
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_flashinfer_ops import (
        kda_prefill_flashinfer as kda_prefill,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.kimi_k3_kda.kda_chunked_bt16 import (
        kda_chunked_mma_reference,
    )

    torch.manual_seed(1901)
    total, n = sum(lengths), len(lengths)

    def sample(shape: tuple[int, ...]) -> torch.Tensor:
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        if distribution == "uniform":
            x.uniform_(-1, 1)
        elif distribution == "sparse":
            x *= torch.rand(shape, device="cuda") < 0.1
        return x * 0.2

    q, k, v, g = [sample((1, total, heads, 128)) for _ in range(4)]
    projection_width = ((128 + heads + 7) // 8) * 8
    beta = sample((1, total, projection_width))[..., 128 : 128 + heads]
    a_log = torch.linspace(-2, 1, heads, device="cuda")
    bias = torch.linspace(-1, 1, heads * 128, device="cuda").reshape(heads, 128)
    slots_backing = torch.arange(n + 1, 0, -1, device="cuda", dtype=torch.int32)
    slots = slots_backing[1:]  # Contiguous, with only natural int32 alignment.
    offsets = torch.tensor([0] + lengths, device="cuda", dtype=torch.int64).cumsum(0)
    row_size = heads * 128 * 128
    backing = sample((n + 2, row_size + 128))
    state = backing[:, :row_size].view(n + 2, heads, 128, 128)
    state[slots[::2].long()] = 0  # Mixed fresh and continuation rows.
    before = backing.clone()
    immutable = [q, k, v, g, beta, a_log, bias, slots, offsets]
    snapshots = [x.clone() for x in immutable]
    expected_out, expected_state = kda_chunked_mma_reference(
        *(x.transpose(1, 2) for x in (q, k, v, g, beta)),
        initial_state=state[slots.long()].transpose(-1, -2),
        a_log=a_log,
        dt_bias=bias,
        cu_seqlens=offsets,
    )
    for mode in ("engine", "decomp", "bt16", "small-bh", "auto"):
        backing.copy_(before)
        output = kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            state,
            slots,
            offsets,
            a_log,
            bias,
            128**-0.5,
            -5.0,
            mode,
            max(lengths),
        )
        torch.testing.assert_close(output, expected_out.transpose(1, 2), rtol=0.01, atol=0.01)
        torch.testing.assert_close(
            state[slots.long()],
            expected_state.transpose(-1, -2).to(torch.bfloat16),
            rtol=0.01,
            atol=0.01,
        )
        torch.testing.assert_close(backing[[0, n + 1]], before[[0, n + 1]], rtol=0, atol=0)
        torch.testing.assert_close(backing[:, row_size:], before[:, row_size:], rtol=0, atol=0)
        for actual, expected in zip(immutable, snapshots):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["engine", "decomp", "bt16", "small-bh", "auto"])
@torch.inference_mode()
def test_indexed_prefill_addresses_rows_past_int32(mode: str) -> None:
    """Rows whose element offset slot * row_stride exceeds int32.

    KV cache manager V2 coalesces every KDA layer's state of a slot, so the per-layer
    pool view has a row stride of layers * H * 128 * 128 elements (Kimi K3: 108,527,616)
    and slot 20 already starts past 2**31. Slot 16 here starts exactly at 2**31. Every
    element outside the batch's rows is NaN: a wrong-row read poisons the output, a
    wrong-row write leaves numbers behind.
    """
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_flashinfer_ops import (
        kda_prefill_flashinfer as kda_prefill,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.kimi_k3_kda.kda_chunked_bt16 import (
        kda_chunked_mma_reference,
    )

    torch.manual_seed(1901)
    heads, lengths = 6, [40, 17]
    row_size, row_stride = heads * 128 * 128, 2**27
    slots = torch.tensor([16, 3], device="cuda", dtype=torch.int32)
    total, n = sum(lengths), len(lengths)
    q, k, v, g = [
        torch.randn(1, total, heads, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        for _ in range(4)
    ]
    beta = torch.randn(1, total, heads, device="cuda", dtype=torch.bfloat16) * 0.2
    a_log = torch.linspace(-2, 1, heads, device="cuda")
    bias = torch.linspace(-1, 1, heads * 128, device="cuda").reshape(heads, 128)
    offsets = torch.tensor([0] + lengths, device="cuda", dtype=torch.int64).cumsum(0)
    backing = torch.full(
        (16 * row_stride + row_size,), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    state = backing.as_strided((17, heads, 128, 128), (row_stride, 128 * 128, 128, 1))
    initial = torch.randn(n, heads, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    state[slots.long()] = initial
    expected_out, expected_state = kda_chunked_mma_reference(
        *(x.transpose(1, 2) for x in (q, k, v, g, beta)),
        initial_state=initial.transpose(-1, -2),
        a_log=a_log,
        dt_bias=bias,
        cu_seqlens=offsets,
    )
    output = kda_prefill(
        q, k, v, g, beta, state, slots, offsets, a_log, bias, 128**-0.5, -5.0, mode, max(lengths)
    )
    torch.testing.assert_close(output, expected_out.transpose(1, 2), rtol=0.01, atol=0.01)
    torch.testing.assert_close(
        state[slots.long()],
        expected_state.transpose(-1, -2).to(torch.bfloat16),
        rtol=0.01,
        atol=0.01,
    )
    state[slots.long()] = float("nan")
    assert bool(torch.isnan(backing).all()), "prefill wrote outside the batch's state rows"


@pytest.mark.parametrize("mode", ["engine", "decomp", "bt16", "small-bh", "auto"])
@torch.inference_mode()
def test_prefill_repeated_token_stays_bounded(mode: str) -> None:
    """One token repeated through a long prefill: identical keys, high beta, weak decay.

    The chunk inverse (I + beta * tril(K K^T))^-1 stays bounded, but its Neumann powers
    L^2..L^8 grow combinatorially; a 16-bit Neumann chain loses their alternating-sign
    cancellation, the chunk's state map gains eigenvalues far above one and the state
    overflows within a few dozen chunks (the autotuner's warmup prefill repeats one
    token). Every route must stay finite and match the reference.
    """
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_flashinfer_ops import (
        kda_prefill_flashinfer as kda_prefill,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.kimi_k3_kda.kda_chunked_bt16 import (
        kda_chunked_mma_reference,
    )

    torch.manual_seed(1907)
    heads, length = 6, 1024
    token = [torch.randn(1, 1, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    q, k, v = (x.expand(1, length, heads, 128).contiguous() for x in token)
    # Decay of -5 * sigmoid(g + dt_bias) per token: ~2e-3 on most channels, ~0.09 on every fourth.
    g = torch.full((1, length, heads, 128), -4.0, device="cuda", dtype=torch.bfloat16)
    g[..., ::4] = 0.0
    beta = torch.full((1, length, heads), 6.0, device="cuda", dtype=torch.bfloat16)
    a_log = torch.zeros(heads, device="cuda")
    bias = torch.full((heads, 128), -4.0, device="cuda")
    offsets = torch.tensor([0, length], device="cuda", dtype=torch.int64)
    state = torch.zeros(2, heads, 128, 128, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([1], device="cuda", dtype=torch.int32)
    expected_out, expected_state = kda_chunked_mma_reference(
        *(x.transpose(1, 2) for x in (q, k, v, g, beta)),
        initial_state=state[slots.long()].transpose(-1, -2),
        a_log=a_log,
        dt_bias=bias,
        cu_seqlens=offsets,
    )
    output = kda_prefill(
        q, k, v, g, beta, state, slots, offsets, a_log, bias, 128**-0.5, -5.0, mode, length
    )
    assert_parity(output, expected_out.transpose(1, 2))
    assert_parity(state[slots.long()], expected_state.transpose(-1, -2).to(torch.bfloat16))


@pytest.mark.parametrize(
    "mode,state_dtype",
    [(mode, torch.bfloat16) for mode in ("engine", "decomp", "auto", "bt16", "small-bh")]
    + [("auto", torch.float32)],
)
@pytest.mark.parametrize("heads", [6, 96])
@pytest.mark.parametrize("beta_layout", ["projection", "head_stride", "unaligned", "token_pitch"])
@torch.inference_mode()
def test_prefill_beta_views_match_compact_eager_and_graph(
    mode: str, state_dtype: torch.dtype, heads: int, beta_layout: str
) -> None:
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_flashinfer_ops import (
        kda_prefill_flashinfer as kda_prefill,
    )

    torch.manual_seed(1901)
    total = 33
    q, k, v, g = [
        torch.randn(1, total, heads, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        for _ in range(4)
    ]
    if beta_layout == "projection":
        width = ((128 + heads + 7) // 8) * 8
        beta = torch.randn(1, total, width, device="cuda", dtype=torch.bfloat16)[
            ..., 128 : 128 + heads
        ]
    elif beta_layout == "head_stride":
        beta = torch.randn(1, total, 2 * heads, device="cuda", dtype=torch.bfloat16)[..., ::2]
    elif beta_layout == "unaligned":
        beta = torch.randn(total * heads + 1, device="cuda", dtype=torch.bfloat16)[1:].view(
            1, total, heads
        )
    else:
        beta = torch.randn(1, total, heads + 1, device="cuda", dtype=torch.bfloat16)[..., :heads]
    compact_beta = beta.clone(memory_format=torch.contiguous_format)
    slots = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 16, total], device="cuda", dtype=torch.int64)
    a_log = torch.linspace(-2, 1, heads, device="cuda")
    bias = torch.linspace(-1, 1, heads * 128, device="cuda").reshape(heads, 128)
    row_size = heads * 128 * 128
    initial = torch.randn(4, row_size + 128, device="cuda", dtype=state_dtype) * 0.1
    initial[2, :row_size] = 0
    backing = initial.clone()
    state = backing[:, :row_size].view(4, heads, 128, 128)
    immutable = [q, k, v, g, beta, slots, a_log, bias]
    snapshots = [tensor.clone() for tensor in immutable]

    def run(beta_input: torch.Tensor) -> torch.Tensor:
        return kda_prefill(
            q, k, v, g, beta_input, state, slots, offsets, a_log, bias, 128**-0.5, -5.0, mode
        )

    run(beta)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run(beta)
    for split in (16, 15):
        offsets[1] = split
        backing.copy_(initial)
        expected = run(compact_beta).clone()
        expected_state = backing.clone()
        backing.copy_(initial)
        actual = run(beta)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(backing, expected_state, rtol=0, atol=0)
        backing.copy_(initial)
        graph.replay()
        torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(backing, expected_state, rtol=0, atol=0)
        torch.testing.assert_close(backing[[0, 3]], initial[[0, 3]], rtol=0, atol=0)
        torch.testing.assert_close(backing[:, row_size:], initial[:, row_size:], rtol=0, atol=0)
    for tensor, snapshot in zip(immutable, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("conv_mode", ["separate", "fused_all"])
@torch.inference_mode()
def test_module_continuation_decode_and_verify(conv_mode: str) -> None:
    heads, hidden = 6, 768
    baseline = use_fla_reference(make_attention(heads, hidden_size=hidden))
    candidate = make_attention(heads, hidden_size=hidden)
    candidate._kda_fi_fuse_conv = conv_mode == "fused_all"
    batch = make_batch(heads, [17, 65], hidden_size=hidden, initial="mixed")
    states = [batch.state.clone(), batch.state.clone()]
    convs = [batch.conv.clone(), batch.conv.clone()]
    for _ in range(3):
        outputs = [
            module.forward_prefill(
                batch.hidden, batch.cu, batch.metadata, batch.n, conv, state, batch.indices
            )
            for module, conv, state in zip((baseline, candidate), convs, states)
        ]
        assert_parity(outputs[1], outputs[0])
        assert_parity(states[1], states[0])
        torch.testing.assert_close(convs[1], convs[0], rtol=0, atol=0)
        batch.metadata.has_initial_states.fill_(True)
        batch.metadata.use_initial_states = True
    for _ in range(3):
        x = torch.randn(batch.n, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
        outputs = [
            module.forward_decode(
                x,
                conv,
                state,
                batch.indices,
                batch.metadata,
                SimpleNamespace(kda_k_cache=None),
                ssm_state_indices=batch.indices,
            )
            for module, conv, state in zip((baseline, candidate), convs, states)
        ]
        assert_parity(outputs[1], outputs[0])
        assert_parity(states[1], states[0])
    steps = 3
    x = torch.randn(batch.n * steps, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    outputs, caches = [], []
    for module, conv, state in zip((baseline, candidate), convs, states):
        cache = SimpleNamespace(
            kda_k_cache=None,
            intermediate_conv_window=torch.empty(
                batch.n, steps, 3 * heads * 128, 3, device="cuda", dtype=torch.bfloat16
            ),
            intermediate_ssm=torch.empty(
                batch.n, steps, heads, 128, 128, device="cuda", dtype=torch.bfloat16
            ),
        )
        outputs.append(module.forward_verify(x, steps, cache, conv, state, batch.indices))
        caches.append(cache)
    assert_parity(outputs[1], outputs[0])
    assert_parity(caches[1].intermediate_ssm, caches[0].intermediate_ssm)


@pytest.mark.parametrize("mode", ["engine", "decomp", "auto", "small-bh"])
@pytest.mark.parametrize("conv_mode", ["separate", "fused_all"])
@torch.inference_mode()
def test_graph_replay_refreshes_packed_lengths(mode: str, conv_mode: str) -> None:
    candidate = make_attention(6, hidden_size=768)
    candidate._kda_fi_mode = mode
    candidate._kda_fi_fuse_conv = conv_mode == "fused_all"
    batch = make_batch(6, [17, 65], hidden_size=768, initial="mixed")
    batch.metadata.kda_max_seqlen = sum([17, 65])
    state, conv = batch.state.clone(), batch.conv.clone()

    def run() -> torch.Tensor:
        return candidate.forward_prefill(
            batch.hidden, batch.cu, batch.metadata, 2, conv, state, batch.indices
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for lengths in ([17, 65], [33, 49], [0, 82], [64, 18]):
        batch.cu.copy_(torch.tensor([0, lengths[0], 82], device="cuda", dtype=torch.int64))
        batch.metadata.query_start_loc.copy_(batch.cu)
        state.copy_(batch.state)
        conv.copy_(batch.conv)
        eager = run().clone()
        expected_state, expected_conv = state.clone(), conv.clone()
        state.copy_(batch.state)
        conv.copy_(batch.conv)
        graph.replay()
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_production_forward_updates_live_cache(state_dtype: torch.dtype) -> None:
    """The model entry point updates recurrent, convolution and replay caches."""
    heads, hidden = 6, 768
    attention = make_attention(heads, hidden_size=hidden)
    reference = use_fla_reference(make_attention(heads, hidden_size=hidden))
    batch = make_batch(heads, [17, 65], hidden_size=hidden, initial="mixed", padded=True)
    # Packed int32 metadata may be a contiguous slice with only natural
    # 4-byte alignment; FlashInfer does not require a 16-byte slot pointer.
    slots = torch.empty(batch.n + 1, dtype=torch.int32, device="cuda")
    slots[1:].copy_(batch.indices)
    batch.indices = slots[1:]
    batch.metadata.state_indices = batch.indices
    state = batch.state.to(state_dtype)
    ref_state, ref_conv = state.clone(), batch.conv.clone()
    cache = SimpleNamespace(conv=batch.conv.clone(), temporal=state)
    cache.kda_k_cache = torch.empty(1, device="cuda")
    for label in ("q", "k", "v"):
        setattr(
            cache,
            f"kda_conv_{label}",
            torch.randn(
                state.shape[0],
                heads * 128,
                7,
                device="cuda",
                dtype=torch.float32,
            ),
        )
    old_replay = [getattr(cache, f"kda_conv_{label}").clone() for label in ("q", "k", "v")]
    metadata = SimpleNamespace(
        mamba_metadata=batch.metadata,
        num_contexts=batch.n,
        num_ctx_tokens=batch.hidden.shape[0],
        seq_lens=torch.tensor([17, 65]),
        kv_cache_manager=SimpleNamespace(mamba_layer_cache=lambda _: cache),
    )
    original = batch.hidden.clone()
    actual = attention(batch.hidden, metadata)
    expected = reference.forward_prefill(
        batch.hidden,
        batch.cu,
        batch.metadata,
        batch.n,
        ref_conv,
        ref_state,
        batch.indices,
    )
    assert_parity(actual, expected)
    assert_parity(state[batch.indices.long()], ref_state[batch.indices.long()])
    torch.testing.assert_close(cache.conv, ref_conv, rtol=0, atol=0)
    torch.testing.assert_close(batch.hidden, original, rtol=0, atol=0)
    untouched = [0, batch.n + 1]
    torch.testing.assert_close(
        state[untouched], batch.state[untouched].to(state_dtype), rtol=0, atol=0
    )
    for i, label in enumerate(("q", "k", "v")):
        replay = getattr(cache, f"kda_conv_{label}")
        selected = cache.conv[batch.indices.long(), i * heads * 128 : (i + 1) * heads * 128]
        torch.testing.assert_close(
            replay[batch.indices.long(), :, :3], selected.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(replay[untouched], old_replay[i][untouched], rtol=0, atol=0)
        torch.testing.assert_close(replay[..., 3:], old_replay[i][..., 3:], rtol=0, atol=0)
