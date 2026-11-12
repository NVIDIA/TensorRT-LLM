# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the Kimi K3 tp1 KDA layer's prefill chain.

The layer runs on FP8 128x128 block-scale q/k/v/g/o weights (K3's checkpoint format), finalized
as the model loader does. The FLA reference (``KimiKDAReference``) shares the layer's FP8
projections, so a comparison measures the convolution, recurrence and gated RMSNorm.

tp1 (every recipe's context servers) runs the token-major fused convolution
(``fused_kda_conv_prefill`` on the QKVG GEMM output) and chunk-pads a single unaligned sequence
in that kernel (with beta), with every KDA head on its rank. The channel-major convolution
(``causal_conv1d_fwd`` + ``fused_kda_post_conv``) it replaced is the dep16 glue's; one test checks
the two against each other. The prefill kernels' own tests are in
``test_tp1_kda_prefill_kernels.py``.
"""

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_kda._kda_kernels import (  # noqa: E402, E501
    is_kda_optimized_supported,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__tp1.modeling import (  # noqa: E402, E501  # noqa: E402, E501
    K3KDALinearAttention,
    _Fp8BlockScaleWeightReadLinear,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm.sm107_linear import (  # noqa: E402, E501
    Bf16Linear,
    Fp8BlockScaleLinear,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.sm107_short_conv import (
    ShortConvWeight,
)
from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization import QuantAlgo  # noqa: E402
from tests.unittest._torch.modules.kimi_kda.kimi_kda_test_utils import KimiKDAReference

NUM_HEADS = 96


HEAD_DIM = 128


CONV_KERNEL_SIZE = 4


HIDDEN_SIZE = 7168


def _has_supported_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    return is_kda_optimized_supported()


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="FlashInfer KDA prefill requires SM100/SM103/SM107",
)


@torch.no_grad()
def _make_kda(
    *, num_heads: int = NUM_HEADS, seed: int = 0, layer_cls=K3KDALinearAttention
) -> K3KDALinearAttention:
    """A layer on seeded FP8 checkpoint-like weights, finalized as the model loader does: the
    fused QKVG and o_proj swapped to the tp1 K3 FP8 weight-read GEMM. ``layer_cls``: the tp1
    layer by default."""
    cfg = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        rms_norm_eps=1e-5,
        linear_attn_config={
            "num_heads": num_heads,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": CONV_KERNEL_SIZE,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    model_config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES))
    kda = layer_cls(cfg, layer_idx=0, model_config=model_config).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(shape, scale):
        return torch.randn(shape, generator=generator, device="cuda") * scale

    for module in kda.modules():
        if isinstance(module, Fp8BlockScaleLinear):
            codes = torch.randint(-32, 33, module.weight.shape, generator=generator, device="cuda")
            module.weight.copy_(codes.to(module.weight.dtype))
            module.weight_scale.fill_(1 / (32 * math.sqrt(module.in_features)))
        elif isinstance(module, Bf16Linear):
            module.weight.copy_(randn(module.weight.shape, (3 * module.in_features) ** -0.5))
        elif isinstance(module, ShortConvWeight):
            module.weight.copy_(randn(module.weight.shape, 0.3))
    kda.o_norm.weight.copy_(1 + randn(kda.o_norm.weight.shape, 0.1))
    kda.dt_bias.zero_()
    kda.finalize_decode_weights()
    kda._build_mtp_conv_weights()
    for name in ("qkvg_proj", "o_proj"):
        module = _Fp8BlockScaleWeightReadLinear.from_linear(getattr(kda, name))
        setattr(kda, name, module)
        module.post_load_weights()
    return kda


class _Columns(torch.nn.Module):
    """Output columns ``[start, end)`` of a projection."""

    def __init__(self, projection: torch.nn.Module, start: int, end: int) -> None:
        super().__init__()
        self.projection, self.start, self.end = projection, start, end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)[..., self.start : self.end]


@torch.no_grad()
def _make_reference(kda: K3KDALinearAttention) -> KimiKDAReference:
    """The FLA reference on ``kda``'s weights: its q / k / v / g projections are the columns of the
    layer's fused FP8 QKVG projection and its o_proj the layer's, the rest copies."""
    reference = KimiKDAReference(
        hidden_size=HIDDEN_SIZE,
        num_heads=kda.num_heads,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL_SIZE,
        use_full_rank_gate=True,
        gate_lower_bound=-5.0,
        rms_norm_eps=1e-5,
        dtype=torch.bfloat16,
    ).to("cuda")
    for name, parameter in reference.named_parameters():
        if not name.startswith(("q_proj.", "k_proj.", "v_proj.", "g_proj.", "o_proj.")):
            parameter.copy_(kda.get_parameter(name))
    d = kda.proj_size
    for index, name in enumerate(("q_proj", "k_proj", "v_proj", "g_proj")):
        setattr(reference, name, _Columns(kda.qkvg_proj, index * d, (index + 1) * d))
    reference.o_proj = kda.o_proj
    return reference


def _run_production_prefill(
    attention: K3KDALinearAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    conv_pool: torch.Tensor | None = None,
    state_pool: torch.Tensor | None = None,
    slot_indices: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
    kda_snapshot=None,
    inforward_stage=None,
    single_sequence_length: int | None = None,
    project_output: bool = True,
) -> torch.Tensor:
    """The prefill output ``[batch, tokens, hidden]``, or with ``project_output=False`` the
    normalized core ``[tokens, heads, head_dim]`` before o_proj."""
    batch_size, sequence_length, _ = hidden_states.shape
    if cu_seqlens is None:
        cu_seqlens = torch.arange(batch_size + 1, device="cuda", dtype=torch.long) * sequence_length
    else:
        batch_size = cu_seqlens.numel() - 1

    projection_size = attention.proj_size
    if conv_pool is None:
        conv_pool = torch.zeros(
            batch_size,
            3 * projection_size,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
    if state_pool is None:
        state_pool = torch.zeros(
            batch_size,
            attention.num_heads,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
    if slot_indices is None:
        slot_indices = torch.arange(batch_size, device="cuda", dtype=torch.long)
    use_initial_states = has_initial_states is not None
    if has_initial_states is None:
        has_initial_states = torch.zeros(batch_size, device="cuda", dtype=torch.bool)
    metadata = SimpleNamespace(
        use_initial_states=use_initial_states,
        has_initial_states=has_initial_states,
        state_indices=slot_indices.to(torch.int32),
        # causal_conv1d wants int32 offsets, unlike the int64 cu_seqlens above.
        query_start_loc=cu_seqlens.to(torch.int32),
        # Row indices of the dense fp32 copy a bf16 state pool is staged through.
        _arange_buffer=torch.arange(batch_size + 1, dtype=torch.int32, device="cuda"),
        kda_snapshot=kda_snapshot,
        kda_single_sequence_length=single_sequence_length,
    )
    core = attention.forward_prefill(
        hidden_states.reshape(-1, HIDDEN_SIZE),
        cu_seqlens.to(torch.int64),
        metadata,
        batch_size,
        conv_pool,
        state_pool,
        slot_indices,
        inforward_stage=inforward_stage,
    )
    if not project_output:
        return core
    output = attention._project_output(core)
    return output.reshape_as(hidden_states)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_float = actual.float()
    expected_float = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    ).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    assert cosine > 0.999
    assert relative_l2 < 3e-2


@torch.no_grad()
@pytest.mark.parametrize("num_tokens", [256, 301])
def test_project_packed_conv_input_layout(num_tokens: int) -> None:
    """The convolution input is the QKVG GEMM's [q | k | v] columns as a [T, 3D] view of its
    output (for fused_kda_conv_prefill); the output gate is the view of the remaining columns."""
    torch.manual_seed(3)
    # Only the projection layout is checked, so two heads keep the weights small.
    kda = _make_kda(num_heads=2)
    d = kda.proj_size
    x = torch.randn(1, num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05

    packed_conv, onorm_g = kda._project_packed_conv_input(x)
    qkvg = kda.qkvg_proj(x).squeeze(0)

    assert packed_conv.shape == (num_tokens, 3 * d) and packed_conv.stride(1) == 1
    assert torch.equal(packed_conv, qkvg[:, : 3 * d])
    assert torch.equal(onorm_g.squeeze(0), qkvg[:, 3 * d :])


@torch.no_grad()
def test_prefill_output_matches_across_projection_layouts() -> None:
    """Aligned and unaligned token counts produce the same prefill output.

    The odd-length request is the aligned one plus one trailing token, so its
    shared prefix must match the aligned request token for token.
    """
    torch.manual_seed(4)
    kda = _make_kda()
    hidden_states = torch.randn(1, 257, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05

    aligned = _run_production_prefill(kda, hidden_states[:, :256].contiguous())
    unaligned = _run_production_prefill(kda, hidden_states)

    torch.testing.assert_close(unaligned[:, :256], aligned, rtol=2e-2, atol=2e-2)


@torch.no_grad()
@pytest.mark.parametrize("num_heads", [NUM_HEADS, 12, 6], ids=["h96", "h12", "h6"])
def test_optimized_prefill_matches_fla_reference(num_heads: int) -> None:
    torch.manual_seed(0)
    optimized = _make_kda(num_heads=num_heads)
    reference = _make_reference(optimized)

    # Reuse the same module across changing token counts and batch sizes.
    for batch_size, sequence_length in [(2, 256), (2, 512), (1, 1024)]:
        hidden_states = (
            torch.randn(
                batch_size,
                sequence_length,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        actual = _run_production_prefill(optimized, hidden_states)
        expected = reference.forward_prefill(hidden_states)
        _assert_close(actual, expected)

    hidden_states = torch.randn(1, 300, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    actual = _run_production_prefill(optimized, hidden_states)
    expected = reference.forward_prefill(hidden_states)
    _assert_close(actual, expected)

    sequence_lengths = [128, 256, 192]
    cumulative_lengths = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )
    hidden_states = (
        torch.randn(
            1,
            sum(sequence_lengths),
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
    expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
    _assert_close(actual, expected)


@torch.no_grad()
def test_kda_mixer_empty_prefill():
    """An empty token payload still publishes the conv state for its slot."""
    optimized = _make_kda()
    hidden_states = torch.empty(1, 0, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    # Seed the pool with a sentinel: the output shape alone would still match if the
    # zero-token state write-back were skipped, so only the pool can catch that.
    conv_pool = torch.full(
        (1, 3 * NUM_HEADS * HEAD_DIM, CONV_KERNEL_SIZE - 1),
        7.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # The engine never runs a forward without tokens, so o_proj (whose GEMMs need a row) is not
    # part of this contract.
    core = _run_production_prefill(
        optimized, hidden_states, conv_pool=conv_pool, project_output=False
    )
    assert core.shape == (0, NUM_HEADS, HEAD_DIM)
    # Without an initial state a zero-token sequence leaves an all-zero causal window.
    assert torch.equal(conv_pool, torch.zeros_like(conv_pool))


@torch.no_grad()
def test_kda_mixer_empty_prefill_preserves_initial_conv_state():
    """A zero-token payload carrying an initial state leaves that state untouched."""
    optimized = _make_kda()
    hidden_states = torch.empty(1, 0, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    conv_seed = (
        torch.randn(
            1,
            3 * NUM_HEADS * HEAD_DIM,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    conv_pool = conv_seed.clone()
    core = _run_production_prefill(
        optimized,
        hidden_states,
        conv_pool=conv_pool,
        has_initial_states=torch.ones(1, dtype=torch.bool, device="cuda"),
        project_output=False,
    )
    assert core.shape == (0, NUM_HEADS, HEAD_DIM)
    # No token shifts into the window, so the write-back must republish it verbatim.
    assert torch.equal(conv_pool, conv_seed)


@torch.no_grad()
def test_kda_prefill_op_partial_final_chunk_large_batch():
    """Large packed batches with short final sequences match the FLA reference.

    Cover the autotuner warmup split [8191, 1] and a mixed-length 8192-token batch.
    """
    optimized = _make_kda()
    reference = _make_reference(optimized)
    for sequence_lengths in ([8191, 1], [8000, 150, 42]):
        cumulative_lengths = torch.tensor(
            [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
            dtype=torch.long,
            device="cuda",
        )
        hidden_states = (
            torch.randn(
                1,
                sum(sequence_lengths),
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
        expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
        _assert_close(actual, expected)


@torch.no_grad()
@pytest.mark.parametrize(
    "sequence_lengths",
    ([1], [30], [65], [6, 12], [64, 64], [1, 2, 3], [64, 64, 63], [6, 12, 20, 25]),
    ids=(
        "single-token",
        "short-single-seq",
        "partial-single-seq",
        "two-short-seqs",
        "two-aligned-seqs",
        "three-short-seqs",
        "three-seqs-partial-tail",
        "four-short-seqs",
    ),
)
def test_kda_prefill_small_varlen_matches_reference(sequence_lengths: list[int]) -> None:
    """Short packed prompts (shorter than the convolution window, partial chunks) on the prefill
    op match the reference."""
    optimized = _make_kda()
    reference = _make_reference(optimized)
    cumulative_lengths = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )
    hidden_states = (
        torch.randn(
            1,
            sum(sequence_lengths),
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )

    actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
    expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
    _assert_close(actual, expected)


@torch.no_grad()
def test_kda_prefill_continuation_matches_one_shot_prefill() -> None:
    """Chunked prefill: the first chunks fresh, the rest from the pool rows they left (mixed
    initial states in one batch, permuted slots), gives the outputs and final rows of one prefill
    of the whole sequences; an unused slot stays untouched."""
    torch.manual_seed(3)
    kda = _make_kda()
    lengths, first = [128, 192, 100, 77], [64, 64, 64, 13]
    slots = 5
    slot_indices = torch.tensor([3, 0, 4, 1], dtype=torch.int32, device="cuda")
    hidden = [
        torch.randn(1, n, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05 for n in lengths
    ]
    projection_size = NUM_HEADS * HEAD_DIM

    def pools():
        conv = torch.randn(
            slots, 3 * projection_size, CONV_KERNEL_SIZE - 1, dtype=torch.bfloat16, device="cuda"
        )
        state = torch.randn(
            slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda"
        )
        return conv, state

    def cumulative(seq_lengths):
        return torch.tensor(
            [0, *torch.tensor(seq_lengths).cumsum(0).tolist()], dtype=torch.long, device="cuda"
        )

    def prefill(chunks, conv, state, has_initial_states):
        seq_lengths = [chunk.shape[1] for chunk in chunks]
        output = _run_production_prefill(
            kda,
            torch.cat(chunks, dim=1),
            cumulative(seq_lengths),
            conv_pool=conv,
            state_pool=state,
            slot_indices=slot_indices,
            has_initial_states=torch.tensor(has_initial_states, device="cuda"),
        )
        return list(output.split(seq_lengths, dim=1))

    conv_seed, state_seed = pools()
    whole_conv, whole_state = conv_seed.clone(), state_seed.clone()
    whole = prefill(hidden, whole_conv, whole_state, [False] * 4)

    conv, state = conv_seed.clone(), state_seed.clone()
    head = prefill([h[:, :n] for h, n in zip(hidden, first)], conv, state, [False] * 4)
    tail = prefill([h[:, n:] for h, n in zip(hidden, first)], conv, state, [True] * 4)

    for i in range(len(lengths)):
        _assert_close(torch.cat([head[i], tail[i]], dim=1), whole[i])
    rows = slot_indices.long()
    # The two batches run the QKVG GEMM on other row counts, so the convolution rows (raw
    # projection outputs) may differ in their last bits.
    _assert_close(conv[rows], whole_conv[rows])
    _assert_close(state[rows], whole_state[rows])
    unused_slot = 2
    assert torch.equal(conv[unused_slot], conv_seed[unused_slot])
    assert torch.equal(state[unused_slot], state_seed[unused_slot])


@torch.no_grad()
def test_kda_prefill_rejects_unaligned_slot_indices() -> None:
    """The prefill op addresses the pool through 16-byte aligned indices (the metadata's state
    indices); an offset int32 view is an error, not a different kernel."""
    kda = _make_kda(num_heads=2)
    slot_storage = torch.tensor([-1, 3, 0], dtype=torch.int32, device="cuda")
    slot_indices = slot_storage[1:]
    assert slot_indices.data_ptr() % 16 != 0
    hidden_states = torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    with pytest.raises(RuntimeError, match="kda_prefill entry"):
        _run_production_prefill(
            kda,
            hidden_states,
            torch.tensor([0, 64, 128], dtype=torch.long, device="cuda"),
            conv_pool=torch.zeros(4, 3 * kda.proj_size, 3, dtype=torch.bfloat16, device="cuda"),
            state_pool=torch.zeros(4, 2, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda"),
            slot_indices=slot_indices,
        )


@torch.no_grad()
@pytest.mark.parametrize("sequence_lengths", [[64, 64, 64, 63], [300]], ids=["varlen", "single"])
def test_fused_conv_prefill_matches_channel_major_chain_with_initial_states(
    sequence_lengths: list[int],
) -> None:
    """tp1's prefill (``fused_kda_conv_prefill`` on the token-major QKVG output; a single unaligned
    sequence chunk-padded in that kernel with beta) carries continuation state like the
    channel-major chain of the dep16 glue (``causal_conv1d_fwd`` + ``fused_kda_post_conv``, the op
    padding a single sequence) on the same weights and projections: the same outputs and recurrent
    rows, the same convolution rows bit for bit, for mixed initial states, permuted slots and an
    unused slot."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__dep16.modeling import (  # noqa: E501
        K3KDALinearAttention as ChannelMajorKDA,
    )

    # Same seeded weights (the constructors draw A_log from the global generator), swapped to
    # the tp1 FP8 weight read: the QKVG GEMM gives the same convolution input to both chains.
    torch.manual_seed(5)
    kda = _make_kda(seed=5)
    torch.manual_seed(5)
    channel_major = _make_kda(seed=5, layer_cls=ChannelMajorKDA)
    for name, parameter in channel_major.named_parameters():
        assert torch.equal(parameter, kda.get_parameter(name)), name
    num_sequences = len(sequence_lengths)
    cumulative_lengths = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()], dtype=torch.long, device="cuda"
    )
    hidden_states = (
        torch.randn(1, sum(sequence_lengths), HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        * 0.05
    )
    slots = 5
    slot_indices = torch.tensor([3, 0, 4, 1][:num_sequences], dtype=torch.int32, device="cuda")
    has_initial_states = torch.tensor([True, False, True, False][:num_sequences], device="cuda")
    conv_seed = (
        torch.randn(
            slots, 3 * kda.proj_size, CONV_KERNEL_SIZE - 1, dtype=torch.bfloat16, device="cuda"
        )
        * 0.05
    )
    state_seed = (
        torch.randn(slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda") * 0.05
    )

    def run(layer):
        """Prefill from fresh copies of the seeded pools; return output and pools."""
        conv_pool, state_pool = conv_seed.clone(), state_seed.clone()
        output = _run_production_prefill(
            layer,
            hidden_states,
            cumulative_lengths,
            conv_pool=conv_pool,
            state_pool=state_pool,
            slot_indices=slot_indices,
            has_initial_states=has_initial_states,
            single_sequence_length=sequence_lengths[0] if num_sequences == 1 else None,
        )
        return output, conv_pool, state_pool

    expected = run(channel_major)
    actual = run(kda)
    rows = slot_indices.long()
    unused_slot = 2
    _assert_close(actual[0], expected[0])
    # Convolution states are copied verbatim out of the projection.
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    _assert_close(actual[2].index_select(0, rows), expected[2].index_select(0, rows))
    assert torch.equal(actual[1][unused_slot], conv_seed[unused_slot])
    assert torch.equal(actual[2][unused_slot], state_seed[unused_slot])


@torch.no_grad()
def test_kda_prefill_op_shape_growth_and_cu_dtype_transitions():
    """Changing token counts and caller offset dtypes preserve reference parity.

    The production adapter prepares int64 packed offsets for FlashInfer from
    either caller dtype. Repeated growth and shrinkage exercise cached kernels.
    """
    torch.manual_seed(0)
    optimized = _make_kda()
    reference = _make_reference(optimized)
    cases = [
        ([517, 654], torch.long),  # small batch, int64 cu (dump-replay-like)
        ([8191], torch.int32),  # growth + dtype conversion
        ([1171], torch.long),  # shrink + flip back
        ([150, 900, 333, 640], torch.int32),  # multi-seq int32
    ]
    for sequence_lengths, cu_dtype in cases:
        cumulative_lengths = torch.tensor(
            [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
            dtype=cu_dtype,
            device="cuda",
        )
        hidden_states = (
            torch.randn(
                1,
                sum(sequence_lengths),
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
        expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
        _assert_close(actual, expected)


@torch.no_grad()
def test_kda_prefill_context_batch_sizes_share_one_k4_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context batches of different sizes run one compiled K4.

    A bf16 state pool reaches K4 as a dense fp32 copy with one row per context
    sequence, so the pool's row count follows the batch size; an fp32 pool is
    addressed in place. Every batch on either pool matches the reference, and
    the two pools end with the same state rows.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm import (
        _kda_prefill as prefill_ops,
    )

    monkeypatch.setattr(prefill_ops, "_k4p_cache", {})
    torch.manual_seed(5)
    optimized = _make_kda()
    reference = _make_reference(optimized)
    slots = 6
    projection_size = NUM_HEADS * HEAD_DIM
    for sequence_lengths in ([200], [64, 130, 77], [96, 33], [150, 64, 40, 129, 10]):
        cumulative_lengths = torch.tensor(
            [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
            dtype=torch.long,
            device="cuda",
        )
        hidden_states = (
            torch.randn(1, sum(sequence_lengths), HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
            * 0.05
        )
        slot_indices = torch.arange(
            slots - len(sequence_lengths), slots, dtype=torch.int32, device="cuda"
        )
        expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
        final_rows = {}
        for pool_dtype in (torch.bfloat16, torch.float32):
            state_pool = torch.zeros(
                slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=pool_dtype, device="cuda"
            )
            conv_pool = torch.zeros(
                slots,
                3 * projection_size,
                CONV_KERNEL_SIZE - 1,
                dtype=torch.bfloat16,
                device="cuda",
            )
            actual = _run_production_prefill(
                optimized,
                hidden_states,
                cumulative_lengths,
                conv_pool=conv_pool,
                state_pool=state_pool,
                slot_indices=slot_indices,
            )
            _assert_close(actual, expected)
            final_rows[pool_dtype] = state_pool.index_select(0, slot_indices).float()
        _assert_close(final_rows[torch.bfloat16], final_rows[torch.float32])
        assert len(prefill_ops._k4p_cache) == 1


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / (expected.float().norm() + 1e-12)).item()


@pytest.mark.parametrize(
    "pool_dtype", [torch.float32, torch.bfloat16], ids=["fp32_pool", "bf16_pool"]
)
@torch.no_grad()
def test_kda_prefill_midchunk_snapshot_matches_prefix_prefill(pool_dtype: torch.dtype) -> None:
    """A mid-chunk snapshot holds the state and convolution window of its sequence's prefix.

    Sequences of 200, 130 and 96 tokens snapshot after 128 tokens (sequence 0) and 64 tokens
    (sequence 2) into their own slots; sequence 1 takes none. Each snapshot row must match a
    prefill of that prefix alone, and lie much closer to it than to the state up to one chunk
    later (sequence 2 ends 32 tokens after its snapshot). The snapshots leave the batch's outputs
    and final states as a run without them.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.common.kda_metadata import (  # noqa: E501
        build_kda_snapshot_plan,
    )

    torch.manual_seed(7)
    optimized = _make_kda()
    projection_size = NUM_HEADS * HEAD_DIM
    lengths = [200, 130, 96]
    starts = [0, 200, 330]
    snapshots = {0: (6, 128), 2: (7, 64)}  # sequence -> (slot, prefix tokens)
    slots = 8
    hidden_states = (
        torch.randn(1, sum(lengths), HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    )

    def pools():
        state = torch.zeros(slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=pool_dtype, device="cuda")
        conv = torch.zeros(
            slots, 3 * projection_size, CONV_KERNEL_SIZE - 1, dtype=torch.bfloat16, device="cuda"
        )
        return state, conv

    def cumulative(seq_lengths):
        return torch.tensor(
            [0, *torch.tensor(seq_lengths).cumsum(0).tolist()], dtype=torch.long, device="cuda"
        )

    slot_indices = torch.tensor([1, 3, 5], dtype=torch.int32, device="cuda")
    plan = build_kda_snapshot_plan(
        [(seq, slot, offset) for seq, (slot, offset) in snapshots.items()],
        lengths,
        torch.device("cuda"),
    )
    state, conv = pools()
    output = _run_production_prefill(
        optimized,
        hidden_states,
        cumulative(lengths),
        conv_pool=conv,
        state_pool=state,
        slot_indices=slot_indices,
        kda_snapshot=plan,
    )
    plain_state, plain_conv = pools()
    plain_output = _run_production_prefill(
        optimized,
        hidden_states,
        cumulative(lengths),
        conv_pool=plain_conv,
        state_pool=plain_state,
        slot_indices=slot_indices,
    )
    _assert_close(output, plain_output)
    _assert_close(state[slot_indices.long()], plain_state[slot_indices.long()])
    _assert_close(conv[slot_indices.long()], plain_conv[slot_indices.long()])

    def prefix_rows(extra_tokens: int):
        """Final state and convolution rows of each snapshotted sequence's first
        ``offset + extra_tokens`` tokens (at most the whole sequence)."""
        prefix_lengths = [
            min(offset + extra_tokens, lengths[seq]) for seq, (_, offset) in snapshots.items()
        ]
        prefix_hidden = torch.cat(
            [
                hidden_states[:, starts[seq] : starts[seq] + n]
                for seq, n in zip(snapshots, prefix_lengths)
            ],
            dim=1,
        )
        prefix_state, prefix_conv = pools()
        prefix_slots = torch.arange(len(snapshots), dtype=torch.int32, device="cuda")
        _run_production_prefill(
            optimized,
            prefix_hidden,
            cumulative(prefix_lengths),
            conv_pool=prefix_conv,
            state_pool=prefix_state,
            slot_indices=prefix_slots,
        )
        return prefix_state[: len(snapshots)], prefix_conv[: len(snapshots)]

    expected_state, expected_conv = prefix_rows(0)
    later_state, _ = prefix_rows(64)
    snapshot_slots = torch.tensor([slot for slot, _ in snapshots.values()], device="cuda")
    _assert_close(state[snapshot_slots], expected_state)
    _assert_close(conv[snapshot_slots], expected_conv)
    for j in range(len(snapshots)):
        assert _relative_l2(state[snapshot_slots[j]], expected_state[j]) < 0.1 * _relative_l2(
            later_state[j], expected_state[j]
        )
    untouched = [s for s in range(slots) if s not in {1, 3, 5, *snapshot_slots.tolist()}]
    assert torch.count_nonzero(state[untouched]) == 0


@pytest.mark.parametrize(
    "pool_dtype", [torch.float32, torch.bfloat16], ids=["fp32_pool", "bf16_pool"]
)
@torch.no_grad()
def test_kda_prefill_inforward_snapshot_swaps_live_and_staging_rows(
    pool_dtype: torch.dtype,
) -> None:
    """In-forward snapshots (``MambaHybridCacheManagerV2`` with block reuse): each planned
    sequence's live slot ends with the state and convolution window of its prefix up to the
    snapshot point, and its staging row with the chunk-end ones, which
    ``finish_inforward_snapshots`` copies back after committing the prefix blocks.

    Sequences of 200, 130 and 96 tokens (live slots 1, 3, 5) snapshot after 128 tokens
    (sequence 0, staging row 1) and 64 tokens (sequence 2, staging row 0). The outputs and the
    unplanned sequence's rows match a run without snapshots.
    """
    from dataclasses import replace

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.common.kda_metadata import (  # noqa: E501
        build_kda_snapshot_plan,
    )

    torch.manual_seed(11)
    optimized = _make_kda()
    projection_size = NUM_HEADS * HEAD_DIM
    lengths = [200, 130, 96]
    starts = [0, 200, 330]
    live = [1, 3, 5]
    planned = {0: (128, 1), 2: (64, 0)}  # sequence -> (prefix tokens, staging row)
    slots = 8
    hidden_states = (
        torch.randn(1, sum(lengths), HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    )

    def pools(rows=slots):
        state = torch.zeros(rows, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=pool_dtype, device="cuda")
        conv = torch.zeros(
            rows, 3 * projection_size, CONV_KERNEL_SIZE - 1, dtype=torch.bfloat16, device="cuda"
        )
        return state, conv

    def cumulative(seq_lengths):
        return torch.tensor(
            [0, *torch.tensor(seq_lengths).cumsum(0).tolist()], dtype=torch.long, device="cuda"
        )

    slot_indices = torch.tensor(live, dtype=torch.int32, device="cuda")
    plan = build_kda_snapshot_plan(
        [(seq, live[seq], offset) for seq, (offset, _) in planned.items()],
        lengths,
        torch.device("cuda"),
    )
    plan = replace(
        plan,
        stage_rows=torch.tensor([row for _, row in planned.values()], device="cuda"),
    )
    state, conv = pools()
    stage_state, stage_conv = pools(len(planned))
    output = _run_production_prefill(
        optimized,
        hidden_states,
        cumulative(lengths),
        conv_pool=conv,
        state_pool=state,
        slot_indices=slot_indices,
        kda_snapshot=plan,
        inforward_stage=(stage_state, stage_conv),
    )
    plain_state, plain_conv = pools()
    plain_output = _run_production_prefill(
        optimized,
        hidden_states,
        cumulative(lengths),
        conv_pool=plain_conv,
        state_pool=plain_state,
        slot_indices=slot_indices,
    )
    _assert_close(output, plain_output)
    _assert_close(state[3], plain_state[3])
    _assert_close(conv[3], plain_conv[3])

    prefix_lengths = [offset for offset, _ in planned.values()]
    prefix_hidden = torch.cat(
        [
            hidden_states[:, starts[seq] : starts[seq] + n]
            for seq, n in zip(planned, prefix_lengths)
        ],
        dim=1,
    )
    prefix_state, prefix_conv = pools()
    _run_production_prefill(
        optimized,
        prefix_hidden,
        cumulative(prefix_lengths),
        conv_pool=prefix_conv,
        state_pool=prefix_state,
        slot_indices=torch.arange(len(planned), dtype=torch.int32, device="cuda"),
    )
    for j, (seq, (_, row)) in enumerate(planned.items()):
        slot = live[seq]
        _assert_close(state[slot], prefix_state[j])
        _assert_close(conv[slot], prefix_conv[j])
        _assert_close(stage_state[row], plain_state[slot])
        _assert_close(stage_conv[row], plain_conv[slot])
        assert _relative_l2(state[slot], prefix_state[j]) < 0.1 * _relative_l2(
            plain_state[slot], prefix_state[j]
        )
    untouched = [s for s in range(slots) if s not in live]
    assert torch.count_nonzero(state[untouched]) == 0
    assert torch.count_nonzero(conv[untouched]) == 0
