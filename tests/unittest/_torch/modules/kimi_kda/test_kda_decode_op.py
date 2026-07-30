# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the optimized Kimi K3 KDA decode op."""

import copy

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda import _kda_decode  # noqa: E402
from tensorrt_llm._torch.modules.kimi_kda._kda_decode import (  # noqa: E402
    run_kda_decode_fusion_cuda,
)
from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import (  # noqa: E402
    KimiKDACachedState,
    KimiKDALinearAttention,
)

# 73: deliberately odd and > 64 to cover non-power-of-two batched decode.
BATCH_SIZE = 73
NUM_HEADS = 96
HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
HIDDEN_SIZE = 7168
SUPPORTED_HEADS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96)
COMPACT_WORK_THRESHOLD = 144


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {(10, 0), (10, 3)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 is supported only on Blackwell (SM100/SM103)",
)


def _make_attention_pair() -> tuple[KimiKDALinearAttention, KimiKDALinearAttention]:
    common = {
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
        "rms_norm_eps": 1e-5,
        "dtype": torch.bfloat16,
    }
    optimized = KimiKDALinearAttention(**common).to("cuda")
    # dt_bias is declared but not initialized (it expects a checkpoint), so it
    # holds whatever the caching allocator last left there. Values large enough
    # to overflow the decay turn the output into NaN, and the comparisons below
    # then fail on NaN != NaN — intermittently, depending on what ran before.
    torch.nn.init.normal_(optimized.dt_bias, std=0.1)
    reference = KimiKDALinearAttention(**common, use_optimized_decode=False).to("cuda")
    reference.load_state_dict(optimized.state_dict())

    assert optimized.decode_kernel_path == "optimized"
    assert reference.decode_kernel_path == "fla"
    assert reference.prefill_kernel_path == optimized.prefill_kernel_path
    return optimized, reference


def _make_cache(batch_size: int = BATCH_SIZE) -> KimiKDACachedState:
    projection_size = NUM_HEADS * HEAD_DIM
    return KimiKDACachedState(
        conv_state_q=(
            torch.randn(
                batch_size,
                projection_size,
                CONV_KERNEL_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        ),
        conv_state_k=(
            torch.randn(
                batch_size,
                projection_size,
                CONV_KERNEL_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        ),
        conv_state_v=(
            torch.randn(
                batch_size,
                projection_size,
                CONV_KERNEL_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        ),
        recurrent_state=(
            torch.randn(
                batch_size,
                NUM_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                dtype=torch.float32,
                device="cuda",
            )
            * 0.05
        ),
    )


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
@pytest.mark.parametrize("batch_size", [1, BATCH_SIZE])
def test_optimized_decode_matches_fla_reference(batch_size: int) -> None:
    torch.manual_seed(0)
    optimized, reference = _make_attention_pair()
    hidden_states = (
        torch.randn(
            batch_size,
            1,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    initial_cache = _make_cache(batch_size)

    actual_output, actual_cache = optimized.forward_decode(
        hidden_states, copy.deepcopy(initial_cache)
    )
    expected_output, expected_cache = reference.forward_decode(
        hidden_states, copy.deepcopy(initial_cache)
    )

    _assert_close(actual_output, expected_output)
    _assert_close(actual_cache.recurrent_state, expected_cache.recurrent_state)
    _assert_close(actual_cache.conv_state_q, expected_cache.conv_state_q)
    _assert_close(actual_cache.conv_state_k, expected_cache.conv_state_k)
    _assert_close(actual_cache.conv_state_v, expected_cache.conv_state_v)
    assert optimized.decode_kernel_source()


@torch.no_grad()
@pytest.mark.parametrize(
    ("batch_size", "slot_gap"),
    [(3, 0), (3, 4096), (1, 0)],
    ids=["many-dense-pool", "many-strided-pool", "compact-dense-pool"],
)
def test_optimized_decode_updates_indexed_recurrent_state_pool_in_place(
    batch_size: int,
    slot_gap: int,
) -> None:
    torch.manual_seed(1)
    optimized, _ = _make_attention_pair()
    hidden_states = (
        torch.randn(
            batch_size,
            1,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    initial_cache = _make_cache(batch_size)

    local_output, local_cache = optimized.forward_decode(
        hidden_states, copy.deepcopy(initial_cache)
    )

    slots = batch_size + 3
    slot_indices = torch.arange(
        slots - 1,
        slots - batch_size - 1,
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    dense_slot_stride = NUM_HEADS * HEAD_DIM * HEAD_DIM
    state_storage = torch.randn(
        slots * (dense_slot_stride + slot_gap),
        dtype=torch.float32,
        device="cuda",
    )
    state_pool = state_storage.as_strided(
        (slots, NUM_HEADS, HEAD_DIM, HEAD_DIM),
        (dense_slot_stride + slot_gap, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    state_pool.mul_(0.2)
    assert state_pool.is_contiguous() is (slot_gap == 0)
    state_pool.index_copy_(0, slot_indices.long(), initial_cache.recurrent_state)
    unselected_indices = torch.arange(3, device="cuda")
    unselected_before = state_pool.index_select(0, unselected_indices).clone()
    indexed_cache = KimiKDACachedState(
        conv_state_q=initial_cache.conv_state_q.clone(),
        conv_state_k=initial_cache.conv_state_k.clone(),
        conv_state_v=initial_cache.conv_state_v.clone(),
        recurrent_state=state_pool,
    )

    indexed_output, indexed_cache = optimized.forward_decode(
        hidden_states,
        indexed_cache,
        ssm_state_indices=slot_indices,
    )

    assert indexed_cache.recurrent_state is state_pool
    torch.testing.assert_close(indexed_output, local_output, rtol=0, atol=0)
    torch.testing.assert_close(
        state_pool.index_select(0, slot_indices.long()),
        local_cache.recurrent_state,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        state_pool.index_select(0, unselected_indices),
        unselected_before,
        rtol=0,
        atol=0,
    )


def _make_direct_decode_args(
    batch_size: int,
    num_heads: int,
    *,
    indexed_state: bool,
) -> dict:
    device = torch.device("cuda")
    projection_size = num_heads * HEAD_DIM
    slots = batch_size + 1 if indexed_state else batch_size
    state = torch.zeros(
        slots,
        num_heads,
        HEAD_DIM,
        HEAD_DIM,
        dtype=torch.float32,
        device=device,
    )
    indices = (
        torch.arange(1, batch_size + 1, dtype=torch.int32, device=device) if indexed_state else None
    )
    return {
        "x_q": torch.randn(1, batch_size, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
        * 0.01,
        "x_k": torch.randn(1, batch_size, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
        * 0.01,
        "x_v": torch.randn(1, batch_size, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
        * 0.01,
        "w_q_t": torch.randn(
            CONV_KERNEL_SIZE,
            projection_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.01,
        "w_k_t": torch.randn(
            CONV_KERNEL_SIZE,
            projection_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.01,
        "w_v_t": torch.randn(
            CONV_KERNEL_SIZE,
            projection_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.01,
        "bias_q": None,
        "bias_k": None,
        "bias_v": None,
        "cs_q": torch.zeros(
            batch_size,
            projection_size,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device=device,
        ),
        "cs_k": torch.zeros(
            batch_size,
            projection_size,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device=device,
        ),
        "cs_v": torch.zeros(
            batch_size,
            projection_size,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device=device,
        ),
        "A_log": torch.zeros(projection_size, dtype=torch.float32, device=device),
        "g": torch.zeros(1, batch_size, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device),
        "dt_bias": torch.zeros(projection_size, dtype=torch.float32, device=device),
        "beta": torch.zeros(1, batch_size, num_heads, dtype=torch.bfloat16, device=device),
        "state": state,
        "onorm_g": torch.zeros(
            1, batch_size, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        ),
        "onorm_weight": torch.ones(HEAD_DIM, dtype=torch.float32, device=device),
        "out": torch.empty(
            batch_size,
            1,
            num_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        ),
        "ssm_state_indices": indices,
        "cu_seqlens": torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        "lower_bound": -5.0,
    }


def _profile_decode_backend(kwargs: dict) -> str:
    _kda_decode.run_kda_decode_fusion_cuda(**kwargs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _kda_decode.run_kda_decode_fusion_cuda(**kwargs)
        torch.cuda.synchronize()

    kernel_names = [
        event.key
        for event in prof.key_averages()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    has_compact = any("kda_decode_fusion_compact_heads_kernel" in name for name in kernel_names)
    has_many = any("kda_decode_fusion_many_heads_kernel" in name for name in kernel_names)
    assert has_compact != has_many, kernel_names
    return "compact" if has_compact else "many"


@torch.no_grad()
@pytest.mark.parametrize("num_heads", SUPPORTED_HEADS)
def test_sm103_selector_dispatches_each_supported_head_at_boundary(num_heads: int) -> None:
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("compact-head selector sweep is tuned only for SM103")

    compact_batch = COMPACT_WORK_THRESHOLD // num_heads
    compact_args = _make_direct_decode_args(
        compact_batch,
        num_heads,
        indexed_state=False,
    )
    many_args = _make_direct_decode_args(
        compact_batch + 1,
        num_heads,
        indexed_state=False,
    )
    assert _profile_decode_backend(compact_args) == "compact"
    assert _profile_decode_backend(many_args) == "many"


@torch.no_grad()
def test_selector_preserves_legacy_compact_heads_off_sm103() -> None:
    if torch.cuda.get_device_capability(0) == (10, 3):
        pytest.skip("non-SM103 fallback requires a different Blackwell target")
    # Off SM103 the H==2 legacy rule dispatches the compact kernel; the
    # SM103-only selector must not change that.
    args = _make_direct_decode_args(1, 2, indexed_state=False)
    assert _profile_decode_backend(args) == "compact"


@torch.no_grad()
@pytest.mark.parametrize(
    ("batch_size", "indexed_state", "expected_backend"),
    [(1, False, "compact"), (2, True, "many")],
)
def test_sm103_selector_is_cuda_graph_safe(
    batch_size: int,
    indexed_state: bool,
    expected_backend: str,
) -> None:
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("compact-head selector sweep is tuned only for SM103")

    args = _make_direct_decode_args(
        batch_size,
        96,
        indexed_state=indexed_state,
    )
    assert _profile_decode_backend(args) == expected_backend

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured_output = _kda_decode.run_kda_decode_fusion_cuda(**args)
    graph.replay()
    torch.cuda.synchronize()
    assert captured_output is args["out"]
    assert torch.isfinite(captured_output).all()


# The compact-heads and many-heads kernels each carry their own copy of the
# conv and per-token addressing, so both need covering. H == 2 picks compact
# and H == 32 picks many under either selector: off SM103 on the legacy H == 2
# rule, on SM103 because shouldUseCompactHeads() only takes B * H work up to
# COMPACT_WORK_THRESHOLD -- which is why every test below decodes B >= 5.
_KERNEL_HEADS = pytest.mark.parametrize("heads", [2, 32], ids=["compact-heads", "many-heads"])


class _DecodeInputs:
    """One full set of ``kda_decode`` arguments, in the kernel's packed layout.

    ``x_q``/``x_k``/``x_v``/``g``/``onorm_g`` are ``[1, B, heads, 128]`` and
    ``beta`` is ``[1, B, heads]``, i.e. what a caller that materialized every
    per-token input separately would hand over.
    """

    def __init__(self, batch: int, heads: int, slots: int, seed: int) -> None:
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def bf16(*shape, scale=0.05):
            return torch.randn(*shape, generator=gen, dtype=torch.bfloat16, device="cuda") * scale

        def fp32(*shape, scale=0.05):
            return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cuda") * scale

        self.batch = batch
        self.heads = heads
        self.slots = slots
        self.dim = heads * HEAD_DIM
        self.x_q = bf16(1, batch, heads, HEAD_DIM)
        self.x_k = bf16(1, batch, heads, HEAD_DIM)
        self.x_v = bf16(1, batch, heads, HEAD_DIM)
        self.onorm_g = bf16(1, batch, heads, HEAD_DIM)
        self.g = bf16(1, batch, heads, HEAD_DIM)
        self.beta = bf16(1, batch, heads)
        self.w_q_t = bf16(CONV_KERNEL_SIZE, self.dim)
        self.w_k_t = bf16(CONV_KERNEL_SIZE, self.dim)
        self.w_v_t = bf16(CONV_KERNEL_SIZE, self.dim)
        self.A_log = fp32(heads, scale=1.0)
        self.dt_bias = fp32(self.dim)
        self.onorm_weight = fp32(HEAD_DIM, scale=1.0)
        self.state = fp32(slots, heads, HEAD_DIM, HEAD_DIM)
        # HF-layout conv pool: [slots, 3 * dim, W] with W contiguous.
        self.conv_pool = bf16(slots, 3 * self.dim, CONV_KERNEL_SIZE)
        self.cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device="cuda")

    def x_new(self) -> torch.Tensor:
        """This step's raw conv inputs, ``[B, 3 * dim]`` in pool section order."""
        return torch.cat(
            [t[0].reshape(self.batch, self.dim) for t in (self.x_q, self.x_k, self.x_v)], dim=-1
        )

    def call(
        self,
        *,
        cs_q,
        cs_k,
        cs_v,
        state,
        slot_indices,
        roll_conv_pool,
        x_q=None,
        x_k=None,
        x_v=None,
        g=None,
        onorm_g=None,
        beta=None,
    ):
        return run_kda_decode_fusion_cuda(
            x_q=self.x_q if x_q is None else x_q,
            x_k=self.x_k if x_k is None else x_k,
            x_v=self.x_v if x_v is None else x_v,
            w_q_t=self.w_q_t,
            w_k_t=self.w_k_t,
            w_v_t=self.w_v_t,
            bias_q=None,
            bias_k=None,
            bias_v=None,
            cs_q=cs_q,
            cs_k=cs_k,
            cs_v=cs_v,
            A_log=self.A_log,
            g=self.g if g is None else g,
            dt_bias=self.dt_bias,
            beta=self.beta if beta is None else beta,
            state=state,
            onorm_g=self.onorm_g if onorm_g is None else onorm_g,
            onorm_weight=self.onorm_weight,
            out=None,
            ssm_state_indices=slot_indices,
            cu_seqlens=self.cu_seqlens,
            scale=HEAD_DIM**-0.5,
            onorm_eps=1e-5,
            lower_bound=-5.0,
            use_beta_sigmoid_in_kernel=True,
            roll_conv_pool=roll_conv_pool,
        )

    def staged_windows(self, conv_pool, slot_indices):
        """The historical W-1 columns, batch-row-dense per section."""
        cs = conv_pool.index_select(0, slot_indices.long())  # [B, 3 * dim, W]
        staged = cs.view(self.batch, 3, self.dim, CONV_KERNEL_SIZE)[:, :, :, 1:].permute(1, 0, 2, 3)
        return cs, staged.contiguous()

    def aten_roll_(self, conv_pool, cs, slot_indices) -> None:
        """Roll the pool the way the unfused decode step does."""
        new_win = torch.cat([cs[:, :, 1:], self.x_new().unsqueeze(-1)], dim=-1)
        conv_pool.index_copy_(0, slot_indices.long(), new_win)


@torch.no_grad()
@_KERNEL_HEADS
def test_decode_reads_row_strided_projection_slices(heads: int) -> None:
    """Column slices of a fused in-projection match separately packed inputs.

    The decode kernel reads its per-token inputs through a batch-row stride,
    so q, k, v, the o-norm gate and beta can be handed over as views into the
    one ``[B, 4 * dim + head_dim + heads]`` row the fused in-projection writes.
    That has to be bit-identical to passing repacked copies, since it is only
    a change of where the kernel reads the same values.
    """
    batch, slots = 5, 8
    inputs = _DecodeInputs(batch, heads, slots, seed=11)
    d, hd = inputs.dim, HEAD_DIM
    slot_indices = torch.tensor([6, 0, 3, 7, 1], dtype=torch.int32, device="cuda")

    # Lay the same values out the way the fused GEMV would: [q|k|v|g|f_a|b].
    proj = torch.empty(batch, 4 * d + hd + heads, dtype=torch.bfloat16, device="cuda")
    for section, tensor in enumerate((inputs.x_q, inputs.x_k, inputs.x_v, inputs.onorm_g)):
        proj[:, section * d : (section + 1) * d] = tensor[0].reshape(batch, d)
    proj[:, 4 * d : 4 * d + hd] = 0.0  # f_a, consumed by a separate projection
    proj[:, 4 * d + hd :] = inputs.beta[0]
    qkvg = proj[:, : 4 * d].unflatten(-1, (4, heads, hd)).permute(1, 0, 2, 3)
    strided_beta = proj[:, 4 * d + hd : 4 * d + hd + heads].unsqueeze(0)
    assert not qkvg[0:1].is_contiguous() and not strided_beta.is_contiguous()

    packed_pool = inputs.conv_pool.clone()
    packed_state = inputs.state.clone()
    _, packed_windows = inputs.staged_windows(packed_pool, slot_indices)
    packed_out = inputs.call(
        cs_q=packed_windows[0],
        cs_k=packed_windows[1],
        cs_v=packed_windows[2],
        state=packed_state,
        slot_indices=slot_indices,
        roll_conv_pool=False,
    )

    strided_pool = inputs.conv_pool.clone()
    strided_state = inputs.state.clone()
    _, strided_windows = inputs.staged_windows(strided_pool, slot_indices)
    strided_out = inputs.call(
        cs_q=strided_windows[0],
        cs_k=strided_windows[1],
        cs_v=strided_windows[2],
        state=strided_state,
        slot_indices=slot_indices,
        roll_conv_pool=False,
        x_q=qkvg[0:1],
        x_k=qkvg[1:2],
        x_v=qkvg[2:3],
        onorm_g=qkvg[3:4],
        beta=strided_beta,
    )

    torch.testing.assert_close(strided_out, packed_out, rtol=0, atol=0)
    torch.testing.assert_close(strided_state, packed_state, rtol=0, atol=0)


@torch.no_grad()
@_KERNEL_HEADS
def test_roll_conv_pool_matches_staged_window_and_aten_roll(heads: int) -> None:
    """The kernel-owned pool roll reproduces staging plus a separate roll.

    ``roll_conv_pool`` folds two things the decode step used to do around the
    kernel — staging each admitted request's history and rolling the pool
    forward — into the kernel's own conv pass. Both the output and the
    resulting pool have to match the unfused sequence exactly.
    """
    batch, slots = 5, 8
    inputs = _DecodeInputs(batch, heads, slots, seed=23)
    d = inputs.dim
    slot_indices = torch.tensor([6, 0, 3, 7, 1], dtype=torch.int32, device="cuda")
    untouched = torch.tensor([2, 4, 5], device="cuda")

    staged_pool = inputs.conv_pool.clone()
    staged_state = inputs.state.clone()
    cs, windows = inputs.staged_windows(staged_pool, slot_indices)
    staged_out = inputs.call(
        cs_q=windows[0],
        cs_k=windows[1],
        cs_v=windows[2],
        state=staged_state,
        slot_indices=slot_indices,
        roll_conv_pool=False,
    )
    inputs.aten_roll_(staged_pool, cs, slot_indices)

    rolled_pool = inputs.conv_pool.clone()
    rolled_state = inputs.state.clone()
    untouched_before = rolled_pool.index_select(0, untouched).clone()
    rolled_out = inputs.call(
        cs_q=rolled_pool[:, :d],
        cs_k=rolled_pool[:, d : 2 * d],
        cs_v=rolled_pool[:, 2 * d :],
        state=rolled_state,
        slot_indices=slot_indices,
        roll_conv_pool=True,
    )

    torch.testing.assert_close(rolled_out, staged_out, rtol=0, atol=0)
    torch.testing.assert_close(rolled_state, staged_state, rtol=0, atol=0)
    torch.testing.assert_close(rolled_pool, staged_pool, rtol=0, atol=0)
    torch.testing.assert_close(
        rolled_pool.index_select(0, untouched), untouched_before, rtol=0, atol=0
    )


@torch.no_grad()
@_KERNEL_HEADS
def test_roll_conv_pool_stays_in_step_over_many_tokens(heads: int) -> None:
    """A one-column drift in the in-place roll accumulates instead of cancelling.

    A single step cannot tell a correct roll from one that repeats or skips a
    column when the neighbouring columns happen to agree, so run long enough
    for any such drift to walk the whole window out of the pool.
    """
    batch, slots, steps = 5, 6, 8
    inputs = _DecodeInputs(batch, heads, slots, seed=37)
    d = inputs.dim
    slot_indices = torch.tensor([4, 1, 5, 0, 2], dtype=torch.int32, device="cuda")

    staged_pool = inputs.conv_pool.clone()
    staged_state = inputs.state.clone()
    rolled_pool = inputs.conv_pool.clone()
    rolled_state = inputs.state.clone()

    for step in range(steps):
        cs, windows = inputs.staged_windows(staged_pool, slot_indices)
        staged_out = inputs.call(
            cs_q=windows[0],
            cs_k=windows[1],
            cs_v=windows[2],
            state=staged_state,
            slot_indices=slot_indices,
            roll_conv_pool=False,
        )
        inputs.aten_roll_(staged_pool, cs, slot_indices)

        rolled_out = inputs.call(
            cs_q=rolled_pool[:, :d],
            cs_k=rolled_pool[:, d : 2 * d],
            cs_v=rolled_pool[:, 2 * d :],
            state=rolled_state,
            slot_indices=slot_indices,
            roll_conv_pool=True,
        )

        torch.testing.assert_close(
            rolled_out, staged_out, rtol=0, atol=0, msg=f"output diverged at step {step}"
        )
        torch.testing.assert_close(
            rolled_pool, staged_pool, rtol=0, atol=0, msg=f"conv pool diverged at step {step}"
        )
        torch.testing.assert_close(
            rolled_state,
            staged_state,
            rtol=0,
            atol=0,
            msg=f"recurrent state diverged at step {step}",
        )


@torch.no_grad()
def test_roll_conv_pool_requires_slot_indices() -> None:
    """Without slot indices the kernel would roll batch rows, not cache slots."""
    inputs = _DecodeInputs(batch=2, heads=4, slots=4, seed=41)
    d = inputs.dim
    with pytest.raises(ValueError, match="ssm_state_indices"):
        inputs.call(
            cs_q=inputs.conv_pool[:, :d],
            cs_k=inputs.conv_pool[:, d : 2 * d],
            cs_v=inputs.conv_pool[:, 2 * d :],
            state=inputs.state,
            slot_indices=None,
            roll_conv_pool=True,
        )
