"""Architecture-aware Triton Sol-Attn forward implementation."""

import torch
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:  # Pointer path remains usable without descriptors.
    TensorDescriptor = None

from sol_attn.interface import (
    _sink_block_range,
    _validate_inputs,
)

from .preprocess import prepare as prepare_ptr


BLOCK = 64
GROUP = 32


def _use_tma(device) -> bool:
    """Use descriptor-backed Triton I/O when the GPU supports TMA."""

    capability = torch.cuda.get_device_capability(device)
    return capability[0] >= 9 and TensorDescriptor is not None


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (1, 2, 3, 4)
    ],
    key=["T"],
)
@triton.jit
def _forward_tma(
    q_desc,
    k_desc,
    v_desc,
    kc_desc,
    vc_desc,
    threshold,
    o_desc,
    scale,
    T,
    sink_start_block,
    sink_end_block,
    HAS_SINK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    v_tile, q_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(
        tl.arange(0, GROUP_SIZE),
        GROUP_SIZE,
    )
    token_offsets = tl.max_contiguous(
        tl.arange(0, BLOCK_SIZE),
        BLOCK_SIZE,
    )
    q_start = q_block * BLOCK_SIZE
    q = q_desc.load([batch, q_start, head, 0]).reshape([BLOCK_SIZE, D])
    q_len = tl.minimum(BLOCK_SIZE, T - q_start).to(tl.float32)

    output = tl.zeros([BLOCK_SIZE, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    tail_length = T - (NT - 1) * BLOCK_SIZE
    route_threshold = tl.load(
        threshold + (batch * NT + q_block) * H + head
    )

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc = kc_desc.load(
            [batch, group_start, head, 0]
        ).reshape([GROUP_SIZE, D])
        vc = vc_desc.load(
            [batch, group_start, head, v_tile * BV]
        ).reshape([GROUP_SIZE, BV])
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        exact = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
        )
        if HAS_SINK:
            exact = exact | (
                (block_indices >= sink_start_block)
                & (block_indices < sink_end_block)
            )
        exact = exact & valid

        approximate = valid & ~exact
        approximate_scores = tl.where(
            approximate[None, :], scores, -float("inf")
        )
        new_max = tl.maximum(row_max, tl.max(approximate_scores, axis=1))
        alpha = tl.math.exp2(
            tl.where(row_max == new_max, 0.0, row_max - new_max)
        )
        approximate_probability = tl.where(
            approximate[None, :],
            tl.math.exp2(approximate_scores - new_max[:, None]),
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(
            approximate_probability.to(vc.dtype),
            vc,
        )
        lengths = tl.where(
            block_indices == NT - 1, tail_length, BLOCK_SIZE
        ).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(
            approximate_probability * lengths[None, :],
            axis=1,
        )
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(
                group_offsets == offset,
                GROUP_SIZE,
                exact_offsets,
            )
            kv_start = block * BLOCK_SIZE
            k = k_desc.load(
                [batch, kv_start, head, 0]
            ).reshape([BLOCK_SIZE, D])
            exact_scores = tl.dot(q, k.T).to(tl.float32) * scale_log2
            exact_scores += tl.where(
                (kv_start + token_offsets)[None, :] < T,
                0.0,
                -float("inf"),
            )
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(
                exact_scores - new_max[:, None]
            )
            row_sum = row_sum * alpha + tl.sum(
                exact_probability,
                axis=1,
            )
            v = v_desc.load(
                [batch, kv_start, head, v_tile * BV]
            ).reshape([BLOCK_SIZE, BV])
            output = output * alpha[:, None] + tl.dot(
                exact_probability.to(v.dtype),
                v,
            )
            row_max = new_max

    o_desc.store(
        [batch, q_start, head, v_tile * BV],
        (output / row_sum[:, None]).to(tl.bfloat16)[None, :, None, :],
    )


@triton.autotune(
    configs=[
        triton.Config({"BV": 128}, num_warps=4, num_stages=1),
        triton.Config({"BV": 128}, num_warps=8, num_stages=1),
        triton.Config({"BV": 128}, num_warps=4, num_stages=2),
        triton.Config({"BV": 64}, num_warps=4, num_stages=1),
    ],
    key=["T"],
)
@triton.jit
def _forward_ptr(
    q_ptr,
    k_ptr,
    v_ptr,
    kc_ptr,
    vc_ptr,
    threshold_ptr,
    o_ptr,
    scale,
    T,
    NPAD,
    sink_start_block,
    sink_end_block,
    HAS_SINK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    v_tile, q_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(
        tl.arange(0, GROUP_SIZE),
        GROUP_SIZE,
    )
    token_offsets = tl.max_contiguous(
        tl.arange(0, BLOCK_SIZE),
        BLOCK_SIZE,
    )
    dims = tl.arange(0, D)
    value_dims = v_tile * BV + tl.arange(0, BV)
    q_tokens = q_block * BLOCK_SIZE + token_offsets
    q_valid = q_tokens < T
    q_offsets = (
        ((batch * T + q_tokens[:, None]).to(tl.int64) * H + head) * D
        + dims[None, :]
    )
    q = tl.load(q_ptr + q_offsets, mask=q_valid[:, None], other=0.0)
    q_len = tl.minimum(BLOCK_SIZE, T - q_block * BLOCK_SIZE).to(tl.float32)

    output = tl.zeros([BLOCK_SIZE, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    route_threshold = tl.load(
        threshold_ptr + (batch * NT + q_block) * H + head
    )

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc_offsets = (
            ((batch * NPAD + block_indices[:, None]) * H + head) * D
            + dims[None, :]
        )
        vc_offsets = (
            ((batch * NPAD + block_indices[:, None]) * H + head) * D
            + value_dims[None, :]
        )
        kc = tl.load(kc_ptr + kc_offsets)
        vc = tl.load(vc_ptr + vc_offsets)
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        exact = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
        )
        if HAS_SINK:
            exact = exact | (
                (block_indices >= sink_start_block)
                & (block_indices < sink_end_block)
            )
        exact = exact & valid

        approximate = valid & ~exact
        has_approximate = tl.sum(approximate.to(tl.int32), axis=0) > 0
        approximate_scores = tl.where(
            approximate[None, :],
            scores,
            -float("inf"),
        )
        safe_scores = tl.where(has_approximate, approximate_scores, 0.0)
        config_max = tl.maximum(row_max, tl.max(safe_scores, axis=1))
        new_max = tl.where(has_approximate, config_max, row_max)
        alpha = tl.math.exp2(
            tl.where(has_approximate, row_max - new_max, 0.0)
        )
        probability = tl.math.exp2(
            safe_scores
            - tl.where(has_approximate, new_max, 0.0)[:, None]
        )
        probability = tl.where(
            has_approximate & approximate[None, :],
            probability,
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(
            probability.to(vc.dtype),
            vc,
        )
        lengths = tl.minimum(
            BLOCK_SIZE,
            tl.maximum(0, T - block_indices * BLOCK_SIZE),
        ).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(
            probability * lengths[None, :],
            axis=1,
        )
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        num_exact = tl.sum(exact.to(tl.int32), axis=0)
        for _ in range(num_exact):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(
                group_offsets == offset,
                GROUP_SIZE,
                exact_offsets,
            )
            kv_tokens = block * BLOCK_SIZE + token_offsets
            kv_valid = kv_tokens < T
            k_offsets = (
                ((batch * T + kv_tokens[:, None]).to(tl.int64) * H + head)
                * D
                + dims[None, :]
            )
            k = tl.load(
                k_ptr + k_offsets,
                mask=kv_valid[:, None],
                other=0.0,
            )
            exact_scores = tl.dot(q, k.T).to(tl.float32) * scale_log2
            exact_scores += tl.where(
                kv_valid[None, :],
                0.0,
                -float("inf"),
            )
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(
                exact_scores - new_max[:, None]
            )
            row_sum = row_sum * alpha + tl.sum(
                exact_probability,
                axis=1,
            )
            v_offsets = (
                ((batch * T + kv_tokens[:, None]).to(tl.int64) * H + head)
                * D
                + value_dims[None, :]
            )
            v = tl.load(
                v_ptr + v_offsets,
                mask=kv_valid[:, None],
                other=0.0,
            )
            output = output * alpha[:, None] + tl.dot(
                exact_probability.to(v.dtype),
                v,
            )
            row_max = new_max

    output_offsets = (
        ((batch * T + q_tokens[:, None]).to(tl.int64) * H + head) * D
        + value_dims[None, :]
    )
    tl.store(
        o_ptr + output_offsets,
        (output / row_sum[:, None]).to(tl.bfloat16),
        mask=q_valid[:, None],
    )


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
    sink_tokens: int = 0,
    sink_start: int | None = None,
) -> torch.Tensor:
    """Run Triton Sol-Attn on contiguous BF16 BTHD inputs."""

    arch = _validate_inputs(
        q,
        k,
        v,
        thresh_type,
        sink_tokens,
        sink_start,
    )
    if arch[0] < 8:
        raise RuntimeError(
            "Triton Sol-Attn requires an NVIDIA GPU with compute "
            f"capability >= 8.0; got SM{arch[0]}{arch[1]}"
        )
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    tau = float(tau)
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK)
    sink_start_block, sink_end_block = _sink_block_range(
        tokens,
        sink_start,
        sink_tokens,
    )
    use_tma = _use_tma(q.device)

    if use_tma:
        # Keep the original descriptor-backed preprocessing on TMA devices.
        # The pointer preprocessing below exists only for older architectures.
        from ..preprocess import prepare as prepare_tma

        kc, vc, threshold = prepare_tma(
            q,
            k,
            v,
            scale=scale,
            tau=tau,
            thresh_type=thresh_type,
        )
        output = torch.empty_like(v)
        block_shape = [1, BLOCK, 1, head_dim]
        summary_shape = [1, GROUP, 1, head_dim]
        _forward_tma[(1, blocks, batch * heads)](
            TensorDescriptor.from_tensor(q, block_shape),
            TensorDescriptor.from_tensor(k, block_shape),
            TensorDescriptor.from_tensor(v, block_shape),
            TensorDescriptor.from_tensor(kc, summary_shape),
            TensorDescriptor.from_tensor(vc, summary_shape),
            threshold,
            TensorDescriptor.from_tensor(output, block_shape),
            scale,
            tokens,
            sink_start_block,
            sink_end_block,
            sink_tokens > 0,
            heads,
            head_dim,
            blocks,
            head_dim,
            BLOCK,
            GROUP,
        )
        return output

    kc, vc, threshold = prepare_ptr(
        q,
        k,
        v,
        scale=scale,
        tau=tau,
        thresh_type=thresh_type,
        tokens=tokens,
    )
    output = torch.empty_like(v)
    grid = lambda meta: (head_dim // meta["BV"], blocks, batch * heads)
    _forward_ptr[grid](
        q,
        k,
        v,
        kc,
        vc,
        threshold,
        output,
        scale,
        tokens,
        kc.shape[1],
        sink_start_block,
        sink_end_block,
        HAS_SINK=sink_tokens > 0,
        H=heads,
        D=head_dim,
        NT=blocks,
        BLOCK_SIZE=BLOCK,
        GROUP_SIZE=GROUP,
    )
    return output


__all__ = ["sol_attn"]
