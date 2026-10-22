# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MTP layer's routed experts, bf16, this rank's 64-expert window: FC1 (gather + grouped
GEMM + SwiGLU) then FC2 (grouped GEMM + weighted finalize, accumulated into the output), in
CuTe DSL."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ...kernels.moe import mtp_moe_gather_grouped_gemm_act_fusion as _fc1
from ...kernels.moe import mtp_moe_grouped_gemm_finalize_fusion as _fc2
from ...kernels.moe.moe_utils import cute_tensor_from_torch as _ctf
from .cute_dsl_moe_nvfp4_experts import PAIR_BLOCK, expert_rows
from .cute_dsl_moe_routing import LOCAL_EXPERTS, TILE, TOP_K, max_tiles

#: bf16: `mma_tiler_n` must hold one up/gate pair of `epi_subtile_n` each; 64 is rejected.
MMA_TILER_N, MMA_TILER_K = 128, 64
EPI_SUBTILE_N = 64


def allocate_c(max_rows: int, inter: int, device) -> torch.Tensor:
    """One FC1 output buffer: bf16 `[1, max_tiles(max_rows) * 128, inter]`."""
    return torch.zeros((1, max_tiles(max_rows) * TILE, inter), dtype=torch.bfloat16, device=device)


@dataclass(frozen=True)
class CompiledMtpExperts:
    """Both GEMMs compiled once for a row capacity. `bias` is the zero fp32
    `[experts, 2 * inter]` operand both declare and neither reads."""

    fc1: object
    fc2: object
    bias: torch.Tensor
    hidden: int
    inter: int
    experts: int
    max_rows: int


def build(
    *,
    x: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor,
    topk_w: torch.Tensor,
    maps: Mapping[str, torch.Tensor],
    fc1_w: torch.Tensor,
    fc2_w: torch.Tensor,
    max_rows: int,
) -> CompiledMtpExperts:
    """Compile FC1 and FC2 once, for up to `max_rows` rows. Call at load; nothing is
    launched.

    The tensors are any operands of the shapes the target passes per call (one slot's
    buffers and the layer's weights): the compile reads their layouts, and pointers stay
    runtime operands. FC1: 128 x 64 MMA tile, rastered along M, every SM. FC2: no bias,
    no block reduction, one buffer, a 64-column epilogue subtile, rastered along M.
    Raises if a compile fails.
    """
    dev = out.device
    experts, n_fc1, hidden = (int(s) for s in fc1_w.shape)
    inter = n_fc1 // 2
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    bias = torch.zeros((experts, n_fc1), dtype=torch.float32, device=dev)
    bf16 = cutlass.BFloat16
    with torch.cuda.device(dev):
        stream = cutlass_torch.current_stream()
        # Built in the stored `(experts, n, k)` order and permuted: `leading_dim=0` wants
        # `stride[0] == 1`, which only the permuted view has.
        fc1 = cute.compile(
            _fc1.host_function,
            _ctf(fc1_w.permute(2, 1, 0), bf16, leading_dim=0),
            _ctf(c.permute(2, 1, 0), bf16, leading_dim=0),
            _ctf(maps["tile_idx_to_expert_idx"], None, leading_dim=0),
            _ctf(bias, None, leading_dim=1),
            _ctf(x.unsqueeze(-1).permute(1, 0, 2), bf16, leading_dim=0),
            _ctf(maps["permuted_idx_to_expanded_idx"], None, leading_dim=0),
            _ctf(maps["tile_idx_to_mn_limit"], None, leading_dim=0),
            hidden,
            n_fc1,
            TOP_K,
            cutlass.Int64(maps["num_non_exiting_tiles"].data_ptr()),
            max_rows * TOP_K,
            MMA_TILER_N,
            MMA_TILER_K,
            True,
            sms,
            stream,
        )
        fc2 = cute.compile(
            _fc2.host_function,
            _ctf(c.permute(2, 1, 0), bf16, leading_dim=0),
            _ctf(fc2_w.permute(2, 1, 0), bf16, leading_dim=0),
            cutlass.Int64(out.data_ptr()),
            cutlass.Int64(maps["tile_idx_to_expert_idx"].data_ptr()),
            cutlass.Int64(bias.data_ptr()),
            cutlass.Int64(maps["permuted_idx_to_expanded_idx"].data_ptr()),
            cutlass.Int64(topk_w.data_ptr()),
            cutlass.Int64(maps["tile_idx_to_mn_limit"].data_ptr()),
            inter,
            hidden,
            TOP_K,
            cutlass.Int64(maps["num_non_exiting_tiles"].data_ptr()),
            max_rows * TOP_K,
            MMA_TILER_N,
            MMA_TILER_K,
            False,
            1,
            False,
            False,
            EPI_SUBTILE_N,
            True,
            sms,
            stream,
        )
    return CompiledMtpExperts(fc1, fc2, bias, hidden, inter, experts, int(max_rows))


class CuteDslMtpMoeExperts(OpWrapper):
    """The MTP layer's routed experts, accumulated into `out`. Returns `out`.

        h_r  = up * silu(gate),  [up | gate] = x[token] x W1[e]^T, rounded to bf16
        out[token] += topk_w[token, slot] * (h_r x W2[e]^T)

    for every valid row r of the maps (`token = expanded // 8`, `slot = expanded % 8`).

    `x` bf16 `[rows, 6144]`; `maps` the six maps for these rows; `topk_w` fp32 `[rows, 8]`
    (or the fused top-k's flat weights, the same memory); `fc1_w` bf16 `[64, 4096, 6144]`
    with 64-row blocks alternating up, gate (`interleave_w13`); `fc2_w` bf16
    `[64, 6144, 2048]`; `out` bf16 `[rows, 6144]`, zeroed by the caller (or the fused
    top-k's zero plane) unless it holds a partial to add onto; `c` the bf16 FC1 output
    scratch (`allocate_c`).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP width 2, one sequence", spec=dict(rows=2, offset=0)),
        Cell(why="the full decode batch, 64 rows", spec=dict(rows=64, offset=64)),
        Cell(
            why="128 rows, a context step padded to a power of two: experts over one tile",
            spec=dict(rows=128, offset=128),
        ),
        Cell(
            why="8192 rows, the token budget: experts over several tiles",
            spec=dict(rows=8192, offset=192),
        ),
    )

    note = """
    The target builds this once for 8192 rows. Decode widths take the router and the
    fused top-k with its zero plane first (`cute_dsl_moe_router`,
    `cute_dsl_moe_topk_routing`); context steps select with `noaux_tc_op`, stage the
    ids, weights and input into the layer's slot buffers, build the maps with
    `cute_dsl_moe_routing` and clear `out` with `zero_()`. Every row count pads to a
    power of two; padding rows go to `pad_expert_id` at weight 0 on context steps.

    The compiled handles take only the runtime arguments, in signature order: FC1 takes
    `topK` as a Constexpr and FC2 at runtime, so a transposition is a wrong answer rather
    than an error. FC2 accumulates with atomics, so the order of a token's eight
    contributions, and their bf16 rounding, varies run to run.
    """

    def __call__(
        self,
        x: torch.Tensor,
        maps: Mapping[str, torch.Tensor],
        topk_w: torch.Tensor,
        fc1_w: torch.Tensor,
        fc2_w: torch.Tensor,
        out: torch.Tensor,
        c: torch.Tensor,
        compiled: CompiledMtpExperts,
    ) -> torch.Tensor:
        bf16 = cutlass.BFloat16
        stream = cutlass_torch.current_stream()
        compiled.fc1(
            _ctf(fc1_w.permute(2, 1, 0), bf16, leading_dim=0),
            _ctf(c.permute(2, 1, 0), bf16, leading_dim=0),
            _ctf(maps["tile_idx_to_expert_idx"], None, leading_dim=0),
            _ctf(compiled.bias, None, leading_dim=1),
            _ctf(x.unsqueeze(-1).permute(1, 0, 2), bf16, leading_dim=0),
            _ctf(maps["permuted_idx_to_expanded_idx"], None, leading_dim=0),
            _ctf(maps["tile_idx_to_mn_limit"], None, leading_dim=0),
            cutlass.Int32(compiled.hidden),
            cutlass.Int32(2 * compiled.inter),
            cutlass.Int64(maps["num_non_exiting_tiles"].data_ptr()),
            cutlass.Int32(compiled.max_rows * TOP_K),
            stream,
        )
        compiled.fc2(
            _ctf(c.permute(2, 1, 0), bf16, leading_dim=0),
            _ctf(fc2_w.permute(2, 1, 0), bf16, leading_dim=0),
            cutlass.Int64(out.data_ptr()),
            cutlass.Int64(maps["tile_idx_to_expert_idx"].data_ptr()),
            cutlass.Int64(compiled.bias.data_ptr()),
            cutlass.Int64(maps["permuted_idx_to_expanded_idx"].data_ptr()),
            cutlass.Int64(topk_w.data_ptr()),
            cutlass.Int64(maps["tile_idx_to_mn_limit"].data_ptr()),
            cutlass.Int32(compiled.inter),
            cutlass.Int32(compiled.hidden),
            cutlass.Int32(TOP_K),
            cutlass.Int64(maps["num_non_exiting_tiles"].data_ptr()),
            cutlass.Int32(compiled.max_rows * TOP_K),
            stream,
        )
        return out

    def reference(
        self,
        x: torch.Tensor,
        maps: Mapping[str, torch.Tensor],
        topk_w: torch.Tensor,
        fc1_w: torch.Tensor,
        fc2_w: torch.Tensor,
        out: torch.Tensor,
        c: torch.Tensor,
        compiled: CompiledMtpExperts,
    ) -> torch.Tensor:
        """fp32 GEMMs (TF32 off) over the gathered rows, the SwiGLU in fp32 then rounded to
        bf16 (FC1 writes bf16), the weighted sum added onto `out` in fp32 and rounded once.
        Fresh tensor; `out` is read, not written. `c` and `compiled` are not read."""
        rows = int(x.shape[0])
        w_flat = topk_w.reshape(-1)[: rows * TOP_K].float()
        acc = out.float().clone()
        xf = x.float()
        with true_fp32_matmul():
            for e, expanded in sorted(expert_rows(maps).items()):
                tokens = expanded // TOP_K
                g = xf[tokens] @ fc1_w[e].float().t()
                r, n = g.shape
                blocks = g.view(r, n // (2 * PAIR_BLOCK), 2, PAIR_BLOCK)
                up = blocks[:, :, 0].reshape(r, n // 2)
                gate = blocks[:, :, 1].reshape(r, n // 2)
                h = (up * gate * torch.sigmoid(gate)).to(torch.bfloat16).float()
                y = h @ fc2_w[e].float().t()
                acc.index_add_(0, tokens, y * w_flat[expanded].unsqueeze(1))
        return acc.to(out.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        maps: Mapping[str, torch.Tensor],
        topk_w: torch.Tensor,
        fc1_w: torch.Tensor,
        fc2_w: torch.Tensor,
        out: torch.Tensor,
        c: torch.Tensor,
        compiled: CompiledMtpExperts,
    ) -> None:
        # FC2 reads the weights, maps and output through raw pointers: a short or strided
        # buffer is read or written as if dense.
        rows = int(x.shape[0])
        hidden, inter, experts = compiled.hidden, compiled.inter, compiled.experts
        assert 1 <= rows <= compiled.max_rows, (
            f"compiled for up to {compiled.max_rows} rows, got {rows}"
        )
        bf = torch.bfloat16
        for name, t, shape, dtype in (
            ("x", x, (rows, hidden), bf),
            ("fc1_w", fc1_w, (experts, 2 * inter, hidden), bf),
            ("fc2_w", fc2_w, (experts, hidden, inter), bf),
            ("out", out, (rows, hidden), bf),
        ):
            assert tuple(t.shape) == shape and t.dtype == dtype, (
                f"{name} must be {dtype} {list(shape)}, got {t.dtype} {tuple(t.shape)}"
            )
            assert t.is_contiguous(), f"{name} must be contiguous"
            assert t.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned"
        assert topk_w.dtype == torch.float32 and topk_w.is_contiguous()
        assert topk_w.numel() >= rows * TOP_K, "topk_w must hold rows * 8 weights"
        tiles = min(LOCAL_EXPERTS, rows * TOP_K) if rows <= TILE else max_tiles(rows)
        for k, size in (
            ("tile_idx_to_expert_idx", tiles),
            ("tile_idx_to_mn_limit", tiles),
            ("permuted_idx_to_expanded_idx", tiles * TILE),
            ("num_non_exiting_tiles", 1),
        ):
            assert maps[k].dtype == torch.int32 and maps[k].is_contiguous()
            assert maps[k].numel() >= size, f"{k} holds {maps[k].numel()}; {rows} rows need {size}"
        assert c.dtype == bf and c.dim() == 3 and c.shape[2] == inter and c.is_contiguous()
        assert c.shape[1] >= tiles * TILE, "c is smaller than these rows' tiles"


cute_dsl_mtp_moe_experts = CuteDslMtpMoeExperts()
