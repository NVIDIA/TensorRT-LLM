# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuteDSL NVFP4 FC1 of the routed experts: gather, grouped GEMM, SwiGLU and NVFP4
requantization of the intermediate, in one trtllm call."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import SWIGLU_LIMIT_SCALAR_DISABLED
from tensorrt_llm._torch.utils import ActivationType

_SWIGLU = int(ActivationType.Swiglu)

#: e2m1 magnitudes by 3-bit code, and the midpoints between neighbours. A value
#: exactly on a midpoint rounds to the even code (`_TIE_UP[i]`: the even side is up).
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)

#: The pairing the fused SwiGLU epilogue reads: the weight's rows `[128 g, 128 g + 64)`
#: are up rows, the next 64 their gate rows, and output column `64 g + r` pairs row
#: `128 g + r` with row `128 g + 64 + r`. Fixed by the epilogue's 64-column subtile,
#: whatever MMA N tile the autotuner picks.
GATE_UP_GROUP = 64


def _ckpt(
    tokens: int,
    tile_size: int,
    mma_n: int = 128,
    dist: str = "router",
    gsf: str = "calibrated",
    zero_tokens: int = 0,
) -> dict:
    """The checkpoint's routed experts: 512 experts, top-10, hidden 2560, I 640, all
    local (tp1), SwiGLU, no clamp, 16-element e4m3 block scales.

    `mma_n` is the autotuner tactic's MMA N tile (128 is also the fallback when the
    tuner has no entry). `dist` names how the test draws the expert ids: `router` =
    each row's top-10 of uniform scores. `gsf` sets `global_sf` against the
    activations' magnitude: `calibrated` = the activation maximum the test
    calibrates to maps to the e4m3 maximum, as the checkpoint's per-layer `1 /
    down_proj.input_scale` does; `subnormal` = 2^-12 of it, so block scales fall
    into e4m3 subnormals and zero. `zero_tokens` makes that many leading tokens
    all-zero rows (zero block scales).
    """
    return dict(
        tokens=tokens,
        tile_size=tile_size,
        mma_n=mma_n,
        dist=dist,
        gsf=gsf,
        zero_tokens=zero_tokens,
        num_experts=512,
        top_k=10,
        hidden=2560,
        intermediate=640,
    )


class Fc1Expected(NamedTuple):
    """What `reference` returns: the op's two outputs, plus what the test's compare reads.

    `c` / `c_sf` are laid out exactly as the op returns them; rows the op does not
    write hold zeros. `rows` marks the permuted rows the op writes and the compare
    gates. `scaled` is `y * global_sf` on those rows, before rounding -- the value
    the block scale and the codes are derived from, which rounding loses.
    """

    c: torch.Tensor
    c_sf: torch.Tensor
    rows: torch.Tensor
    scaled: torch.Tensor


class CuteDslNvfp4GatherGroupedGemmActFusion(OpWrapper):
    """`torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell`:
    FC1 of the routed experts on the CuteDSL MoE path, one kernel launch.

    For every permuted row `p` the sort filled -- tile `i = p // tile_size` below
    `num_non_exiting_tiles` and `p < tile_idx_to_mn_limit[i]`:

        t        = permuted_idx_to_expanded_idx[p] // top_k        the token
        e        = tile_idx_to_group_idx[i]                        the local expert
        acc      = alpha[e] * deq(input[t]) @ deq(weight[e]).T     [2 I], fp32
        y[p, j]  = up * silu(gate),  up = acc[128 g + r], gate = acc[128 g + 64 + r],
                   j = 64 g + r
        out[p], out_sf[p] = NVFP4(y[p] * global_sf)                16-element blocks

    where `deq` is e2m1 times its e4m3 block scale, and NVFP4 takes per block
    `sf = e4m3(max|v| / 6)` (saturating at 448) and codes `e2m1(v / sf)` (saturating
    at 6). So `out * out_sf / global_sf` approximates `y`.

    Arguments, as the CuteDSL MoE passes them (`run_moe_nvfp4_impl`,
    `_torch/moe/fused_moe/fused_moe_cute_dsl.py`):

    - `input` `[T, H/2]` float4_e2m1fn_x2 and `input_scale` `[T, H/16]` uint8: the
      un-permuted activation as `fp4_quantize(x, 1 / input_scale_1, 16, False,
      False)` returns it -- LINEAR (row-major) e4m3 scales, viewed 2-D. Gathered
      inside; the caller never permutes it.
    - `weight` `[E, 2I, H/2]` float4_e2m1fn_x2: per expert `cat([up, gate])`
      interleaved in `GATE_UP_GROUP`-row groups (up rows 0:64, gate rows 0:64, up
      rows 64:128, ...). No 32-row shuffle (that is trtllm-gen's layout).
    - `weight_scale` `[E, 2I, H/16]` uint8: the e4m3 block scales with the same row
      interleave, then 128x4-swizzled per expert (`block_scale_interleave`).
    - `alpha` `[E]` fp32: per expert, `input_scale_1 * weight_scale_2[e]` -- one
      factor for both halves, applied to the accumulator before SwiGLU.
    - the four sort tensors, exactly as `moe_sort` returns them for the same
      `top_k` and `tile_tokens_dim == tile_size`.
    - `global_sf` fp32, one element: the FC2 input's global scale, `1 /
      max(down_proj.input_scale)` of the layer.
    - `num_experts`, `num_local_experts`, `local_expert_offset`: read only by the
      autotuner's shape buckets and cache key; the kernel takes the expert count
      from `weight.size(0)` and the expert from the tile.

    Returns `(c, c_sf)`: `c` `[M, I/2]` float4_e2m1fn_x2 and `c_sf` `[M * I/16]`
    uint8, `M = permuted_idx_to_expanded_idx.numel()`, one row per PERMUTED row and
    the block scales 128x4-swizzled over `[M, I/16]` -- exactly the `input` /
    `input_scale` the plain NVFP4 grouped GEMM (FC2) takes, with the same four sort
    tensors. Rows the sort did not fill are not part of the result (see `note`).

    Fusion boundary: the gather of the activation rows, the grouped GEMM, the
    per-expert alpha, SwiGLU and the requantization happen inside. The caller owns
    the activation quantization, the sort, FC2 and the routing-weighted combine;
    routing weights are not read here.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tile 128: decode, 1 request without MTP",
            spec=_ckpt(1, 128),
        ),
        Cell(
            why="tile 128: 8192 rows, max_num_tokens prefill",
            spec=_ckpt(8192, 128),
        ),
        Cell(
            why="tile 128: tiny global_sf and 7 all-zero token rows; subnormal and zero "
            "block scales",
            spec=_ckpt(2048, 128, gsf="subnormal", zero_tokens=7),
        ),
    )

    note = """
    Path certified: the CuTe DSL kernel `BlockScaledContiguousGatherGroupedGemmKernel`
    (`_torch/moe/kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py`),
    built by `Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner`
    (`_torch/custom_ops/cute_dsl_custom_ops.py`), which JIT-compiles it with
    `cute.compile` on the first call per (tile, tactic, top_k, activation) and
    keeps it for the process. The op is registered only when the CuTe DSL imports
    (`IS_CUTLASS_DSL_AVAILABLE`), and the runner accepts SM 100 and 103 only.

    Metadata consumed: none beyond the arguments. Stateless: allocates `c` and
    `c_sf` on every call, on the current stream, with no host synchronization, so
    it captures into a CUDA graph together with moe_sort.

    Call sites: the engine's CuteDSL MoE (`run_moe_nvfp4_impl`) and the target's
    routed-expert step, eager for prefill and mixed batches (T up to
    max_num_tokens 8192) and captured for decode (T = requests x MTP verify width,
    up to 32 x 4 = 128). Tile 128 is the target's and the cells' tile; 256 is
    certified through the per-pair invariance below.

    Tactics. The op asks the AutoTuner for an MMA tile of (tile_size, 128) or
    (tile_size, 256) and with no tuning entry falls back to (tile_size, 128);
    serving warmup tunes per row bucket and may pick either. Every pair's codes
    and block scales are bitwise identical across tile 128 / 256 x MMA N 128 / 256
    x the fallback (the test checks it at 8192 rows), so cells driven at one
    tactic certify the others.

    Determinism. A (token, expert) pair's output row depends on that pair alone:
    it is bitwise unchanged by reordering an expert's rows (moe_sort orders them by
    atomics from T = 5), by repeating the call, and by cutting a batch to a prefix.

    What is written: the rows the sort filled hold the result; the padding rows of
    a written tile -- from its tile_idx_to_mn_limit to its end -- get zero block
    scales and zero codes, so a plain grouped GEMM over whole tiles (FC2) reads
    zeros there; the tiles at and past num_non_exiting_tiles are not written at
    all, and under a graph keep whatever the buffer last held. FC2 must be tiled by
    the same tile arrays, and the combine reads only
    expanded_idx_to_permuted_idx's rows.

    Scale range: with a tiny global_sf the block scales fall to e4m3 subnormals
    and to zero (a cell). An all-zero token gives scale 0 and codes of magnitude
    0 (the same cell). With global_sf 25x the calibrated value the block scales
    clip at 448 and the codes at 6 (no NaN) -- measured, not certified. A block
    whose scale is 0 dequantizes to zero whatever its codes hold, and the test's
    compare does not read them.

    Preconditions the caller owns (a wrong preparation of each operand -- the
    weight's rows, its scales, the activation's scales, alpha -- runs without
    error, with finite scales, and puts most block scales and codes outside the
    test's gate):
    - `weight` / `weight_scale` in the layout the class docstring states: not
      trtllm-gen's row interleave, up rows first, scales swizzled after the same
      row interleave as the weight, alpha not shifted by an expert.
    - One `alpha` per expert scales BOTH halves, so up_proj and gate_proj must
      share `weight_scale_2`, and `input_scale` must be one value per layer; this
      checkpoint satisfies both. `global_sf` inverts the largest
      `down_proj.input_scale`; the spread of the rest is FC2's `alpha`.
    - `input_scale` LINEAR. Swizzled scales have the same size when T % 128 == 0
      and are read as garbage without error.
    - The sort from moe_sort with the same `top_k` and tile_tokens_dim ==
      `tile_size`, over the tokens `input` holds. Nothing checks `top_k` against it.

    Refused by `is_valid`, each silently wrong unguarded: a non-contiguous input,
    input_scale, weight, weight_scale or alpha (data_ptr() read as dense); an
    alpha shorter than weight.size(0); a weight_scale with fewer bytes than the
    weight needs (read past its end: zeros of the same allocation come back as zero
    experts, a fresh allocation faults).

    Loud (AssertionError from the runner before the launch): tile_size 256 with a
    tile-128 sort and 128 with a tile-256 sort; input_scale not H/16 wide;
    global_sf of 2 elements; alpha not fp32; weight_scale typed float8_e4m3fn;
    input typed uint8. tile_size 64: ValueError.

    Measured at bring-up and not re-driven by the test: the CUDA-graph capture
    with moe_sort, the determinism and write statements, the wrong preparations'
    distance from the gate, the short-alpha and short-weight_scale refusals and
    the loud rejections. The test re-drives the cells, the tile / tactic
    invariance and the `input` contiguity refusal.

    Uncertified: Relu2 and SiTU; a SwiGLU clamp (swiglu_limit_scalar >= 0);
    expert_counts (DeepEP count-native metadata); expert parallelism (the kernel
    reads neither num_experts nor local_expert_offset, so EP changes only the sort
    and the weight slice); other hidden / intermediate sizes and top_k;
    scaling_vector_size other than 16; other architectures.

    Hard limits (sm_103)
      written rows x I/16 B of c_sf   boundary 2^31 B: an illegal memory access
                      once rows written reach 53687092 at I = 640 (T ~ 5.37M at
                      top-10; T = 8192 writes at most 212480 rows)
      input, input_scale, weight      past 2^31 elements / bytes: exact (probed
                      up to 1536 experts)
      permuted index  int32, within moe_sort's 2^31-pair limit; no index passes
                      through fp32
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        scaling_vector_size: int = 16,
        activation_type: int = _SWIGLU,
        swiglu_limit_scalar: float = SWIGLU_LIMIT_SCALAR_DISABLED,
        expert_counts: Optional[torch.Tensor] = None,
        expert_capacity: int = 0,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
            input=input,
            weight=weight,
            input_scale=input_scale,
            weight_scale=weight_scale,
            alpha=alpha,
            tile_idx_to_group_idx=tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=global_sf,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            tile_size=tile_size,
            scaling_vector_size=scaling_vector_size,
            activation_type=activation_type,
            swiglu_limit_scalar=swiglu_limit_scalar,
            expert_counts=expert_counts,
            expert_capacity=expert_capacity,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    def reference(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        scaling_vector_size: int = 16,
        activation_type: int = _SWIGLU,
        swiglu_limit_scalar: float = SWIGLU_LIMIT_SCALAR_DISABLED,
        expert_counts: Optional[torch.Tensor] = None,
        expert_capacity: int = 0,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
    ) -> Fc1Expected:
        """Gather, dequantize, GEMM in float64, SwiGLU, exact NVFP4 rounding.

        Products of e2m1 and e4m3 values are exact, and float64 sums of 2560 of
        them are exact to ~1e-16 relative, so this side carries no accumulation
        error of its own; the block scale and codes are the exact round-to-nearest-
        even of the exact `y * global_sf`. Where a block scale rounds to zero the
        codes here are zero: they dequantize to zero whatever they hold, and the
        test's compare does not read them.

        Certified surface only: SwiGLU without a clamp, the sort-metadata path
        (`expert_counts` None), no SiTU. Anything else raises NotImplementedError.
        `num_experts`, `num_local_experts`, `local_expert_offset` and
        `expert_capacity` do not enter the computation.
        """
        if (
            int(activation_type) != _SWIGLU
            or swiglu_limit_scalar >= 0
            or expert_counts is not None
            or situ_beta is not None
            or situ_linear_beta is not None
        ):
            raise NotImplementedError(
                "reference covers SwiGLU without a clamp on sort metadata only"
            )
        vec = scaling_vector_size
        dev = input.device
        m = permuted_idx_to_expanded_idx.numel()
        n2 = weight.shape[1]
        inter = n2 // 2
        hidden = input.shape[1] * 2
        n_tiles = int(num_non_exiting_tiles.reshape(-1)[0])
        span = n_tiles * tile_size

        rows = torch.zeros(m, dtype=torch.bool, device=dev)
        limit = tile_idx_to_mn_limit[:n_tiles].long().repeat_interleave(tile_size)
        rows[:span] = torch.arange(span, device=dev) < limit
        real = torch.nonzero(rows).squeeze(1)
        token = permuted_idx_to_expanded_idx.long()[real] // top_k
        expert = tile_idx_to_group_idx[:n_tiles].long().repeat_interleave(tile_size)[real]

        a = _dequant(input.view(torch.uint8)[token], input_scale.view(torch.uint8)[token], vec)
        y = torch.empty(real.numel(), inter, dtype=torch.float64, device=dev)
        w_sf = weight_scale.view(torch.uint8).reshape(weight.shape[0], -1)
        group = GATE_UP_GROUP
        for e in torch.unique(expert).tolist():
            sel = torch.nonzero(expert == e).squeeze(1)
            w = _dequant(weight[e].view(torch.uint8), unswizzle(w_sf[e], n2, hidden // vec), vec)
            acc = (a[sel] @ w.T) * float(alpha[e])
            acc = acc.view(-1, n2 // (2 * group), 2, group)
            up = acc[:, :, 0].reshape(-1, inter)
            gate = acc[:, :, 1].reshape(-1, inter)
            y[sel] = up * gate * torch.sigmoid(gate)
        scaled = y * float(global_sf.reshape(-1)[0])

        blocks = scaled.view(-1, inter // vec, vec)
        pre = blocks.abs().amax(-1) / E2M1_MAX
        sf = pre.clamp(max=E4M3_MAX).float().to(torch.float8_e4m3fn)
        sf_val = sf.double().unsqueeze(-1)
        q = torch.where(sf_val > 0, blocks / sf_val, torch.zeros_like(blocks))
        codes = e2m1_codes(q.reshape(-1, inter))
        c = torch.zeros(m, inter // 2, dtype=torch.uint8, device=dev)
        c[real] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        sf_rows = torch.zeros(m, inter // vec, dtype=torch.uint8, device=dev)
        sf_rows[real] = sf.view(torch.uint8)
        return Fc1Expected(
            c=c.view(torch.float4_e2m1fn_x2), c_sf=swizzle(sf_rows), rows=rows, scaled=scaled
        )

    def is_valid(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        scaling_vector_size: int = 16,
        activation_type: int = _SWIGLU,
        swiglu_limit_scalar: float = SWIGLU_LIMIT_SCALAR_DISABLED,
        expert_counts: Optional[torch.Tensor] = None,
        expert_capacity: int = 0,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
    ) -> None:
        # The runner hands data_ptr() to the kernel with dense row-major layouts
        # assumed; it checks dims and dtypes, never strides.
        for name, t in (
            ("input", input),
            ("input_scale", input_scale),
            ("weight", weight),
            ("weight_scale", weight_scale),
            ("alpha", alpha),
        ):
            assert t.is_contiguous(), (
                f"{name} must be contiguous; the kernel reads it from data_ptr() as a "
                "dense row-major buffer and silently reads the wrong elements"
            )
        # One alpha per expert of `weight`; the runner checks only alpha.dim() == 1.
        assert alpha.numel() == weight.size(0), (
            f"alpha has {alpha.numel()} elements for {weight.size(0)} experts; the "
            "kernel reads alpha[expert] past the end without raising"
        )
        # The runner checks weight_scale.dim() == 3 only, never its size.
        expected = weight.numel() * 2 // scaling_vector_size
        assert weight_scale.numel() == expected, (
            f"weight_scale holds {weight_scale.numel()} bytes, the weight needs "
            f"{expected}; the kernel reads past its end -- silently within the "
            "allocation, with an illegal memory access beyond it"
        )


# ── native-torch NVFP4 helpers for reference() and the test's compare ────────


def unswizzle(buf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """128x4-swizzled scale bytes -> `[rows, cols]` (rows % 128 == 0, cols % 4 == 0).

    Byte `(r, c)` sits at `(r // 128) * 128 * cols + (c // 4) * 512 + (r % 32) * 16
    + ((r % 128) // 32) * 4 + c % 4`.
    """
    return buf.reshape(rows // 128, cols // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rows, cols)


def swizzle(sf: torch.Tensor) -> torch.Tensor:
    """`[rows, cols]` scale bytes -> the flat 128x4-swizzled buffer; inverse of `unswizzle`."""
    rows, cols = sf.shape
    return sf.reshape(rows // 128, 4, 32, cols // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1)


def e4m3_value(byte: torch.Tensor) -> torch.Tensor:
    return byte.to(torch.uint8).view(torch.float8_e4m3fn).double()


def _dequant(packed: torch.Tensor, sf: torch.Tensor, vec: int) -> torch.Tensor:
    """`[R, K/2]` packed e2m1 (element 2j in the low nibble) and `[R, K/vec]` e4m3
    bytes -> float64 `[R, K]`. Exact."""
    lut = torch.tensor(E2M1, dtype=torch.float64, device=packed.device)
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    vals = lut[codes & 7] * torch.where(codes >= 8, -1.0, 1.0).double()
    return vals * e4m3_value(sf).repeat_interleave(vec, dim=1)


def e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """float -> 4-bit e2m1 codes (bit 3 = sign), round-to-nearest-even, saturating at 6."""
    mid = torch.tensor(_E2M1_MIDPOINTS, device=v.device, dtype=v.dtype)
    tie_up = torch.tensor(_TIE_UP, device=v.device)
    a = v.abs().unsqueeze(-1)
    code = ((a > mid).sum(-1) + ((a == mid) & tie_up).any(-1)).to(torch.uint8)
    negative = (v < 0) | ((v == 0) & torch.signbit(v))
    return code | (negative.to(torch.uint8) << 3)


cute_dsl_nvfp4_gather_grouped_gemm_act_fusion = CuteDslNvfp4GatherGroupedGemmActFusion()
