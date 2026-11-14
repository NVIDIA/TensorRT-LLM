# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context step's projections in MXF8: the bf16 activation quantized to e4m3 with e8m0
1x32 block scales (one CuTe DSL launch, or none when the producer already wrote the operand),
then a block-scaled tcgen05 GEMM against the weight quantized once at load."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.gemm import self_attn_prefill_projections_mxf8 as _kernel

#: Entry -> the kernel's ABI. `qkva` is the fused q_a | kv_a | k_pe projection.
ENTRIES: dict[str, str] = {
    "qkva": "kv_a_proj_with_mqa",
    "o_proj": "o_proj",
    # The fp8 context attention's query: e4m3 out on a weight with the Q scale folded in.
    "q_b_e4m3": "q_b_proj_e4m3",
    # The fp8 context attention's K / V expansion: an A operand that is already MXF8 (the
    # `[cached | new]` latent rows `cute_dsl_mha_kv_gather_mxf8` writes), e4m3 out.
    "kv_b_k8": "kv_b_expand_k_mxf8",
    "kv_b_v8": "kv_b_expand_v512_mxf8",
    # Both in one launch: `[K | V]` out of the K operand, the V tiles on its first 512 columns.
    "kv_b_kv8": "kv_b_expand_kv_mxf8",
    # The dense MLP (layers 0-2): gate|up (gate rows first) and down, this rank's 3072-wide
    # slice of the intermediate.
    "dense_gate_up": "dense_gate_up_mxf8",
    "dense_down": "dense_down_mxf8",
    # The MTP module's replicated eh_proj on context rows; A is the operand the two input
    # norms write (`cute_dsl_mtp_eh_norm_cat_mxf8`).
    "mtp_eh": "mtp_eh_proj_mxf8",
}
#: A weight in MXF8: e4m3 bytes `[N, K]` and its blocked e8m0 scales.
QuantizedWeight = _kernel.QuantizedWeight


def quantize_weight(weight: torch.Tensor) -> QuantizedWeight:
    """Quantize a row-major bf16 `[N, K]` weight once (the activation kernel's rule)."""
    return _kernel.quantize_weight(weight)


@dataclass(frozen=True)
class Mxf8ProjectionRunner:
    """One entry's launcher: any row count, the weight per call (one runner serves every
    layer). `launch.choice_hits` counts launches per tile."""

    kernel: str
    n: int
    k: int
    out_dtype: torch.dtype
    launch: _kernel.Mxf8Projection | _kernel.Mxf8PrequantProjection

    @property
    def prequant(self) -> bool:
        """The A operand is MXF8 already (`x` uint8 e4m3 plus `sfa`): no activation quant."""
        return isinstance(self.launch, _kernel.Mxf8PrequantProjection)


def build(kernel: str, weight: QuantizedWeight) -> Mxf8ProjectionRunner:
    """Compile `kernel`'s activation quant and both GEMM tiles (256x128 up to the entry's
    small-T bound, 256x256 above) and launch each once on zeros, so no context step meets a
    first compile or module load. `weight` is any one layer's quantized weight."""
    abi = _kernel.ABIS[ENTRIES[kernel]]
    dev = weight.q.device
    launch = _kernel.build(abi.name, dev.index)
    out_dtype = torch.float8_e4m3fn if abi.out_e4m3 else torch.bfloat16
    a_dtype = torch.uint8 if abi.a_prequant else torch.bfloat16
    with torch.cuda.device(dev):
        for rows in launch.warm_rows():
            x = torch.zeros([rows, abi.in_features], dtype=a_dtype, device=dev)
            out = torch.empty([rows, abi.out_features], dtype=out_dtype, device=dev)
            launch(x, weight, out.view(torch.uint8) if abi.out_e4m3 else out)
        torch.cuda.synchronize(dev)
    launch.choice_hits = {t: 0 for t in launch.choice_hits}
    if hasattr(launch, "pre_hits"):
        launch.pre_hits = 0
    return Mxf8ProjectionRunner(kernel, abi.out_features, abi.in_features, out_dtype, launch)


class CuteDslProjectionMxf8(OpWrapper):
    """`out = x @ W^T` with both operands in MXF8 and fp32 accumulation.

    `x` is the step's bf16 `[M, K]` activation (unit inner stride, a row stride that is a
    multiple of 8, 16-byte aligned); one launch quantizes it per 32-element block along K:
    the scale is the block's amax / 448 rounded up to a power of two (so nothing
    saturates) and the values are `e4m3(x / scale)`. `pre = (xq, sfa)` is that quantization
    already written by `x`'s producer (a fused epilogue writing the same bytes): the quant
    launch is skipped and `x` is read for its row count only (it may then be `xq` itself,
    where no bf16 activation exists). `weight` is the
    `QuantizedWeight` of the row-major `[N, K]` bf16 weight (`quantize_weight`, at load).
    `out` is a contiguous `[M, N]` bf16, or e4m3 for the e4m3-out entries, written in place
    and returned.

        qkva      [M, 6144] x [2624, 6144]
        o_proj    [M, 4096] x [6144, 4096]
        q_b_e4m3  [M, 2048] x [4096, 2048]   e4m3 out (nearest, saturating)
        kv_b_k8   [M,  576] x [4096,  576]   A pre-quantized, e4m3 out
        kv_b_v8   [M,  512] x [4096,  512]   A pre-quantized, e4m3 out
        kv_b_kv8  [M,  576] x [8192,  576]   A pre-quantized, e4m3 out, rows >= 4096 on K < 512
        dense_gate_up  [M, 6144] x [6144, 6144]
        dense_down     [M, 3072] x [6144, 3072]
        mtp_eh         [M, 12288] x [6144, 12288]

    The pre-quantized entries take `x` as contiguous uint8 e4m3 bytes and `sfa` its blocked
    e8m0 scales (K padded to 128 for the layout; None means every block at 2^0).

    `kv_b_kv8` is `kv_b_k8` and `kv_b_v8` in one launch: its weight is the K weight stacked
    over the V weight zero-padded to 576 columns (`quantize_weight` of that `[8192, 576]`
    matrix), and the output rows hold K then V. Its V tiles issue only the k-blocks below 512,
    so with `x` / `sfa` the gather's `a_k` / `sfa_k` the output is the two launches' outputs
    (on `a_v` / `sfa_v`, the same bytes) side by side, bitwise.

    Relative error against the bf16 product is e4m3's (about 2^-4 per operand element,
    1e-2 .. 3e-2 relative RMS of a real layer's output); against the same MXF8 operands
    in fp32 it is the accumulation order only.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target's context calls: rows above the decode buckets up to max_num_tokens. The
    # cells cover each entry's two tiles (the small-T tile's last row count and the first
    # one past it) and the token budget; the producer-quantized and pre-quantized forms.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="qkva at 65 rows, the first context count: the 256x128 tile",
            spec=dict(kernel="qkva", rows=65),
        ),
        Cell(
            why="qkva at 7255 rows, a served context step: the 256x256 tile",
            spec=dict(kernel="qkva", rows=7255),
        ),
        Cell(
            why="qkva at 7255 rows on its producer's MXF8 operand (the closing all-reduce)",
            spec=dict(kernel="qkva", rows=7255, pre=True),
        ),
        Cell(
            why="o_proj at 1024 rows, the small tile's last count",
            spec=dict(kernel="o_proj", rows=1024),
        ),
        Cell(
            why="o_proj at 1025 rows, the large tile's first, a partial last tile",
            spec=dict(kernel="o_proj", rows=1025),
        ),
        Cell(
            why="o_proj at 8192 rows, the token budget",
            spec=dict(kernel="o_proj", rows=8192),
        ),
        Cell(
            why="o_proj at 3001 rows on its producer's MXF8 operand (the MHA's O epilogue)",
            spec=dict(kernel="o_proj", rows=3001, pre=True),
        ),
        Cell(
            why="q_b_e4m3 at 513 rows, e4m3 out on the large tile",
            spec=dict(kernel="q_b_e4m3", rows=513),
        ),
        Cell(
            why="q_b_e4m3 at 987 rows on the q_a tail of a 2624-wide row",
            spec=dict(kernel="q_b_e4m3", rows=987, x_row_stride=2624),
        ),
        Cell(
            why="q_b_e4m3 at 7255 rows on the q_a norm's MXF8 operand",
            spec=dict(kernel="q_b_e4m3", rows=7255, pre=True),
        ),
        Cell(
            why="kv_b_k8 at 1024 rows: K = 576 is not a multiple of the 128-element k-block",
            spec=dict(kernel="kv_b_k8", rows=1024),
        ),
        Cell(
            why="kv_b_k8 at 7300 rows with block scales other than 2^0",
            spec=dict(kernel="kv_b_k8", rows=7300, sfa=True),
        ),
        Cell(
            why="kv_b_v8 at 1025 rows with block scales, the large tile's first count",
            spec=dict(kernel="kv_b_v8", rows=1025, sfa=True),
        ),
        Cell(
            why="kv_b_kv8 at 1024 rows: K and V in one launch on the small tile",
            spec=dict(kernel="kv_b_kv8", rows=1024, sfa=True),
        ),
        Cell(
            why="kv_b_kv8 at 7300 rows with block scales: the large tile, V tiles on K < 512",
            spec=dict(kernel="kv_b_kv8", rows=7300, sfa=True),
        ),
        # The served form (`kv_strided`): `kv_b_kv8` on the gather's `a_k` / `sfa_k`, in the
        # target's row slices of the runner's capacity, its `[T_k, 8192]` rows read as two
        # halves by the `kv_strided` MHA handle. The test also pins these bitwise to the two
        # launches `kv_b_k8` + `kv_b_v8` on `a_k` / `a_v`.
        Cell(
            why="kv_strided: 513 K/V rows, the first context row count past the decode buckets",
            spec=dict(kernel="kv_b_kv8", rows=513, sfa=True, kv_strided=True),
        ),
        Cell(
            why="kv_strided: 7340 K/V rows, a served context step ending mid-tile",
            spec=dict(kernel="kv_b_kv8", rows=7340, sfa=True, kv_strided=True),
        ),
        Cell(
            why="kv_strided: 8192 K/V rows, the token budget with no cached prefix",
            spec=dict(kernel="kv_b_kv8", rows=8192, sfa=True, kv_strided=True),
        ),
        Cell(
            why="kv_strided: 16448 K/V rows (a cached prefix), two row slices of 16384 + 64",
            spec=dict(kernel="kv_b_kv8", rows=16448, sfa=True, kv_strided=True),
        ),
        Cell(
            why="dense_gate_up at 65 rows through its own activation quant",
            spec=dict(kernel="dense_gate_up", rows=65),
        ),
        Cell(
            why="dense_gate_up at 7310 rows on the pre-MLP all-reduce's MXF8 operand",
            spec=dict(kernel="dense_gate_up", rows=7310, pre=True),
        ),
        Cell(
            why="dense_down at 7310 rows on the SwiGLU's operand, no bf16 activation",
            spec=dict(kernel="dense_down", rows=7310, pre=True, x_is_pre=True),
        ),
        Cell(
            why="mtp_eh at 65 rows through its own activation quant",
            spec=dict(kernel="mtp_eh", rows=65),
        ),
        Cell(
            why="mtp_eh at 7310 rows on the input norms' operand, no bf16 concat",
            spec=dict(kernel="mtp_eh", rows=7310, pre=True, x_is_pre=True),
        ),
    )

    note = """
    The target quantizes the trunk layers' qkva and o_proj weights once at load and,
    when the fp8 context attention is built, the q_b weight copy with that attention's Q
    scale folded in (its bf16 copy is then not kept) and the folded K / V expansion
    weights, and the dense layers' gate|up and down weights. Context-path calls of those
    projections (rows above the decode buckets:
    65..8192) take this op; a mixed step's generation rows ride the context rows' launch.
    Where the activation's producer can write the MXF8 operand -- the closing two-shot
    all-reduce for qkva, the q_a norm for q_b, the fp8 MHA's O epilogue for o_proj -- the
    call passes it as `pre`. Decode cells (rows <= 64, the CUDA graphs), the MTP layer and
    the absorbed path's q_b stay bf16. The dense down reads the operand
    `cute_dsl_swiglu_mxf8` writes, passed as both `x` and `pre`. If an entry fails to build, the target keeps the
    bf16 prefill GEMM for it.

    The launcher checks every call's shapes and dtypes on the host; `is_valid` states the
    operand contract.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        out: torch.Tensor,
        runner: Mxf8ProjectionRunner,
        pre: tuple[torch.Tensor, torch.Tensor] | None = None,
        sfa: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dst = out.view(torch.uint8) if out.dtype == torch.float8_e4m3fn else out
        if runner.prequant:
            runner.launch(x, weight, dst, sfa)
        else:
            runner.launch(x, weight, dst, pre=pre)
        return out

    def reference(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        out: torch.Tensor,
        runner: Mxf8ProjectionRunner,
        pre: tuple[torch.Tensor, torch.Tensor] | None = None,
        sfa: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The product of the same MXF8 operands in fp32 (`x` quantized by the kernel's
        torch twin, or `pre` / the pre-quantized `x` and `sfa` as given; `weight`
        dequantized), rounded once to `out`'s dtype (e4m3 saturating), as a new tensor; `out`
        is not written."""
        split = _kernel.ABIS[ENTRIES[runner.kernel]].n_split_k
        if pre is not None:
            xq, sf = pre
            res = _kernel.emulate(xq, weight, runner.k, sfa=sf, n_split_k=split)
        else:
            res = _kernel.emulate(x, weight, runner.k, sfa=sfa, n_split_k=split)
        if out.dtype == torch.float8_e4m3fn:
            res = res.clamp(-448.0, 448.0)
        return res.to(out.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: QuantizedWeight,
        out: torch.Tensor,
        runner: Mxf8ProjectionRunner,
        pre: tuple[torch.Tensor, torch.Tensor] | None = None,
        sfa: torch.Tensor | None = None,
    ) -> None:
        rows = int(x.shape[0])
        assert x.dim() == 2 and int(x.shape[1]) == runner.k, (
            f"x must be [M, {runner.k}], got {tuple(x.shape)}"
        )
        if runner.prequant:
            assert x.dtype == torch.uint8 and x.is_contiguous(), (
                f"{runner.kernel} takes its A operand as contiguous uint8 e4m3 bytes"
            )
            assert pre is None, f"{runner.kernel} has no activation quant to skip"
            if sfa is not None:
                need = _kernel.unit_sf_bytes(rows, runner.k)
                assert sfa.dtype == torch.uint8 and sfa.is_contiguous() and sfa.numel() >= need, (
                    f"sfa must be contiguous uint8 of at least {need} bytes"
                )
        else:
            assert sfa is None, f"{runner.kernel} quantizes its own A operand"
            if pre is None or x is not pre[0]:
                assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
                assert x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0, (
                    "x needs a unit inner stride, a row stride that is a multiple of 8 and "
                    f"16-byte alignment, got {tuple(x.stride())}"
                )
            if pre is not None:
                xq, sf = pre
                need = _kernel._quant.sf_bytes(rows, runner.k)
                assert (
                    xq.dtype == torch.uint8
                    and tuple(xq.shape) == (rows, runner.k)
                    and xq.is_contiguous()
                    and sf.dtype == torch.uint8
                    and sf.is_contiguous()
                    and sf.numel() >= need
                ), f"pre must be uint8 [{rows}, {runner.k}] and at least {need} scale bytes"
        assert tuple(weight.q.shape) == (runner.n, runner.k), (
            f"weight must be the MXF8 [N, K] = [{runner.n}, {runner.k}], "
            f"got {tuple(weight.q.shape)}"
        )
        split = _kernel.ABIS[ENTRIES[runner.kernel]].n_split_k
        if split is not None:
            # The split tiles never read these weight bytes: the contract is that they are zero.
            assert not bool(weight.q[split[0] :, split[1] :].any()), (
                f"{runner.kernel}: weight rows >= {split[0]} must be zero at columns >= {split[1]}"
            )
        assert out.dtype == runner.out_dtype, f"{runner.kernel} writes {runner.out_dtype}"
        assert tuple(out.shape) == (rows, runner.n) and out.is_contiguous(), (
            f"out must be a contiguous [{rows}, {runner.n}], got {tuple(out.shape)}"
        )


cute_dsl_projection_mxf8 = CuteDslProjectionMxf8()
