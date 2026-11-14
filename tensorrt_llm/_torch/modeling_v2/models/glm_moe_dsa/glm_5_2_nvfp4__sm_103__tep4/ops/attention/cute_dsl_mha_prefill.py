# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context step's attention in the non-absorbed MLA form: causal varlen MHA, d = 256,
BF16 or E4M3 operands, one CuTe DSL launch."""

from __future__ import annotations

import math

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import self_attn_prefill_mha_varlen_h16_d256_bf16 as _kernel

NUM_HEADS = _kernel.NUM_HEADS
#: QK and V head dims: nope 192 + rope 64, and v 256.
HEAD_DIM = _kernel.HEAD_DIM
#: 1 / sqrt(192 + 64), the served MLA's softmax scale.
SOFTMAX_SCALE = _kernel.SOFTMAX_SCALE

CompiledMhaPrefill = _kernel.PrefillMha
#: The fp8 variant's per-head factor table `[2, 16]` fp32 from the dequantization scales
#: `(s_q, s_k[16], s_v[16])`: row 0 the softmax exp2 scale, row 1 the output scale.
fp8_head_scales = _kernel.fp8_head_scales


#: The fp8 handles' kernel options: P converted to e4m3 four values per register, and on the
#: MXF8-O handle an epilogue that holds a whole O chunk in registers, hands it back to the MMA,
#: then rounds and block-scales it with packed converts; and tail-first Q tiling, each
#: request's partial 256-row tile pair first (its causal kv loop the shortest, the others' up
#: to the shift shorter); on the MXF8-O handle only (the bf16-O launcher refuses it), the
#: epilogue offloaded to four added warps that block-scale and store O while the correction
#: warps serve the next tile (-0.4 .. -0.55 ms per context step on B300). Same output bytes
#: as without them.
_FP8_VARIANT = {"cvt4": True, "tail_first": True}
_Q8_VARIANT = {"cvt4": True, "epi_fast": True, "tail_first": True, "epi_offload": True}


def sf_bytes(rows: int) -> int:
    """Bytes of the blocked e8m0 scales of an MXF8 O operand of `rows` rows."""
    return _kernel._mxf8.sf_bytes(rows, NUM_HEADS * HEAD_DIM)


def compile(  # noqa: A001
    device_index: int,
    fp8: bool = False,
    q8: bool = False,
    epi_offload: bool = True,
    kv_strided: bool = False,
) -> CompiledMhaPrefill:
    """Compile the kernel for `device_index` and return the handle the op takes as
    `compiled`. Call once at load. `fp8=True` builds the e4m3 variant: q / k / v e4m3,
    out bf16, and a `head_scales` table (`fp8_head_scales`) on every call. `q8=True` (fp8
    only) builds the variant whose epilogue writes O as the o_proj GEMM's MXF8 operand
    (`q8_out`) instead of `out`: o_proj is the only reader of the context rows' attention.
    `epi_offload=False` builds that variant without the offloaded epilogue (same bytes).
    `kv_strided=True` builds the handle that reads `k` / `v` at their own token-row strides
    (the two halves of the one-launch K | V expansion's `[T_k, 8192]` rows,
    `cute_dsl_projection_mxf8` `kv_b_kv8`) instead of packed rows; same output bytes.

    One compile serves every step: query and K/V row counts, request count and the
    longest query are runtime values. Costs one CuTe DSL compile, then one launch in
    each form the target calls -- K/V rows equal to the query rows, and K/V rows past
    them (a cached prefix) -- so no context step meets a first module load.
    """
    if q8 and not fp8:
        raise ValueError("the MXF8-O epilogue is an fp8-handle option")
    strided = {"kv_strided": True} if kv_strided else {}
    if q8:
        launcher = _kernel.build(
            int(device_index),
            fp8=True,
            q8=True,
            q8_store_o=False,
            variant={**_Q8_VARIANT, "epi_offload": bool(epi_offload), **strided},
        )
    else:
        launcher = _kernel.build(
            int(device_index),
            fp8=bool(fp8),
            variant={**(_FP8_VARIANT if fp8 else {}), **strided} or None,
        )
    dev = torch.device("cuda", int(device_index))
    shape = (NUM_HEADS, HEAD_DIM)
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    hs = fp8_head_scales(1.0, torch.ones(NUM_HEADS), torch.ones(NUM_HEADS)).to(dev) if fp8 else None
    with torch.cuda.device(dev):
        for q_rows, kv_rows in ((1, 1), (1, 2)):
            q = torch.zeros((q_rows, *shape), dtype=dtype, device=dev)
            k = torch.zeros((kv_rows, *shape), dtype=dtype, device=dev)
            v = torch.zeros((kv_rows, *shape), dtype=dtype, device=dev)
            if kv_strided:
                kv = torch.zeros((kv_rows, 2, *shape), dtype=dtype, device=dev)
                k, v = kv[:, 0], kv[:, 1]
            out = torch.empty((q_rows, *shape), dtype=torch.bfloat16, device=dev)
            cu = torch.tensor([0, q_rows], dtype=torch.int32, device=dev)
            cu_k = None
            if kv_rows != q_rows:
                cu_k = torch.tensor([0, kv_rows], dtype=torch.int32, device=dev)
            q8_out = None
            if q8:
                q8_out = (
                    torch.empty(q_rows, NUM_HEADS * HEAD_DIM, dtype=torch.uint8, device=dev),
                    torch.empty(sf_bytes(q_rows), dtype=torch.uint8, device=dev),
                )
            launcher(
                q, k, v, out, cu, q_rows, validate=True, cu_seqlens_k=cu_k, head_scales=hs,
                q8_out=q8_out,
            )  # fmt: skip
        torch.cuda.synchronize(dev)
    return launcher


class CuteDslMhaPrefill(OpWrapper):
    """Causal attention per request over packed token-major rows: `out[t] = softmax(
    q[t] k^T / 16) v` over the request's K/V rows the causal mask admits.

    `q` and `out` are contiguous `[T_q, 16, 256]` bf16 (query rows of the context
    requests, back to back); `k` and `v` contiguous `[T_k, 16, 256]` bf16 (the
    `kv_strided` handle: any token-row stride that is a multiple of 256 elements).
    `cu_seqlens` (int32 `[B + 1]`) delimits each request's query rows and
    `cu_seqlens_k` (int32 `[B + 1]`, default `cu_seqlens`) its K/V rows: a cached
    prefix first, then the rows of this step, so `kv_len >= q_len`. Query `i` of a
    request attends K/V positions `<= i + kv_len - q_len` (bottom-right aligned).
    `max_seqlen` is the longest request's query row count; it sizes the grid.

    fp32 QK and PV accumulation, bf16 P, exp2 softmax with the scale folded in, and
    an FA4-style conditional rescale (the reference max moves only when a tile raises
    the row max by more than 2^8): exact after the final 1 / row_sum, not bitwise
    against an exact-max softmax. Returns `out`.

    The fp8 handle (`compile(fp8=True)`): `q`, `k`, `v` e4m3 holding `q / s_q`,
    `k / s_k[h]`, `v / s_v[h]` (power-of-two scales the caller folds into the GEMMs that
    write them), `out` bf16, and `head_scales = fp8_head_scales(s_q, s_k, s_v)`. Both
    products run in e4m3 with fp32 accumulation; P is e4m3, and the conditional rescale
    threshold is 0.75 so P stays below e4m3's 448.

    The MXF8-O handle (`compile(fp8=True, q8=True)`) does not write `out`; it writes O as
    `q8_out = (q8 [T_o >= T_q, 4096] uint8, sf)`, the activation quant's exact bytes of the
    bf16 O the fp8 handle writes (per 32-element block a power-of-two scale and e4m3 values;
    rows `< T_q` written).
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target's context call: 16 heads, d 256. The cells vary what a step varies:
    # request lengths (a partial last 128-row tile, requests shorter than a tile, a
    # 2048-row one), several requests in one launch, and cached prefixes of block-reuse
    # steps, which make K/V longer than the query rows.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request of 987 rows, no cached prefix: a partial last 128-row tile",
            spec=dict(new=[987], cached=[0]),
        ),
        Cell(
            why="eight requests of 987 rows, the shape of a served context step",
            spec=dict(new=[987] * 8, cached=[0] * 8),
        ),
        Cell(
            why="one-row and seven-row requests next to a 2048-row one: tile pairs past "
            "a request's end are skipped",
            spec=dict(new=[1, 7, 2048], cached=[0, 0, 0]),
        ),
        Cell(
            why="cached prefixes of 17, 64 and 300 rows: K/V past the query rows, "
            "bottom-right causal",
            spec=dict(new=[64, 300, 1], cached=[17, 64, 300]),
        ),
        Cell(
            why="a one-row prefix next to an uncached request, both mid-tile",
            spec=dict(new=[129, 383], cached=[1, 0]),
        ),
        Cell(
            why="fp8: eight requests of 987 rows, the served context step's shape",
            spec=dict(new=[987] * 8, cached=[0] * 8, fp8=True),
        ),
        Cell(
            why="fp8: cached prefixes of 17, 64 and 300 rows next to a one-row request",
            spec=dict(new=[64, 300, 1], cached=[17, 64, 300], fp8=True),
        ),
        Cell(
            why="MXF8-O: eight requests of 987 rows written as o_proj's operand",
            spec=dict(new=[987] * 8, cached=[0] * 8, fp8=True, q8=True),
        ),
        Cell(
            why="MXF8-O: cached prefixes, and generation rows after the context rows",
            spec=dict(new=[64, 300, 1], cached=[17, 64, 300], fp8=True, q8=True, tail=16),
        ),
        # The served handles (`kv_strided`): K and V are the two halves of the one-launch
        # expansion's `[T_k, 8192]` rows (`cute_dsl_projection_mxf8` `kv_b_kv8`). The test
        # also pins each to the packed handle on the same K / V made contiguous, bitwise.
        Cell(
            why="kv_strided MXF8-O: one request of 513 rows, the first context row count",
            spec=dict(new=[513], cached=[0], fp8=True, q8=True, kv_strided=True),
        ),
        Cell(
            why="kv_strided MXF8-O: 7340 rows in eight requests, one shorter, no prefix",
            spec=dict(new=[987] * 7 + [431], cached=[0] * 8, fp8=True, q8=True, kv_strided=True),
        ),
        Cell(
            why="kv_strided MXF8-O: eight 1024-row requests over 1032-row cached prefixes "
            "(16448 K/V rows), and generation rows after the context rows",
            spec=dict(
                new=[1024] * 8, cached=[1032] * 8, fp8=True, q8=True, kv_strided=True, tail=16
            ),
        ),
        Cell(
            why="kv_strided bf16 O: four requests of 1032 rows",
            spec=dict(new=[1032] * 4, cached=[0] * 4, fp8=True, kv_strided=True),
        ),
        Cell(
            why="kv_strided bf16 O: cached prefixes of 17, 64 and 300 rows",
            spec=dict(new=[64, 300, 1], cached=[17, 64, 300], fp8=True, kv_strided=True),
        ),
    )

    note = """
    The target compiles one handle at load and takes this op on a dense-causal
    context step (every context request within index_topk, so the indexer
    selects the whole causal window) for the trunk layers: q is the q_b
    projection's output with rope(q_pe) written in place by the context rope
    kernel's `q_in_place` mode, and K/V come from the prefill GEMM's kv_b
    expansion entries. The MTP layer and sparse steps keep the absorbed path.
    When the fp8 handle and the e4m3 GEMM entries build, the trunk layers take the
    fp8 form: q from the q_b GEMM's e4m3 entry (rope in place on e4m3), K/V from the
    e4m3 expansion entries, with the scales folded into those weights at load.

    The kernel checks dtypes, shapes, contiguity and alignment on the host when
    called with `validate=True` (the target's warm-up does), not per serving
    call. What it never checks is in `is_valid`: prefix sums that do not
    delimit the rows, a `max_seqlen` short of the longest request (its tail
    rows would not be computed), and a request with fewer K/V rows than query
    rows.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        compiled: CompiledMhaPrefill,
        cu_seqlens_k: torch.Tensor | None = None,
        head_scales: torch.Tensor | None = None,
        q8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        compiled(
            q, k, v, out, cu_seqlens, max_seqlen, cu_seqlens_k=cu_seqlens_k,
            head_scales=head_scales, q8_out=q8_out,
        )  # fmt: skip
        return out

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        compiled: CompiledMhaPrefill,
        cu_seqlens_k: torch.Tensor | None = None,
        head_scales: torch.Tensor | None = None,
        q8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The attention per request in fp32, rounded once to `out`'s dtype, as a new
        tensor; `out` is not written and `compiled` is ignored. With `head_scales`, on
        the dequantized e4m3 operands: `s_q * s_k[h]` (row 0 over log2(e) / 16) folded
        into q and `s_v[h]` (row 1) into v. For the MXF8-O handle this is the bf16 O whose
        activation quant `q8_out` holds."""
        if head_scales is not None:
            hs = head_scales.float()
            qk = hs[0] / (math.log2(math.e) * SOFTMAX_SCALE)
            q = q.float() * qk.view(1, -1, 1)
            k = k.float()
            v = v.float() * hs[1].view(1, -1, 1)
        return _kernel.reference(q, k, v, cu_seqlens, cu_seqlens_k).to(out.dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        compiled: CompiledMhaPrefill,
        cu_seqlens_k: torch.Tensor | None = None,
        head_scales: torch.Tensor | None = None,
        q8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Reads the prefix sums back to the host: never on the serving path."""
        try:
            compiled.check(q, k, v, out, cu_seqlens, cu_seqlens_k, head_scales)
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc
        assert (q8_out is not None) == bool(compiled.q8), (
            "q8_out is the MXF8-O handle's output, and that handle's only one"
        )
        if q8_out is not None:
            q8, sf = q8_out
            assert (
                q8.dtype == torch.uint8 and q8.dim() == 2 and q8.is_contiguous()
                and int(q8.shape[1]) == NUM_HEADS * HEAD_DIM and q8.shape[0] >= q.shape[0]
                and q8.data_ptr() % 16 == 0
            ), f"q8 must be a contiguous uint8 [T_o >= {q.shape[0]}, 4096], 16-byte aligned"  # fmt: skip
            assert sf.dtype == torch.uint8 and sf.numel() >= sf_bytes(int(q8.shape[0])), (
                f"sf must hold at least {sf_bytes(int(q8.shape[0]))} scale bytes"
            )
        cu = cu_seqlens.cpu().long()
        cu_k = cu if cu_seqlens_k is None else cu_seqlens_k.cpu().long()
        q_len, kv_len = cu[1:] - cu[:-1], cu_k[1:] - cu_k[:-1]
        assert int(cu[0]) == 0 and int(cu[-1]) == q.shape[0] and bool((q_len >= 0).all()), (
            f"cu_seqlens must rise from 0 to the {q.shape[0]} query rows, got {cu.tolist()}"
        )
        assert int(cu_k[0]) == 0 and int(cu_k[-1]) == k.shape[0], (
            f"cu_seqlens_k must rise from 0 to the {k.shape[0]} K/V rows, got {cu_k.tolist()}"
        )
        assert bool((kv_len >= q_len).all()), (
            "a request has fewer K/V rows than query rows; the cached prefix comes first"
        )
        assert int(max_seqlen) >= int(q_len.max()), (
            f"max_seqlen {max_seqlen} is short of the longest request ({int(q_len.max())} rows)"
        )


cute_dsl_mha_prefill = CuteDslMhaPrefill()
