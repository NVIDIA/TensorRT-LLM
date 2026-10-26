# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeoX RoPE from a precomputed fp32 cos|sin table, in place on q and k."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: Head widths FlashInfer serves with its per-head-dim kernels
#: (`BatchQKApplyRotaryPosIdsCosSinCacheKernel` and its head-parallel twin).
#: Every other width is routed to `RopeQuantizeKernel`.
_HEAD_DIM_KERNELS = (64, 128, 256, 512)

#: `RopeQuantizeKernel` gives each lane one 32-byte vector: 16 elements of a
#: 16-bit dtype.
_ROPE_QUANTIZE_VEC_BYTES = 32

#: Step-5's `max_seq_len`: tables span positions 0..9215.
TABLE_ROWS = 9216


def _cell(why: str, **spec) -> Cell:
    return Cell(why=why, spec=dict(spec))


def _main_full(**kw) -> dict:
    # full_attention layers: partial rotary 192 / 3 = 64 dims (32 pairs),
    # theta 1e7 with the llama3 block at factor 1.0, on the 192-wide head.
    return dict(head_size=192, width=192, rotary_dim=64, theta=1e7, llama3=True, **kw)


def _main_sliding(**kw) -> dict:
    # sliding_attention layers: full rotary 192 (96 pairs), theta 1e4, on the
    # 192-wide head.
    return dict(head_size=192, width=192, rotary_dim=192, theta=1e4, llama3=False, **kw)


def _main_sliding_padded(**kw) -> dict:
    # sliding_attention layers: full rotary 192 (96 pairs), theta 1e4, on heads
    # already zero-padded to the 256-wide attention layout.
    return dict(head_size=256, width=256, live=192, rotary_dim=192, theta=1e4, llama3=False, **kw)


def _main_full_padded(**kw) -> dict:
    return dict(head_size=256, width=256, live=192, rotary_dim=64, theta=1e7, llama3=True, **kw)


def _indexer(**kw) -> dict:
    # Sparse-layer indexer: proxy_dim 256, leading 32 dims (16 pairs), the
    # layer's theta 1e7 with its llama3 block; one k head.
    return dict(
        head_size=256, width=256, rotary_dim=32, theta=1e7, llama3=True, layout="split", **kw
    )


class FlashinferApplyRopeWithCosSinCache(OpWrapper):
    """Rotate the leading `rotary_dim` dims of every q and k head, NeoX-paired,
    in place. Returns `(query, key)`, the same tensors, mutated.

    `rotary_dim = cos_sin_cache.shape[1]`; row `p` of the table is
    `[cos(p * f_0..f_{R/2-1}) | sin(p * f_0..f_{R/2-1})]`. For token `t` at
    position `p = positions[t]` and each head, with `h = rotary_dim // 2`:
    `x[d] <- x[d] * cos_d - x[d + h] * sin_d` and
    `x[d + h] <- x[d + h] * cos_d + x[d] * sin_d` for `d < h`; dims at and
    past `rotary_dim` are not written. `query` is `[T, Hq * head_size]` (or any
    view `.view(T, -1, head_size)` accepts), `key` `[T, Hk * head_size]`.

    Wraps `torch.ops.trtllm.flashinfer_apply_rope_with_cos_sin_cache_inplace`
    (`custom_ops/flashinfer_custom_ops.py`), which forwards to FlashInfer.
    Head widths 64/128/256/512 take one in-place launch of
    `BatchQKApplyRotaryPosIdsCosSinCache{,HeadParallelism}Kernel`; every other
    width takes `RopeQuantizeKernel`, in place when a token's lane count
    `ceil(rotary_dim / 16)` divides 32, and otherwise out of place through
    FlashInfer's `apply_rope_with_cos_sin_cache` with the results copied back
    into `query` and `key` (`_rope_quantize_lanes_straddle_warps` decides).
    The rotation is evaluated in fp32 and rounded once to the input dtype.

    Fusion boundary: the rotation only. The table, the QK norm before it and
    the padding, KV append and attention after it are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: head_size (the op's argument), width (each head's stride in the
    # buffer), live (dims holding data, the rest zero: 192 in the padded
    # 256-wide layout; width when absent), rotary_dim, theta,
    # llama3 (Step-5's factor-1.0 llama3 block applied to the frequencies),
    # hq, hk, tokens, positions ("prefill" 0..T-1, "decode" spread over
    # 0..9215 with both ends, "chunk" a 512-token chunk at 8704..9215),
    # layout ("packed": q and k are column views of one [T, (hq + 2 hk) *
    # width] qkv buffer; "split": two contiguous tensors), table ("fp32": cos
    # and sin as fp32 computes them; "bf16": the same rounded to bf16 and
    # widened back, the values Step-5's reference rotates with).
    CELLS: tuple[Cell, ...] = (
        # full layers on the 192-wide heads (RopeQuantizeKernel, 4 lanes/token)
        _cell(
            "full dp 64/4 h192 r64: prefill 8192, envelope's largest row",
            **_main_full(
                hq=64, hk=4, tokens=8192, positions="prefill", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "full dp 64/4 h192 r64: decode 128, captured batch",
            **_main_full(
                hq=64, hk=4, tokens=128, positions="decode", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "full dp 64/4 h192 r64: decode 1 at the last row",
            **_main_full(hq=64, hk=4, tokens=1, positions="decode", layout="packed", table="fp32"),
        ),
        _cell(
            "full dp 64/4 h192 r64: chunk 512 at 8704, split",
            **_main_full(hq=64, hk=4, tokens=512, positions="chunk", layout="split", table="fp32"),
        ),
        _cell(
            "full tp4 16/1 h192 r64: prefill 8192",
            **_main_full(
                hq=16, hk=1, tokens=8192, positions="prefill", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "full tp4 16/1 h192 r64: decode 128, split",
            **_main_full(hq=16, hk=1, tokens=128, positions="decode", layout="split", table="fp32"),
        ),
        _cell(
            "full tp4 16/1 h192 r64: decode 1",
            **_main_full(hq=16, hk=1, tokens=1, positions="decode", layout="packed", table="fp32"),
        ),
        _cell(
            "full dp h192 r64 bf16-valued table: 8192",
            **_main_full(
                hq=64, hk=4, tokens=8192, positions="prefill", layout="packed", table="bf16"
            ),
        ),
        # sliding layers on the 192-wide heads (RopeQuantizeKernel at 12
        # lanes/token, which the trtllm op runs out of place)
        _cell(
            "sliding dp 64/4 h192 r192: prefill 8192",
            **_main_sliding(
                hq=64, hk=4, tokens=8192, positions="prefill", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "sliding dp 64/4 h192 r192: decode 128",
            **_main_sliding(
                hq=64, hk=4, tokens=128, positions="decode", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "sliding tp4 16/1 h192 r192: prefill 8192",
            **_main_sliding(
                hq=16, hk=1, tokens=8192, positions="prefill", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "sliding tp4 16/1 h192 r192: chunk 512 split",
            **_main_sliding(
                hq=16, hk=1, tokens=512, positions="chunk", layout="split", table="fp32"
            ),
        ),
        # sliding layers on heads zero-padded to 256 (per-head-dim kernel)
        _cell(
            "sliding dp 64/4 pad256 r192: prefill 8192",
            **_main_sliding_padded(
                hq=64, hk=4, tokens=8192, positions="prefill", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "sliding dp 64/4 pad256 r192: decode 128",
            **_main_sliding_padded(
                hq=64, hk=4, tokens=128, positions="decode", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "sliding dp 64/4 pad256 r192: decode 1",
            **_main_sliding_padded(
                hq=64, hk=4, tokens=1, positions="decode", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "sliding tp4 16/1 pad256 r192: prefill 8192",
            **_main_sliding_padded(
                hq=16, hk=1, tokens=8192, positions="prefill", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "sliding tp4 16/1 pad256 r192: chunk 512 split",
            **_main_sliding_padded(
                hq=16, hk=1, tokens=512, positions="chunk", layout="split", table="fp32"
            ),
        ),
        _cell(
            "sliding tp4 16/1 pad256 r192: decode 128",
            **_main_sliding_padded(
                hq=16, hk=1, tokens=128, positions="decode", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "sliding dp pad256 r192 bf16-valued table: 8192",
            **_main_sliding_padded(
                hq=64, hk=4, tokens=8192, positions="prefill", layout="packed", table="bf16"
            ),
        ),
        # full layers on the padded heads
        _cell(
            "full dp 64/4 pad256 r64: prefill 8192",
            **_main_full_padded(
                hq=64, hk=4, tokens=8192, positions="prefill", layout="packed", table="fp32"
            ),
        ),
        _cell(
            "full tp4 16/1 pad256 r64: decode 128",
            **_main_full_padded(
                hq=16, hk=1, tokens=128, positions="decode", layout="packed", table="fp32"
            ),
        ),
        # sparse-layer indexer, q (Hi heads) and k (1 head) at proxy_dim 256
        _cell(
            "indexer dp 16/1 h256 r32: prefill 8192",
            **_indexer(hq=16, hk=1, tokens=8192, positions="prefill", table="fp32"),
        ),
        _cell(
            "indexer dp 16/1 h256 r32: decode 128",
            **_indexer(hq=16, hk=1, tokens=128, positions="decode", table="fp32"),
        ),
        _cell(
            "indexer dp 16/1 h256 r32: decode 1",
            **_indexer(hq=16, hk=1, tokens=1, positions="decode", table="fp32"),
        ),
        _cell(
            "indexer dp 16/1 h256 r32: chunk 512 at 8704",
            **_indexer(hq=16, hk=1, tokens=512, positions="chunk", table="fp32"),
        ),
        _cell(
            "indexer tp4 4/1 h256 r32: prefill 8192",
            **_indexer(hq=4, hk=1, tokens=8192, positions="prefill", table="fp32"),
        ),
        _cell(
            "indexer tp4 4/1 h256 r32: decode 128",
            **_indexer(hq=4, hk=1, tokens=128, positions="decode", table="fp32"),
        ),
        _cell(
            "indexer dp h256 r32 bf16-valued table: 8192",
            **_indexer(hq=16, hk=1, tokens=8192, positions="prefill", table="bf16"),
        ),
    )

    note = """
    Metadata consumed: `positions` (int32 or int64, contiguous, one per token;
    the cells pass int32, the engine's position dtype) indexes rows of
    `cos_sin_cache` (fp32, contiguous, `[rows, rotary_dim]`, cos half first).
    In place: `query` and `key` are both read and written; the views keep
    their storage, so a q/k slice of a packed qkv buffer is rotated where it
    lies (the v columns and every dim at or past `rotary_dim` are asserted
    bit-unchanged per cell, the zero pad dims of the 256-wide layout included).

    Kernel route on head widths other than 64/128/256/512: FlashInfer's
    `RopeQuantizeKernel`, one 16-element vector per lane. The trtllm op runs
    it in place when a token's lane count ceil(rotary_dim / 16) divides 32
    (the full layers' r64: 4 lanes), and otherwise out of place followed by
    a copy back into q and k (the sliding layers' r192: 12 lanes; also r96
    and r160): one op call, three launches, and two temporaries the size of
    q and k allocated per call. Repeats of the out-of-place route are
    bit-equal to each other and to FlashInfer's `apply_rope_with_cos_sin_cache`
    on the same inputs. Rotary dims whose half is not a whole 16-element
    vector (16, 48, 80, 112, 144, 176 on 192-wide heads) are wrong through
    either route, with no error: guarded. Heads zero-padded to 256 take the
    per-head-dim kernel (one warp per token, in place), right at every
    multiple of 16 up to 256; the padded cells certify that call form.

    Table values: Step-5's reference builds cos/sin in fp32 and stores them
    in bf16, at most 2^-8 relative (one bf16 ulp) from the fp32 values. A
    caller that passes the bf16-rounded values widened to fp32 gets results
    bit-equal to the once-rounded exact rotation of those values (the
    bf16-valued cells: 0 elements off). Passing the fp32-built table instead
    moves the result off Step-5's arithmetic: at 128 decode rows, 6.7e-2 of
    elements (full r64), 2.6e-2 (indexer r32) and 2.1e-1 (sliding r192),
    by at most 2 bf16 ulps of the head's largest element. Which table a
    target passes is its choice; the entry is right for either, and the
    band tells the two apart. The `llama3` spec key applies Step-5's
    factor-1.0 block to the frequencies: it is not bit-identical to the plain
    frequencies (one of r64's 32 moves by an fp32 ulp, 9.8e-8 relative).

    Guards (`is_valid`, each observed answering wrongly with no error):
    on head widths other than 64/128/256/512, a rotary_dim whose half is not
    a whole 32-byte vector (above); positions at or past the table's rows
    (reads the next rows of whatever storage holds the table); fewer positions than
    tokens (reads past the end of `positions`); a key dtype other than the
    query's (an fp16 key under a bf16 query is rotated as bf16 bits: max
    error 2.75 in the guard test). The range check is skipped while the
    stream is capturing a CUDA graph, where it cannot read positions back.

    Loud rejections (measured, not guarded): bf16 table (ValueError
    "cos_sin_cache should be float32"); non-contiguous table or positions;
    fp32 q/k; int16 positions; rotary_dim > head_size; a key whose width is
    not a multiple of head_size; q/k token counts that differ; q/k with a
    non-unit last stride. A negative position faults the context
    ("operation not supported on global/shared address space" or an illegal
    address): the kernel forms `pos * rotary_dim` in 32-bit unsigned.

    Measured, not certified: int64 positions (bit-equal to int32); fp16 q/k
    (max abs error 9.7e-4 on unit-normal input); a 192-wide view of the
    256-wide padded buffer passed with head_size 192 at r64 (within the band,
    pad dims untouched).

    Kernel selection (torch.profiler, sm_103): head 192 at r64 runs
    `RopeQuantizeKernel<false, 16, 1, bf16, int, bf16>`; head 192 at r192
    runs the same kernel out of place, then two `direct_copy_kernel_cuda`
    copies; head 256 runs `BatchQKApplyRotaryPosIdsCosSinCacheKernel` when
    rows / 4 reach the device's resident-CTA count (8192 rows) and the
    head-parallel kernel below it (1 and 128 rows). Cost per call under CUDA
    graph replay on sm_103, dp 64/4 (tp4 16/1): sliding h192 r192 10.32
    (7.70) us at 128 rows and 334.56 (81.17) us at 8192; the same rotation on
    heads padded to 256 3.44 (2.11) us and 122.94 (21.57) us; full h192 r64
    2.92 (2.31) us and 104.70 (18.19) us. The padded form is the cheaper
    sliding call.

    CUDA graphs: capture and replay at 128 decode rows (h192 r64, h192
    r192, pad256 r192, h256 r32) is bitwise equal to eager after the
    positions and q/k are refilled; the out-of-place route's temporaries are
    allocated from the graph's pool at capture. The first call JIT-loads
    FlashInfer's rope module, so warm up before capture.

    Hard limits
      positions x rotary_dim  32-bit unsigned product in the kernel: wraps
      at pos >= 2^32 / rotary_dim (22.4M at r192, 134M at r32); the table
      has 9216 rows.
      positions through fp32  the caller's table builder: exact to 2^24;
      envelope reaches 9215.
      q/k element offsets  size_t in the kernel; envelope's largest buffer
      8192 x 72 x 256 = 1.5e8 elements, below 2^31 anyway.

    Not certified: rotary dims other than the cells' on 192-wide heads
    (r96 and r160 are measured right and deterministic at 128 decode rows,
    r32 and r128 at 8192 prefill, none a cell); is_neox False; fp16 q/k;
    int64 positions; token counts other than 1, 128, 512, 8192; positions
    outside the cells' ranges; tables of other row counts; torch.compile;
    sm_100.
    """

    def __call__(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.ops.trtllm.flashinfer_apply_rope_with_cos_sin_cache_inplace(
            positions, query, key, head_size, cos_sin_cache, is_neox
        )
        return query, key

    def reference(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The rotation in fp64, rounded once to the input dtype; new
        tensors shaped like `query` and `key`, which are not modified.

        Products of a 16-bit input and an fp32 table entry are exact in fp64,
        so the only rounding is the final one. NeoX pairing only: the cells
        never pass `is_neox=False`.
        """
        assert is_neox, "reference covers NeoX pairing only"
        rotary_dim = cos_sin_cache.shape[1]
        half = rotary_dim // 2
        rows = cos_sin_cache.double()[positions[: query.shape[0]].long()]
        cos = rows[:, None, :half]
        sin = rows[:, None, half:rotary_dim]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            heads = x.reshape(x.shape[0], -1, head_size)
            out = heads.clone()
            x1 = heads[..., :half].double()
            x2 = heads[..., half:rotary_dim].double()
            out[..., :half] = (x1 * cos - x2 * sin).to(x.dtype)
            out[..., half:rotary_dim] = (x2 * cos + x1 * sin).to(x.dtype)
            return out.reshape(x.shape)

        return rotate(query), rotate(key)

    def is_valid(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ) -> None:
        rotary_dim = cos_sin_cache.shape[1]
        if head_size not in _HEAD_DIM_KERNELS:
            vec = _ROPE_QUANTIZE_VEC_BYTES // query.element_size()
            assert (rotary_dim // 2) % vec == 0, (
                f"flashinfer_apply_rope_with_cos_sin_cache: rotary_dim {rotary_dim} on "
                f"{head_size}-wide heads takes RopeQuantizeKernel, whose {vec}-element vectors "
                "do not split at the rotary half; pad the heads to 256 first"
            )
        tokens = query.shape[0]
        assert positions.numel() >= tokens, (
            f"flashinfer_apply_rope_with_cos_sin_cache: {positions.numel()} positions for "
            f"{tokens} tokens"
        )
        # The range check reads the positions back, which a stream under
        # CUDA-graph capture cannot do; the capture test checks it eagerly.
        if tokens and not torch.cuda.is_current_stream_capturing():
            used = positions.reshape(-1)[:tokens]
            low, high = int(used.min()), int(used.max())
            assert 0 <= low and high < cos_sin_cache.shape[0], (
                f"flashinfer_apply_rope_with_cos_sin_cache: positions {low}..{high} outside "
                f"the table's {cos_sin_cache.shape[0]} rows"
            )
        assert key.dtype == query.dtype, (
            f"flashinfer_apply_rope_with_cos_sin_cache: key dtype {key.dtype} differs from "
            f"query dtype {query.dtype}"
        )


flashinfer_apply_rope_with_cos_sin_cache = FlashinferApplyRopeWithCosSinCache()
