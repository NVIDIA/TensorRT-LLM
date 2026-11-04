# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the FP8 unified target's Gated-DeltaNet entries, each over its certified cells.

Geometry is a Qwen3.8-Flash-Next linear-attention layer (key and value heads of 128, conv
width 4): the whole layer's 16 / 48 heads at tp1 and on a dep4 rank, 8 / 24 on a tp2/tep2
rank, 4 / 12 on a tp4/tep4 rank. A pool is a contiguous tensor (compared whole, so slots the
call did not name are gated too), or the stock manager's coalesced page: 36 GDN layers'
states a slot, the cell's view at the last layer's sub-page. At an engine's measured slot
count such a page spans tens to hundreds of GB, so it is a reserved virtual range with
physical memory mapped under the touched slots only (`_SparsePage`); the reference runs on a
dense copy of those slots, and every other byte of their pages is checked unchanged.

- `causal_conv1d_update`: the decode conv over the first `dim` columns of the projection
  output, the pool the last layer's view of a coalesced page; the activation within one bf16
  step of the fp32 reference, the pool bitwise. Which kernel each member reaches (the CuTe
  kernel from 64 rows at 5120 and 10240 channels, never at a tp4 rank's 2560), the tp2 call
  compiling its own CuTe kernel, and that kernel bitwise the TRT-LLM one.
- `gdn_prefill_conv_post`: the prefill conv + q/k norm + gating against the fp32/fp64
  reference, the conv pool bitwise, the rest of its page untouched; the cells reach slots past
  the 32-bit offset, and `linear_gate=True` is `torch.exp` of the log gate.
- `flashinfer_chunk_gated_delta_rule`: FlashInfer's GDN prefill (in place, chunked or the CP
  pipeline, the path each cell names asserted), and `has_initial_state` as the reset followed
  by the plain call.
- `gdn_decode_rows_gated_norm` and `gdn_decode_rows`: the decode recurrence (with and without
  the gated norm) against the fp64 reference; a fixed row bitwise across batch sizes over a
  bf16 pool, a stale-read check under graph replay (a producer rewrites the projection inside
  the graph right before the op), the bulk-copy and register variants bitwise each other, and
  the variant choice over an unaligned pool -- each at the head counts that bracket the
  members, 16 / 48 and 4 / 12 (tp2's 8 / 24 runs the same two variants, at every row count
  the variant choice separates, in its cells).
- `fused_recurrent_gated_delta_rule_cached_replay_update`: stateful across verify steps, so
  every cell drives consecutive steps the way the engine does (work items, the op per layer,
  the all-layer commit, `_advance_replay_state` with a random partial acceptance), each call
  against `reference` and every output against an exact twin that never wraps; a `capture`
  cell replays every step in a CUDA graph bitwise to eager. A dep4 reference handed the key
  heads reversed must fall outside the output gate.
"""

from __future__ import annotations

import math
import warnings
import zlib

import cuda.bindings.driver as cuda_drv
import pytest
import torch
import torch.nn.functional as F
from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import should_use_cp_host
from flashinfer.utils import get_compute_capability, get_device_name, get_device_sm_count

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm import (  # noqa: E501
    gdn_decode_rows_gated_norm as rows_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.causal_conv1d_update import (  # noqa: E501
    _DECODE_CONV,
    CKPT_DIM,
    CKPT_X_STRIDE,
    GDN_PAGE_LAYERS,
    RANK_DIM,
    RANK_X_STRIDE,
    TP2_DIM,
    TP2_X_STRIDE,
    _decode_conv_ok,
    _spread,
    causal_conv1d_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.flashinfer_chunk_gated_delta_rule import (  # noqa: E501
    INPLACE_MAX_TOKENS,
    flashinfer_chunk_gated_delta_rule,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.fused_recurrent_gated_delta_rule_cached_replay_update import (  # noqa: E501
    ALL_LAYER_COMMIT_MIN_BATCH,
    HISTORY_SIZE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.fused_recurrent_gated_delta_rule_cached_replay_update import (  # noqa: E501
    fused_recurrent_gated_delta_rule_cached_replay_update as replay_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.gdn_decode_rows_gated_norm import (  # noqa: E501
    gdn_decode_rows,
    gdn_decode_rows_gated_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.gdn_prefill_conv_post import (  # noqa: E501
    A_LOG_RANGE,
    CONV_WIDTH,
    DT_BIAS_RANGE,
    HEAD_K_DIM,
    HEAD_V_DIM,
    IN_PROJ_WIDTH,
    NUM_K_HEADS,
    NUM_V_HEADS,
    SOFTPLUS_THRESHOLD,
    _engine_slots,
    gdn_prefill_conv_post,
)
from tensorrt_llm._torch.modules.fla.cache_manager import (
    ReplayStateUpdateMetadata,
    _advance_replay_state,
)
from tensorrt_llm._torch.modules.fla.cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    commit_gdn_cached_replay_history_layers,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import _build_replay_work_items_triton

__extra_import_path__ = ["../../.."]
from _validating import ULP, certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "the Gated-DeltaNet entries require a CUDA device"

DEV = "cuda"
#: Key heads, value heads, key and value head dims of the whole layer (tp1, a dep4 rank).
H, HV, K, V = NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM


# ------------------------------------------------- reserved pages, mapped where touched ---


def _scattered(n: int, slots: int) -> list[int]:
    """`n` distinct slots of a `slots`-slot pool: both ends first, then a stride-37 walk
    over the interior (37 is coprime to `slots - 2` for every pool here)."""
    rest = [(i * 37 + 4) % (slots - 2) + 1 for i in range(max(n - 2, 0))]
    picked = ([slots - 1, 0] + rest)[:n]
    assert len(set(picked)) == n, (n, slots)
    return picked


def _check(result):
    err, *value = result
    assert err == cuda_drv.CUresult.CUDA_SUCCESS, err
    return value[0] if len(value) == 1 else value


#: Array-interface type of each pool dtype; bf16 has none, so it is read as int16 and viewed.
_TYPESTR = {torch.float32: ("<f4", torch.float32), torch.bfloat16: ("<i2", torch.int16)}


class _ArrayInterface:
    def __init__(self, ptr: int, shape, strides_bytes, typestr: str):
        self.__cuda_array_interface__ = dict(
            shape=tuple(shape),
            strides=tuple(strides_bytes),
            typestr=typestr,
            data=(ptr, False),
            version=2,
        )


class _SparsePage:
    """A reserved virtual range of `elems` elements of `dtype` with physical memory mapped
    under `regions` (element ranges) only. Views of it are torch tensors over that range;
    an access to an unmapped address faults, so a stray write is loud."""

    def __init__(self, elems: int, regions, dtype: torch.dtype):
        self.dtype, self.es = dtype, torch.empty((), dtype=dtype).element_size()
        dev = torch.cuda.current_device()
        prop = cuda_drv.CUmemAllocationProp()
        prop.type = cuda_drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = dev
        gran = _check(
            cuda_drv.cuMemGetAllocationGranularity(
                prop,
                cuda_drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
        es = self.es
        self.size = -(-elems * es // gran) * gran
        self.va = int(_check(cuda_drv.cuMemAddressReserve(self.size, gran, 0, 0)))
        spans = sorted(
            ((lo * es) // gran * gran, min(-(-hi * es // gran) * gran, self.size))
            for lo, hi in regions
        )
        merged = []
        for lo, hi in spans:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        access = cuda_drv.CUmemAccessDesc()
        access.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = dev
        access.flags = cuda_drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.maps = []
        for lo, hi in merged:
            handle = _check(cuda_drv.cuMemCreate(hi - lo, prop, 0))
            ptr = cuda_drv.CUdeviceptr(self.va + lo)
            _check(cuda_drv.cuMemMap(ptr, hi - lo, 0, handle, 0))
            self.maps.append((ptr, hi - lo, handle))
            _check(cuda_drv.cuMemSetAccess(ptr, hi - lo, [access], 1))

    def view(self, offset: int, shape, strides) -> torch.Tensor:
        """Tensor of the page's dtype at element `offset`, element `strides`."""
        typestr, carrier = _TYPESTR[self.dtype]
        es = self.es
        t = torch.as_tensor(
            _ArrayInterface(self.va + es * offset, shape, [es * s for s in strides], typestr),
            device=DEV,
        )
        assert t.dtype == carrier
        return t.view(self.dtype)

    def close(self) -> None:
        torch.cuda.synchronize()
        for ptr, size, handle in self.maps:
            _check(cuda_drv.cuMemUnmap(ptr, size))
            _check(cuda_drv.cuMemRelease(handle))
        _check(cuda_drv.cuMemAddressFree(cuda_drv.CUdeviceptr(self.va), self.size))


# ------------------------------------------------------------- causal_conv1d_update ---


def _bf16_steps(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """`|ordinal(out) - ordinal(ref)|`: how many representable bf16 values apart each
    element is (+0 and -0 the same value)."""
    assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
    b_out, b_ref = out.view(torch.int16).int(), ref.view(torch.int16).int()
    ordinal = [torch.where(b < 0, -(b & 0x7FFF), b) for b in (b_out, b_ref)]
    return (ordinal[0] - ordinal[1]).abs()


def _conv_compare(
    out: tuple[torch.Tensor, torch.Tensor], ref: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """`(x, conv_state)` pairs: the returned `x` with the pool as the call
    left it, against `reference`'s pair.

    The activation at torch's default band and within one bf16 step of the
    fp32 reference, element by element. The kernels evaluate the taps as an
    fma chain from 0 and the SiLU as `v * __frcp_rn(1 + __expf(-v))`; the
    reference sums the products and takes torch's sigmoid, so an fp32 result
    that lands near a bf16 rounding midpoint can round the other way.
    Measured over every cell: 99.9988% to 100% of elements bitwise (the
    1-row cells all of them), every other one exactly one step away -- none
    two or more, near zero included. The pool is a copy of input values, so
    it is gated bitwise -- over the whole pool, which is the lock on every
    slot the batch did not name.
    """
    (act, state), (ref_act, ref_state) = out, ref
    assert act.dtype == ref_act.dtype and act.shape == ref_act.shape
    torch.testing.assert_close(act, ref_act)
    far = int((_bf16_steps(act, ref_act) > 1).sum())
    assert far == 0, f"{far} activations more than one bf16 step from the reference"
    assert torch.equal(state, ref_state), "the advanced state must be bitwise exact"


PAD_SLOT_ID = -1
#: Rows of prefill ahead of the decode rows in the in_proj output.
_PREFILL_ROWS = 37


def _conv_build(spec, seed):
    """`__call__`'s arguments for one spec, plus the buffers x and the pool view.

    x is the first `dim` columns of the decode rows of a `[rows, x_row_stride]`
    projection output (its z | a | b columns past them); the pool is the last
    layer's view of a `[slots, pool_layers, dim, 3]` page, as the V2 manager
    coalesces the GDN layers' conv states inside each slot.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dim, batch, layers = spec["dim"], spec["batch"], spec.get("pool_layers", 1)
    row_stride = spec.get("x_row_stride", CKPT_X_STRIDE)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16)

    buf = randn(_PREFILL_ROWS + batch, row_stride)
    x = buf[_PREFILL_ROWS:, :dim]
    assert x.stride() == (row_stride, 1)
    pool_buf = randn(spec["pool_slots"], layers, dim, 3)
    conv_state = pool_buf[:, layers - 1]
    assert conv_state.stride() == (layers * dim * 3, 3, 1)
    weight = randn(dim, 4)
    slots = spec["slots"]
    if slots == "random":
        indices = torch.randperm(spec["pool_slots"], generator=g, device="cuda")[:batch]
        indices = indices.to(torch.int32)
    else:
        indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
    args = [x, conv_state, weight, None, True, None, indices, PAD_SLOT_ID]
    return args, dict(buf=buf, pool_buf=pool_buf)


def _conv_run(args, extras):
    """Drive the op on copies of x and the pool, through the same views."""
    got = list(args)
    got_buf, got_pool = extras["buf"].clone(), extras["pool_buf"].clone()
    got[0] = got_buf[_PREFILL_ROWS:, : args[0].shape[1]]
    got[1] = got_pool[:, got_pool.shape[1] - 1]
    returned = causal_conv1d_update(*got)
    assert returned is got[0], "returns the x it was handed"
    assert torch.equal(got_buf[:_PREFILL_ROWS], extras["buf"][:_PREFILL_ROWS])
    dim = args[0].shape[1]
    assert torch.equal(got_buf[:, dim:], extras["buf"][:, dim:]), "columns past x written"
    assert torch.equal(got_pool[:, :-1], extras["pool_buf"][:, :-1]), "another layer written"
    return returned, got[1]


@pytest.mark.parametrize(
    "cell", causal_conv1d_update.CELLS, ids=[c.why[:44] for c in causal_conv1d_update.CELLS]
)
def test_causal_conv1d_update_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(causal_conv1d_update)
    args, extras = _conv_build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    weight = args[2].clone()
    ref = causal_conv1d_update.reference(*args)
    returned, state = _conv_run(args, extras)
    assert torch.equal(args[2], weight), "weight is read-only"
    _conv_compare((returned, state), ref)
    exact = (_bf16_steps(returned, ref[0]) == 0).float().mean().item()
    print(
        f"  {cell.why[:44]}: x {100 * exact:.4f}% bitwise, the rest one bf16 step; "
        f"{args[1].shape[0]}-slot pool, top named slot {max(args[6].tolist())}"
    )
    # Non-vacuity: every slot the batch named moved (the whole-pool gate above
    # holds every other slot, and every pad row, unchanged).
    for slot in args[6].tolist():
        if slot != PAD_SLOT_ID:
            assert not torch.equal(state[slot], args[1][slot]), slot


def test_which_conv_kernel_each_member_runs() -> None:
    """The path the note states each member's cells certify: a tp4/tep4 rank's 2560
    channels never reach the CuTe kernel (2560 is not a multiple of 1024), so its 128-row
    cells certify the TRT-LLM kernel; a dep4 rank's 10240 and a tp2/tep2 rank's 5120 do
    from 64 rows, and below 64 run the TRT-LLM kernel."""
    for dim, row_stride, batch, expect in (
        (RANK_DIM, RANK_X_STRIDE, 128, False),
        (CKPT_DIM, CKPT_X_STRIDE, 128, True),
        (TP2_DIM, TP2_X_STRIDE, 64, True),
        (TP2_DIM, TP2_X_STRIDE, 63, False),
    ):
        spec = dict(dim=dim, x_row_stride=row_stride, batch=batch, pool_slots=256, slots="random")
        args = _conv_build(spec, seed=dim)[0]
        assert _decode_conv_ok(*args[:7]) is expect, (dim, batch, expect)


def test_a_tp2_conv_call_from_64_rows_compiles_the_cute_kernel() -> None:
    """A tp2/tep2 call from 64 rows reaches the CuTe kernel: it JIT-compiles under the
    rank's own (channels, x row stride) key, which no other member's call shares."""
    spec = dict(dim=TP2_DIM, x_row_stride=TP2_X_STRIDE, batch=64, pool_slots=256, slots="random")
    args, extras = _conv_build(spec, seed=TP2_DIM)
    # Dropped first, so that a cell compiling it earlier in the session cannot answer for
    # this call.
    _DECODE_CONV.pop((TP2_DIM, TP2_X_STRIDE), None)
    _conv_run(args, extras)
    assert (TP2_DIM, TP2_X_STRIDE) in _DECODE_CONV


def test_the_cute_conv_kernel_is_bitwise_the_trtllm_kernel_at_tp2() -> None:
    """The note's claim that the CuTe kernel is the TRT-LLM kernel instruction for
    instruction, held at a tp2/tep2 rank's 5120 / 8240 call of 128 rows with pad rows among
    them: x and the whole page bitwise equal after the entry (the CuTe kernel) and after the
    raw TRT-LLM op on the same operands."""
    batch, pad_every = 128, 21
    pool_slots = 512
    spec = dict(
        dim=TP2_DIM,
        x_row_stride=TP2_X_STRIDE,
        batch=batch,
        pool_slots=pool_slots,
        slots=_spread(batch, pool_slots, pad_every),
        pool_layers=GDN_PAGE_LAYERS,
    )
    args, extras = _conv_build(spec, seed=batch + pad_every)
    assert _decode_conv_ok(*args[:7])
    cute_x, cute_state = _conv_run(args, extras)

    trt_buf, trt_page = extras["buf"].clone(), extras["pool_buf"].clone()
    trt_x = trt_buf[_PREFILL_ROWS:, :TP2_DIM]
    torch.ops.trtllm.causal_conv1d_update(
        trt_x.unsqueeze(-1), trt_page[:, -1], args[2], None, True, None, args[6], PAD_SLOT_ID
    )
    assert torch.equal(cute_x, trt_x), "x differs between the CuTe and TRT-LLM kernels"
    assert torch.equal(cute_state, trt_page[:, -1]), "the pool differs between the two kernels"
    assert torch.equal(trt_page[:, :-1], extras["pool_buf"][:, :-1])


# ------------------------------------------------------------ gdn_prefill_conv_post ---

DTYPE = torch.bfloat16
PAD = -1
L2_NORM_EPS = 1e-6
#: `g`'s band: two SASS
#: approximate transcendentals (~2^-22 relative each), the softplus bounded by
#: the threshold, every term scaled by exp(A_log).
_APPROX_ULP = 2.0**-22
G_ATOL = 4 * _APPROX_ULP * SOFTPLUS_THRESHOLD * math.exp(A_LOG_RANGE[1])


def _conv_post_geometry(spec):
    """`(num_k_heads, num_v_heads, head_dim, qkv_dim, projection width)` of a cell: its
    own, or the tp1 layer's."""
    hk = spec.get("num_k_heads", NUM_K_HEADS)
    hv = spec.get("num_v_heads", NUM_V_HEADS)
    d = spec.get("head_dim", HEAD_K_DIM)
    assert d == HEAD_K_DIM == HEAD_V_DIM
    qkv = 2 * hk * d + hv * d
    width = spec.get("x_row_stride", IN_PROJ_WIDTH)
    assert width == qkv + hv * d + 2 * hv, (
        f"x_row_stride {width} is not the [q|k|v|z|a|b] projection of {hk}/{hv} heads"
    )
    return hk, hv, d, qkv, width


def _conv_post_build(spec, seed):
    """`__call__`'s positional arguments for one cell."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    lengths = spec["lengths"]
    n_tok = sum(lengths)
    hk, hv, d, qkv, width = _conv_post_geometry(spec)
    a_col = qkv + hv * d

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=DEV).to(DTYPE)

    # "in_proj" and "rows" both carve x, a and b out of one projection; "rows"
    # leaves `extra_rows` trailing rows the call must not read.
    proj = randn(n_tok + spec.get("extra_rows", 0), width)
    x = proj[:n_tok, :qkv]
    a = proj[:n_tok, a_col : a_col + hv]
    b = proj[:n_tok, a_col + hv : a_col + 2 * hv]
    assert x.stride(0) == a.stride(0) == width
    weight = (torch.randn(qkv, CONV_WIDTH, generator=g, device=DEV) * 0.5).to(DTYPE)
    pool_slots = spec["pool_slots"]
    # The pool is a per-layer view of a multi-layer buffer, as a cache manager
    # hands it out: two layers for the tp1 cells (slot stride twice the slot's own
    # size); for a rank's cells the stock page, `page_layers` (36) GDN layers'
    # states a slot, viewed at the last -- the strides and offset the manager's
    # `get_conv_states` returns, measured.
    layers = spec.get("page_layers", 2)
    conv_states = randn(pool_slots, layers, qkv, CONV_WIDTH - 1)[:, layers - 1]
    slots = spec.get("slots") or _engine_slots(len(lengths))
    bounds = [0]
    for n in lengths:
        bounds.append(bounds[-1] + n)
    qsl = torch.tensor(bounds, dtype=torch.int32, device=DEV)
    si = torch.tensor(slots, dtype=torch.int32, device=DEV)
    hi = torch.tensor(spec["initial"], dtype=torch.bool, device=DEV)
    lo, hi_ = A_LOG_RANGE
    A_log = torch.rand(hv, generator=g, device=DEV) * (hi_ - lo) + lo
    lo, hi_ = DT_BIAS_RANGE
    dt_bias = torch.rand(hv, generator=g, device=DEV) * (hi_ - lo) + lo
    return [
        x,
        weight,
        conv_states,
        qsl,
        si,
        hi,
        a,
        b,
        A_log,
        dt_bias,
        hk,
        d,
        hv,
        d,
        PAD,
        L2_NORM_EPS,
        SOFTPLUS_THRESHOLD,
        spec.get("linear_gate", False),
    ]


def _conv_post_compare(out, pool, want, what=""):
    q, k, v, g, beta = out
    q_r, k_r, v_r, g_r, beta_r, pool_r = want
    for got, ref in zip(out, want[:5]):
        assert got.dtype == ref.dtype and got.shape == ref.shape and got.is_contiguous()
    for name, got, ref in (("q", q, q_r), ("k", k, k_r), ("v", v, v_r)):
        d = (got.float() - ref.float()).abs()
        flips = (d > 0).float().mean().item() if d.numel() else 0.0
        print(
            f"{what} {name}: max abs {d.max().item() if d.numel() else 0:.3e}, "
            f"differing fraction {flips:.2e}"
        )
        torch.testing.assert_close(got, ref)
    torch.testing.assert_close(g, g_r, rtol=8 * _APPROX_ULP, atol=G_ATOL)
    torch.testing.assert_close(beta, beta_r)
    assert torch.equal(pool, pool_r), "conv pool differs"


@pytest.mark.parametrize(
    "cell", gdn_prefill_conv_post.CELLS, ids=[c.why[:44] for c in gdn_prefill_conv_post.CELLS]
)
def test_gdn_prefill_conv_post_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(gdn_prefill_conv_post)
    args = _conv_post_build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    want = gdn_prefill_conv_post.reference(*args)
    inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in args]
    pool = args[2]
    page = pool._base
    page_before = page.clone()
    out = gdn_prefill_conv_post(*args)
    for i, (arg, before) in enumerate(zip(args, inputs)):
        if isinstance(arg, torch.Tensor) and i != 2:
            assert torch.equal(arg, before), f"input {i} must be read-only"
    _conv_post_compare(out, pool, want, what=f"{cell.why[:30]} vs reference")
    # The rest of the buffer -- the other layers' sub-pages of every slot -- is untouched.
    layer = (pool.storage_offset() - page.storage_offset()) // page.stride(1)
    page_before[:, layer] = want[5]
    assert torch.equal(page, page_before), "the call wrote outside its layer's view of the pool"


def test_the_conv_post_cells_carry_from_slots_past_the_int32_offset() -> None:
    """The cells read a carried window at a slot whose element offset on the stock page
    passes 2^31 at every rank geometry, and 2^32 at 16 / 48 and 8 / 24 heads -- where a
    32-bit slot offset faults or wraps back into the page -- so the 64-bit slot base stays
    exercised."""
    reach = {}
    for cell in gdn_prefill_conv_post.CELLS:
        spec = cell.spec
        if "page_layers" not in spec:
            continue
        hk, _, _, qkv, _ = _conv_post_geometry(spec)
        slot_stride = spec["page_layers"] * qkv * (CONV_WIDTH - 1)
        carried = [s for s, init in zip(spec["slots"], spec["initial"]) if init]
        reach[hk] = max([reach.get(hk, 0)] + [s * slot_stride for s in carried])
    assert reach[NUM_K_HEADS] >= 2**32 and reach[NUM_K_HEADS // 4] >= 2**31, reach
    assert reach[NUM_K_HEADS // 2] >= 2**32, reach


def test_the_conv_post_linear_gate_is_torch_exp_of_the_log_gate() -> None:
    """`linear_gate=True` stores `torch.exp` of the log-space decay bitwise, and leaves
    every other output and the conv pool as without it."""
    spec = next(c for c in gdn_prefill_conv_post.CELLS if c.spec.get("linear_gate")).spec
    args = _conv_post_build(spec, seed=77)
    log_args = list(args)
    log_args[-1] = False
    log_args[2] = args[2].clone()
    pool_lin = args[2]
    log_out = gdn_prefill_conv_post(*log_args)
    lin_out = gdn_prefill_conv_post(*args)
    assert torch.equal(lin_out[3], torch.exp(log_out[3]))
    for got, want in zip(lin_out[:3] + lin_out[4:], log_out[:3] + log_out[4:]):
        assert torch.equal(got, want)
    assert torch.equal(pool_lin, log_args[2])


# ------------------------------------------------ flashinfer_chunk_gated_delta_rule ---

#: The band the Triton chunk op is certified at (rtol bf16 default; atol 2e-3
#: on `o`, 2e-2 on states), derived for a chunked WY reformulation with bf16
#: operands. This op is the same class of sum and sits inside it against fp64, CP
#: pipeline and chunked kernel alike; a value head reading the wrong key head at 4/12
#: lands far outside it (measured).
CHUNK_RTOL, CHUNK_ATOL_OUTPUT, CHUNK_ATOL_STATE = 1.6e-2, 2e-3, 2e-2
#: Rows past the call's own in the output buffer of a `slice` cell: the target passes
#: `out[:, :num_tokens]` of a wider buffer, and the rows past it must stay as they were.
_CHUNK_OUTPUT_TAIL = 16


def _chunk_compare(out: tuple, ref: tuple) -> None:
    """`(o, final_state, pool)` against `reference`'s triple; `None` must match."""
    for got, want, atol in zip(out, ref, (CHUNK_ATOL_OUTPUT, CHUNK_ATOL_STATE, CHUNK_ATOL_STATE)):
        assert (got is None) == (want is None), (got is None, want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape, (
            (got.dtype, got.shape),
            (want.dtype, want.shape),
        )
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got.float(), want.float(), rtol=CHUNK_RTOL, atol=atol)


def _chunk_stats(name: str, got: torch.Tensor, want: torch.Tensor) -> str:
    """Relative L2, cosine, p99.9 and max absolute error of `got` against `want`."""
    g, w = got.double().flatten(), want.double().flatten()
    err = (g - w).abs()
    rel = (err.norm() / w.norm().clamp_min(1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(g, w, dim=0).item()
    p999 = torch.quantile(err[:: max(1, err.numel() // 2_000_000)].float(), 0.999).item()
    return f"{name} rel-L2 {rel:.3e} cos {cos:.6f} p99.9 {p999:.3e} max {err.max().item():.3e}"


def _flashinfer_path(spec) -> str:
    """Which FlashInfer kernel the call reaches: `inplace` (the chunked kernel over the pool
    itself) for a `has_initial_state` call of at most `INPLACE_MAX_TOKENS` tokens, else `cp`
    (the context-parallel pipeline) when FlashInfer's own heuristic picks it for
    `num_seqs x num_v_heads` on this device."""
    if "initial" in spec and sum(spec["seqlens"]) <= INPLACE_MAX_TOKENS:
        return "inplace"
    device = torch.device(DEV)
    cp = should_use_cp_host(
        len(spec["seqlens"]) * spec["geometry"][1],
        get_device_sm_count(device),
        get_device_name(device),
        device_capability=get_compute_capability(device),
    )
    return "cp" if cp else "chunked"


class _ChunkEnginePool:
    """The stock manager's coalesced bf16 state page at `slots` slots (slot stride:
    `layers` `[HV, V, K]` states), this layer's view the last layer's sub-page, mapped
    under the named slots: each one's whole page and the next slot's first layer. The
    view's own base (slot 0's sub-page) is mapped too: torch reads the device off a
    tensor's base pointer."""

    def __init__(self, slots: int, layers: int, used, v_heads: int, head_v: int, head_k: int, gen):
        layer = v_heads * head_v * head_k
        stride = layers * layer
        total = slots * stride
        base = (layers - 1) * layer
        self.used = sorted(set(used))
        self.spans = [(s * stride, min((s + 1) * stride + layer, total)) for s in self.used]
        self.page = _SparsePage(total, [*self.spans, (base, base + layer)], torch.bfloat16)
        self.own = base
        self.layer = layer
        self.pool = self.page.view(
            base, (slots, v_heads, head_v, head_k), (stride, head_v * head_k, head_k, 1)
        )
        self.flat = [self.page.view(lo, (hi - lo,), (1,)) for lo, hi in self.spans]
        for f in self.flat:
            f.copy_(torch.randn(f.numel(), generator=gen, device=DEV) * 0.1)
        torch.cuda.synchronize()
        self.before = [f.clone() for f in self.flat]

    def dense(self, slots: torch.Tensor):
        """A dense copy of the named slots and `slots` renumbered into it."""
        index = {s: i for i, s in enumerate(self.used)}
        compact = torch.stack([self.pool[s] for s in self.used]).contiguous()
        renamed = [index[s] for s in slots.tolist()]
        return compact, torch.tensor(renamed, dtype=slots.dtype, device=DEV)

    def touched(self) -> torch.Tensor:
        return torch.stack([self.pool[s] for s in self.used])

    def assert_rest_unchanged(self) -> None:
        """Every byte of a named slot's page but this layer's state, and the next slot's
        first layer, is what it was."""
        torch.cuda.synchronize()
        for f, b, s in zip(self.flat, self.before, self.used):
            keep = torch.ones(f.numel(), dtype=torch.bool, device=DEV)
            keep[self.own : self.own + self.layer] = False
            assert torch.equal(f[keep], b[keep]), f"slot {s}: the call wrote outside its state"

    def close(self) -> None:
        del self.pool, self.flat
        self.page.close()


def _chunk_build(spec, seed):
    """`(args, engine)`: keyword arguments in the domain a GDN post-conv step produces
    (unit-norm q/k, g < 0 in log space, beta a sigmoid), and the `_ChunkEnginePool` behind the
    pool (or None)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = spec["geometry"]
    seqlens = spec["seqlens"]
    total = sum(seqlens)
    q = torch.randn(1, total, num_k_heads, head_k_dim, generator=gen, device=DEV)
    k = torch.randn(1, total, num_k_heads, head_k_dim, generator=gen, device=DEV)
    q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    q, k = q.bfloat16(), k.bfloat16()
    v = torch.randn(
        1, total, num_v_heads, head_v_dim, generator=gen, device=DEV, dtype=torch.bfloat16
    )
    a_log = torch.log(torch.rand(num_v_heads, generator=gen, device=DEV) * 15 + 1)
    x = torch.randn(1, total, num_v_heads, generator=gen, device=DEV) + torch.randn(
        num_v_heads, generator=gen, device=DEV
    )
    g = (-torch.exp(a_log) * torch.nn.functional.softplus(x)).float()
    beta = torch.rand(1, total, num_v_heads, generator=gen, device=DEV).sigmoid().float()
    engine = None
    if spec.get("page_layers"):
        engine = _ChunkEnginePool(
            spec["slots"],
            spec["page_layers"],
            spec["indices"],
            num_v_heads,
            head_v_dim,
            head_k_dim,
            gen,
        )
        pool = engine.pool
    else:
        pool = (
            torch.randn(
                spec["slots"], num_v_heads, head_v_dim, head_k_dim, generator=gen, device=DEV
            )
            * 0.1
        ).to(spec.get("pool_dtype", torch.float32))
    cu = torch.tensor([0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.long, device=DEV)
    indices = torch.tensor(spec["indices"], dtype=torch.int32, device=DEV)
    output = None
    if spec.get("output") == "buffer":
        output = torch.full(
            (1, total, num_v_heads, head_v_dim), -7.0, device=DEV, dtype=torch.bfloat16
        )
    elif spec.get("output") == "slice":
        output = torch.full(
            (1, total + _CHUNK_OUTPUT_TAIL, num_v_heads, head_v_dim),
            -7.0,
            device=DEV,
            dtype=torch.bfloat16,
        )[:, :total]
    extra = {}
    if "initial" in spec:
        extra["has_initial_state"] = torch.tensor(spec["initial"], dtype=torch.bool, device=DEV)
    if spec.get("linear_gate"):
        extra["linear_gate"] = True
        g = torch.exp(g)
    args = dict(
        **extra,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=None,
        initial_state=pool,
        initial_state_indices=indices,
        inplace_indexed_state_update=True,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
        output=output,
        state_workspace=None,
    )
    return args, engine


@pytest.mark.parametrize(
    "cell",
    flashinfer_chunk_gated_delta_rule.CELLS,
    ids=[c.why[:44] for c in flashinfer_chunk_gated_delta_rule.CELLS],
)
def test_flashinfer_chunk_gated_delta_rule_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(flashinfer_chunk_gated_delta_rule)
    args, engine = _chunk_build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    try:
        if engine is None:
            pool = args["initial_state"]
            pool_before = pool.clone()
            ref = flashinfer_chunk_gated_delta_rule.reference(**args)
        else:
            compact, renamed = engine.dense(args["initial_state_indices"])
            ref = flashinfer_chunk_gated_delta_rule.reference(
                **dict(args, initial_state=compact, initial_state_indices=renamed)
            )
        output = args["output"]
        # A `slice` cell's output is a prefix view of the wider buffer the target owns.
        full = None if output is None else output._base
        tail_before = None if full is None else full[:, output.shape[1] :].clone()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            o, final_state = flashinfer_chunk_gated_delta_rule(**args)
        # FlashInfer falls back from a CP heuristic match with a RuntimeWarning, so a CP
        # cell certifies the CP pipeline only if none was raised.
        fallback = [str(w.message) for w in caught if "falling back to non-CP" in str(w.message)]
        path = "chunked" if fallback else _flashinfer_path(cell.spec)

        got_pool = pool if engine is None else engine.touched()
        _chunk_compare((o, final_state, got_pool), ref)
        print(
            f"\n[{cell.why}] path={path} sms={get_device_sm_count(torch.device(DEV))}; "
            f"{_chunk_stats('o', o, ref[0])}; {_chunk_stats('state', got_pool, ref[2])}"
        )
        if "fi_path" in cell.spec:
            assert path == cell.spec["fi_path"], (path, fallback)
        if output is not None:
            assert o.data_ptr() == output.data_ptr(), "output= must be written in place"
        if tail_before is not None:
            assert torch.equal(full[:, output.shape[1] :], tail_before), "rows past output written"
        if engine is None:
            named = set(args["initial_state_indices"].tolist())
            unnamed = [s for s in range(pool.shape[0]) if s not in named]
            assert torch.equal(pool[unnamed], pool_before[unnamed]), "unnamed slots written"
        else:
            engine.assert_rest_unchanged()
    finally:
        if engine is not None:
            engine.close()


_CHUNK_CARRIED = next(c for c in flashinfer_chunk_gated_delta_rule.CELLS if "initial" in c.spec)


def test_chunk_has_initial_state_is_the_reset_then_the_plain_call() -> None:
    """The flags replace `reset_recurrent_state_rows` before the call: zeroing the fresh
    slots and running the plain in-place call gives the same output and pool, bitwise; a
    linear gate that is `torch.exp` of the plain call's log gate changes nothing."""
    args, _ = _chunk_build(_CHUNK_CARRIED.spec, seed=5)
    flags = args.pop("has_initial_state")
    linear_gate = args.pop("linear_gate", False)
    pool = args["initial_state"]
    fresh = args["initial_state_indices"][~flags].long()
    pool[fresh] = torch.randn_like(pool[fresh])  # stale rows the flags must not read
    plain = dict(args, initial_state=pool.clone(), output=args["output"].clone())
    plain["initial_state"][fresh] = 0.0
    if linear_gate:
        plain["g"] = torch.log(args["g"])
        # The linear gate of the target call is torch.exp of the log gate, bitwise.
        args["g"] = torch.exp(plain["g"])

    o_plain, _ = flashinfer_chunk_gated_delta_rule(**plain)
    o, final_state = flashinfer_chunk_gated_delta_rule(
        **args, has_initial_state=flags, linear_gate=linear_gate
    )

    assert final_state is None
    assert torch.equal(o, o_plain)
    assert torch.equal(pool, plain["initial_state"])


# ------------------------------------------------------- gdn_decode_rows_gated_norm ---

SLOTS, EPS = 461, 1e-6


#: GDN layers whose states share one slot page in the stock manager (48 layers, every 4th
#: full attention); the view is at the last one's sub-page, the page's highest addresses.
PAGE_LAYERS = 36
#: (k heads, v heads) of a tp4/tep4 rank.
RANK_HEADS = (4, 12)
#: The properties below hold at the head counts that bracket the members' (16 / 48 and 4 / 12);
#: tp2/tep2's 8 / 24 runs the same two variants, at every row count the choice separates
#: (1, 2, 3 and 128 rows over its 7194-slot page), in its cells.
_BRACKET = pytest.mark.parametrize("k_heads, v_heads", [(H, HV), RANK_HEADS], ids=["16x48", "4x12"])


def _heads(spec) -> tuple[int, int]:
    """(k heads, v heads): tp1's 16 / 48 where the cell names none."""
    return spec.get("k_heads", H), spec.get("v_heads", HV)


def _width(k_heads: int, v_heads: int) -> int:
    """`[q | k | v | z | a | b]`: 16480 at 16 / 48, 4120 at a tp4/tep4 rank's 4 / 12."""
    return 2 * k_heads * K + 2 * v_heads * V + 2 * v_heads


class _DecodeEnginePool:
    """The stock manager's coalesced state page at `slots` slots (slot stride: 36 layers'
    `[HV, V, K]` states), this layer's view the last layer's sub-page, mapped under the
    touched slots: each one's whole page and the next slot's first layer. The view's own
    base (slot 0's sub-page) is mapped too: torch reads the device off a tensor's base
    pointer."""

    def __init__(self, slots: int, used, v_heads: int, dtype: torch.dtype, gen):
        layer = v_heads * V * K
        stride = PAGE_LAYERS * layer
        total = slots * stride
        base = (PAGE_LAYERS - 1) * layer
        self.used = sorted(set(used))
        self.spans = [(s * stride, min((s + 1) * stride + layer, total)) for s in self.used]
        self.page = _SparsePage(total, [*self.spans, (base, base + layer)], dtype)
        # This layer's state within each mapped span: every span starts at a slot's page.
        self.own = base
        self.layer = layer
        self.pool = self.page.view(base, (slots, v_heads, V, K), (stride, V * K, K, 1))
        self.flat = [self.page.view(lo, (hi - lo,), (1,)) for lo, hi in self.spans]
        for f in self.flat:
            f.normal_(0.0, 0.05, generator=gen)
        torch.cuda.synchronize()
        self.before = [f.clone() for f in self.flat]

    def dense(self, slots: torch.Tensor):
        """A dense copy of the touched slots and `slots` renumbered into it (-1 stays)."""
        index = {s: i for i, s in enumerate(self.used)}
        compact = torch.stack([self.pool[s] for s in self.used]).contiguous()
        renamed = [index[s] if s >= 0 else -1 for s in slots.tolist()]
        return compact, torch.tensor(renamed, dtype=torch.int32, device=DEV)

    def touched(self) -> torch.Tensor:
        return torch.stack([self.pool[s] for s in self.used])

    def assert_rest_unchanged(self) -> None:
        """Every byte of a touched slot's page but this layer's state, and the next slot's
        first layer, is what it was."""
        torch.cuda.synchronize()
        lo, hi = self.own, self.own + self.layer
        for f, b, s in zip(self.flat, self.before, self.used):
            assert torch.equal(f[:lo], b[:lo]) and torch.equal(f[hi:], b[hi:]), (
                f"slot {s}: the call wrote outside its state"
            )

    def close(self) -> None:
        del self.pool, self.flat
        self.page.close()


def _decode_build(spec, seed):
    """`(args, engine)`: the call's arguments, and the `_DecodeEnginePool` behind them (or None)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows = spec["rows"]
    k_heads, v_heads = _heads(spec)
    width = _width(k_heads, v_heads)
    assert spec.get("proj_width", width) == width, (spec, width)
    state = torch.bfloat16 if spec.get("state") == "bf16" else torch.float32
    proj = torch.randn(rows, width, generator=g, device=DEV).to(torch.bfloat16)
    engine = None
    if spec.get("pool_slots"):
        n = spec["pool_slots"]
        picked = _scattered(rows, n)
        engine = _DecodeEnginePool(n, picked, v_heads, state, g)
        pool = engine.pool
        slots = torch.tensor(picked, dtype=torch.int32, device=DEV)
    elif spec.get("pool") == "coalesced":
        # The KV-cache-v2 manager's view: every GDN layer's state of a slot in one page,
        # this layer's state one sub-page of it, so the slot stride is 36 layers' states
        # and slot 100 sits ~2.8G elements past the base (past int32).
        layers, num = PAGE_LAYERS, 102
        page = torch.empty(num * layers * v_heads * V * K, device=DEV, dtype=state)
        page[: v_heads * V * K * 4].normal_(0.0, 0.05, generator=g)
        pool = page.as_strided((num, v_heads, V, K), (layers * v_heads * V * K, V * K, K, 1))
        for s in (100, 101):
            pool[s].copy_(torch.randn(v_heads, V, K, generator=g, device=DEV) * 0.05)
        slots = torch.tensor([100, 101][:rows], dtype=torch.int32, device=DEV)
    else:
        pool = (torch.randn(SLOTS, v_heads, V, K, generator=g, device=DEV) * 0.05).to(state)
        slots = torch.tensor(_scattered(rows, SLOTS), dtype=torch.int32, device=DEV)
    if spec.get("pad"):
        slots[1] = -1
    A_log = torch.empty(v_heads, device=DEV).uniform_(-5.5, 1.0, generator=g)
    dt_bias = torch.randn(v_heads, generator=g, device=DEV) * 4.0
    w = (1.0 + 0.2 * torch.randn(V, generator=g, device=DEV)).to(torch.bfloat16)
    return [proj, pool, slots, A_log, dt_bias, w, EPS, k_heads, K], engine


def _decode_compare(out, pool, ref_out, ref_pool) -> None:
    assert out.dtype == torch.bfloat16 and out.shape == ref_out.shape and out.is_contiguous()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(pool, ref_pool)


def _decode_stats(out, ref_out, pool, ref_pool) -> str:
    """What `note` quotes: the output against its bf16-rounded reference, the state against
    the reference rounded to the pool's dtype. Elements are counted by value (a pad row's
    -0.0 equals the reference's 0.0)."""
    o, r = out.float().flatten(), ref_out.float().flatten()
    cos = torch.nn.functional.cosine_similarity(o, r, dim=0).item()
    rel = ((o - r).norm() / r.norm().clamp_min(1e-30)).item()
    differ = (o != r).sum().item()
    p, q = pool.float().flatten(), ref_pool.float().flatten()
    p_rel = ((p - q).norm() / q.norm().clamp_min(1e-30)).item()
    p_differ = (p != q).sum().item()
    return (
        f"out max|d| {(o - r).abs().max().item():.3e}, {differ}/{o.numel()} elements differ, "
        f"cos {cos:.8f}, rel L2 {rel:.3e}; state max|d| {(p - q).abs().max().item():.3e}, "
        f"{p_differ}/{p.numel()} differ, rel L2 {p_rel:.3e}"
    )


@pytest.mark.parametrize(
    "cell",
    gdn_decode_rows_gated_norm.CELLS,
    ids=[c.why[:44] for c in gdn_decode_rows_gated_norm.CELLS],
)
def test_gdn_decode_rows_gated_norm_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(gdn_decode_rows_gated_norm)
    args, engine = _decode_build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    try:
        if engine is None:
            ref_out, ref_pool = gdn_decode_rows_gated_norm.reference(*args)
            out = gdn_decode_rows_gated_norm(*args)
            pool = args[1]
        else:
            dense, renamed = engine.dense(args[2])
            ref_out, ref_pool = gdn_decode_rows_gated_norm.reference(
                args[0], dense, renamed, *args[3:]
            )
            out = gdn_decode_rows_gated_norm(*args)
            torch.cuda.synchronize()
            pool = engine.touched()
        _decode_compare(out, pool, ref_out, ref_pool)
        if engine is not None:
            engine.assert_rest_unchanged()
        print(f"[{cell.why}] {_decode_stats(out, ref_out, pool, ref_pool)}")
        if cell.spec.get("pad"):
            assert (out[1] == 0).all(), "a pad row's output is zero"
    finally:
        if engine is not None:
            engine.close()


#: Decode batch sizes up to the engine's largest CUDA-graph batch.
_BATCHES = (1, 2, 4, 5, 16, 33, 128, 256)
_FIXED_SLOT = 301


@_BRACKET
def test_decode_bf16_pool_is_batch_invariant(k_heads, v_heads) -> None:
    """The property the bf16 decode step relies on: a row's output and state are bitwise
    independent of the batch it rides in. One fixed row (its projection and slot 301 of a
    bf16 pool) sits first and last in random batches of every size, across the batch where
    the launch switches to the bulk-copy variant (7 rows at 48 v heads, 26 at 12)."""
    width = _width(k_heads, v_heads)
    gen = torch.Generator(device=DEV).manual_seed(3)
    cpu = torch.Generator().manual_seed(3)
    pool0 = (torch.randn(SLOTS, v_heads, V, K, generator=gen, device=DEV) * 0.05).to(torch.bfloat16)
    row = torch.randn(width, generator=gen, device=DEV).to(torch.bfloat16)
    rest = _decode_build(dict(rows=1, k_heads=k_heads, v_heads=v_heads), seed=3)[0][3:]
    first = None
    for n in _BATCHES:
        for pos in sorted({0, n - 1}):
            g = torch.Generator(device=DEV).manual_seed(1000 * n + pos)
            proj = torch.randn(n, width, generator=g, device=DEV).to(torch.bfloat16)
            proj[pos] = row
            others = [s for s in torch.randperm(SLOTS, generator=cpu).tolist() if s != _FIXED_SLOT]
            others = others[:n]
            others[pos] = _FIXED_SLOT
            slots = torch.tensor(others, dtype=torch.int32, device=DEV)
            pool = pool0.clone()
            out = gdn_decode_rows_gated_norm(proj, pool, slots, *rest)
            got = (out[pos].clone(), pool[_FIXED_SLOT].clone())
            first = first or got
            assert torch.equal(got[0], first[0]) and torch.equal(got[1], first[1]), (
                f"the fixed row differs at batch {n}, position {pos}"
            )


@pytest.mark.parametrize(
    "spec",
    (
        dict(rows=16),
        dict(rows=3, k_heads=4, v_heads=12, state="bf16"),
        dict(rows=128, k_heads=4, v_heads=12, state="bf16"),
    ),
    ids=["16x48-16rows", "4x12-3rows-bf16", "4x12-128rows-bf16"],
)
def test_decode_reads_after_the_wait_under_graph_replay(spec) -> None:
    """A producer rewrites the projection inside the graph right before the op; with PDL
    a read of it before `griddepcontrol.wait` would see the previous replay's row. The
    pool advances one step per replay, as in decode. At 4 / 12 heads 3 rows take the
    register variant, 128 the bulk copy."""
    args = _decode_build(spec, seed=41)[0]
    src = args[0].clone()

    def step():
        args[0].copy_(src * 1.0)
        return gdn_decode_rows_gated_norm(*args)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    for trial in range(3):
        src.copy_(_decode_build(spec, seed=100 + trial)[0][0])
        before = args[1].clone()
        graph.replay()
        torch.cuda.synchronize()
        after = args[1].clone()
        args[1].copy_(before)
        want = gdn_decode_rows_gated_norm(src.clone(), *args[1:])
        assert torch.equal(graph_out, want), f"replay {trial} output"
        assert torch.equal(after, args[1]), f"replay {trial} pool"


_VARIANT_SPECS = (
    dict(rows=33, state="bf16"),
    dict(rows=33, pad=True),
    dict(rows=2, pool="coalesced", state="bf16", pad=True),
    dict(rows=3, k_heads=4, v_heads=12, state="bf16", pad=True),
    dict(rows=128, k_heads=4, v_heads=12, state="bf16"),
    dict(rows=3, k_heads=4, v_heads=12, state="bf16", pool_slots=rows_mod.TEP4_SLOTS),
)


@pytest.mark.parametrize("norm", [True, False], ids=["norm", "rows"])
@pytest.mark.parametrize("spec", _VARIANT_SPECS, ids=lambda s: "-".join(map(str, s.values())))
def test_decode_bulk_and_register_variants_agree(spec, norm) -> None:
    """The bulk-copy and the register variants: outputs and state bitwise equal.

    The bulk-copy variant stages the state rows in shared memory; the register variant holds
    them in registers; both compute the same instructions. Over contiguous bf16 and fp32 pools,
    the tp1 coalesced page (slot 100, ~2.8G elements past the base, by the bulk copy) and
    tep4's engine page (slot 1873, ~1.3e10 elements past the base), with pad rows, at 16 / 48
    and 4 / 12 heads.
    """
    args, engine = _decode_build(spec, seed=zlib.crc32(repr(spec).encode()) % (2**31))
    try:
        proj, pool, slots, k_heads = args[0], args[1], args[2], args[7]
        assert rows_mod._bulk_eligible(pool, K)
        live = slots[slots >= 0].long()
        before = pool[live].clone()
        got = []
        for bulk in (False, True):
            pool[live] = before
            w = args[5] if norm else None
            out = rows_mod.launch(
                proj, pool, slots, args[3], args[4], w, EPS, k_heads, K, bulk=bulk
            )
            got.append((out.clone(), pool[live].clone()))
        assert torch.equal(got[0][0], got[1][0]), "outputs differ between the variants"
        assert torch.equal(got[0][1], got[1][1]), "states differ between the variants"
        if engine is not None:
            engine.assert_rest_unchanged()
    finally:
        if engine is not None:
            engine.close()


@_BRACKET
def test_decode_variant_choice_and_unaligned_pool(k_heads, v_heads) -> None:
    """The variant choice, and the fallback for a pool the bulk copy cannot take.

    The bulk variant from the batch the register variant needs a second wave for; a pool 8-
    but not 16-byte aligned stays on the register variant whatever is asked, and matches the
    aligned pool's result.
    """
    sms = torch.cuda.get_device_properties(DEV).multi_processor_count
    edge = (2 * sms) // v_heads
    assert not rows_mod._use_bulk(edge, v_heads, DEV) and rows_mod._use_bulk(edge + 1, v_heads, DEV)
    print(f"[{v_heads} v heads] {sms} SMs: the bulk-copy variant from {edge + 1} rows")
    args = _decode_build(dict(rows=33, k_heads=k_heads, v_heads=v_heads, state="bf16"), seed=19)[0]
    pool = args[1]
    flat = torch.empty(pool.numel() + 4, dtype=pool.dtype, device=DEV)
    shifted = flat[4:].view(pool.shape)
    shifted.copy_(pool)
    assert not rows_mod._bulk_eligible(shifted, K)
    want = rows_mod.launch(args[0], pool, *args[2:], bulk=True)
    got = rows_mod.launch(args[0], shifted, *args[2:], bulk=True)
    assert torch.equal(got, want) and torch.equal(shifted, pool)


def _rows_args(spec, seed):
    """`(args, buf, engine)`: the entry's arguments, `out` rows 5.. of a `[1, rows + 9, HV, V]`
    buffer (the step's output, the call's rows a range of it), that buffer, and the
    `_EnginePool` behind the pool (or None)."""
    (proj, pool, slots, A_log, dt_bias, _, _, h, k), engine = _decode_build(spec, seed)
    rows, v_heads = proj.shape[0], pool.shape[1]
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    buf = torch.randn(1, rows + 9, v_heads, V, generator=g, device=DEV).to(torch.bfloat16)
    out = buf[0, 5 : 5 + rows].view(rows, v_heads * V)
    return [proj, pool, slots, A_log, dt_bias, h, k, out], buf, engine


@pytest.mark.parametrize(
    "cell", gdn_decode_rows.CELLS, ids=[c.why[:44] for c in gdn_decode_rows.CELLS]
)
def test_gdn_decode_rows_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(gdn_decode_rows)
    args, buf, engine = _rows_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    try:
        outside = torch.cat([buf[0, :5], buf[0, 5 + args[0].shape[0] :]]).clone()
        if engine is None:
            ref_o, ref_pool = gdn_decode_rows.reference(*args)
            got = gdn_decode_rows(*args)
            pool = args[1]
        else:
            dense, renamed = engine.dense(args[2])
            ref_o, ref_pool = gdn_decode_rows.reference(args[0], dense, renamed, *args[3:])
            got = gdn_decode_rows(*args)
            torch.cuda.synchronize()
            pool = engine.touched()
        assert got.data_ptr() == args[-1].data_ptr(), "written into out"
        torch.testing.assert_close(got, ref_o)
        torch.testing.assert_close(pool, ref_pool)
        if engine is not None:
            engine.assert_rest_unchanged()
        print(f"[{cell.why}] {_decode_stats(got, ref_o, pool, ref_pool)}")
        assert torch.equal(torch.cat([buf[0, :5], buf[0, 5 + args[0].shape[0] :]]), outside), (
            "rows outside the range untouched"
        )
        if cell.spec.get("pad"):
            assert (got[1] == 0).all(), "a pad row's output is zero"
    finally:
        if engine is not None:
            engine.close()


# ----------------------------- fused_recurrent_gated_delta_rule_cached_replay_update ---

HIST = HISTORY_SIZE
BF16_ULP = ULP[torch.bfloat16]

#: This checkpoint's 36 GDN layers of `linear_attn.A_log` / `linear_attn.dt_bias`
#: (bf16, 48 each): mean, standard deviation, range.
A_LOG_STATS = (1.006, 1.487, (-7.125, 5.0625))
DT_BIAS_STATS = (-4.411, 1.732, (-11.4375, 2.53125))

#: Element gate of the bf16 members, in bf16 ulps of the block they belong to.
_ELEMENT_ULPS = 4.0
#: Relative-RMS gate of the output, in bf16 ulps.
_OUTPUT_RMS_ULPS = 1.5
#: Element gate of an fp32 checkpoint state, relative to its (slot, head) scale.
_STATE_TOL = 2.0**-13
#: Element gate of a bf16 checkpoint state, in bf16 ulps of its (slot, head) scale.
_BF16_STATE_ULPS = 2.0
#: Gate of the stored cumulative log-decays: |dG| <= atol + rtol * |G|.
_G_ATOL, _G_RTOL = 2.0**-11, 2.0**-18

#: The largest deviation each gated member reached in the running cell, in units of
#: its gate's measure (printed per cell, so the margins are on record).
_MEASURED: dict[str, float] = {}


def _record(key: str, value: float) -> None:
    _MEASURED[key] = max(_MEASURED.get(key, 0.0), value)


def _gate(name, got, want, element, rms=None):
    """`got` against `want` per block of the last two dims (a token's
    `[heads, dim]`, or one head's `[V, K]` state): every element within
    `element` of its block's absmax, NaN blocks of `want` excluded."""
    block = got.shape[-2] * got.shape[-1]
    g, w = got.float().reshape(-1, block), want.float().reshape(-1, block)
    keep = ~torch.isnan(w).any(1)
    g, w = g[keep], w[keep]
    scale = w.abs().amax(1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    ratio = ((g - w).abs() / scale).nan_to_num(nan=float("inf"))
    worst = ratio.max().item() if ratio.numel() else 0.0
    _record(f"{name} element/gate", worst / element)
    assert worst <= element, (
        f"{name}: an element is {worst:.3e} of its block's absmax from the reference "
        f"(gate {element:.3e})"
    )
    if rms is not None and w.numel():
        rel = ((g - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt().clamp_min(1e-30)).item()
        _record(f"{name} rms/gate", rel / rms)
        assert rel <= rms, f"{name}: relative RMS {rel:.3e} exceeds {rms:.3e}"


def _replay_compare(out: tuple, ref: tuple) -> None:
    """`(returned, ssm_states, old_u, old_k, old_G)` as the call left them,
    against `reference`'s five; a member `None` on both sides is skipped (the
    exact twin is compared on the output alone).

    The kernel is not fp32 throughout, and the bands say where. It rounds the
    L2-normalized q and k to bf16 before every product (2^-9 relative each),
    rounds the history-mixing coefficients to bf16 for its tensor-core dots, and
    stores the output and the history's u and k in bf16. So the output, `old_u`
    and `old_k` are gated per block -- one token's `[heads, dim]` -- at 4 bf16
    ulps of the block's absmax, which keeps a near-silent head from turning
    rounding into relative noise, and the output's relative RMS at 1.5 ulps. An fp32
    checkpoint state is fp32 by construction: the commit splits the decayed
    history update into a bf16 hi and lo part, ~16 significant bits per term, and
    up to 16 terms each within 2^-17 give the 2^-13 gate, relative to one (slot,
    head) state's absmax. A bf16 checkpoint is that fp32 commit rounded once, as the
    reference's is, so the two stored values are equal or adjacent, and adjacent bf16
    values are at most 2 bf16 ulps of their head's absmax apart: the bf16 gate is one
    representable step. `G` only ever enters as exp(G_t - G_j), so an absolute
    error is a relative decay error: the kernel's softplus runs on approximate
    transcendentals whose fp32 sum drops exp(x)'s low bits; atol 2^-11 is an
    eighth of a bf16 ulp of the u it scales, and rtol 2^-18 covers the cumulative
    sum's rounding where |G| reaches thousands on fast-forgetting heads.
    """
    names = ("output", "ssm_states", "old_u", "old_k", "old_G")
    assert len(out) == len(ref) == len(names)
    for name, got, want in zip(names, out, ref):
        assert (got is None) == (want is None), f"{name}: present on one side only"
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape, (
            name,
            (got.dtype, tuple(got.shape)),
            (want.dtype, tuple(want.shape)),
        )
        if name == "old_G":
            bound = _G_ATOL + _G_RTOL * want.abs()
            _record("old_G |dG|/gate", ((got - want).abs() / bound).max().item())
            torch.testing.assert_close(got, want, rtol=_G_RTOL, atol=_G_ATOL, msg=name)
        elif name == "ssm_states":
            bf16 = got.dtype == torch.bfloat16
            _gate(name, got, want, _BF16_STATE_ULPS * BF16_ULP if bf16 else _STATE_TOL)
        else:
            rms = _OUTPUT_RMS_ULPS * BF16_ULP if name == "output" else None
            _gate(name, got, want, _ELEMENT_ULPS * BF16_ULP, rms)


def _seed(why):
    return zlib.crc32(why.encode())


def _gating(gen, layers, hv=HV):
    """Per-layer A_log / dt_bias drawn around the checkpoint's own statistics;
    head 0 pinned to the corner that forgets everything every token
    (g ~ -400), head 1 to the one that hardly decays (g ~ -1e-8)."""
    (a_mean, a_std, a_range), (d_mean, d_std, d_range) = A_LOG_STATS, DT_BIAS_STATS
    a_log = torch.randn(layers, hv, generator=gen, device=DEV) * a_std + a_mean
    dt_bias = torch.randn(layers, hv, generator=gen, device=DEV) * d_std + d_mean
    a_log, dt_bias = a_log.clamp(*a_range), dt_bias.clamp(*d_range)
    a_log[:, 0], dt_bias[:, 0] = a_range[1], d_range[1]
    a_log[:, 1], dt_bias[:, 1] = a_range[0], d_range[0]
    return a_log.contiguous(), dt_bias.contiguous()


def _unit(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


class _ReplayEnginePage:
    """The stock manager's coalesced state page at `slots` slots: a slot's page holds
    `page_layers` layers' `[HV, V, K]` states, and the bench's `layers` are its top
    sub-pages, the page's highest addresses. Memory is mapped under each touched slot's
    page plus the next slot's first sub-page (where a write past the page would land),
    and under slot 0's page, which holds the views' base."""

    def __init__(self, slots, page_layers, layers, used, hv, dtype, gen):
        state = hv * V * K
        stride = page_layers * state
        total = slots * stride
        self.state, self.stride, self.layers = state, stride, layers
        self.top = (page_layers - layers) * state
        self.used = sorted(set(used))
        self.mapped = sorted(set(self.used) | {0})
        self.spans = [(s * stride, min((s + 1) * stride + state, total)) for s in self.mapped]
        self.page = _SparsePage(total, self.spans, dtype)
        self.pools = [
            self.page.view(self.top + j * state, (slots, hv, V, K), (stride, V * K, K, 1))
            for j in range(layers)
        ]
        self.flat = [self.page.view(lo, (hi - lo,), (1,)) for lo, hi in self.spans]
        for f in self.flat:
            f.normal_(0.0, 0.5, generator=gen)
        torch.cuda.synchronize()
        self.before = [f.clone() for f in self.flat]

    def assert_rest_unchanged(self) -> None:
        """Every mapped byte outside the touched slots' own sub-pages is what it was."""
        torch.cuda.synchronize()
        lo, hi = self.top, self.top + self.layers * self.state
        for s, f, b in zip(self.mapped, self.flat, self.before):
            if s in self.used:
                same = torch.equal(f[:lo], b[:lo]) and torch.equal(f[hi:], b[hi:])
            else:
                same = torch.equal(f, b)
            assert same, f"slot {s}: the page was written outside the cell's sub-pages"

    def close(self) -> None:
        del self.pools, self.flat, self.before
        self.page.close()


class _ReplayBench:
    """The state the cache manager keeps for `layers` GDN layers, laid out as
    `MambaHybridCacheManagerV2` lays it out, driven the way the engine drives
    it: work items before the forward (`_build_replay_work_items_triton`, what
    mamba metadata's prepare runs for a partitioned batch), the op per layer,
    then `update_mamba_states`'s order -- the all-layer commit, then
    `_advance_replay_state`.

    With `exact=True` it also carries the exact twin: the entry's `reference` driven on the
    same checkpoint and history held in fp32 and deep enough never to wrap, advanced by
    `p += accepted` alone. Its state is the plain token-by-token recurrence over every accepted
    token, independent of the wrap, the double buffer and the commit, so a wrong commit or
    wrong bookkeeping surfaces as an output that drifts off it.
    Over an engine page (`page_layers`) the reference and the twin see a dense copy
    of the touched slots: `idx` the slots in order, `position` a slot's row in it.
    """

    def __init__(self, spec, seed, exact=True):
        self.spec = spec
        self.path, self.n, self.t = spec["path"], spec["n"], spec["t"]
        self.h, self.hv = spec.get("H", H), spec.get("HV", HV)
        assert spec.get("history_size", HIST) == HIST, spec
        self.width = 2 * self.h * K + self.hv * V
        self.layers = spec.get("layers", 2)
        self.slots = spec.get("slots", 64)
        self.steps = spec.get("steps", 8)
        self.gen = torch.Generator(device=DEV).manual_seed(seed)
        self.cpu = torch.Generator().manual_seed(seed)
        n, t, L, S = self.n, self.t, self.layers, self.slots
        h, hv = self.h, self.hv
        page_layers = spec.get("page_layers")

        if page_layers:
            chosen = torch.tensor(_scattered(n, S))
        elif spec.get("edge_slots"):
            inner = torch.randperm(S - 2, generator=self.cpu)[: n - 2] + 1
            chosen = torch.cat([torch.tensor([S - 1]), inner, torch.tensor([0])])
        else:
            chosen = torch.randperm(S, generator=self.cpu)[:n]
        chosen = chosen.to(torch.int32)
        live_rows = torch.ones(n, dtype=torch.bool)
        dup = tuple(spec.get("dup", ()))
        if dup:
            chosen[list(dup)] = int(chosen[dup[0]])
            # The padding rows' shared slot is nobody's state: the bookkeeping never names it.
            live_rows[list(dup)] = False
        self.state_indices = chosen.to(DEV)
        self.live = self.state_indices[live_rows.to(DEV)].clone()

        state_dtype = spec.get("state_dtype", torch.float32)
        self.page = self.idx = self.position = None
        if page_layers:
            self.page = _ReplayEnginePage(
                S, page_layers, L, chosen.tolist(), hv, state_dtype, self.gen
            )
            self.pool = self.page.pools
            self.layer_stride, self.slot_stride = self.page.state, self.page.stride
            self.idx = torch.tensor(self.page.used, dtype=torch.long, device=DEV)
            self.position = torch.full((S,), -1, dtype=torch.long, device=DEV)
            self.position[self.idx] = torch.arange(self.idx.numel(), device=DEV)
        else:
            # [slots, layers, HV, V, K]: each layer's state is a slot-strided view,
            # the affine layout whose strides the all-layer commit is handed.
            self.backing = (self._randn(S, L, hv, V, K) * 0.5).to(state_dtype)
            self.pool = [self.backing[:, i] for i in range(L)]
            self.layer_stride, self.slot_stride = self.backing.stride(1), self.backing.stride(0)
        self.old_x = torch.zeros(L, S, 2, HIST, hv, V, device=DEV, dtype=torch.bfloat16)
        self.old_B = torch.zeros(L, S, 2, HIST, h, K, device=DEV, dtype=torch.bfloat16)
        self.old_dt = torch.zeros(L, S, 2, hv, HIST, device=DEV)
        self.old_dA = self._randn(L, S, 2, hv, HIST)  # handed over as old_beta; never read
        self.pnat = torch.zeros(S, dtype=torch.int32, device=DEV)
        self.buf = torch.zeros(S, dtype=torch.int32, device=DEV)
        self.a_log, self.dt_bias = _gating(self.gen, L, hv)
        if spec.get("start", "warm") == "warm":
            self.old_x.copy_(self._randn(L, S, 2, HIST, hv, V) * 0.5)
            self.old_B.copy_(_unit(self._randn(L, S, 2, HIST, h, K)))
            raw_a = self._randn(L, S, 2, hv, HIST)
            g = -torch.exp(self.a_log)[:, None, None, :, None] * F.softplus(
                raw_a + self.dt_bias[:, None, None, :, None]
            )
            self.old_dt.copy_(g.cumsum(-1))
            pnat = torch.randint(0, HIST + 1, (S,), generator=self.cpu, dtype=torch.int32)
            live = self.live.long().cpu()
            pnat[live[0]] = HIST  # a wrap at step 0
            if live.numel() > 1:
                pnat[live[1]] = 0  # an append at step 0
            self.pnat.copy_(pnat)
            self.buf.copy_(torch.randint(0, 2, (S,), generator=self.cpu, dtype=torch.int32))
        self.meta = ReplayStateUpdateMetadata(self.pnat, self.buf, t, HIST)
        self.work_items = self.n_writes = None
        if self.path == "all_layer":
            # The engine hands the op work items exactly from the batch the op's all-layer
            # commit starts at.
            assert n >= ALL_LAYER_COMMIT_MIN_BATCH == CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE, n
            self.work_items = torch.zeros(n, 4, dtype=torch.int32, device=DEV)
            self.n_writes = torch.zeros(1, dtype=torch.int32, device=DEV)

        self.exact = exact
        if exact:
            self.depth = HIST + (self.steps + 1) * t
            self.x_pool = [self.compact(p).clone() for p in self.pool]
            rows = self.x_pool[0].shape[0]
            self.x_u = torch.zeros(L, rows, 1, self.depth, hv, V, device=DEV)
            self.x_k = torch.zeros(L, rows, 1, self.depth, h, K, device=DEV)
            self.x_G = torch.zeros(L, rows, 1, hv, self.depth, device=DEV)
            self.x_pnat = self.compact(self.pnat).clone()
            self.x_buf = torch.zeros_like(self.x_pnat)
            for s in self.live.tolist():
                p, b = int(self.pnat[s]), int(self.buf[s])
                c = s if self.position is None else int(self.position[s])
                self.x_u[:, c, 0, :p] = self.old_x[:, s, b, :p].float()
                self.x_k[:, c, 0, :p] = self.old_B[:, s, b, :p].float()
                self.x_G[:, c, 0, :, :p] = self.old_dt[:, s, b, :, :p]

    def _randn(self, *shape, dtype=torch.float32):
        return torch.randn(*shape, generator=self.gen, device=DEV, dtype=dtype)

    def compact(self, per_slot: torch.Tensor) -> torch.Tensor:
        """A per-slot tensor as the reference sees it: whole over a dense pool, the touched
        slots' rows (a copy, in `idx` order) over an engine page."""
        return per_slot if self.idx is None else per_slot[self.idx]

    def compact_indices(self, slots: torch.Tensor) -> torch.Tensor:
        """Slot ids as rows of `compact`'s output (negative ids stay negative)."""
        if self.position is None:
            return slots
        renamed = torch.where(slots >= 0, self.position[slots.long().clamp(min=0)], -1)
        return renamed.to(slots.dtype)

    def close(self) -> None:
        if self.page is not None:
            self.pool = []
            self.page.close()

    def inputs(self):
        """One layer's fresh activations: post-conv `[q|k|v]` rows and the raw
        `a`/`b` projections, as the verify step hands them over."""
        n, t, h, hv = self.n, self.t, self.h, self.hv
        packed = self._randn(n * t, self.width, dtype=torch.bfloat16)
        rows = packed.view(n, t, self.width)
        q = rows[..., : h * K].view(n, t, h, K)
        k = rows[..., h * K : 2 * h * K].view(n, t, h, K)
        v = rows[..., 2 * h * K :].view(n, t, hv, V)
        a = self._randn(n, t, hv, dtype=torch.bfloat16)
        b = self._randn(n, t, hv, dtype=torch.bfloat16)
        return dict(q=q, k=k, v=v, a=a, b=b, packed=packed)

    def kwargs(self, layer, act):
        return dict(
            q=act["q"],
            k=act["k"],
            v=act["v"],
            g=act["a"],
            beta=act["b"],
            ssm_states=self.pool[layer],
            state_indices=self.state_indices,
            old_u=self.old_x[layer],
            old_k=self.old_B[layer],
            old_G=self.old_dt[layer],
            old_beta=self.old_dA[layer],
            cache_buf_idx=self.buf,
            prev_num_accepted_tokens=self.pnat,
            history_size=HIST,
            scale=None,
            use_qk_l2norm_in_kernel=True,
            A_log=self.a_log[layer],
            dt_bias=self.dt_bias[layer],
            launch_with_pdl=False,
            replay_work_items=self.work_items,
            n_writes=self.n_writes,
            packed_qkv=act["packed"],
            use_all_layer_commit=self.path == "all_layer",
            output=None,
        )

    def reference(self, kw):
        """The entry's `reference` on a call's arguments, every per-slot buffer through
        `compact`: the touched slots' rows, the rows' slots renumbered into them."""
        if self.idx is None:
            return replay_update.reference(**kw)
        per_slot = (
            "ssm_states",
            "old_u",
            "old_k",
            "old_G",
            "old_beta",
            "cache_buf_idx",
            "prev_num_accepted_tokens",
        )
        compact = {name: self.compact(kw[name]) for name in per_slot}
        return replay_update.reference(
            **dict(kw, **compact, state_indices=self.compact_indices(kw["state_indices"]))
        )

    def exact_kwargs(self, layer, kw):
        return dict(
            kw,
            ssm_states=self.x_pool[layer],
            state_indices=self.compact_indices(kw["state_indices"]),
            old_u=self.x_u[layer],
            old_k=self.x_k[layer],
            old_G=self.x_G[layer],
            cache_buf_idx=self.x_buf,
            prev_num_accepted_tokens=self.x_pnat,
            history_size=self.depth,
            replay_work_items=None,
            n_writes=None,
            use_all_layer_commit=False,
            output=None,
        )

    def adopt_exact(self, layer, ref):
        assert torch.equal(ref[1], self.x_pool[layer]), "the twin must never wrap"
        self.x_u[layer].copy_(ref[2])
        self.x_k[layer].copy_(ref[3])
        self.x_G[layer].copy_(ref[4])

    def state(self, layer):
        return self.pool[layer], self.old_x[layer], self.old_B[layer], self.old_dt[layer]

    def observed(self, layer):
        """`state(layer)` through `compact`, as `reference`'s results are laid out."""
        return tuple(self.compact(x) for x in self.state(layer))

    def unwritten(self, wrapped):
        """Rows of `observed`'s pool the call must leave as they were: all of them on the
        partitioned path, all but the wrapping slots' when the call commits in-kernel."""
        rows = self.slots if self.idx is None else self.idx.numel()
        keep = torch.ones(rows, dtype=torch.bool, device=DEV)
        if self.path == "small":
            keep[self.compact_indices(wrapped).long()] = False
        return keep

    def prepare(self):
        if self.path == "all_layer":
            self.work_items.zero_()
            _build_replay_work_items_triton(
                self.live, self.pnat, self.buf, self.work_items, self.n_writes, self.t, HIST
            )

    def commit(self):
        if self.path == "all_layer":
            commit_gdn_cached_replay_history_layers(
                ssm_states=self.pool[0],
                ssm_state_num_layers=self.layers,
                ssm_state_layer_stride=self.layer_stride,
                ssm_state_slot_stride=self.slot_stride,
                old_u=self.old_x,
                old_k=self.old_B,
                old_G=self.old_dt,
                replay_work_items=self.work_items,
                n_writes=self.n_writes,
                history_size=HIST,
            )

    def draw_acceptance(self):
        """Accepted tokens per live row, bonus token included, by `spec["accept"]`."""
        mode = self.spec.get("accept", "random")
        rows = self.live.numel()
        if mode == "full":
            return torch.full((rows,), self.t, dtype=torch.int32, device=DEV)
        assert mode == "random", mode
        return torch.randint(1, self.t + 1, (rows,), generator=self.cpu, dtype=torch.int32).to(DEV)

    def accept(self, accepted):
        _advance_replay_state(self.meta, self.live, accepted)
        if self.exact:
            self.x_pnat[self.compact_indices(self.live).long()] += accepted

    def writes(self):
        """Where this step's calls append, by the documented rule, which slots wrap, and how
        many append exactly up to the history's end: `[slots, 2, HIST]` mask, wrapping slot
        ids, the `pnat + t == HIST` count."""
        s = self.live.long()
        p, b = self.pnat[s].long(), self.buf[s].long()
        wrap = p + self.t > HIST
        half, start = torch.where(wrap, 1 - b, b), torch.where(wrap, 0, p)
        mask = torch.zeros(self.slots, 2, HIST, dtype=torch.bool, device=DEV)
        for t in range(self.t):
            mask[s, half, start + t] = True
        return mask, s[wrap], int((p + self.t == HIST).sum())


class _ReplayCaptured:
    """The cell's verify step captured once in a CUDA graph, as the engine's padded decode
    graph holds it: on a twin bench built from the cell's own seed (so its state starts
    bitwise the eager bench's), the op over every layer and, on the partitioned path, the
    manager's all-layer commit after them, over static activation buffers. The work items
    are built outside the graph before every replay, as mamba metadata's prepare builds
    them. Each step copies the eager step's activations in and replays."""

    def __init__(self, spec, seed):
        # The Triton kernels compile on first call: run one eager step on a throwaway bench
        # of the same layout, so nothing compiles under capture.
        warm = _ReplayBench(spec, seed, exact=False)
        try:
            warm.prepare()
            for layer in range(warm.layers):
                replay_update(**warm.kwargs(layer, warm.inputs()))
            warm.commit()
            torch.cuda.synchronize()
        finally:
            warm.close()
        self.bench = bench = _ReplayBench(spec, seed, exact=False)
        self.acts = [bench.inputs() for _ in range(bench.layers)]
        bench.prepare()
        torch.cuda.synchronize()
        self.replays = 0
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outs = [
                replay_update(**bench.kwargs(layer, self.acts[layer]))
                for layer in range(bench.layers)
            ]
            bench.commit()
        torch.cuda.synchronize()

    def assert_same_start(self, eager) -> None:
        for layer in range(eager.layers):
            for x, y in zip(eager.observed(layer), self.bench.observed(layer)):
                assert torch.equal(x, y), "the twin bench must start bitwise the eager one"
        for name in ("pnat", "buf", "state_indices"):
            assert torch.equal(getattr(eager, name), getattr(self.bench, name)), name

    def step(self, acts) -> None:
        for static, fresh in zip(self.acts, acts):
            for key in ("packed", "a", "b"):  # q, k, v are views of `packed`
                static[key].copy_(fresh[key])
        self.bench.prepare()
        self.graph.replay()
        torch.cuda.synchronize()
        self.replays += 1

    def assert_matches(self, eager, outs, rows, keep) -> None:
        """The replay against the eager step, bitwise: each layer's output `rows`, the `keep`
        rows of every per-slot buffer as `observed` lays them out, the bookkeeping, and every
        byte of the page outside the touched slots' sub-pages."""
        for layer, (e, g) in enumerate(zip(outs, self.outs)):
            assert torch.equal(e[rows], g[rows]), f"layer {layer}: the replayed output differs"
            names = ("ssm_states", "old_u", "old_k", "old_G")
            for name, x, y in zip(names, eager.observed(layer), self.bench.observed(layer)):
                assert torch.equal(x[keep], y[keep]), f"layer {layer}: the replayed {name} differs"
        for name in ("pnat", "buf"):
            assert torch.equal(getattr(eager, name), getattr(self.bench, name)), name
        if self.bench.page is not None:
            self.bench.page.assert_rest_unchanged()

    def accept(self, accepted) -> None:
        self.bench.accept(accepted)

    def close(self) -> None:
        del self.graph, self.outs
        self.bench.close()


def _check_duplicate_padding_slot(cell, bench, graph=None) -> None:
    """The last rows of a batch share one slot, as CUDA-graph padding repeats the manager's
    dummy slot: the rows on unique slots, and every slot but the shared one, are held to
    the reference run on those rows alone -- the padding rows change nothing about them.
    The shared slot's state and the padding rows' outputs are unspecified. Over an engine
    page the slots are `observed`'s rows; every byte outside their sub-pages is unchanged.
    With `graph`, every step is also replayed and held bitwise to the eager step, outside
    the shared slot and the padding rows."""
    dup = cell.spec["dup"]
    unique = bench.n - len(dup)
    assert dup == tuple(range(unique, bench.n)), "the padding rows trail the batch"
    shared = int(bench.state_indices[unique])
    assert (bench.state_indices[unique:] == shared).all()
    assert shared not in bench.live.tolist()
    rows = bench.slots if bench.idx is None else bench.idx.numel()
    others = torch.ones(rows, dtype=torch.bool, device=DEV)
    others[bench.compact_indices(torch.tensor([shared], device=DEV)).long()] = False
    if graph is not None:
        graph.assert_same_start(bench)
    wraps = appends = 0
    for _ in range(bench.steps):
        _, wrapped, _ = bench.writes()
        wraps += wrapped.numel()
        appends += bench.live.numel() - wrapped.numel()
        acts, outs = [], []
        for layer in range(bench.layers):
            act = bench.inputs()
            kw = bench.kwargs(layer, act)
            alone = dict(
                kw,
                q=kw["q"][:unique],
                k=kw["k"][:unique],
                v=kw["v"][:unique],
                g=kw["g"][:unique],
                beta=kw["beta"][:unique],
                state_indices=kw["state_indices"][:unique],
                packed_qkv=act["packed"][: unique * bench.t],
            )
            ref = bench.reference(alone)
            got = replay_update(**kw)
            _replay_compare(
                (got[:unique], *(x[others] for x in bench.observed(layer))),
                (ref[0], *(x[others] for x in ref[1:])),
            )
            acts.append(act)
            outs.append(got)
        if bench.page is not None:
            bench.page.assert_rest_unchanged()
        if graph is not None:
            graph.step(acts)
            graph.assert_matches(bench, outs, slice(0, unique), others)
        accepted = bench.draw_acceptance()
        bench.accept(accepted)
        if graph is not None:
            graph.accept(accepted)
    assert wraps > 0 and appends > 0, (
        f"the cell never wrapped or never appended ({wraps}, {appends})"
    )
    assert graph is None or graph.replays == bench.steps, "a step was not replayed"


@pytest.mark.parametrize("cell", replay_update.CELLS, ids=[c.why[:44] for c in replay_update.CELLS])
def test_cached_replay_update_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(replay_update)
    _MEASURED.clear()
    dup = "dup" in cell.spec
    bench = _ReplayBench(cell.spec, seed=_seed(cell.why), exact=not dup)
    graph = None
    try:
        if cell.spec.get("capture"):
            graph = _ReplayCaptured(cell.spec, _seed(cell.why))
        if dup:
            _check_duplicate_padding_slot(cell, bench, graph)
        else:
            _drive_replay_cell(cell, bench, graph)
    finally:
        if graph is not None:
            graph.close()
        bench.close()
    replayed = f"; {cell.spec['steps']} steps replayed bitwise" if cell.spec.get("capture") else ""
    print(
        f"\n[{cell.why}] "
        + ", ".join(f"{k} {v:.3f}" for k, v in sorted(_MEASURED.items()))
        + replayed
    )


def _drive_replay_cell(cell, bench, graph=None) -> None:
    if graph is not None:
        graph.assert_same_start(bench)
    wraps = appends = to_the_end = 0
    for _ in range(bench.steps):
        mask, wrapped, at_end = bench.writes()
        wraps += wrapped.numel()
        to_the_end += at_end
        appends += bench.live.numel() - wrapped.numel()
        bench.prepare()
        after_commit, acts = [], []
        for layer in range(bench.layers):
            act = bench.inputs()
            acts.append(act)
            kw = bench.kwargs(layer, act)
            _, u, k, G = bench.state(layer)
            before = dict(
                pool=bench.observed(layer)[0].clone(),
                u=u.clone(),
                k=k.clone(),
                G=G.clone(),
                old_beta=bench.old_dA[layer].clone(),
            )
            ref = bench.reference(kw)
            exact = replay_update.reference(**bench.exact_kwargs(layer, kw))
            committed = None
            if bench.path == "all_layer":
                committed = bench.reference(
                    dict(kw, replay_work_items=None, n_writes=None, use_all_layer_commit=False)
                )
            got = replay_update(**kw)
            _replay_compare((got, *bench.observed(layer)), ref)
            # The locks: in place, and nothing outside the documented writes.
            _, u, k, G = bench.state(layer)
            assert torch.equal(bench.old_dA[layer], before["old_beta"]), "old_beta is never written"
            assert torch.equal(u[~mask], before["u"][~mask]), "old_u written outside the append"
            assert torch.equal(k[~mask], before["k"][~mask]), "old_k written outside the append"
            Gt, Gt0 = G.transpose(-1, -2), before["G"].transpose(-1, -2)
            assert torch.equal(Gt[~mask], Gt0[~mask]), "old_G written outside the append"
            kept = bench.unwritten(wrapped)
            assert torch.equal(bench.observed(layer)[0][kept], before["pool"][kept]), (
                "the pool moved where the call commits nothing (a partitioned call commits none)"
            )
            bench.adopt_exact(layer, exact)
            after_commit.append((got, exact, committed))
        bench.commit()
        for layer, (got, exact, committed) in enumerate(after_commit):
            if committed is not None:
                # The partitioned call plus the manager's commit is the call
                # that commits in-kernel.
                _replay_compare((got, *bench.observed(layer)), committed)
            _replay_compare((got, None, None, None, None), (exact[0], None, None, None, None))
        if bench.page is not None:
            bench.page.assert_rest_unchanged()
        if graph is not None:
            graph.step(acts)
            everything = torch.ones(bench.observed(0)[0].shape[0], dtype=torch.bool, device=DEV)
            outs = [got for got, _, _ in after_commit]
            graph.assert_matches(bench, outs, slice(None), everything)
        accepted = bench.draw_acceptance()
        bench.accept(accepted)
        if graph is not None:
            graph.accept(accepted)
    assert wraps > 0 and appends > 0, (
        f"the cell never wrapped or never appended ({wraps}, {appends})"
    )
    assert graph is None or graph.replays == bench.steps, "a step was not replayed"
    if cell.spec.get("accept") == "full":
        live = bench.live.numel()
        assert wraps >= 3 * live and to_the_end >= 3 * live, (
            f"full acceptance should wrap every row at least 3 times and append up to the "
            f"history's end as often ({wraps}, {to_the_end} over {live} rows)"
        )


def test_the_replay_output_gate_refuses_reversed_key_heads_at_dep4() -> None:
    """At a dep4 rank's 16 / 48 heads each group of three value heads reads its own key head
    (`hv // 3`). A reference handed the new tokens' key heads reversed -- value heads 0..2
    on key head 15 -- stands for a wrong head mapping: the output gate must refuse it."""
    cell = next(c for c in replay_update.CELLS if c.why.startswith("dep4 all_layer n=16 "))
    bench = _ReplayBench(cell.spec, seed=_seed(cell.why), exact=False)
    try:
        bench.prepare()
        kw = bench.kwargs(0, bench.inputs())
        right = bench.reference(kw)
        wrong = bench.reference(dict(kw, q=kw["q"].flip(2), k=kw["k"].flip(2), packed_qkv=None))
        got = replay_update(**kw)
        torch.cuda.synchronize()
    finally:
        bench.close()
    _replay_compare((got, None, None, None, None), (right[0], None, None, None, None))
    with pytest.raises(AssertionError):
        _replay_compare((got, None, None, None, None), (wrong[0], None, None, None, None))
    g, w = got.float(), wrong[0].float()
    rel = ((g - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt()).item()
    print(f"\nreversed key heads: output relative RMS {rel / BF16_ULP:.1f} bf16 ulps off")
