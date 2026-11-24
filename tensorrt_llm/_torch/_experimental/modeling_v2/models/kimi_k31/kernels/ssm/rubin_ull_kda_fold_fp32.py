# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
"""The K3.1 flat chain's folded KDA verify pair for three or four requests' q8 rows (24 / 32
rows): MR8's ``KDA_FOLD64`` (``KdaInProj64``: the in-projection runs ``KdaSplit``'s input
prologue in its epilogue; ``KdaVerifyCols``: the verify recurrence reads those operands) at FP32
state, with fewer than MR8's eight requests in the 64-column tile and bounded F_A waits.

Derived from MR8 ``cutedsl/kda/inproj64.py`` (``KdaInProj64``, ``compile_inproj64``,
``PreparedKdaInProj64``), ``cutedsl/kda/verify.py`` (``KdaVerifyCols``) and ``cutedsl/kda/prepared.py``
(``executable``'s prepared branch); see ``kernels/SOURCE.json``. Changes:

- ``KdaInProjFold`` (``KdaInProj64`` at q8): its entry loads read the slots / accepted counts of
  ``requests`` (3 or 4) requests only. MR8's reads all eight a 64-column tile holds; here the
  token columns past ``requests`` x 8 are padding (x's rows past them are the tensor map's zero
  fill), so those requests neither read nor write a conv page, as a padding request (slot <= 0)
  does in MR8. Their published rows are not stored (C has ``requests`` x 8 columns); their
  prepared rows land in rows the verify does not read (``prepared`` keeps 64 rows). Every other
  instruction is MR8's ``KdaInProj64``.
- ``KdaVerifyColsFp32`` (``KdaVerifyCols``): the gate warp's F_A wait (``finish_fa``: the light
  poll, then the full looks) traps after ``spin_trap.BOUND_NS`` (sites ``SITE_LIGHT`` /
  ``SITE_LOOK``) instead of spinning forever; every other instruction is MR8's.
- ``compile_verify_cols``: ``executable``'s prepared branch for ``KdaVerifyColsFp32`` at an explicit
  schedule (``SCHEDULE``): early loads, 8 lanes per row, one pass, two CTAs per SM, eight-CTA
  clusters. Rubin's plain B4 schedule (16 lanes, two passes) does not admit the prepared verify.

Numerics against the plain pair (``SplitKTmemGemm64`` in-projection at the same tactic ->
``KdaSplit``): the projection's split and rank-order sums are the same, and the prologue and the
recurrence run ``KdaSplit``'s operations in its order (MR8's ``KdaInProj64`` / ``KdaVerifyCols``
docstrings), so outputs, checkpoints and conv windows are bitwise the pair's.
"""

from __future__ import annotations

from functools import cache, reduce
from operator import or_

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..common import rubin_ull_spin_trap as spin_trap
from ..vendor.vllm_mr8 import module

_inproj = module("cutedsl.kda.inproj64")
_verify = module("cutedsl.kda.verify")
_host = module("cutedsl.common.host")
_prim = module("cutedsl.common.primitives")
_gemm = module("cutedsl.gemm.splitk")
_proj = module("cutedsl.gemm.projection")

#: ``spin_trap`` sites of the verify's F_A wait: the light poll and the full looks.
SITE_LIGHT = 213
SITE_LOOK = 214
#: Verify length and the requests this pair serves (``requests`` x q8 rows).
TOKENS = 8
REQUESTS = (3, 4)
#: The 64-column tile's requests (MR8's ``KdaInProj64``): the prepared buffer's rows.
TILE_ROWS = 64
#: ``KdaSplit`` schedule of the prepared verify by request count: (early, lanes, occupancy,
#: cluster). 8 lanes x 16 rows: one pass (``KdaVerifyCols`` requires it); two CTAs per SM (the
#: 144 / 192 CTAs spread over Rubin's 212 SMs; three per SM packs them onto fewer SMs and
#: lengthens the token loop); eight-CTA clusters (one owned token per CTA).
SCHEDULE = {3: (1, 8, 2, 8), 4: (1, 8, 2, 8)}
#: ``KdaVerifyCols``' spare warps (MR8 ``VERIFY_COLS`` at B8: 1; 0 is unvalidated).
SPARE = 1


class KdaInProjFold(_inproj.KdaInProj64):
    """MR8's ``KdaInProj64`` (q8) whose entry loads read ``requests`` requests' metadata: the
    tile's token columns of requests >= ``requests`` are padding (see the module docstring)."""

    def __init__(self, *, tactic, heads=6, has_bias=False, k=7168, requests):
        super().__init__(tactic=tactic, heads=heads, has_bias=has_bias, k=k)
        assert 1 <= requests <= TILE_ROWS // TOKENS
        self.requests = requests

    @cute.jit
    def prefetch(self, tid, conv_w, bias, conv_state, slots, accepted):
        """MR8's ``KdaInProj64.prefetch`` with the request loads (and the conv page they select)
        guarded by ``request < requests``: a padding column's request stays not live, slot 0."""
        dim: cutlass.Constexpr = self.dim
        tap = cute.make_rmem_tensor((2, 3), BFloat16)
        coeff = cute.make_rmem_tensor(5, Float32)  # 4 taps' weights, bias
        meta = cute.make_rmem_tensor((2, 3), Int32)  # offset, live, conv slot
        for rq in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(3):
                tap[rq, j] = BFloat16(0)
                meta[rq, j] = Int32(0)
        for i in cutlass.range_constexpr(5):
            coeff[i] = Float32(0)
        bidx, _, _ = cute.arch.block_idx()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        row0 = bidx * 128 + rank * _inproj.SHARE
        section = row0 // dim
        if (tid >= 128) & (rank < _inproj.OWNERS) & (section < 3):
            lane, g = (tid - 128) % _inproj.SHARE, (tid - 128) // _inproj.SHARE
            channel = row0 % dim + lane
            row = section * dim + channel
            for i in cutlass.range_constexpr(4):
                coeff[i] = Float32(conv_w[section, i, channel])
            if cutlass.const_expr(self.has_bias):
                coeff[4] = Float32(bias[row])
            for rq in cutlass.range_constexpr(2):
                request = 2 * g + rq
                if request < self.requests:
                    offset = accepted[request] - 1
                    live = Int32(slots[request, offset] > 0)
                    meta[rq, 0] = offset
                    meta[rq, 1] = live
                    meta[rq, 2] = slots[request, 0] * live
                    if live != 0:
                        page = conv_state[Int64(meta[rq, 2]), None, None]
                        for j in cutlass.range_constexpr(3):
                            tap[rq, j] = page[row, offset + j]
        self.pf_tap, self.pf_coeff, self.pf_meta = tap, coeff, meta


class KdaVerifyColsFp32(_verify.KdaVerifyCols):
    """MR8's ``KdaVerifyCols`` whose F_A wait traps after ``spin_trap.BOUND_NS``;
    ``trap_record``: ``spin_trap.record_address()`` (set before compiling)."""

    trap_record = 0

    @cute.jit
    def finish_fa(self, f_a, token, quarter, values):
        """MR8's ``KdaVerifyRows.finish_fa`` with both loops bounded (sites ``SITE_LIGHT`` /
        ``SITE_LOOK``): done if the first look found no ``FA_SENT``; else the light poll, then
        full looks until none is."""
        words = [values[c, k] for c in range(8) for k in range(2)]
        dirty = Int32(cute.arch.vote_any_sync(reduce(or_, map(_prim.front_dirty, words))))
        if dirty != 0:
            if quarter == 0:
                waiting = Int32(1)
                spins, since = spin_trap.start()
                while waiting != 0:
                    spins, since = spin_trap.tick(
                        spins, since, SITE_LIGHT, self.trap_record, token, waiting
                    )
                    pointers = [_prim.at(f_a, token, 32 * owner + 30) for owner in range(4)]
                    watched = [_prim.load_volatile(p)[0] for p in pointers]
                    waiting = Int32(reduce(or_, map(_prim.front_dirty, watched)))
                    if waiting != 0:
                        _prim.sleep_ns(_verify.FA_BACKOFF_NS)
            cute.arch.sync_warp()
            spins, since = spin_trap.start()
            while dirty != 0:
                spins, since = spin_trap.tick(
                    spins, since, SITE_LOOK, self.trap_record, token, quarter
                )
                self.look_fa(f_a, token, quarter, values)
                words = [values[c, k] for c in range(8) for k in range(2)]
                dirty = Int32(
                    cute.arch.vote_any_sync(reduce(or_, map(_prim.front_dirty, words)))
                )
        return [(values[c, 0], values[c, 1]) for c in range(8)]


@cache
def compile_fold(tactic, heads, has_bias, specs, arch, requests):
    """MR8's ``compile_inproj64`` for ``KdaInProjFold`` at ``requests`` requests x q8."""

    def matrix(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    kernel = KdaInProjFold(
        tactic=_gemm.SplitKTactic(*tactic[:4]), heads=heads, has_bias=has_bias, requests=requests
    )
    tensors = [_host.fake(shape, stride, dtype) for shape, stride, dtype in specs]
    args = (matrix(1), matrix(1), matrix(0), *tensors, Float32(1.0), cutlass.Int64(0))
    return cute.experimental.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


@cache
def compile_verify_cols(specs, heads, gate_mode, has_bias, device, requests, schedule, record):
    """MR8 ``prepared.executable``'s prepared branch (``KdaVerifyCols``) for
    ``KdaVerifyColsFp32`` at ``requests`` x q8, ``schedule`` (``SCHEDULE``'s), FP32 state,
    ``SPARE`` spare warps; ``record``: the ``spin_trap`` record's device address."""
    args = tuple(_host.fake(shape, stride, dtype) for shape, stride, dtype in specs)
    if specs[9][2] != torch.float32:
        raise TypeError("KdaVerifyColsFp32 needs the FP32 KDA state")
    kernel = KdaVerifyColsFp32(
        heads, requests, gate_mode, has_bias, schedule, TOKENS, False, SPARE
    )
    kernel.trap_record = record
    scalars = (Float32(v) for v in (-5, 128**-0.5, 1e-6))
    with torch.accelerator.device_index(device):
        compiled = cute.compile(kernel, *args[:13], *scalars, *args[13:], make_fake_stream())
    carveout = getattr(kernel, "shared_carveout", None)
    if carveout is not None:
        compiled = _host.prepare_shared_carveout(compiled, device, carveout)
    return compiled


class PreparedKdaFold(_inproj.PreparedKdaInProj64):
    """A KDA layer's folded verify pair at ``REQUESTS`` x q8 rows: ``KdaInProjFold`` (the
    in-projection at ``tactic``, the 64-row ``SplitKTmemGemm64`` tactic, with ``KdaSplit``'s
    prologue in its epilogue) -> ``KdaVerifyColsFp32`` on the prepared operands. ``core`` is the
    layer's ``PreparedKda`` (FP32 state; conv weight, bias, layout, q scale; its ``readers``
    must hold >= 4 zeros). ``projected``: the layer's own [>= 32, pitch] BF16 buffer for these
    rows only, its F_A columns armed (``arm``) before the first run and re-armed by each
    verify's last reader. ``prepared``: an FP32 [64, ``PREPARED_PITCH``] scratch (any layer's)."""

    def __init__(self, weight, core, tactic, projected, prepared):
        super().__init__(weight, core, tactic, TOKENS)
        if projected.dtype != torch.bfloat16 or projected.shape[0] < TOKENS * max(REQUESTS):
            raise ValueError("PreparedKdaFold needs a BF16 [>= 32, pitch] projection buffer")
        if prepared.dtype != torch.float32 or prepared.shape != (
            TILE_ROWS,
            _inproj.PREPARED_PITCH,
        ):
            raise ValueError("PreparedKdaFold needs an FP32 [64, PREPARED_PITCH] scratch")
        self.projected, self.prepared = projected, prepared
        self.codes: dict[int, object] = {}  # rows -> the in-projection
        self.verify: dict[int, object] = {}  # rows -> the prepared verify

    @staticmethod
    def arm(projected, dim):
        """Arm ``projected``'s F_A columns (every BF16 element 0xFFFF, ``FA_SENT``)."""
        projected[:, 4 * dim : 4 * dim + 128].view(torch.int16).fill_(-1)

    def prepare_caches(self, caches, rows=tuple(TOKENS * r for r in REQUESTS)):
        """Compile both kernels for each verify row count of ``rows`` (eager, before capture)."""
        if caches[1].dtype != torch.float32:
            raise TypeError("PreparedKdaFold needs the FP32 KDA state")
        if self.core.readers.numel() < max(REQUESTS) or self.core.conv_dim_first is not True:
            raise ValueError("PreparedKdaFold needs readers for 4 requests and dim-first conv")
        meta = dict(device="meta")
        arch, record = _host.architecture(self.device), spin_trap.record_address()
        pitch, dim = self.projected.shape[1], self.core.dim
        for r in rows:
            requests = r // TOKENS
            if r % TOKENS or requests not in REQUESTS:
                raise ValueError(f"PreparedKdaFold runs {REQUESTS} requests x q{TOKENS} only")
            if r in self.codes:
                continue
            slots = torch.empty((requests, TOKENS), dtype=torch.int32, **meta)
            accepted = torch.empty((requests,), dtype=torch.int32, **meta)
            prepared = torch.empty(TILE_ROWS, _inproj.PREPARED_PITCH, dtype=torch.float32, **meta)
            tensors = self._tensors(prepared, caches, slots, accepted)
            specs = tuple(map(_host.layout, tensors))
            heads, bias = self.core.heads, self.core.has_bias
            with torch.accelerator.device_index(self.device.index):
                code = compile_fold(self.tactic, heads, bias, specs, arch, requests)
            _, function = _host.resolve_kernel(code, self.device.index)
            count = _inproj._cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS
            (registers,) = _host._checked(_inproj._cuda.cuFuncGetAttribute(count, function))
            if registers > _inproj.REGISTERS:
                raise RuntimeError(
                    f"KdaInProjFold uses {registers} registers per thread, over {_inproj.REGISTERS}"
                )
            self.codes[r] = code
            # the verify: PreparedKda's operands with ``prepared`` first and the readers last
            projected = torch.empty((r, pitch), dtype=torch.bfloat16, **meta)
            out = torch.empty(r, dim, dtype=torch.bfloat16, **meta)
            operands = self.core._operands(caches, projected, slots, out, accepted)
            rows_prepared = torch.empty(r, 4 * dim + 8, dtype=torch.float32, **meta)
            readers = self.core.readers[:requests]
            vspecs = tuple(map(_host.layout, (rows_prepared, *operands[1:], readers)))
            self.verify[r] = compile_verify_cols(
                vspecs, heads, 0, bias, self.device.index, requests, SCHEDULE[requests], record
            )
        return self

    def run_fold(self, x, out, *, caches, slots, accepted):
        """The folded pair for ``x`` BF16 [r, hidden] (r in ``self.codes``): writes ``out``
        [r, dim], the state rows of ``slots`` [r / 8, 8] and the conv pages; returns ``out``."""
        r = x.shape[0]
        requests = r // TOKENS
        if slots.stride() != (slots.shape[1], 1):
            slots = slots.contiguous()
        slots, accepted = slots[:requests], accepted[:requests]
        projected = self.projected[:r]
        tensors = self._tensors(self.prepared, caches, slots, accepted)
        views = [_host.to_cute(t, 16) for t in tensors]
        y = _proj.matrix_view(projected, output=True)
        maps = (cutlass.Int64(self.map.data_ptr()),)
        self.codes[r](
            self.view, _proj.matrix_view(x), y, *views, self.core.scalars[1], *maps,
            _host.stream(self.device),
        )  # fmt: skip
        operands = self.core._operands(caches, projected, slots, out, accepted)
        operands = (self.prepared[:r], *operands[1:], self.core.readers[:requests])
        vviews = tuple(_host.to_cute(t, 16) for t in operands)
        self.verify[r](*vviews[:13], *self.core.scalars, *vviews[13:], _host.stream(self.device))
        return out
