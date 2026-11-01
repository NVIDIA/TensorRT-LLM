# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused expert finalize and tensor-parallel all-reduce for Muse Spark Open.

`FinalizeIpcWorkspace.finalize` normalizes every selected expert row on its own
(weight-first RMSNorm with zero-centered gains), combines the rows of a token with
its routing weights, rounds that per-rank partial to BF16, and sums the partials
of the four ranks of a single-node TP group in FP32. One kernel launch does all
of it, and every rank ends up with the same BF16 result.

Communication is one-shot over CUDA IPC. Each rank stores its BF16 partial into
its own slab of every peer's receive buffer and then reads the slabs addressed to
it. A 32-bit word that still holds the sentinel `0x80000000` has not arrived yet;
producers turn BF16 negative zero into positive zero, so published data never
looks like the sentinel. The reader writes the sentinel back as soon as it has a
value.

Two receive buffers alternate between calls. Before a peer can write call
`n + 2` into a buffer, it must have read this rank's call `n + 1`, which this
rank only publishes after call `n` (including its resets) completed. A buffer is
therefore never rewritten while it still holds unread data, and a value left over
from an earlier call can never satisfy a later read.

Each rank owns four int32 control words: the current buffer index and an arrival
counter. Every CTA reads the index once and then arrives on the counter; the last
arrival flips the index. `atom.inc` wraps the counter to zero in the same step,
so no counter grows without bound and graph replays advance the state on device.

Workspace per rank: `2 x 4 x 128 x 4096` BF16 receive slots (8 MiB) plus the
control words. Construction is collective over the TP group, compiles and loads
the kernel, and either succeeds on every rank or raises the same exception type
on every rank. Nothing compiles afterwards.

Release never frees memory that a peer may still map. CUDA leaves freeing an IPC
export undefined while an importer still has it open, so `close()` is
collective: once the owner has quiesced every rank, each rank synchronizes its
device, then closes its mappings of peer memory, and only after the whole group
confirmed both steps does any rank free its own buffer. Dropping the workspace
without `close()` stays local: it waits for this rank's device and closes its
mappings, but leaves its own buffer allocated until the process exits.
"""

import functools
import importlib.metadata
import math
import os
import socket
import sys
import zlib

import numpy as np
import torch

from tensorrt_llm._ipc_utils import _ipc_mem_handle_from_bytes, _ipc_mem_handle_to_bytes
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

# Problem shape fixed by the supported model configuration.
_HIDDEN = 4096
_TOP_K = 8
_MAX_TOKENS = 128
_RANKS = 4
# BF16 values moved per 16-byte access.
_VECTOR = 8
_SLAB_BYTES = _MAX_TOKENS * _HIDDEN * 2
_BUFFERS = 2
_SENTINEL = 0x80000000
_CONTROL_WORDS = 4
# Bump when the device protocol changes so mixed builds refuse to pair up.
_PROTOCOL_VERSION = 1
# CuTe DSL release this kernel was validated with.
_VALIDATED_DSL = "4.8.0"
_IPC_ALIGNMENT = 2 << 20

# Construction status codes exchanged across the TP group.
_OK = 0
_UNSUPPORTED = 1
_FAILED = 2


def _dsl_version_supported() -> bool:
    from packaging.version import InvalidVersion, Version

    try:
        installed = Version(importlib.metadata.version("nvidia-cutlass-dsl"))
    except (importlib.metadata.PackageNotFoundError, InvalidVersion):
        return False
    return installed >= Version(_VALIDATED_DSL)


try:
    import cuda.bindings.driver as cuda
    import cuda.bindings.runtime as cudart
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    from cutlass import BFloat16, Float32, Int32, Int64, Uint32
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op
except ImportError:
    HAS_CUTEDSL = False
else:
    HAS_CUTEDSL = _dsl_version_supported()


def _pdl_enabled() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


if HAS_CUTEDSL:
    _FOUR_WORDS = "!llvm.struct<(i32,i32,i32,i32)>"

    def _asm(result_type, operands, text, constraints, loc, ip, side_effects=True):
        return llvm.inline_asm(
            result_type,
            operands,
            text,
            constraints,
            has_side_effects=side_effects,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    def _u32(value, loc, ip):
        return Uint32(value).ir_value(loc=loc, ip=ip)

    def _i64(value, loc, ip):
        return Int64(value).ir_value(loc=loc, ip=ip)

    def _f32(value, loc, ip):
        return Float32(value).ir_value(loc=loc, ip=ip)

    def _split_words(result, loc, ip):
        return tuple(
            Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4)
        )

    # Every global access below is a side-effecting asm statement, so none of
    # them can move above griddepcontrol.wait or be merged across poll attempts.

    @dsl_user_op
    def _load_input_x4(address, enabled, *, loc=None, ip=None):
        """16-byte input load; yields zeros without touching memory when disabled."""
        result = _asm(
            ir.Type.parse(_FOUR_WORDS),
            [_i64(address, loc, ip), _u32(enabled, loc, ip)],
            "{\n.reg .pred p;\n.reg .b32 a, b, c, d;\n"
            "mov.b32 a, 0;\nmov.b32 b, 0;\nmov.b32 c, 0;\nmov.b32 d, 0;\n"
            "setp.ne.u32 p, $5, 0;\n"
            "@p ld.global.nc.v4.b32 {a, b, c, d}, [$4];\n"
            "mov.b32 $0, a;\nmov.b32 $1, b;\nmov.b32 $2, c;\nmov.b32 $3, d;\n}",
            "=r,=r,=r,=r,l,r",
            loc,
            ip,
        )
        return _split_words(result, loc, ip)

    @dsl_user_op
    def _load_input_s32(address, *, loc=None, ip=None):
        result = _asm(
            T.i32(), [_i64(address, loc, ip)], "ld.global.nc.b32 $0, [$1];", "=r,l", loc, ip
        )
        return Int32(result)

    @dsl_user_op
    def _load_input_f32(address, *, loc=None, ip=None):
        result = _asm(
            T.f32(), [_i64(address, loc, ip)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc, ip
        )
        return Float32(result)

    @dsl_user_op
    def _load_peer_x4(address, *, loc=None, ip=None):
        """Relaxed system-scope 16-byte load, as four independent 32-bit words."""
        result = _asm(
            ir.Type.parse(_FOUR_WORDS),
            [_i64(address, loc, ip)],
            "ld.relaxed.sys.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            loc,
            ip,
        )
        return _split_words(result, loc, ip)

    @dsl_user_op
    def _store_peer_x4(address, w0, w1, w2, w3, *, loc=None, ip=None):
        """Relaxed system-scope 16-byte store, as four independent 32-bit words."""
        _asm(
            None,
            [_i64(address, loc, ip)] + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
            "st.relaxed.sys.global.v4.b32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _store_x4(address, w0, w1, w2, w3, *, loc=None, ip=None):
        _asm(
            None,
            [_i64(address, loc, ip)] + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
            "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _load_control(address, *, loc=None, ip=None):
        result = _asm(
            T.i32(),
            [_i64(address, loc, ip)],
            "ld.relaxed.gpu.global.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
        )
        return Uint32(result)

    @dsl_user_op
    def _store_control(address, value, *, loc=None, ip=None):
        _asm(
            None,
            [_i64(address, loc, ip), _u32(value, loc, ip)],
            "st.relaxed.gpu.global.b32 [$0], $1;",
            "l,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _arrive(address, last, *, loc=None, ip=None):
        """Arrive on the counter; it wraps to zero when the old value equals `last`."""
        result = _asm(
            T.i32(),
            [_i64(address, loc, ip), _u32(last, loc, ip)],
            "atom.acq_rel.gpu.global.inc.u32 $0, [$1], $2;",
            "=r,l,r",
            loc,
            ip,
        )
        return Uint32(result)

    @dsl_user_op
    def _unpack_bf16x2(word, *, loc=None, ip=None):
        """Return the FP32 values of the low and high BF16 halves of `word`."""
        result = _asm(
            ir.Type.parse("!llvm.struct<(f32,f32)>"),
            [_u32(word, loc, ip)],
            "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xffff0000;\n"
            "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
            "=f,=f,r",
            loc,
            ip,
            side_effects=False,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        )

    @dsl_user_op
    def _pack_bf16x2(low, high, *, loc=None, ip=None):
        """Round to nearest-even BF16 and pack; `low` lands in the low half."""
        result = _asm(
            T.i32(),
            [_f32(high, loc, ip), _f32(low, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def _pack_publishable_bf16x2(low, high, *, loc=None, ip=None):
        """Like `_pack_bf16x2`, but negative zeros become positive zeros.

        The sentinel word is a +0 low half next to a -0 high half, so a payload
        without negative zeros can never be mistaken for it.
        """
        result = _asm(
            T.i32(),
            [_f32(high, loc, ip), _f32(low, loc, ip)],
            "{\n.reg .b32 w, m;\n.reg .pred p;\n"
            "cvt.rn.bf16x2.f32 w, $1, $2;\n"
            "and.b32 m, w, 0xffff;\nsetp.eq.u32 p, m, 0x8000;\n@p and.b32 w, w, 0xffff0000;\n"
            "and.b32 m, w, 0xffff0000;\nsetp.eq.u32 p, m, 0x80000000;\n@p and.b32 w, w, 0xffff;\n"
            "mov.b32 $0, w;\n}",
            "=r,f,f",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def _zero_unless(value, enabled, *, loc=None, ip=None):
        result = _asm(
            T.f32(),
            [_f32(value, loc, ip), _u32(enabled, loc, ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $2, 0;\nselp.f32 $0, $1, 0f00000000, p;\n}",
            "=f,f,r",
            loc,
            ip,
            side_effects=False,
        )
        return Float32(result)

    class _FinalizeAllReduceKernel:
        """One-shot finalize + all-reduce across the four ranks.

        CTA `b` handles token `b`; each thread owns one 16-byte column vector of
        the row for all `_TOP_K` slots.
        """

        def __init__(self, pdl: bool):
            self.threads = _HIDDEN // _VECTOR
            self.warps = self.threads // 32
            self.pdl = pdl

        @cute.jit
        def __call__(
            self,
            gemm: cute.Pointer,
            indices: cute.Pointer,
            weights: cute.Pointer,
            norm: cute.Pointer,
            out: cute.Pointer,
            control: cute.Pointer,
            receive: cute.Pointer,
            peer0: cute.Pointer,
            peer1: cute.Pointer,
            peer2: cute.Pointer,
            peer3: cute.Pointer,
            rows: Int32,
            row_stride: Int32,
            tokens: Int32,
            rank: Int32,
            eps: Float32,
            dry: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                gemm,
                indices,
                weights,
                norm,
                out,
                control,
                receive,
                peer0,
                peer1,
                peer2,
                peer3,
                rows,
                row_stride,
                tokens,
                rank,
                eps,
                dry,
            ).launch(
                grid=[tokens, 1, 1],
                block=[self.threads, 1, 1],
                smem=128 + 32 * self.warps,
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.kernel
        def kernel(
            self,
            gemm: cute.Pointer,
            indices: cute.Pointer,
            weights: cute.Pointer,
            norm: cute.Pointer,
            out: cute.Pointer,
            control: cute.Pointer,
            receive: cute.Pointer,
            peer0: cute.Pointer,
            peer1: cute.Pointer,
            peer2: cute.Pointer,
            peer3: cute.Pointer,
            rows: Int32,
            row_stride: Int32,
            tokens: Int32,
            rank: Int32,
            eps: Float32,
            dry: Int32,
        ):
            peers = (peer0, peer1, peer2, peer3)
            tid = cute.arch.thread_idx()[0]
            token = cute.arch.block_idx()[0]
            blocks = cute.arch.grid_dim()[0]
            column = tid * _VECTOR
            lane = tid % 32
            warp = tid // 32

            smem = cutlass_utils.SmemAllocator()
            snapshot = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
            warp_sums = smem.allocate_tensor(
                Float32, cute.make_layout((self.warps * _TOP_K,)), byte_alignment=16
            )
            sums = smem.allocate_tensor(Float32, cute.make_layout((_TOP_K,)), byte_alignment=16)

            # A dry launch only makes the driver load the module.
            if dry == 0:
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()

                # Selected rows and routing weights; invalid slots read nothing.
                selection = Int64(token) * _TOP_K
                gemm_address = gemm.toint()
                row_index = []
                row_weight = []
                row_words = []
                for k in cutlass.range_constexpr(_TOP_K):
                    index = _load_input_s32(indices.toint() + (selection + k) * 4)
                    row_index.append(index)
                    row_weight.append(_load_input_f32(weights.toint() + (selection + k) * 4))
                    valid = Uint32((index >= 0) & (index < rows))
                    address = gemm_address + (Int64(index) * Int64(row_stride) + Int64(column)) * 2
                    row_words.append(_load_input_x4(address, valid))
                norm_words = _load_input_x4(norm.toint() + Int64(column) * 2, Uint32(1))

                # Read the buffer index once, then arrive; the last arrival flips it.
                # Issued after the input loads so both round trips overlap; the
                # reduction barrier below makes the snapshot visible to every thread.
                control_address = control.toint()
                if tid == 0:
                    current = _load_control(control_address)
                    snapshot[0] = Int32(current)
                    previous = _arrive(control_address + 4, Uint32(blocks - 1))
                    if previous == Uint32(blocks - 1):
                        _store_control(control_address, current ^ Uint32(1))

                gains = []
                for w in cutlass.range_constexpr(4):
                    low, high = _unpack_bf16x2(norm_words[w])
                    gains.append(low + Float32(1.0))
                    gains.append(high + Float32(1.0))

                # Sum of squares of the scaled row, per slot.
                partials = []
                for k in cutlass.range_constexpr(_TOP_K):
                    square_sum = Float32(0.0)
                    for w in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(row_words[k][w])
                        scaled_low = low * gains[2 * w]
                        scaled_high = high * gains[2 * w + 1]
                        square_sum = (
                            square_sum + scaled_low * scaled_low + scaled_high * scaled_high
                        )
                    partials.append(cute.arch.warp_reduction_sum(square_sum))
                if lane == 0:
                    for k in cutlass.range_constexpr(_TOP_K):
                        warp_sums[warp * _TOP_K + k] = partials[k]
                cute.arch.barrier()
                if tid < _TOP_K:
                    total = Float32(0.0)
                    for w in cutlass.range_constexpr(self.warps):
                        total = total + warp_sums[w * _TOP_K + tid]
                    sums[tid] = total
                cute.arch.barrier()

                # Normalize each slot, weight it, and accumulate in FP32.
                result = [Float32(0.0)] * _VECTOR
                for k in cutlass.range_constexpr(_TOP_K):
                    valid = Uint32((row_index[k] >= 0) & (row_index[k] < rows))
                    scale = row_weight[k] * cute.math.rsqrt(sums[k] / Float32(_HIDDEN) + eps)
                    scale = _zero_unless(scale, valid)
                    for w in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(row_words[k][w])
                        result[2 * w] = result[2 * w] + scale * (low * gains[2 * w])
                        result[2 * w + 1] = result[2 * w + 1] + scale * (high * gains[2 * w + 1])
                own = []
                for w in cutlass.range_constexpr(4):
                    own.append(_pack_publishable_bf16x2(result[2 * w], result[2 * w + 1]))

                # Publish this rank's partial into its slab of every peer's buffer.
                buffer = Int64(snapshot[0])
                slot = (Int64(token) * _HIDDEN + Int64(column)) * 2
                own_slab = (buffer * _RANKS + Int64(rank)) * _SLAB_BYTES
                for p in cutlass.range_constexpr(_RANKS):
                    if rank != p:
                        _store_peer_x4(
                            peers[p].toint() + own_slab + slot, own[0], own[1], own[2], own[3]
                        )

                # Consume every rank's partial in rank order and reset its slot.
                total = [Float32(0.0)] * _VECTOR
                receive_address = receive.toint()
                for p in cutlass.range_constexpr(_RANKS):
                    w0, w1, w2, w3 = own[0], own[1], own[2], own[3]
                    if rank != p:
                        address = receive_address + (buffer * _RANKS + p) * _SLAB_BYTES + slot
                        w0, w1, w2, w3 = _load_peer_x4(address)
                        while (
                            (w0 == _SENTINEL)
                            | (w1 == _SENTINEL)
                            | (w2 == _SENTINEL)
                            | (w3 == _SENTINEL)
                        ):
                            w0, w1, w2, w3 = _load_peer_x4(address)
                        _store_x4(address, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL)
                    words = (w0, w1, w2, w3)
                    for w in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(words[w])
                        total[2 * w] = total[2 * w] + low
                        total[2 * w + 1] = total[2 * w + 1] + high

                packed = []
                for w in cutlass.range_constexpr(4):
                    packed.append(_pack_bf16x2(total[2 * w], total[2 * w + 1]))
                _store_x4(out.toint() + slot, packed[0], packed[1], packed[2], packed[3])

                # Dependents may start only after the last access to the buffers.
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_launch_dependents()

    @functools.cache
    def _compiled_kernel(device_index: int, pdl: bool):
        def placeholder(dtype, alignment):
            return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=alignment)

        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                _FinalizeAllReduceKernel(pdl),
                placeholder(BFloat16, 16),
                placeholder(Int32, 4),
                placeholder(Float32, 4),
                placeholder(BFloat16, 16),
                placeholder(BFloat16, 16),
                placeholder(Uint32, 16),
                placeholder(Uint32, 16),
                *[placeholder(Uint32, 16) for _ in range(_RANKS)],
                Int32(0),
                Int32(_HIDDEN),
                Int32(1),
                Int32(0),
                Float32(1e-6),
                Int32(1),
                stream,
                options="--enable-tvm-ffi",
            )


def _fingerprint() -> int:
    dsl = importlib.metadata.version("nvidia-cutlass-dsl") if HAS_CUTEDSL else "none"
    text = f"{_PROTOCOL_VERSION}|{dsl}|{_RANKS}|{_HIDDEN}|{_TOP_K}|{_MAX_TOKENS}"
    return zlib.crc32(text.encode())


def _describe(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _host_id() -> int:
    return zlib.crc32(socket.gethostname().encode())


def _device_id(device: torch.device) -> int:
    return zlib.crc32(str(torch.cuda.get_device_properties(device).uuid).encode())


class FinalizeIpcWorkspace:
    """Persistent state of the fused finalize + all-reduce for one TP4 group on one node.

    One instance can be shared by every layer of a model; calls must be issued
    in the same order on every rank, ordered on one stream (or explicitly) per
    rank. `close()` releases it on every rank; see its documentation for what
    dropping it does instead.
    """

    # What the destructor sees if __init__ failed before setting up any state.
    _closed = True

    def __init__(self, mapping: Mapping):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"mapping must be a Mapping, got {type(mapping).__name__}")
        self._ranks = mapping.tp_size
        self._rank = mapping.tp_rank
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._export = 0
        self._imports = []
        self._control = None
        self._kernel = None
        self._peer_args = ()
        self._receive = 0
        self._closed = False

        from tensorrt_llm._torch.distributed.communicator import Distributed

        self._dist = Distributed.get(mapping)
        self._agree("setup", *self._guarded(self._prepare, mapping))
        self._allocate_ipc()
        self._agree("initialization", *self._guarded(self._initialize))

    # Construction stages. Each stage turns any local failure into a status
    # and agrees on it with the whole group, so that every rank either goes on
    # or raises the same exception type; a rank never waits for peers that
    # already gave up.

    @staticmethod
    def _guarded(stage, *args):
        """Run this rank's part of a stage; an exception of any type becomes a failure status.

        A rank that raised past the following agreement would leave its peers waiting in it.
        TVM-FFI, for one, reports launch failures as ValueError, TypeError and other types.
        """
        try:
            return stage(*args)
        except Exception as error:  # noqa: BLE001 - every failure must reach the agreement
            return _FAILED, _describe(error)

    def _prepare(self, mapping: Mapping):
        if not HAS_CUTEDSL:
            return _UNSUPPORTED, "CuTe DSL is unavailable or older than the validated release"
        nodes = len({rank // mapping.gpus_per_node for rank in mapping.tp_group})
        if self._ranks != _RANKS or nodes != 1:
            return _UNSUPPORTED, "the fusion needs TP4 on one node"
        if torch.cuda.get_device_capability(self._device) != (10, 3):
            return _UNSUPPORTED, "the kernel is validated on SM103 only"
        try:
            self._kernel = _compiled_kernel(self._device.index, _pdl_enabled())
        # Compilation can fail in many ways; every one must reach the agreement.
        except Exception as error:  # noqa: BLE001
            return _FAILED, f"kernel compilation failed: {error!r}"
        return _OK, ""

    def _agree(self, stage: str, status: int, detail: str) -> None:
        identity = [status, _fingerprint(), self._rank, _host_id(), _device_id(self._device)]
        table = self._dist.tp_allgather_int64(identity)
        problems = []
        if status != _OK:
            problems.append(f"rank {self._rank}: {detail}")
        statuses = table[:, 0]
        if (table[:, 1] != table[0, 1]).any():
            statuses = np.maximum(statuses, _UNSUPPORTED)
            problems.append("ranks run different kernel builds or configurations")
        if sorted(table[:, 2].tolist()) != list(range(self._ranks)):
            statuses = np.maximum(statuses, _UNSUPPORTED)
            problems.append("TP ranks are not distinct")
        hosts = len(set(table[:, 3].tolist()))
        if hosts != 1:
            statuses = np.maximum(statuses, _UNSUPPORTED)
            problems.append(f"the TP group spans {hosts} hosts")
        devices = {(int(host), int(device)) for host, device in table[:, 3:5]}
        if len(devices) != self._ranks:
            statuses = np.maximum(statuses, _UNSUPPORTED)
            problems.append("two ranks share one GPU")
        worst = int(statuses.max())
        if worst == _OK:
            return
        message = f"MuseSparkOpen finalize+all-reduce {stage} failed on rank(s) " + ", ".join(
            str(i) for i, s in enumerate(statuses) if s != _OK
        )
        if problems:
            message += ": " + "; ".join(problems)
        self._abandon(ValueError if worst == _UNSUPPORTED else RuntimeError, message)

    def _allocate_ipc(self) -> None:
        size = math.ceil(_BUFFERS * self._ranks * _SLAB_BYTES / _IPC_ALIGNMENT) * _IPC_ALIGNMENT
        status, detail, handle = _OK, "", bytes(64)
        try:
            error, pointer = cudart.cudaMalloc(size)
            if error != cudart.cudaError_t.cudaSuccess:
                status, detail = _FAILED, f"cudaMalloc failed: {error}"
            else:
                self._export = int(pointer)
                error, ipc_handle = cudart.cudaIpcGetMemHandle(pointer)
                if error != cudart.cudaError_t.cudaSuccess:
                    status, detail = _FAILED, f"cudaIpcGetMemHandle failed: {error}"
                else:
                    handle = _ipc_mem_handle_to_bytes(ipc_handle)
        except Exception as error:  # noqa: BLE001 - every failure must reach the agreement
            status, detail, handle = _FAILED, _describe(error), bytes(64)
        words = np.frombuffer(handle.ljust(64, b"\0")[:64], dtype=np.int64)
        table = self._dist.tp_allgather_int64([status, *words.tolist()])
        failed = [i for i, s in enumerate(table[:, 0]) if s != _OK]
        if failed:
            self._abandon(
                RuntimeError,
                f"MuseSparkOpen finalize+all-reduce allocation failed on rank(s) {failed}"
                + (f": {detail}" if detail else ""),
            )

        status, detail = _OK, ""
        peers = []
        try:
            for peer in range(self._ranks):
                if peer == self._rank:
                    peers.append(self._export)
                    continue
                data = np.ascontiguousarray(table[peer, 1:]).tobytes()
                error, pointer = cudart.cudaIpcOpenMemHandle(
                    _ipc_mem_handle_from_bytes(data), cudart.cudaIpcMemLazyEnablePeerAccess
                )
                if error != cudart.cudaError_t.cudaSuccess:
                    status, detail = _FAILED, f"cudaIpcOpenMemHandle(rank {peer}) failed: {error}"
                    break
                self._imports.append(int(pointer))
                peers.append(int(pointer))
        except Exception as error:  # noqa: BLE001 - every failure must reach the agreement
            status, detail = _FAILED, _describe(error)
        table = self._dist.tp_allgather_int64([status])
        failed = [i for i, s in enumerate(table[:, 0]) if s != _OK]
        if failed:
            self._abandon(
                RuntimeError,
                f"MuseSparkOpen finalize+all-reduce peer mapping failed on rank(s) {failed}"
                + (f": {detail}" if detail else ""),
            )
        self._receive = self._export
        self._peer_args = tuple(peers)

    def _initialize(self):
        # Runs through _guarded: any exception becomes this rank's failure status.
        with torch.cuda.device(self._device):
            size = _BUFFERS * self._ranks * _SLAB_BYTES
            (error,) = cuda.cuMemsetD32(self._receive, _SENTINEL, size // 4)
            if error != cuda.CUresult.CUDA_SUCCESS:
                return _FAILED, f"sentinel fill failed: {error}"
            self._control = torch.zeros(_CONTROL_WORDS, dtype=torch.int32, device=self._device)
            # Pointer arguments that stay fixed for the workspace's lifetime.
            self._fixed = (self._control.data_ptr(), self._receive, *self._peer_args)
            # A dry launch loads the module now, not inside a CUDA graph capture.
            dummy = torch.zeros(_HIDDEN, dtype=torch.bfloat16, device=self._device).data_ptr()
            self._launch(dummy, dummy, dummy, dummy, dummy, 0, _HIDDEN, 1, 1.0, 1)
            torch.cuda.synchronize(self._device)
        return _OK, ""

    def _launch(self, gemm, indices, weights, norm, out, rows, row_stride, tokens, eps, dry=0):
        # TVM-FFI accepts device addresses as plain ints for pointer parameters.
        self._kernel(
            gemm,
            indices,
            weights,
            norm,
            out,
            *self._fixed,
            rows,
            row_stride,
            tokens,
            self._rank,
            eps,
            dry,
            cuda.CUstream(torch.cuda.current_stream(self._device).cuda_stream),
        )

    def finalize(
        self,
        gemm2_output: torch.Tensor,
        indices: torch.Tensor,
        routing_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        num_tokens: int,
    ) -> torch.Tensor:
        """Enqueue the fused finalize + all-reduce on the current stream.

        Returns a new `[num_tokens, 4096]` BF16 tensor holding the full group
        sum. Every rank must make the matching call with the same `num_tokens`
        and `eps`. Requirements, checked before anything is launched:

        * `gemm2_output`: CUDA BF16 `[R, W]`, `W >= 4096`, unit column stride,
          row stride a multiple of 8 and a 16-byte aligned base when `R > 0`.
          Only rows addressed by valid slots and their first 4096 columns are
          read.
        * `indices`: contiguous int32 with `8 * num_tokens` entries; a slot is
          valid when `0 <= index < R`.
        * `routing_weights`: contiguous FP32 with `8 * num_tokens` entries.
        * `norm_weight`: contiguous, 16-byte aligned BF16 with 4096 entries,
          stored as offsets from one.
        * `eps`: positive and finite. `num_tokens`: in `[1, 128]`.

        Raises `RuntimeError` once the workspace is closed.
        """
        if self._closed:
            raise RuntimeError("this MuseSparkOpen finalize workspace is closed")
        rows, row_stride = self._validate(
            gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens
        )
        out = torch.empty((num_tokens, _HIDDEN), dtype=torch.bfloat16, device=self._device)
        self._launch(
            gemm2_output.data_ptr() if rows > 0 else 0,
            indices.data_ptr(),
            routing_weights.data_ptr(),
            norm_weight.data_ptr(),
            out.data_ptr(),
            rows,
            row_stride,
            num_tokens,
            float(eps),
        )
        return out

    def _validate(self, gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens):
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
            raise ValueError(f"num_tokens must be an int, got {type(num_tokens).__name__}")
        if not 1 <= num_tokens <= _MAX_TOKENS:
            raise ValueError(f"num_tokens must be in [1, {_MAX_TOKENS}], got {num_tokens}")
        if not (isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0):
            raise ValueError(f"eps must be positive and finite, got {eps!r}")
        if (
            gemm2_output.device != self._device
            or gemm2_output.dtype != torch.bfloat16
            or gemm2_output.dim() != 2
            or gemm2_output.shape[1] < _HIDDEN
        ):
            raise ValueError(
                "gemm2_output must be a [R, >=4096] BF16 tensor on the workspace device"
            )
        rows = gemm2_output.shape[0]
        row_stride = _HIDDEN
        if rows > 0:
            if gemm2_output.stride(1) != 1 or gemm2_output.data_ptr() % 16:
                raise ValueError("gemm2_output needs unit column stride and a 16-byte aligned base")
            if rows > 1:
                row_stride = gemm2_output.stride(0)
                if row_stride % _VECTOR:
                    raise ValueError("gemm2_output row stride must be a multiple of 8 elements")
        expected = num_tokens * _TOP_K
        for name, tensor, dtype in (
            ("indices", indices, torch.int32),
            ("routing_weights", routing_weights, torch.float32),
        ):
            if (
                tensor.device != self._device
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.numel() != expected
            ):
                raise ValueError(
                    f"{name} must be a contiguous {dtype} tensor with {expected} entries"
                )
        if (
            norm_weight.device != self._device
            or norm_weight.dtype != torch.bfloat16
            or not norm_weight.is_contiguous()
            or norm_weight.numel() != _HIDDEN
            or norm_weight.data_ptr() % 16
        ):
            raise ValueError(
                "norm_weight must be a contiguous, 16-byte aligned BF16 tensor of 4096"
            )
        return rows, row_stride

    # Release.

    def close(self) -> None:
        """Release the workspace on every rank of the TP group. Collective.

        The owner first stops, on every rank, all `finalize` calls and all
        replays of CUDA graphs that captured one. Then every rank calls
        `close()` while the TP group's MPI communicator is still alive:

        1. each rank synchronizes its device, and the group confirms that all did;
        2. each rank closes its mappings of peer memory, and the group confirms
           that all did, so no rank maps any peer's buffer any more;
        3. each rank frees its own buffer, and the group confirms that all did.

        If any rank fails a step, every rank raises `RuntimeError` and nothing
        a peer may still map is freed; it stays allocated until the process
        exits. The workspace is unusable afterwards, and calling `close()`
        again does nothing.

        Dropping the workspace without `close()` involves no other rank: it
        waits for this rank's device and then closes this rank's mappings of
        peer memory. It never frees this rank's own buffer, which peers may
        still map; that stays allocated until the process exits.
        """
        if self._closed:
            return
        self._closed = True
        problem = self._release_together()
        if problem:
            raise RuntimeError(f"MuseSparkOpen finalize workspace close failed: {problem}")

    def _abandon(self, error_type: type, message: str) -> None:
        """Undo a failed construction on every rank, then raise the agreed error."""
        problem = self._release_together()
        self._closed = True
        if problem:
            message += f" (cleanup: {problem})"
        raise error_type(message)

    def _release_together(self) -> str:
        """Release this rank's resources in an order that is safe for every peer.

        Collective. Returns an empty string on success, or what failed.
        """
        held = bool(self._export or self._imports)
        failed = self._failed_ranks(self._sync() if held else True)
        if failed:
            self._retain()
            return f"device synchronization failed on rank(s) {failed}"
        failed = self._failed_ranks(self._close_imports())
        if failed:
            self._retain()
            return f"unmapping peer memory failed on rank(s) {failed}"
        # No rank maps a peer's buffer any more, so every rank may free its own.
        failed = self._failed_ranks(self._free_export())
        if failed:
            return f"freeing workspace memory failed on rank(s) {failed}"
        return ""

    def _failed_ranks(self, ok: bool) -> list[int]:
        table = self._dist.tp_allgather_int64([int(ok)])
        return [rank for rank, value in enumerate(table[:, 0]) if not value]

    def _sync(self) -> bool:
        try:
            torch.cuda.synchronize(self._device)
        except RuntimeError as error:
            logger.warning(f"MuseSparkOpen finalize workspace release: {error}")
            return False
        return True

    def _close_imports(self) -> bool:
        ok = True
        for pointer in self._imports:
            (error,) = cudart.cudaIpcCloseMemHandle(pointer)
            if error != cudart.cudaError_t.cudaSuccess:
                logger.warning(
                    f"MuseSparkOpen finalize workspace release: unmapping failed: {error}"
                )
                ok = False
        self._imports = []
        return ok

    def _free_export(self) -> bool:
        if not self._export:
            return True
        (error,) = cudart.cudaFree(self._export)
        if error != cudart.cudaError_t.cudaSuccess:
            logger.warning(f"MuseSparkOpen finalize workspace release: cudaFree failed: {error}")
            return False
        self._export = 0
        return True

    def _retain(self) -> None:
        """Keep everything a peer may still map allocated until the process exits."""
        self._export, self._imports = 0, []

    def __del__(self):
        # Local only, so it frees nothing that a peer may still map (see close()).
        if self._closed or sys.is_finalizing():
            return
        self._closed = True
        if not (self._export or self._imports):
            return
        if not self._sync():
            self._retain()
            return
        self._close_imports()
        if self._export:
            logger.debug(
                "MuseSparkOpen finalize workspace dropped without close(); its IPC buffer "
                "stays allocated until the process exits"
            )
            self._export = 0
