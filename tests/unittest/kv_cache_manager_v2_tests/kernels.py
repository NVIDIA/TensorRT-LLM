# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import ctypes
from collections.abc import Sequence
from functools import lru_cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, Iterator

import cuda.bindings.driver as drv

try:
    from cuda.core import Kernel, ObjectCode, Program, ProgramOptions
except ImportError:
    from cuda.core.experimental import Kernel, Program, ProgramOptions
    from cuda.core.experimental._module import ObjectCode

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2._common import CudaStream, LayerId, MemAddress, TokenIdExt
    from kv_cache_manager_v2._utils import _unwrap, div_up, exact_div
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
        CudaStream,
        LayerId,
        MemAddress,
        TokenIdExt,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import _unwrap, div_up, exact_div

_SLEEP_TIME_NS: int = 0


@contextlib.contextmanager
def enable_kernel_delay() -> Iterator[None]:
    global _SLEEP_TIME_NS
    _SLEEP_TIME_NS = 30_000
    yield
    _SLEEP_TIME_NS = 0


@lru_cache(maxsize=None)
def get_program(debug: bool, max_tokens: int, sleep_time: int) -> ObjectCode:
    assert max_tokens > 0 and (max_tokens & (max_tokens - 1)) == 0, (
        "max_tokens must be a power of 2"
    )
    code = r"""
#if !defined(__CUDACC_RTC__)
#include <cassert>
#include <cstdio>
#endif

#ifdef NDEBUG
__device__ inline void check(bool condition) {
    if (!condition) {
        asm volatile("trap;" ::: "memory");
    }
}
#else
#define check assert
#endif

using uint32_t = unsigned int;
using uint16_t = unsigned short;

constexpr uint32_t sleepTime = SLEEP_TIME_NS;

struct alignas(16) Value {
    uint32_t token;
    uint32_t layer;
    uint32_t role;
    uint32_t beam;

    __device__ inline bool operator==(Value const& other) const {
        return token == other.token && layer == other.layer && role == other.role && beam == other.beam;
    }
    __device__ inline bool operator!=(Value const& other) const {
        return !(*this == other);
    }
};

constexpr uint32_t kMAX_TOKENS = MAX_TOKENS;

struct Tokens {
    uint32_t tokens[kMAX_TOKENS];
};

extern "C" __global__ void fillValues(Value* data, uint32_t valuesPerToken, uint32_t layer,
        uint32_t buf_id, uint32_t beam, __grid_constant__ const Tokens tokens, uint32_t numTokens) {
    if (sleepTime > 0) {
        __nanosleep(sleepTime);
    }
    check(numTokens <= kMAX_TOKENS);
    auto const tid = (static_cast<uint32_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    auto const idxToken = tid / valuesPerToken;
    if (idxToken >= numTokens) {
        return;
    }
    auto const token = tokens.tokens[idxToken];
    auto const value = Value{token, layer, buf_id, beam};
    data[tid] = value;
}

__device__ inline void assertEq(Value const& a, Value const& b) {
#ifndef NDEBUG
    if (a != b) {
        printf("(%d, %d, %d, %d) != (%d, %d, %d, %d)\n",
                a.token, a.layer, a.role, a.beam,
                b.token, b.layer, b.role, b.beam);
    }
#endif
    check(a == b);
}

extern "C" __global__ void checkValues(Value const* data, uint32_t valuesPerToken, uint32_t layer,
        uint32_t buf_id, uint32_t beam, __grid_constant__ const Tokens tokens, uint32_t numTokens) {
    if (sleepTime > 0) {
        __nanosleep(sleepTime);
    }
    check(numTokens <= kMAX_TOKENS);
    auto const tid = (static_cast<uint32_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    auto const idxToken = tid / valuesPerToken;
    if (idxToken >= numTokens) {
        return;
    }
    auto const token = tokens.tokens[idxToken];
    auto const value = Value{token, layer, buf_id, beam};
    assertEq(data[tid], value);
}
    """
    macros = [("MAX_TOKENS", str(max_tokens)), ("SLEEP_TIME_NS", str(sleep_time))]
    program_options = ProgramOptions(std="c++17", lineinfo=True, debug=debug, define_macro=macros)  # type: ignore[arg-type]
    if not debug:
        program_options.use_fast_math = True
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("fillValues", "checkValues"))
    return mod


def get_kernel(name: str, num_tokens: int, sleep_time: int) -> tuple[Kernel, int]:
    assert num_tokens > 0

    @lru_cache(maxsize=None)
    def impl(name: str, max_tokens: int, sleep_time: int) -> Kernel:
        assert name in ("fillValues", "checkValues")
        assert max_tokens != 0 and (max_tokens & (max_tokens - 1)) == 0, (
            "max_tokens must be a power of 2"
        )
        debug = False
        # debug = not NDEBUG
        return get_program(debug, max_tokens, sleep_time).get_kernel(name)

    # Round up to the next power of two
    max_tokens = 2 ** ((num_tokens - 1).bit_length())
    return impl(name, max_tokens, sleep_time), max_tokens


class Value(ctypes.Structure):
    _fields_ = [
        ("token", ctypes.c_uint32),
        ("layer", ctypes.c_uint32),
        ("buf_id", ctypes.c_uint32),
        ("beam", ctypes.c_uint32),
    ]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Value):
            return NotImplemented
        return (
            self.token == other.token
            and self.layer == other.layer
            and self.buf_id == other.buf_id
            and self.beam == other.beam
        )

    def __str__(self) -> str:
        return (
            f"Value(token={self.token}, layer={self.layer}, buf_id={self.buf_id}, beam={self.beam})"
        )


@lru_cache(maxsize=None)
def _get_ctypes_struct(max_tokens: int) -> type[ctypes.Structure]:
    class Tokens(ctypes.Structure):
        _fields_ = [
            ("tokens", ctypes.c_uint32 * max_tokens),
        ]

    Tokens.__name__ = f"Tokens_{max_tokens}"
    return Tokens


def _make_tokens(tokens: Sequence[TokenIdExt], max_tokens: int) -> ctypes.Structure:
    assert len(tokens) <= max_tokens
    padded = list(tokens) + [0] * (max_tokens - len(tokens))
    Tokens = _get_ctypes_struct(max_tokens)
    return Tokens(
        tokens=(ctypes.c_uint32 * max_tokens)(
            *[
                t if isinstance(t, int) else int.from_bytes(t[:4], "little", signed=False)
                for t in padded
            ]
        )
    )


def fill_values(
    address: MemAddress,
    bytes_per_token: int,
    layer: LayerId,
    buf_id: int,
    beam: int,
    tokens: Sequence[TokenIdExt],
    stream: CudaStream,
):
    values_per_token = exact_div(bytes_per_token, ctypes.sizeof(Value))
    num_tokens = len(tokens)
    if num_tokens == 0:
        return
    kernel, max_tokens = get_kernel("fillValues", len(tokens), _SLEEP_TIME_NS)
    args = (
        address,
        values_per_token,
        layer,
        buf_id,
        beam,
        _make_tokens(tokens, max_tokens),
        num_tokens,
    )
    arg_types = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        None,
        ctypes.c_uint32,
    )
    num_threads = values_per_token * num_tokens
    cta_size = 256
    _unwrap(
        drv.cuLaunchKernel(
            kernel._handle,
            div_up(num_threads, cta_size),
            1,
            1,
            cta_size,
            1,
            1,
            0,
            stream,
            (args, arg_types),
            0,
        )
    )


def check_values(
    address: MemAddress,
    bytes_per_token: int,
    layer: LayerId,
    buf_id: int,
    beam: int,
    tokens: Sequence[TokenIdExt],
    stream: CudaStream,
):
    values_per_token = exact_div(bytes_per_token, ctypes.sizeof(Value))
    num_tokens = len(tokens)
    if num_tokens == 0:
        return
    kernel, max_tokens = get_kernel("checkValues", len(tokens), _SLEEP_TIME_NS)
    args = (
        address,
        values_per_token,
        layer,
        buf_id,
        beam,
        _make_tokens(tokens, max_tokens),
        num_tokens,
    )
    arg_types = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        None,
        ctypes.c_uint32,
    )
    num_threads = values_per_token * num_tokens
    cta_size = 256
    _unwrap(
        drv.cuLaunchKernel(
            kernel._handle,
            div_up(num_threads, cta_size),
            1,
            1,
            cta_size,
            1,
            1,
            0,
            stream,
            (args, arg_types),
            0,
        )
    )


def debug_dump_tokens(
    addr: MemAddress, token_bytes: int, num_tokens: int, stream: CudaStream
) -> Iterator[Value]:
    if num_tokens == 0:
        return
    val_size = ctypes.sizeof(Value)
    values_per_token = exact_div(token_bytes, val_size)
    host_buf = (Value * values_per_token * num_tokens)()
    ptr = ctypes.addressof(host_buf)
    _unwrap(drv.cuMemcpyDtoHAsync(ptr, addr, num_tokens * token_bytes, stream))
    _unwrap(drv.cuStreamSynchronize(stream))
    for i in range(num_tokens):
        token = host_buf[i]
        value = Value.from_buffer_copy(token[0])
        for j in range(1, values_per_token):
            assert token[j] == token[0]
        yield value
