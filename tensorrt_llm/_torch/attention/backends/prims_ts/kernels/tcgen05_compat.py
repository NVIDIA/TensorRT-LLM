# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Narrow compatibility helpers for tcgen05 primitives used by PrimTS kernels."""

import functools
import importlib.metadata

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import arith as arith_dialect
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims


def _ld_32x32b_max_ptx(num: int) -> str:
    registers = ", ".join(f"{{$w{i}}}" for i in range(num))
    return (
        f"tcgen05.ld.red.sync.aligned.32x32b.x{num}.f32.max "
        f"{{{registers}}}, {{$w{num}}}, [{{$r0}}];"
    )


@functools.cache
def dsl_supports_ldtm_stat() -> bool:
    """LDTM.STAT uses a native wrapper in DSL 4.8 or inline PTX in 4.7."""
    try:
        dsl_version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        return False
    from packaging import version as pkg_version

    try:
        # Use .release so 4.7.0.dev* includes the inline-PTX path.
        return pkg_version.Version(dsl_version).release >= (4, 7, 0)
    except pkg_version.InvalidVersion:
        return False


def ldtm_stat_supported(compute_capability: tuple[int, int]) -> bool:
    """Whether kernels may take score maxima from LDTM.STAT on this GPU.

    tcgen05.ld.red.max (LDTM.STAT) exists on B300 (SM103) and Rubin (SM107),
    not B200 (SM100). DSL 4.7's Rubin escape hatch targets sm_100f, which
    cannot emit it, so SM107 also needs the native DSL wrapper.
    """
    if not dsl_supports_ldtm_stat():
        return False
    capability = tuple(compute_capability)
    if capability == (10, 3):
        return True
    if capability == (10, 7):
        return hasattr(prims, "tcgen05_ld_red")
    return False


@cute.jit
def tcgen05_ld_32x32b_max(tmem_addr: Int32, num: cutlass.Constexpr[int]) -> tuple:
    """Load ``num`` FP32 TMEM columns per lane and their maximum (LDTM.STAT).

    Returns the ``num`` loaded values followed by their maximum. The native
    DSL wrapper is used when present; inline PTX covers CUTLASS DSL 4.7, which
    has none. The caller gates the architecture with ``ldtm_stat_supported``
    and waits for TMEM loads before reading any result.
    """
    if cutlass.const_expr(hasattr(prims, "tcgen05_ld_red")):
        loaded_words, red_word = prims.tcgen05_ld_red(
            prims.Tcgen05LdStShape.SHAPE_32X32B,
            prims.make_tmem_ptr(tmem_addr, Float32),
            prims.ReductionKind.MAX,
            num=num,
        )
        result = tuple(loaded_words[i].bitcast(Float32) for i in range(num)) + (
            Float32(arith_dialect.bitcast(Float32.mlir_type, red_word.ir_value())),
        )
    else:
        result = cute.arch.inline_ptx(
            _ld_32x32b_max_ptx(num),
            write_only_types=[Float32] * (num + 1),
            read_only_args=[tmem_addr],
        )
    return result


@dsl_user_op
def tcgen05_mma_ws(
    mma_kind,
    d,
    a,
    b,
    idesc,
    enable_input_d,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue WS MMA across the CUTLASS DSL 4.7 keyword mismatch.

    CUTLASS DSL 4.7 exposes ``col_b_zero_mask`` in the public wrapper but
    forwards it under the rejected name ``zero_col_mask``. Prefer the public
    primitive so newer releases stay on their supported API; import the private
    binding only after observing that exact compatibility failure.
    """

    try:
        prims.tcgen05_mma_ws(
            mma_kind,
            d,
            a,
            b,
            idesc,
            enable_input_d,
            col_b_zero_mask=None,
            loc=loc,
            ip=ip,
        )
        return
    except TypeError as error:
        if "zero_col_mask" not in str(error):
            raise

    from cutlass.experimental.primitives import nvvm_wrapper as prims_nvvm

    prims_nvvm._assert_tensor_mem(d, "tcgen05.mma.ws")
    prims_nvvm._nvvm.tcgen05_mma_ws(
        prims_nvvm._TCGEN05_MMA_KIND_TO_DIALECT[mma_kind],
        d,
        a,
        cutlass.Int64(b),
        cutlass.Int32(idesc),
        cutlass.Boolean(enable_input_d),
        collector_b_buffer=None,
        collector_op=None,
        col_b_zero_mask=None,
        loc=loc,
        ip=ip,
    )


__all__ = [
    "dsl_supports_ldtm_stat",
    "ldtm_stat_supported",
    "tcgen05_ld_32x32b_max",
    "tcgen05_mma_ws",
]
