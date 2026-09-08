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

import cutlass
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims


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


__all__ = ["tcgen05_mma_ws"]
