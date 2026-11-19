# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_mxfp8_gemm and rubin_mxfp8_linear entries: the dense MXFP8 GEMM on sm_107,
and rubin_mxfp8_linear's sm_103 path (the engine's MXFP8 GEMM).

A sample of each entry's cells runs the entry against its reference: rubin_mxfp8_gemm on MXFP8
operands, rubin_mxfp8_linear (the ModelOpt MXFP8 W8A8 projections: the KDA projections and the shared
experts at each layout's shapes) on a BF16 input it quantizes the kernel's way. One call of the NVFP4
default target runs rubin_mxfp8_linear on sm_103 and sm_107; the sampled cells are the sm_107 targets'
and run there.
"""

import pytest

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E501
    mxfp8_scales_to_r128c4,
    quantize_mxfp8_reference,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E501
    rubin_mxfp8_gemm as op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear import (  # noqa: E501
    rubin_mxfp8_linear as linear,
)

__extra_import_path__ = [".", "../..", "../../../.."]
from _k31_utils import on_sm, sample_cells  # noqa: E402 — needs the path declared above
from _rubin_gemm_util import (  # noqa: E402 — needs the path declared above
    activation,
    generator,
    mxfp8_weight,
    requires_sm107,
)
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

pytestmark = on_sm(103, 107)


def _mnk(m: int, n: int, k: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(m=m, n=n, k=k))


#: The NVFP4 default target's rubin_mxfp8_linear call that runs on sm_103 and sm_107: N zero-padded
#: to a multiple of 32, which the sm_103 GEMM takes.
SM10X_LINEAR_CELL = _mnk(
    40, 6144, 7168, "KDA [q | k | v | g] at TP8, a verify step of 8 requests x 5 rows"
)

#: (entry, cell): a sample of the GEMM's cells and of rubin_mxfp8_linear's (catalog MXFP8 quantizer + the GEMM),
#: on sm_107; the NVFP4 default target's linear call on sm_103 and sm_107.
_ENTRY_CELLS = [
    *(
        pytest.param(
            entry, cell, marks=requires_sm107, id=f"{type(entry).__name__}-{cell.why[:44]}"
        )
        for entry in (op, linear)
        for cell in sample_cells(entry.CELLS, size_key="m")[:2]
    ),
    pytest.param(linear, SM10X_LINEAR_CELL, id=f"sm10x-{SM10X_LINEAR_CELL.why[:44]}"),
]


def _operands(spec, gen):
    a, a_scale = quantize_mxfp8_reference(activation(spec["m"], spec["k"], gen))
    b, b_scale = mxfp8_weight(spec["n"], spec["k"], gen)
    return a, b, mxfp8_scales_to_r128c4(a_scale), b_scale


def _linear_args(spec, gen):
    x = activation(spec["m"], spec["k"], gen)
    weight, weight_scale = mxfp8_weight(spec["n"], spec["k"], gen)
    return x, weight, weight_scale


def _check(out, ref) -> None:
    # E4M3 x E4M3 products and the power-of-two scales are exact in FP32, so both sides sum the
    # same exact terms in FP32 in another order and round once to BF16: the band of one BF16
    # rounding, as for a BF16 GEMM (2 units of 2^-8 of the row's largest magnitude per element,
    # half a unit in RMS). rubin_mxfp8_linear's reference quantizes the input to the kernel's bytes,
    # so the same band holds for it.
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


@pytest.mark.parametrize("entry, cell", _ENTRY_CELLS)
def test_certified_cells(entry, cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(entry)
    gen = generator(cell.why)
    args = _operands(cell.spec, gen) if entry is op else _linear_args(cell.spec, gen)
    out = entry(*args)
    _check(out, entry.reference(*args))
