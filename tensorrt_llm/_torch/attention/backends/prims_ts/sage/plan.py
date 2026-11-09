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

"""Host-side state of a plan's prepared Sage scale buffer.

A plan that runs the Sage prepare as its own launch before attention keeps
the buffer it writes (``_SageScaleLayout``), the scale shapes a run must
supply and the compiled prepare launch; the contiguous context plan does.
The decode kernel compiles its prepare into the attention launch instead.
"""

import functools
from dataclasses import dataclass
from typing import Callable

import torch

from .config import SageAttentionConfig, sage_scale_shapes
from .layout import _SageScaleLayout

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"


@dataclass(frozen=True)
class SagePreparedScales:
    """A plan's prepared Sage scale buffer and the launch that writes it."""

    layout: _SageScaleLayout
    # Expected shape of every scale tensor a run supplies, by field name.
    scale_shapes: dict[str, tuple[int, ...]]
    # The buffer the Sage prepare writes before attention (``_SageScaleLayout``).
    scales: torch.Tensor
    prepare: Callable[..., None]


@functools.cache
def get_compiled_sage_prepare(
    device_index: int,
    layout: _SageScaleLayout,
    num_qo_heads: int,
    num_kv_heads: int,
) -> Callable[..., None]:
    """Compile the launch that writes a plan's prepared Sage scales.

    The compiled callable takes ``(q_scale, k_scale, v_scale, v_mean, scales)``
    torch tensors (``v_mean`` is ``None`` for a recipe without one) and runs on
    the current stream. The batch size is a runtime extent, as for the
    attention kernel, so plans of one layout share the compilation.
    """

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv

    from .prepare import _PrepareSageScales

    prepare = _PrepareSageScales(layout=layout, num_kv_heads=num_kv_heads)

    @cute.jit
    def prepare_adapter(
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        v_scale: cute.Tensor,
        v_mean: cute.Tensor | None,
        scales: cute.Tensor,
        stream: cuda_drv.CUstream,
    ) -> None:
        prepare(q_scale, k_scale, None, v_scale, v_mean, scales, stream)

    def fake_compact(shape):
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=16,
        )

    v_rows_shape = (cute.sym_int(), num_kv_heads, layout.v_rows.headdim)
    with torch.cuda.device(device_index):
        return cute.compile(
            prepare_adapter,
            fake_compact((num_qo_heads, cute.sym_int())),
            fake_compact((num_kv_heads, cute.sym_int())),
            fake_compact(v_rows_shape),
            fake_compact(v_rows_shape) if layout.v_rows.has_mean else None,
            fake_compact((cute.sym_int(),)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=_COMPILE_OPTIONS,
        )


def build_sage_prepared_scales(
    layout: _SageScaleLayout,
    sage: SageAttentionConfig,
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    device_index: int,
) -> SagePreparedScales:
    """Allocate a plan's prepared scale buffer and compile its prepare launch."""

    scale_shapes = sage_scale_shapes(
        sage,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    scales = torch.empty(
        layout.total_words(batch_size * num_kv_heads),
        dtype=torch.float32,
        device=device,
    )
    prepare = get_compiled_sage_prepare(
        device_index, layout, num_qo_heads, num_kv_heads
    )
    return SagePreparedScales(
        layout=layout,
        scale_shapes=scale_shapes,
        scales=scales,
        prepare=prepare,
    )
