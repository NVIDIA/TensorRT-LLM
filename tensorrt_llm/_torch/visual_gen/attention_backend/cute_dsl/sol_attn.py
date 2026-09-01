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
"""
Sol-Attn backend for visual generation models.

Sol-Attn (https://arxiv.org/abs/2607.24027) is dynamic block routing +
sparse computation + approximation correction folded into one online-softmax
pass. The kernel is vendored from its reference implementation
(https://github.com/NVlabs/Sana, branch
https://github.com/NVlabs/Sana/tree/sol-engine, pinned at commit
https://github.com/NVlabs/Sana/commit/5fe5feb -- see
``cute_dsl_kernels/blackwell/sol_attn/THIRD_PARTY_NOTICES.md`` for the pin
and its currency-check note) under ``..cute_dsl_kernels.blackwell.sol_attn``
/ ``sol_attn_backend.py``. Only the sm100 (B200/GB200) and sm120 (RTX
Blackwell) kernels are carried; the upstream sm89/sm90 kernels and the Triton
reference path are not, and the FlashAttention CuTe helpers they needed come
from the ``flash-attn-4`` dependency rather than a vendored copy.

This file is only the TRT-LLM AttentionBackend adapter around that kernel's
public BTHD entry point, plus the dense_layers layer-skip guard.

``disabled_until_timestep`` is the dense-prefix control, and mirrors
skip_softmax's field of the same name: sparse attention stays disabled (that
is, the layer runs dense SDPA) while the normalized denoising timestep is at
or above the cutoff, and switches to the sparse kernel once it drops below.

The timestep arrives as a forward kwarg -- ``modules/attention.py`` already
threads it to every backend, and all VisualGen pipelines normalize it to
``[0, 1]`` by ``num_train_timesteps`` per the ``BaseDiffusionModel.forward``
contract. Nothing has to be wired per pipeline, and there is no process-wide
state to keep in sync.

"""

from typing import Any, Optional

import torch

from tensorrt_llm.logger import logger

from ..interface import AttentionBackend, AttentionTensorLayout

_sol_attn_import_error = None
try:
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn_backend import (
        _run_sol_attn_bthd as _sol_attn_run,
    )
except (ImportError, OSError) as e:
    _sol_attn_run = None
    _sol_attn_import_error = e


def _as_float(timestep: Any) -> Optional[float]:
    """Coerce a scalar/0-d/1-element timestep to float, else None."""
    if timestep is None:
        return None
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 0:
            return None
        return float(timestep.reshape(-1)[0].item())
    try:
        return float(timestep)
    except (TypeError, ValueError):
        return None


def sol_attn_graph_phase(
    timestep: Any, *, disabled_until_timestep: Optional[float]
) -> Optional[int]:
    """Return 1 once descending timesteps cross the cutoff, 0 before, else None.

    Same contract and sense as
    ``SkipSoftmaxScheduler.get_graph_phase_for_timestep``: phase 0 is the dense
    prefix, phase 1 the sparse phase, and ``None`` means there is no phase to
    distinguish so the CUDA-graph runner omits the key part.
    """
    if disabled_until_timestep is None:
        return None
    value = _as_float(timestep)
    if value is None:
        return None
    return int(value < disabled_until_timestep)


def _parse_dense_layers(spec: Optional[str]) -> frozenset:
    layers: set = set()
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(item))
    return frozenset(layers)


class SolAttnAttention(AttentionBackend):
    """Sol-Attn dynamic block-routing sparse attention (CuTeDSL, sm100/sm120).

    The kernel wrapper already falls back to dense SDPA on any unsupported
    shape/dtype/arch (see ``_run_sol_attn_bthd``); this class only adds the
    ``dense_layers`` layer-skip guard (evaluated at construction time, no
    external plumbing needed) and forwards the routing knobs from config.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 128,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        sparse_attention_config=None,
        **kwargs,
    ):
        if _sol_attn_run is None:
            raise ImportError(
                "SolAttnAttention requires the vendored sol_attn kernel "
                f"package; import failed: {_sol_attn_import_error}"
            )
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        assert self.num_kv_heads == self.num_heads, (
            f"Sol-Attn is MHA-only (num_kv_heads == num_heads), got "
            f"num_kv_heads={self.num_kv_heads}, num_heads={self.num_heads}. "
            f"GQA/MQA is not supported."
        )
        self.dtype = dtype
        cfg = sparse_attention_config
        self.tau = getattr(cfg, "tau", 1.0)
        self.thresh_type = getattr(cfg, "thresh_type", "diag")
        self.kv_splits = getattr(cfg, "kv_splits", "auto")
        self.disabled_until_timestep = getattr(cfg, "disabled_until_timestep", None)
        self.dense_layers = _parse_dense_layers(getattr(cfg, "dense_layers", None))

    # The `.item()` in here would graph-break the enclosing block once per
    # attention layer, so keep it in eager (as cute_dsl/fmha.py and VSA's
    # `_get_vsa_inputs` do). Returns a host-side bool, so the dense and sparse
    # phases still compile as separate graphs -- they run different kernels.
    @torch.compiler.disable
    def _dense_by_step(self, timestep) -> bool:
        phase = sol_attn_graph_phase(
            timestep,
            disabled_until_timestep=self.disabled_until_timestep,
        )
        if phase is None:
            # Fail open, matching the CuTeDSL skip-softmax path: without a
            # timestep we cannot tell which phase we are in, so run the
            # sparse kernel rather than silently forcing dense forever.
            # This degrades quality rather than raising, so say so once.
            logger.warning_once(
                "SolAttnAttentionConfig.disabled_until_timestep="
                f"{self.disabled_until_timestep} is set, but no `timestep` reached "
                "the Sol-Attn forward call. The dense prefix it requests will not "
                "be applied. Ensure the pipeline passes a normalized timestep, or "
                "unset disabled_until_timestep.",
                key="sol_attn_missing_timestep",
            )
            return False
        return phase == 0

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """q, k, v: [B, S, H, D] (NHD), same original token order in and out."""
        dense_by_layer = self.layer_idx in self.dense_layers
        dense_by_step = False
        if self.disabled_until_timestep is not None:
            dense_by_step = self._dense_by_step(kwargs.get("timestep"))
        if dense_by_layer or dense_by_step:
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
        return _sol_attn_run(
            q,
            k,
            v,
            tau=self.tau,
            thresh_type=self.thresh_type,
            kv_splits=self.kv_splits,
        )

    @classmethod
    def support_lse(cls) -> bool:
        return False

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
