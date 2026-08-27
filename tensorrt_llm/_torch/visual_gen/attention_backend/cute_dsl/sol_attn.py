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
pass. The kernel itself is vendored wholesale from its reference
implementation (https://github.com/NVlabs/Sana, branch
https://github.com/NVlabs/Sana/tree/sol-engine, pinned at commit
https://github.com/NVlabs/Sana/commit/5fe5feb -- see
``cute_dsl_kernels/blackwell/sol_attn/THIRD_PARTY_NOTICES.md`` for the pin
and its currency-check note) under ``..cute_dsl_kernels.blackwell.sol_attn``
/ ``sol_attn_backend.py`` -- this file is only the TRT-LLM AttentionBackend
adapter around that kernel's public BTHD entry point, plus the dense_layers
layer-skip guard.

``dense_steps`` (an absolute-step dense-prefix warmup, the sol_attn
counterpart to skip_softmax's normalized-timestep disabled_until_timestep)
requires a per-denoising-step counter threaded from each model's forward
loop -- the same way VSA's ``set_vsa_forward_context()`` is called once per
step by each pipeline. ``SolAttnStepContext.advance_step()`` is now called
once per denoising step by ``visual_gen/models/wan/pipeline_wan.py`` (mirrors
the VSA call site exactly: incremented once per ``forward_fn`` invocation,
before that step's layers run), and reset once per ``generate()`` call so a
warmup generation never leaks a stale step count into the timed run. Any
pipeline that does NOT call ``advance_step()`` leaves ``current_step()`` at
-1 forever, which would make ``dense_steps > 0`` silently dense every step
forever -- so ``forward()`` treats "never advanced" as a hard error instead
of a silent no-op (see ``_require_step_context`` below).
"""

import threading
from typing import Optional

import torch

from ..interface import AttentionBackend, AttentionTensorLayout

_sol_attn_import_error = None
try:
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn_backend import (
        _run_sol_attn_bthd as _sol_attn_run,
    )
except (ImportError, OSError) as e:
    _sol_attn_run = None
    _sol_attn_import_error = e


class SolAttnStepContext:
    """Process-wide denoising-step counter for the dense-prefix guard.

    Wired into visual_gen/models/wan/pipeline_wan.py; see module docstring.
    Other pipelines (Hunyuan, Flux, ...) do not call advance_step() yet --
    forward() below refuses dense_steps > 0 in that case rather than
    silently forcing dense every step (see _advanced).
    """

    _step = -1
    _advanced = False
    _lock = threading.Lock()

    @classmethod
    def advance_step(cls) -> None:
        with cls._lock:
            cls._step += 1
            cls._advanced = True

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._step = -1
            cls._advanced = False

    @classmethod
    def current_step(cls) -> int:
        return cls._step

    @classmethod
    def is_advancing(cls) -> bool:
        return cls._advanced


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
    """Sol-Attn dynamic block-routing sparse attention (CuTeDSL, sm90/sm100).

    The vendored kernel already falls back to dense SDPA on any unsupported
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
        self.dense_steps = getattr(cfg, "dense_steps", 0)
        self.dense_layers = _parse_dense_layers(getattr(cfg, "dense_layers", None))

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
        if self.dense_steps > 0:
            if not SolAttnStepContext.is_advancing():
                raise RuntimeError(
                    f"SolAttnAttentionConfig.dense_steps={self.dense_steps} requires "
                    "the active pipeline to call SolAttnStepContext.advance_step() "
                    "once per denoising step; it never has. Without that call, "
                    "current_step() would stay at -1 forever and every step would "
                    "silently run dense instead of the requested dense-prefix "
                    "window. Wire advance_step() into this model's forward loop "
                    "(mirrors VSA's set_vsa_forward_context() call site), or set "
                    "dense_steps=0."
                )
            dense_by_step = SolAttnStepContext.current_step() < self.dense_steps
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
