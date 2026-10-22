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
"""deferred_norm_evict_hints next-layer projection: deferred_norm's staged projection with
deferred_norm_evict_hints's schedule knobs.

``EvictHintsProjection`` is ``DeferredNormProjection`` (same module binding, weights read in
place, same outputs bit for bit) launching ``EvictHintsProjKernel``. The launch
configuration is deferred_norm's served one plus ``EvictHintsProjConfig``'s fields.
"""

from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple

import torch

from ..deferred_norm.projection import (
    _N_KVA,
    MAX_ROWS,
    SERVED_N1,
    DeferredNormProjConfig,
    DeferredNormProjection,
    _kernel_config,
    _sm_count,
)

# EvictHintsProjKernel's keys beyond deferred_norm's launch configuration.
EVICT_HINTS_CONFIG_KEYS = (
    "maxrreg",
    "maxntid",
    "copy_hint",
    "split_chunk",
    "split_delay_ns",
    "copy_warps",
    "prod_warp",
    "defer_tail",
    "defer_wait",
)


@dataclass(frozen=True)
class EvictHintsProjConfig(DeferredNormProjConfig):
    """deferred_norm's launch configuration plus deferred_norm_evict_hints's knobs (``proj_kernel.py``).

    The defaults are deferred_norm's launch with ``copy_hint="evict_first"`` (the
    weight copies carry an ``L2::evict_first`` policy). ``maxrreg`` caps the
    registers per thread (0: none). ``prod_warp`` adds a copy-issuing 17th
    warp, so that ``split_chunk`` / ``split_delay_ns`` / ``defer_tail`` /
    ``defer_wait`` defer part of the staging without holding an MMA warp.
    """

    maxrreg: int = 0
    maxntid: int = 0
    copy_hint: str = "evict_first"
    split_chunk: int = 0
    split_delay_ns: int = 0
    copy_warps: bool = False
    prod_warp: bool = False
    defer_tail: int = 0
    defer_wait: bool = False


def parse_cfg(text: str, cls=None):
    """``"key=value,..."`` into a config (bool / int / str by field type)."""
    cls = cls or EvictHintsProjConfig
    kwargs = {}
    types = {f.name: f.type for f in fields(cls)}
    for item in filter(None, text.replace(";", ",").split(",")):
        key, value = item.split("=")
        typ = types[key]
        if typ in (bool, "bool"):
            kwargs[key] = value.lower() in ("1", "true")
        elif typ in (int, "int"):
            kwargs[key] = int(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def config_for(config: EvictHintsProjConfig, n1: int) -> EvictHintsProjConfig:
    """The configuration deferred_norm_evict_hints launches for a layer kind.

    ``config`` serves the shared-indexer layers (``n1 = 0``), the only
    boundaries deferred_norm serves under its ``shared`` scope. A full-indexer layer
    (``n1 > 0``, scope ``all`` only) keeps deferred_norm's served launch configuration (no
    register cap, no copy hint, one CTA per SM).
    """
    return config if n1 == 0 else EvictHintsProjConfig(copy_hint="none")


def _evict_hints_kernel_config(
    config: EvictHintsProjConfig, n1: int, ctas: int, use_pdl: bool, deferred: bool, debug_ts: bool
) -> dict:
    cfg = _kernel_config(config, n1, ctas, use_pdl, deferred, debug_ts)
    z = asdict(config)
    for key in EVICT_HINTS_CONFIG_KEYS:
        cfg[key] = z[key]
    return cfg


def served_kernel_configs(device, config: Optional[EvictHintsProjConfig] = None) -> list:
    """Kernel configurations the deferred_norm_evict_hints consumer launches (one per ``SERVED_N1``)."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    config = config or EvictHintsProjConfig()
    use_pdl = get_env_enable_pdl()
    out = []
    for n1 in SERVED_N1:
        kind_config = config_for(config, n1)
        ctas = kind_config.ctas or _sm_count(device)
        cfg = _evict_hints_kernel_config(
            kind_config, n1, ctas, use_pdl, deferred=True, debug_ts=False
        )
        cfg["b_layout"] = "frag"
        out.append(cfg)
    return out


def precompile_served(device, config: Optional[EvictHintsProjConfig] = None) -> Optional[str]:
    """Compile every served deferred_norm_evict_hints projection kernel (no GPU work); see deferred_norm's."""
    from ..deferred_norm.proj_kernel import projection_supported
    from .proj_kernel import compiled_evict_hints_projection

    cfgs = served_kernel_configs(device, config)
    for cfg in cfgs:
        why = projection_supported(
            cfg["k"],
            cfg["n0"] + cfg["n1"],
            cfg["ctas"],
            cfg["threads"],
            cfg["smem_rows"],
            cfg["nchunk"],
        )
        if why is not None:
            return f"projection n1={cfg['n1']}: {why}"
    for cfg in cfgs:
        compiled_evict_hints_projection(**cfg)
    return None


class EvictHintsProjection(DeferredNormProjection):
    """``DeferredNormProjection`` launching the deferred_norm_evict_hints kernel."""

    def __init__(
        self,
        mla,
        config: Optional[EvictHintsProjConfig] = None,
        kw_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__(mla, config or EvictHintsProjConfig(), kw_weight)
        self.config = config_for(self.config, self.n1)
        self.ctas = self.config.ctas or _sm_count(self.w0.device)

    def kernel_config(self, deferred: bool = True, debug_ts: bool = False) -> dict:
        return _evict_hints_kernel_config(
            self.config, self.n1, self.ctas, self.use_pdl, deferred, debug_ts
        )

    def precompile(self) -> None:
        from .proj_kernel import compiled_evict_hints_projection

        compiled_evict_hints_projection(**self.served_config())

    def run(
        self,
        act: torch.Tensor,
        sumsq: Optional[torch.Tensor],
        eps: float,
        rows: Optional[int] = None,
        ts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Launch on the current stream (arguments as ``DeferredNormProjection.run``)."""
        from .proj_kernel import compiled_evict_hints_projection

        frag = act.dim() == 1
        if not frag:
            rows = act.shape[0]
        assert rows is not None and 1 <= rows <= MAX_ROWS
        assert act.is_contiguous() and act.dtype == torch.bfloat16
        kva = act.new_empty(rows, _N_KVA)
        kw = act.new_empty(rows, self.n1) if self.n1 else None
        deferred = sumsq is not None
        cfg = self.kernel_config(deferred, ts is not None)
        cfg["b_layout"] = "frag" if frag else "plain"
        compiled = compiled_evict_hints_projection(**cfg)
        pf = [(t.data_ptr(), t.numel() * t.element_size()) for t in self.prefetch[:3]]
        pf += [(0, 0)] * (3 - len(pf))
        compiled(
            act.data_ptr(),
            sumsq.data_ptr() if deferred else 0,
            self.w0.data_ptr(),
            0 if self.w1 is None else self.w1.data_ptr(),
            kva.data_ptr(),
            0 if kw is None else kw.data_ptr(),
            0 if ts is None else ts.data_ptr(),
            pf[0][0],
            pf[0][1],
            pf[1][0],
            pf[1][1],
            pf[2][0],
            pf[2][1],
            kva.view(torch.int32).view(-1),
            rows,
            float(eps),
        )
        return kva, kw


__all__ = [
    "EvictHintsProjConfig",
    "EvictHintsProjection",
    "config_for",
    "parse_cfg",
    "precompile_served",
    "served_kernel_configs",
]
