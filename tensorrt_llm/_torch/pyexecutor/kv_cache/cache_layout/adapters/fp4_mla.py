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
"""The layout of a manager under ``Fp4MlaV2CacheLayoutPolicy``: FP4 MLA latents, the V sidecars
the importer derives from them, and a high-precision tail per model layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from ...kv_cache_manager_v2 import Role
from ..adapter import CacheLayoutAdapter
from ..page_layout import Tensor, replicated

if TYPE_CHECKING:
    from ....llm_request import LlmRequest


class Fp4MlaLayoutAdapter(CacheLayoutAdapter):
    def buffer_tensors(
        self, layer_id: int, role: str, size: int, tokens: int, state: bool
    ) -> tuple[Tensor, ...]:
        """Latents token-major by head, as the FP4 MLA kernels write them; the high-precision
        tail alike on every rank."""
        if role == Role.MLA_HP_TAIL:
            return replicated(size)
        return self.head_split(layer_id, size, tokens, state, token_major_=True)

    def derived_roles(self) -> frozenset[str]:
        """V scales and packed V derive from K and K scales; the generation worker rebuilds them
        locally (``rebuild_imported``)."""
        return frozenset((str(Role.MLA_V_SCALE), str(Role.MLA_V_PACKED)))

    def rebuild_imported(self, requests: Iterable[LlmRequest]) -> None:
        """Rebuild the derived V sidecars of requests whose prompt KV was imported."""
        from tensorrt_llm._torch.attention.backends.fp4_mla.v_cache import (
            rebuild_fp4_mla_disagg_imported_cache,
        )

        requests = list(requests)
        rebuild_fp4_mla_disagg_imported_cache(
            self.manager,
            [request.py_request_id for request in requests],
            [request.prompt_len for request in requests],
        )

    def layer_ids(self) -> dict[int, tuple[int, int]]:
        """A model layer's cache and its high-precision tail are two internal layers: global
        layers ``2 * m`` and ``2 * m + 1``, both of model layer ``m``."""
        layer_ids = {}
        for layer_group_id, layers in enumerate(self.manager.impl.layer_grouping):
            for layer, (model_layer, tail) in zip(
                layers, self._model_layers_and_tails(layer_group_id)
            ):
                layer_ids[int(layer)] = (2 * model_layer + int(tail), model_layer)
        return layer_ids

    def _model_layers_and_tails(self, layer_group_id: int) -> list[tuple[int, bool]]:
        """Per internal layer of the layer group, its model layer and whether it holds that
        layer's high-precision tail."""
        m = self.manager
        local_layer_ids = list(m.impl.layer_grouping[layer_group_id])
        cache_local = {
            int(layer_id): m._fp4_mla_compact_to_local[compact_layer]
            for compact_layer, layer_id in enumerate(m._cache_manager_layer_ids)
        }
        cache_local.update(
            {int(layer_id): int(layer_id) for layer_id in m._bf16_mla_manager_layer_ids}
        )
        hp_local = {
            int(layer_id): m._fp4_mla_compact_to_local[compact_layer]
            for compact_layer, layer_id in enumerate(m._hp_manager_layer_ids)
        }
        result = []
        for layer_id in local_layer_ids:
            internal_layer = int(layer_id)
            if internal_layer in cache_local:
                result.append((int(m.pp_layers[cache_local[internal_layer]]), False))
            elif internal_layer in hp_local:
                result.append((int(m.pp_layers[hp_local[internal_layer]]), True))
            else:
                raise ValueError(f"Unknown FP4 MLA V2 internal layer {internal_layer}.")
        return result
