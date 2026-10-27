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
"""JIT compile ahead for Kimi K3 (``KimiLinearForCausalLM``): the FP8 GEMMs
its weight loading creates, predicted from the constructed model, so the
DeepGEMM helpers build their kernels while the weights load
(``jit_prefetch_deep_gemm.FP8LinearDeepGemmProvider``)."""

from typing import Set, Tuple

from torch import nn


def fp8_weight_read_shapes(model) -> Set[Tuple[int, int]]:
    """``(N, K)`` of the FP8 weight-read GEMMs ``_finalize_weight_load``
    creates in the Kimi K3 ``model``, read from the constructed model.

    The weight-read projections exist only once the weights are loaded: the
    KDA layers' fused ``[q | k | v | g]`` projection and the converted
    shared-expert and latent MLP projections. Each rule mirrors the
    conversion it predicts; a shape it misses is planned from the loaded
    model instead, when warmup starts.
    """
    from . import modeling_kimi_linear as mkl

    shapes: Set[Tuple[int, int]] = set()
    if not mkl.is_sm_100f() or mkl._rubin_mxfp8_is_available():
        return shapes  # the weight read then runs the CuTe DSL MXFP8 GEMM

    def created(module) -> bool:
        return isinstance(module, nn.Linear) or (
            isinstance(module, mkl.TrtllmLinear) and module._weights_created
        )

    def fp8_linear(module) -> bool:
        return (
            isinstance(module, mkl.TrtllmLinear)
            and module._weights_created
            and module.has_fp8_block_scales
        )

    def nk(module) -> Tuple[int, int]:
        return int(module.weight.shape[0]), int(module.weight.shape[1])

    attention_dp = model.model_config.mapping.enable_attention_dp
    for layer in model.model.layers:
        if not mkl._has_weights(layer):
            continue
        if getattr(layer, "is_kda", False):
            attention = layer.linear_attn
            # ``finalize_decode_weights`` fuses [q | k | v | g] first
            # (``_fuse_checkpoint_projections``); a low-rank gate has no
            # ``g_proj`` and keeps them apart.
            qkvg = [getattr(attention, n, None) for n in ("q_proj", "k_proj", "v_proj", "g_proj")]
            if (
                attention.use_full_rank_gate
                and attention._dispatch.decode_kernel_path == "optimized"
                and all(fp8_linear(module) for module in qkvg)
            ):
                shapes.add((sum(m.out_features for m in qkvg), qkvg[0].in_features))
        else:
            attention = getattr(getattr(layer, "self_attn", None), "mixer", None)
        if attention is not None:
            shapes.update(nk(m) for m in attention.children() if fp8_linear(m))
        if not model._fp8_weight_read_moe_mlp:
            continue
        # ``_convert_moe_mlps_to_fp8_weight_read``.
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        converted = [
            getattr(moe, a, None) for a in ("routed_expert_down_proj", "routed_expert_up_proj")
        ]
        converted = [child for child in converted if isinstance(child, nn.Linear)]
        shared = getattr(moe, "shared_experts", None)
        if shared is not None:
            names = ("gate_proj", "up_proj", "gate_up_proj", "down_proj")
            for name in names if attention_dp else names[:2] + names[3:]:
                child = getattr(shared, name, None)
                if isinstance(child, nn.Linear) or (
                    isinstance(child, mkl.TrtllmLinear) and child.tp_size == 1
                ):
                    converted.append(child)
        shapes.update(nk(child) for child in converted if created(child))
    return shapes
