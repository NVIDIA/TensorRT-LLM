# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Graph transform to inject LoRA delta ops into the AutoDeploy graph.

This transform runs at the PATTERN_MATCHER stage, after attention/RoPE/RMSNorm
matching but BEFORE quantization and SwiGLU matching. At this point:
- Linears are still torch_linear_simple with original HF weight names
- Attention ops are already matched (Q/K/V linears feed into torch_attention_sdpa)
- Inserting lora_delta + add nodes naturally blocks downstream fusion patterns
  (SwiGLU, GEMM fusion) on LoRA targets — correct behavior

The lora_delta op reads runtime state from _GlobalLoraPlanner (populated each
forward by _prepare_inputs), so no LoRA metadata needs to be a graph input.

AutoDeploy LoRA assumes each adapter target maps to an HF module name that is
present as a real linear weight in the pre-fusion exported graph. It does not
synthesize fused adapter targets such as attn_qkv from separate q/k/v linears.
"""

import re
from typing import Optional, Tuple, Type

import torch
from pydantic import BaseModel, Field
from torch.fx import GraphModule

from ...custom_ops.lora.lora_delta import lora_delta  # noqa: F401 — registers the op
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_weight_name, is_linear_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _parse_weight_name(weight_name: str, hf_module_names: set) -> Optional[Tuple[int, str]]:
    """Parse a weight name to extract (layer_id, hf_module_name).

    Weight names follow the pattern:
        model.layers.{N}.self_attn.{module}.weight
        model.layers.{N}.mlp.{module}.weight

    Args:
        weight_name: e.g., "model.layers.3.self_attn.q_proj.weight"
        hf_module_names: set of HF module names to match (e.g., {"q_proj", "k_proj"})

    Returns:
        (layer_id, hf_module_name) or None if not a LoRA target.
    """
    # Extract the module name (second-to-last segment before .weight)
    parts = weight_name.split(".")
    if len(parts) < 3 or parts[-1] != "weight":
        return None

    hf_module = parts[-2]
    if hf_module not in hf_module_names:
        return None

    # Extract layer_id from "model.layers.{N}...."
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match is None:
        return None

    layer_id = int(match.group(1))
    return layer_id, hf_module


class InjectLoraTargetConfig(BaseModel):
    """Resolved LoRA target for one HF graph module."""

    hf_module_name: str = Field(
        description="HF module name to match in graph weight names, e.g. q_proj."
    )
    module_id: int = Field(description="Runtime LoRA module id passed to auto_deploy::lora_delta.")


class InjectLoraConfig(TransformConfig):
    """Configuration for the inject_lora transform."""

    lora_targets: list[InjectLoraTargetConfig] = Field(
        default_factory=list,
        description="Resolved LoRA targets keyed by HF graph module name and runtime module id.",
    )


@TransformRegistry.register("inject_lora")
class InjectLora(BaseTransform):
    """Inject auto_deploy::lora_delta ops after LoRA-target linear nodes.

    For each target linear, inserts:
        linear_out = torch_linear_simple(x, weight, bias)
        lora_out = auto_deploy::lora_delta(x, layer_id, module_id, output_size)
        combined = aten.add(linear_out, lora_out)

    Downstream users of the linear are rewired to use the combined output.

    This naturally blocks SwiGLU/GEMM fusion on LoRA targets because the
    lora_delta + add nodes break the expected fusion patterns.
    """

    config: InjectLoraConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return InjectLoraConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        lora_targets = self.config.lora_targets
        if not lora_targets:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        target_by_hf_module = {}
        for target in lora_targets:
            if target.hf_module_name in target_by_hf_module:
                raise ValueError(
                    f"Duplicate AutoDeploy LoRA target for HF module name: {target.hf_module_name}."
                )
            target_by_hf_module[target.hf_module_name] = target.module_id

        hf_module_names = set(target_by_hf_module.keys())

        matched_hf_modules = {
            parsed[1]
            for node in gm.graph.nodes
            if is_linear_op(node)
            and (weight_name := extract_weight_name(node))
            and isinstance(weight_name, str)
            and (parsed := _parse_weight_name(weight_name, hf_module_names)) is not None
        }
        missing_hf_modules = hf_module_names - matched_hf_modules
        if missing_hf_modules:
            missing_targets = {
                target.hf_module_name: target.module_id
                for target in lora_targets
                if target.hf_module_name in missing_hf_modules
            }
            raise ValueError(
                "AutoDeploy LoRA requires every adapter target to map to an HF linear weight "
                "present in the pre-fusion exported graph. Missing target modules: "
                f"{missing_targets}. Synthetic fused targets are not created by inject_lora."
            )

        cnt = 0
        # Walk all linear nodes and inject lora_delta where appropriate
        for node in list(gm.graph.nodes):
            if not is_linear_op(node):
                continue

            weight_name = extract_weight_name(node)
            if not weight_name or not isinstance(weight_name, str):
                continue

            parsed = _parse_weight_name(weight_name, hf_module_names)
            if parsed is None:
                continue

            layer_id, hf_module = parsed
            module_id = target_by_hf_module[hf_module]

            # Get the input to the linear (first arg)
            linear_input = node.args[0]

            # Insert lora_delta node after the linear.
            # Pass the linear's output node (node) as linear_out so that the fake
            # implementation can use torch.empty_like(linear_out) to share symbolic shapes.
            with gm.graph.inserting_after(node):
                lora_node = gm.graph.call_function(
                    torch.ops.auto_deploy.lora_delta.default,
                    args=(linear_input, node, layer_id, module_id),
                )

            # Insert add node combining linear output + lora delta
            with gm.graph.inserting_after(lora_node):
                add_node = gm.graph.call_function(
                    torch.ops.aten.add.Tensor,
                    args=(node, lora_node),
                )
                # Tag the add node so fusion passes can distinguish it from residual adds
                add_node.meta["is_lora_add"] = True

            # Rewire downstream users: everyone who used the linear output now uses the add.
            # replace_all_uses_with replaces ALL references to node, including those inside
            # add_node and lora_node — we must restore those after.
            node.replace_all_uses_with(add_node)
            add_node.replace_input_with(add_node, node)  # add still reads from linear
            lora_node.replace_input_with(add_node, node)  # lora_delta still reads linear_out

            ad_logger.info(
                f"Injected LoRA delta for {weight_name} (layer={layer_id}, module_id={module_id})"
            )
            cnt += 1

        if cnt > 0:
            gm.recompile()

        return gm, TransformInfo(
            skipped=False,
            num_matches=cnt,
            is_clean=False,
            has_valid_shapes=(cnt == 0),
        )
