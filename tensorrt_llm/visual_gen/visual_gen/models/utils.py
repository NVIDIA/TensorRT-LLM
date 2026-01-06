# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from visual_gen.layers.linear import ditLinear
from visual_gen.utils.logger import get_logger

logger = get_logger(__file__)


@contextmanager
def disable_weight_management():
    """
    Disable the pingpong logic in the weight management.
    This is useful if we just use parts of the transformer blocks, and don't want to break the pingpong order.
    In this context, we will copy the host weight to device in the forward pass and thus at the cost of latency.
    """
    prev = os.environ.get("DISABLE_WEIGHT_MANAGEMENT", None)
    os.environ["DISABLE_WEIGHT_MANAGEMENT"] = "True"
    try:
        yield
    finally:
        if prev is None:
            os.environ["DISABLE_WEIGHT_MANAGEMENT"] = "False"
        else:
            os.environ["DISABLE_WEIGHT_MANAGEMENT"] = prev


def check_transformer_blocks(blocks: nn.ModuleList):
    """
    Check if the model is a transformer model.
    We based on the assumption that the transformer blocks are the same structure.
    """
    if not isinstance(blocks, nn.ModuleList):
        return False

    def compare_module_structure(model1, model2):
        modules1 = dict(model1.named_modules())
        modules2 = dict(model2.named_modules())
        if set(modules1.keys()) != set(modules2.keys()):
            return False
        for name in modules1.keys():
            if type(modules1[name]) is not type(modules2[name]):
                return False

            params1 = list(modules1[name].parameters())
            params2 = list(modules2[name].parameters())
            if len(params1) != len(params2):
                return False
            for p1, p2 in zip(params1, params2):
                if p1.shape != p2.shape:
                    return False
                if p1.dtype != p2.dtype:
                    return False
        return True

    is_transformer_blocks = True
    for i in range(1, len(blocks)):
        if not compare_module_structure(blocks[i], blocks[0]):
            is_transformer_blocks = False
            break

    return is_transformer_blocks


class WeightManagedBlocks(torch.nn.ModuleList):
    """Manage Weights in a ModuleList
    The params with name ends of "weight" will be managed.
    Weights between different modules in the ModuleList can share same GPU buffer. If the shared GPU buffer is occupied by one block, then other blocks' weights will retain on CPU buffer.
    Note: We assume the weights in the ModuleList are all the same name, shape and dtype.
    """

    def __init__(self, modules: torch.nn.ModuleList) -> None:
        super().__init__(list(modules))
        self._check_block_weight()
        self.offloading_stride = 1

    def _check_block_weight(self):
        """
        Check if the weights in the ModuleList are all the same name, shape and dtype.
        """

        if not check_transformer_blocks(self):
            raise ValueError("The weights in the ModuleList are not all the same and thus cannot be managed.")

    def _set_next_offloading_layer_weight(self):
        """
        Set next_offloading_layer_weight to the next offloading block's same layer's host weight.
        """
        # 除了最后一个offloading block，设置next_offloading_layer_weight，这记录了下一个block的host weight，在执行当前层计算时，将下一层的host weight拷贝到pingpong buffer
        for i in range(len(self.offloading_block_index_list)):
            if i < len(self.offloading_block_index_list) - 1:
                current_block_idx = self.offloading_block_index_list[i]
                next_block_idx = self.offloading_block_index_list[i + 1]
                assert current_block_idx + self.offloading_stride == next_block_idx
            else:
                current_block_idx = self.offloading_block_index_list[i]
                next_block_idx = 0

            current_block = self[current_block_idx]
            next_block = self[next_block_idx]

            for name in self.managed_weight_layers:
                visual_gen_linear_current_block = current_block.get_submodule(name.replace("-", "."))
                visual_gen_linear_next_block = next_block.get_submodule(name.replace("-", "."))
                assert isinstance(visual_gen_linear_current_block, ditLinear) and isinstance(
                    visual_gen_linear_next_block, ditLinear
                )
                visual_gen_linear_current_block.next_offloading_layer_weight = visual_gen_linear_next_block.host_weight
                logger.debug(
                    f"set block[{current_block_idx}].{name}.next_offloading_layer_weight to block[{next_block_idx}].{name}.host_weight"
                )

    def _set_cuda_event_and_stream(self):
        """
        Set cuda event and stream for the ModuleList.
        """

        for idx, block in enumerate(self):
            if idx in self.offloading_block_index_list:
                for name in self.managed_weight_layers:
                    # set offloading_stream, offloading_event, device_weight for the offloading layer
                    offloading_layer = block.get_submodule(name.replace("-", "."))
                    setattr(offloading_layer, "offloading_stream", self.offloading_stream)
                    event = getattr(self, name + "-event")
                    setattr(offloading_layer, "offloading_event", event)
                    pingpong_buffers = getattr(self, name + "-weight-pingpong")
                    setattr(offloading_layer, "device_weight", pingpong_buffers)

    def _set_pingpong_in(self, block: torch.nn.Module, pingpong_in: int):
        """
        Set pingpong_in for the ModuleList.
        """
        for name in self.managed_weight_layers:
            offloading_layer = block.get_submodule(name.replace("-", "."))
            setattr(offloading_layer, "pingpong_in", pingpong_in)

    def setup_offloading(self, offloading_stride: int):
        """
        Setup offloading for the ModuleList.
        """
        self.offloading_stride = offloading_stride

        offloading_list = [True if i % self.offloading_stride == 0 else False for i in range(len(self))]
        self.offloading_block_index_list = [
            i for i, v in enumerate(offloading_list) if v
        ]  # offloading block index list, e.g. [0, 2, 4, 6]
        if len(self.offloading_block_index_list) <= 1:
            logger.debug(f"Too less offloading blocks, no need to setup offloading: {self.offloading_block_index_list}")
            self.offloading_block_index_list = []
            return

        for i in self.offloading_block_index_list:
            for module in self[i].modules():
                if isinstance(module, ditLinear):
                    module.set_offloading()

        # record which layers in the block need to be offloaded
        self.managed_weight_layers = list()
        device = torch.cuda.current_device()
        # Same layer in different blocks share the same pingpong buffer, event
        first_offloading_block = self[self.offloading_block_index_list[0]]
        for name, module in first_offloading_block.named_modules():
            if isinstance(module, ditLinear):
                name = name.replace(".", "-")
                self.managed_weight_layers.append(name)
                param = module.get_offloading_weights()
                setattr(
                    self,
                    name + "-weight-pingpong",
                    [
                        param.clone()
                        .detach()
                        .to(device),  # the first offloading block will get weight from pingpong_in
                        torch.nn.Parameter(
                            torch.empty(param.shape, dtype=param.dtype, device=device), requires_grad=False
                        ),
                    ],
                )
                logger.debug(f"register pingpong buffer for {name}")
                setattr(self, name + "-event", torch.cuda.Event())
                logger.debug(f"register event for {name}")
        # All layers share the same offloading stream
        self.offloading_stream = torch.cuda.Stream()

        self._set_next_offloading_layer_weight()
        self._set_cuda_event_and_stream()

        self.pingpong_in, self.pingpong_out = 0, 1
        self.current_block_idx = 0

        # Patch all blocks' forward methods to inject pingpong logic
        def make_wrapped_forward(orig_forward, block, idx, self_ref):
            def wrapped_forward(*args, **kwargs):
                is_offloading_block = idx in self_ref.offloading_block_index_list
                if is_offloading_block:
                    self_ref._set_pingpong_in(block, self_ref.pingpong_in)
                    # Ensure offloading block is executed in order
                    assert (
                        self_ref.current_block_idx == idx
                    ), f"current_block_idx {self_ref.current_block_idx} != idx {idx}"
                    self_ref.current_block_idx += self_ref.offloading_stride
                    if self_ref.current_block_idx >= len(self_ref):
                        self_ref.current_block_idx = 0
                out = orig_forward(*args, **kwargs)
                if is_offloading_block:
                    # swap pingpong_in and pingpong_out after forward
                    self_ref.pingpong_in, self_ref.pingpong_out = self_ref.pingpong_out, self_ref.pingpong_in
                return out

            return wrapped_forward

        for idx, block in enumerate(self):
            # Only patch if not already patched
            if not hasattr(block, "_original_forward_for_pingpong_patch"):
                block._original_forward_for_pingpong_patch = block.forward
                block.forward = make_wrapped_forward(block.forward, block, idx, self)


class QKVLinearMerger:
    """
    A utility class for merging consecutive Q, K, V linear layers in a model

    Functions:
    - Automatically find Q, K, V linear layers in the model
    - Merge three linear layers into one, with corresponding concatenation of weights and biases
    - Split the output of the merged layer into original Q, K, V tensors
    """

    @staticmethod
    def find_qkv_linear_layers(
        module: nn.Module,
        q_names: List[str] = ["to_q", "q_proj"],
        k_names: List[str] = ["to_k", "k_proj"],
        v_names: List[str] = ["to_v", "v_proj"],
        attn_layers_to_merge: List[str] = ["attn1"],
    ) -> List[Dict[str, Any]]:
        """
        Traverse the module to find all submodules containing Q, K, V linear layers

        Args:
            module: PyTorch module to search
            q_names: List of possible attribute names for Q layers (e.g., "to_q")
            k_names: List of possible attribute names for K layers (e.g., "to_k")
            v_names: List of possible attribute names for V layers (e.g., "to_v")
            attn_layers_to_merge: List of possible attribute names for attention layers to merge (e.g., "attn1")

        Returns:
            List where each element is a dictionary containing QKV information:
            {
                "q_layer": Q linear layer,
                "k_layer": K linear layer,
                "v_layer": V linear layer,
                "path": Module path (e.g., "blocks.0.attn")
            }
        """
        qkv_list = []

        def _recursive_search(current_module: nn.Module, current_path: str = ""):
            # If the current path is not in the attn_layers_to_merge, skip
            found_attn_layer = False
            for attn_layer in attn_layers_to_merge:
                if attn_layer in current_path:
                    found_attn_layer = True
                    break
            if not found_attn_layer:
                return

            # Find Q, K, V layers in current module
            q_layer = None
            for name in q_names:
                if hasattr(current_module, name) and isinstance(getattr(current_module, name), nn.Linear):
                    q_layer = getattr(current_module, name)
                    break

            k_layer = None
            for name in k_names:
                if hasattr(current_module, name) and isinstance(getattr(current_module, name), nn.Linear):
                    k_layer = getattr(current_module, name)
                    break

            v_layer = None
            for name in v_names:
                if hasattr(current_module, name) and isinstance(getattr(current_module, name), nn.Linear):
                    v_layer = getattr(current_module, name)
                    break

            # If complete QKV layers are found, record the information
            if q_layer and k_layer and v_layer:
                qkv_list.append({"q_layer": q_layer, "k_layer": k_layer, "v_layer": v_layer, "path": current_path})

            # Recursively search submodules
            for child_name, child_module in current_module.named_children():
                new_path = f"{current_path}.{child_name}" if current_path else child_name
                _recursive_search(child_module, new_path)

        _recursive_search(module)
        return qkv_list

    @staticmethod
    def merge_qkv_layers(
        q_layer: nn.Linear, k_layer: nn.Linear, v_layer: nn.Linear
    ) -> Tuple[nn.Linear, Tuple[int, int, int]]:
        """
        Merge Q, K, V linear layers into a single linear layer

        Args:
            q_layer: Q linear layer
            k_layer: K linear layer
            v_layer: V linear layer

        Returns:
            Merged linear layer + tuple of output dimensions (q_dim, k_dim, v_dim)
        """
        # Verify input dimension consistency
        assert (
            q_layer.in_features == k_layer.in_features == v_layer.in_features
        ), "QKV layers must have the same input dimension"

        # Calculate dimensions
        in_features = q_layer.in_features
        q_dim = q_layer.out_features
        k_dim = k_layer.out_features
        v_dim = v_layer.out_features
        out_features = q_dim + k_dim + v_dim

        # Create new linear layer (on the same device as original layers)
        device = q_layer.weight.device
        from visual_gen.layers.linear import ditLinear

        merged_layer = ditLinear(  # nn.Linear(
            in_features=in_features, out_features=out_features, bias=q_layer.bias is not None
        ).to(device)

        # Merge weights and biases
        merged_layer.weight.data = torch.cat(
            [q_layer.weight.data, k_layer.weight.data, v_layer.weight.data], dim=0  # Concatenate along output dimension
        )
        if q_layer.bias is not None:
            merged_layer.bias.data = torch.cat([q_layer.bias.data, k_layer.bias.data, v_layer.bias.data], dim=0)

        return merged_layer, (q_dim, k_dim, v_dim)

    @staticmethod
    def split_qkv_output(
        merged_output: torch.Tensor, q_dim: int, k_dim: int, v_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the output of the merged layer into Q, K, V tensors

        Args:
            merged_output: Output tensor from the merged layer (batch, seq, q_dim+k_dim+v_dim)
            q_dim: Feature dimension of Q
            k_dim: Feature dimension of K
            v_dim: Feature dimension of V

        Returns:
            Three tensors: Q, K, V
        """
        q, k, v = torch.chunk(merged_output, 3, dim=-1)

        # Verify split dimensions (optional, for debugging)
        assert q.size(-1) == q_dim, f"Q dimension mismatch: expected {q_dim}, got {q.size(-1)}"
        assert k.size(-1) == k_dim, f"K dimension mismatch: expected {k_dim}, got {k.size(-1)}"
        assert v.size(-1) == v_dim, f"V dimension mismatch: expected {v_dim}, got {v.size(-1)}"

        return q, k, v

    @staticmethod
    def replace_qkv_layers(
        model: nn.Module,
        qkv_info: Dict[str, Any],
        merged_layer: nn.Linear,
        dims: Tuple[int, int, int],  # (q_dim, k_dim, v_dim)
    ) -> None:
        """
        Replace the original Q, K, V layers in the model with the merged layer

        Args:
            model: The entire model
            qkv_info: Dictionary containing QKV layer information (from find_qkv_linear_layers)
            merged_layer: The merged linear layer
            dims: Output dimensions of Q, K, V
        """
        # 1. Parse module path (e.g., "blocks.13.attn2" -> parent path "blocks.13" + module name "attn2")
        full_path = qkv_info["path"]
        path_parts = full_path.split(".")
        module_name = path_parts[-1] if path_parts else ""
        parent_path = ".".join(path_parts[:-1]) if len(path_parts) > 1 else ""

        # 2. Get target module (the module containing QKV layers, e.g., attn2)
        target_module = model
        if parent_path:
            # Traverse parent path to find the parent of the target module
            for part in parent_path.split("."):
                target_module = getattr(target_module, part)
        # If path is empty, target module is the model itself
        if module_name:
            target_module = getattr(target_module, module_name)

        # 3. Save original layers (optional, for debugging or rollback)
        target_module.original_qkv = {
            "to_q": qkv_info["q_layer"],
            "to_k": qkv_info["k_layer"],
            "to_v": qkv_info["v_layer"],
            "dims": dims,
        }

        # 4. Remove original QKV layer attributes
        if hasattr(target_module, "to_q"):
            delattr(target_module, "to_q")
        if hasattr(target_module, "to_k"):
            delattr(target_module, "to_k")
        if hasattr(target_module, "to_v"):
            delattr(target_module, "to_v")

        # 5. Add merged layer and dimension information
        target_module.merged_qkv = merged_layer
        target_module.qkv_dims = dims  # (q_dim, k_dim, v_dim)

    @staticmethod
    def forward_with_merged_qkv(
        module: nn.Module, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward computation using the merged layer (replacing original Q, K, V logic)

        Args:
            module: Module containing merged_qkv and qkv_dims
            hidden_states: Input feature tensor

        Returns:
            Split Q, K, V tensors
        """
        # Call the merged layer
        merged_output = module.merged_qkv(hidden_states)
        # Split the result
        q, k, v = QKVLinearMerger.split_qkv_output(merged_output, *module.qkv_dims)  # Unpack (q_dim, k_dim, v_dim)
        return q, k, v


# New module to replace MockGELU, preserving only the proj operation
class MockGELUWithoutActivation(nn.Module):
    def __init__(self, original_proj_module):
        super().__init__()
        self.proj = original_proj_module

    def forward(self, x):
        return self.proj(x)


class GemmGeluProcessor:
    """
    An integrated tool class for detection, marking, and removal.
    1. Detect GEMM (ditLinear) -> GELU patterns.
    2. Set hasgelu=True for detected GEMM modules.
    3. Remove all GELU activation operations (preserve other operations).
    """

    def __init__(self):
        self.detected_gelus: List[Dict] = []
        self.updated_gemms: List[Dict] = []
        self.model: Optional[nn.Module] = None

    @staticmethod
    def is_gelu_module(module: nn.Module) -> bool:
        return isinstance(module, nn.GELU)

    @staticmethod
    def is_custom_gelu_module(module: nn.Module) -> bool:
        if isinstance(module, nn.GELU):
            return False
        module_name = module.__class__.__name__.lower()
        return "gelu" in module_name

    @staticmethod
    def is_gemm(module: nn.Module) -> bool:
        return isinstance(module, ditLinear)

    def _reset_hasgelu(self, module: nn.Module):
        """Recursively reset hasgelu attribute to False for all ditLinear modules"""
        if self.is_gemm(module):
            module.hasgelu = False
        for child in module.children():
            self._reset_hasgelu(child)

    def _find_and_mark_previous_gemm(
        self, gelu_path: str, parent_module: nn.Module, gelu_index: int = None, gelu_name: str = None
    ):
        """
        Find the preceding GEMM module for a GELU in the parent module and mark its hasgelu attribute.
        """
        children = list(parent_module.named_children())
        child_modules = [child for _, child in children]
        child_names = [name for name, _ in children]

        if gelu_index is not None:
            if gelu_index > 0:
                prev_module = child_modules[gelu_index - 1]
                prev_path = (
                    f"{gelu_path.rsplit('.', 1)[0]}.{child_names[gelu_index - 1]}"
                    if "." in gelu_path
                    else child_names[gelu_index - 1]
                )
                if self.is_gemm(prev_module):
                    prev_module.hasgelu = True
                    self.updated_gemms.append({"gemm_path": prev_path, "gelu_path": gelu_path})
        elif gelu_name is not None:
            try:
                gelu_index_in_children = child_names.index(gelu_name)
                for i in range(gelu_index_in_children - 1, -1, -1):
                    if self.is_gemm(child_modules[i]):
                        prev_path = (
                            f"{gelu_path.rsplit('.', 1)[0]}.{child_names[i]}" if "." in gelu_path else child_names[i]
                        )
                        child_modules[i].hasgelu = True
                        self.updated_gemms.append({"gemm_path": prev_path, "gelu_path": gelu_path})
                        break
            except ValueError:
                print(f"  -> Warning: Child module named {gelu_name} not found in parent module.")

    def _traverse_analyze_and_remove(self, module: nn.Module, parent_path: str = ""):
        """
        Recursively traverse the model to analyze and remove GELU operations.
        """
        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            children = list(module.children())
            child_names = [name for name, _ in module.named_children()]

            new_children = []
            for i, (child_name, child_module) in enumerate(zip(child_names, children)):
                current_path = f"{parent_path}.{child_name}" if parent_path else child_name
                if self.is_gelu_module(child_module):
                    self.detected_gelus.append({"path": current_path, "type": "independent_module"})
                    self._find_and_mark_previous_gemm(current_path, module, gelu_index=i)
                elif self.is_custom_gelu_module(child_module):
                    self.detected_gelus.append({"path": current_path, "type": "custom_module"})
                    if hasattr(child_module, "proj") and self.is_gemm(child_module.proj):
                        child_module.proj.hasgelu = True
                        self.updated_gemms.append({"gemm_path": f"{current_path}.proj", "gelu_path": current_path})

                    self._find_and_mark_previous_gemm(current_path, module, gelu_index=i)

                    new_module = MockGELUWithoutActivation(child_module.proj)
                    new_children.append(new_module)
                else:
                    self._traverse_analyze_and_remove(child_module, current_path)
                    new_children.append(child_module)

            if new_children != children:
                if isinstance(module, nn.Sequential):
                    module.__init__(nn.ModuleList(new_children))
                elif isinstance(module, nn.ModuleList):
                    module.__init__(new_children)
            return

        for name, child in module.named_children():
            current_path = f"{parent_path}.{name}" if parent_path else name

            if self.is_gelu_module(child):
                self.detected_gelus.append({"path": current_path, "type": "independent_module"})
                self._find_and_mark_previous_gemm(current_path, module, gelu_name=name)
                setattr(module, name, nn.Identity())

            elif self.is_custom_gelu_module(child):
                self.detected_gelus.append({"path": current_path, "type": "custom_module"})

                if hasattr(child, "proj") and self.is_gemm(child.proj):
                    child.proj.hasgelu = True
                    self.updated_gemms.append({"gemm_path": f"{current_path}.proj", "gelu_path": current_path})

                self._find_and_mark_previous_gemm(current_path, module, gelu_name=name)

                new_module = MockGELUWithoutActivation(child.proj)
                setattr(module, name, new_module)

            else:
                self._traverse_analyze_and_remove(child, current_path)

    def process_model(self, model: nn.Module) -> Tuple[List[Dict], List[Dict]]:
        """
        Core method: Process the model.
        """
        self.detected_gelus = []
        self.updated_gemms = []

        self._reset_hasgelu(model)
        self._traverse_analyze_and_remove(model)

        print(f"Total detected and removed GELU operations: {len(self.detected_gelus)}")
        print(f"Total GEMM modules marked with hasgelu=True: {len(self.updated_gemms)}")


def apply_async_cpu_offloading(
    model: nn.Module, transformer_blocks_name: str = "transformer_blocks", offloading_stride: int = 1
):
    """
    Apply async cpu offloading to the model.
    """
    transformer_blocks = getattr(model, transformer_blocks_name)
    transformer_blocks = WeightManagedBlocks(transformer_blocks)
    transformer_blocks.setup_offloading(offloading_stride)
    for name, param in model.named_parameters():
        parent_name = ".".join(name.split(".")[:-1])
        parent = model
        for n in parent_name.split("."):
            parent = getattr(parent, n)
        if not (isinstance(parent, ditLinear) and parent.offloading and "weight" in name):
            param.data = param.data.to(torch.cuda.current_device())
    for name, buffer in model.named_buffers():
        buffer.data = buffer.data.to(torch.cuda.current_device())
    setattr(model, transformer_blocks_name, transformer_blocks)
