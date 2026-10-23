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
"""Model creation on meta tensors and the later allocation of real CUDA memory.

`ModelLoader.load()` first builds the model with meta tensors, then calls
this file to give every tensor CUDA memory before checkpoint weights arrive.
"""

import copy
import functools
import traceback
import warnings
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, MetaInitMode
from tensorrt_llm._torch.virtual_memory import RestoreMode
from tensorrt_llm._torch.virtual_memory import scope as virtual_memory_scope
from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType, LoadFormat
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.virtual_memory import _MultiPoolProxy

# Weight memory tag -> the engine memory tag that holds weights already on
# CUDA before the weight allocation stage.
_WEIGHT_TAG_TO_ENGINE_TAG = {
    ExecutorMemoryType.MODEL_WEIGHTS_MAIN: ExecutorMemoryType.MODEL_ENGINE_MAIN,
    ExecutorMemoryType.MODEL_WEIGHTS_DRAFT: ExecutorMemoryType.MODEL_ENGINE_DRAFT,
}


def create_model_with_meta_tensors(
    config: ModelConfig,
) -> tuple[DecoderModelForCausalLM, ModelConfig, bool]:
    """Try to build the LLM model with meta tensors.

    This changes every uninitialized tensors in the model to meta tensors.
    Meta tensor: tensor with shape and dtype info, but no GPU allocation.
    This avoids many costly CPU tensor allocation inside torch.nn.Module.__init__().

    Note input ``config`` may be in place for some models.

    Returns:
        - The model
        - The config it was built from (the model may have changed it): the
          copy after meta init, the original ``config`` after the fallback
        - bool: Whether the model was built with meta tensors.
    """
    # Config will be modified in-place for some models, like Qwen2.
    # To ensure the fallback code in the except block uses the original, un-corrupt config,
    # we make a deepcopy here.
    config_copy = copy.deepcopy(config)
    try:
        with MetaInitMode():
            model = AutoModelForCausalLM.from_config(config_copy)
        return model, config_copy, True
    except Exception:
        logger.info(f"Fallback to regular model init: {traceback.format_exc(limit=10)}")
        model = AutoModelForCausalLM.from_config(config)
        return model, config, False


def allocate_model_on_cuda(
    model: DecoderModelForCausalLM,
    *,
    is_meta_init: bool,
    load_format: LoadFormat,
    model_weights_memory_tag: Optional[ExecutorMemoryType],
    model_weights_restore_mode: Optional[RestoreMode],
) -> Optional["_MultiPoolProxy"]:
    """Move every model tensor to CUDA.

    This is called before checkpoint weights are loaded.

    If `model_weights_memory_tag` is not None, it allocates model weights and
    buffers in their respective virtual memory scope for memory management at
    runtime.

    Args:
        model: model built by `create_model_with_meta_tensors`
        is_meta_init: whether ``model`` was built with meta tensors
        load_format: load format of this load
        model_weights_memory_tag: virtual memory tag for the weights, or None
        model_weights_restore_mode: restore mode of that tag

    Returns:
        The memory pools of the weight scope, which must stay alive, or None
        when there is no weight memory tag.
    """
    weight_pool = None
    if model_weights_memory_tag is not None:
        weight_pool = _allocate_weights_in_tagged_scope(
            model,
            is_meta_init=is_meta_init,
            load_format=load_format,
            model_weights_memory_tag=model_weights_memory_tag,
            model_weights_restore_mode=model_weights_restore_mode,
        )
    elif is_meta_init:
        init_meta_tensors_on_cuda(model)

    # Ensure everything is at least on CUDA
    # No-op if worked as expected
    model.to("cuda")

    logger.info(
        f"Use {get_rank_model_storage(model) / (1024**3):.2f} GB on CUDA for model weights."
    )
    return weight_pool


def init_meta_tensors_on_cuda(model: torch.nn.Module) -> None:
    """Allocate empty CUDA storage for every meta tensor of ``model``.

    Tensors that are not meta stay as they are.
    """
    # Shared tensors map to one CUDA tensor.
    meta_to_cuda_tensor: dict[torch.Tensor, torch.Tensor] = {}
    model._apply(
        functools.partial(_init_meta_tensor_on_cuda, meta_to_cuda_tensor=meta_to_cuda_tensor)
    )


def _init_meta_tensor_on_cuda(
    tensor: torch.Tensor, meta_to_cuda_tensor: dict[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Return an empty CUDA tensor for a meta tensor; keep other tensors as is.

    Called through ``model._apply`` by `init_meta_tensors_on_cuda`.
    """
    if tensor.device != torch.device("meta"):
        return tensor
    if tensor not in meta_to_cuda_tensor:
        meta_to_cuda_tensor[tensor] = torch.empty_like(tensor, device="cuda")
    return meta_to_cuda_tensor[tensor]


def _allocate_weights_in_tagged_scope(
    model: DecoderModelForCausalLM,
    *,
    is_meta_init: bool,
    load_format: LoadFormat,
    model_weights_memory_tag: ExecutorMemoryType,
    model_weights_restore_mode: Optional[RestoreMode],
) -> "_MultiPoolProxy":
    """Allocate model weights in the tagged virtual memory scope.

    Note model buffers are not allocated in the tagged virtual memory scope.
    They are allocated normally in the current virtual memory scope.
    """
    # Old tensor -> its CUDA tensor; each CUDA tensor also maps to itself.
    old_to_cuda_tensor: dict[torch.Tensor, torch.Tensor] = {}

    # Allocate buffers to the outer virtual_memory_scope,
    # but parameters (weights) to the dedicated inner virtual_memory_scope.
    _apply_to_buffers_only(
        model, functools.partial(_allocate_buffer_on_cuda, old_to_cuda_tensor=old_to_cuda_tensor)
    )

    # Weights that are not meta already hold values. Keep them when the load
    # does not overwrite every weight (not AUTO or DUMMY), or when meta init
    # left them real on purpose (e.g. created with ``ones`` or ``full``).
    need_initialized_weights = load_format not in (LoadFormat.AUTO, LoadFormat.DUMMY)
    copy_real_weight_values = need_initialized_weights or is_meta_init
    with virtual_memory_scope(model_weights_memory_tag, model_weights_restore_mode) as pool:
        model._apply(
            functools.partial(
                _allocate_weight_on_cuda,
                old_to_cuda_tensor=old_to_cuda_tensor,
                copy_real_weight_values=copy_real_weight_values,
                model_weights_memory_tag=model_weights_memory_tag,
            )
        )
    return pool


def _apply_to_buffers_only(model: torch.nn.Module, fn):
    """Apply *fn* to every buffer in *model*, skipping parameters."""
    for module in model.modules():
        for key, buf in module._buffers.items():
            if buf is not None:
                module._buffers[key] = fn(buf)


def _allocate_buffer_on_cuda(
    tensor: torch.Tensor, old_to_cuda_tensor: dict[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Return the CUDA tensor for a buffer: empty for meta, a copy otherwise."""
    if tensor not in old_to_cuda_tensor:
        if tensor.device == torch.device("meta"):
            cuda_tensor = torch.empty_like(tensor, device="cuda")
        else:
            cuda_tensor = tensor.cuda()
        old_to_cuda_tensor[tensor] = cuda_tensor
        old_to_cuda_tensor[cuda_tensor] = cuda_tensor
    return old_to_cuda_tensor[tensor]


def _allocate_weight_on_cuda(
    tensor: torch.Tensor,
    old_to_cuda_tensor: dict[torch.Tensor, torch.Tensor],
    *,
    copy_real_weight_values: bool,
    model_weights_memory_tag: ExecutorMemoryType,
) -> torch.Tensor:
    """Return a new CUDA tensor for a weight, keeping its values when asked."""
    if tensor not in old_to_cuda_tensor:
        cuda_tensor = torch.empty_like(tensor, device="cuda")
        if tensor.device != torch.device("meta") and copy_real_weight_values:
            if tensor.is_cuda:
                warnings.warn(
                    f"A weight tensor of shape {tensor.shape} is already allocated on CUDA device before "
                    f"the weight allocation stage. This will cause extra CUDA memory usage in the "
                    f"'{_WEIGHT_TAG_TO_ENGINE_TAG[model_weights_memory_tag]}' scope."
                )
            cuda_tensor.copy_(tensor)
        old_to_cuda_tensor[tensor] = cuda_tensor
        old_to_cuda_tensor[cuda_tensor] = cuda_tensor
    return old_to_cuda_tensor[tensor]


def get_rank_model_storage(model) -> int:
    """Compute total CUDA memory used by the model in the current rank"""
    total_bytes = 0
    for _, param in model.named_parameters():
        if param.device.type == "cuda" and param.device.index == torch.cuda.current_device():
            total_bytes += param.element_size() * param.nelement()
    for _, buf in model.named_buffers():
        if buf.device.type == "cuda" and buf.device.index == torch.cuda.current_device():
            total_bytes += buf.element_size() * buf.nelement()
    return total_bytes
