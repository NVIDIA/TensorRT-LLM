# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

_LORA_CKPT_SOURCES = ("hf", "nemo")


def _decode_string_scalar(value):
    """Decode a Triton STRING-tensor scalar to a Python str.

    A STRING tensor's `.item()` may return `bytes`, `numpy.bytes_`, or
    `str` depending on whether the array dtype is object/bytes-based or
    unicode-based, and on numpy version. Normalize to `str`.
    """
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def convert_request_input_to_dict(request, param_mappings, default_values,
                                  batch_size, batch_index):
    kwargs = {}
    for source_name, target_name in param_mappings.items():
        value = get_input_scalar_by_name(request, source_name, batch_size,
                                         batch_index)
        if value is None and source_name in default_values:
            kwargs[target_name] = default_values[source_name]
        elif value is not None:
            kwargs[target_name] = value

    return kwargs


def get_sampling_params_from_request(request, batch_size=1, batch_index=0):
    """Get trtllm.SamplingParams (LLMAPI) parameters from request.

    Used in llmapi/tensorrt_llm.
    """
    sampling_params_args = [
        'best_of',
        'temperature',
        'top_k',
        'top_p',
        'frequency_penalty',
        'presence_penalty',
        'max_tokens',
        'seed',
        'exclude_input_from_output',
        'return_perf_metrics',
    ]
    param_mappings = {}
    for arg in sampling_params_args:
        param_mappings[f"sampling_param_{arg}"] = arg
    default_values = {
        'sampling_param_best_of': 1,
        'sampling_param_exclude_input_from_output': False,
        'sampling_param_return_perf_metrics': False,
    }
    kwargs = convert_request_input_to_dict(request, param_mappings,
                                           default_values, batch_size,
                                           batch_index)
    # Pasrse stop as a list of strings not scalar
    kwargs['stop'] = get_input_tensor_by_name(request, 'sampling_param_stop')
    return kwargs


def get_output_config_from_request(request, batch_size=1, batch_index=0):
    """Get trtllm output-config parameters from request.

    Used in llmapi/tensorrt_llm.
    """
    output_config_args = [
        'return_finish_reason', 'return_stop_reason',
        'return_cumulative_logprob'
    ]
    param_mappings = {}
    for arg in output_config_args:
        param_mappings[arg] = arg

    default_values = {
        'return_finish_reason': False,
        'return_stop_reason': False,
        'return_cumulative_logprob': False
    }
    kwargs = convert_request_input_to_dict(request, param_mappings,
                                           default_values, batch_size,
                                           batch_index)
    return kwargs


def get_streaming_from_request(request, batch_size=1, batch_index=0):
    """Get the streaming flag from request.

    Used in llmapi/tensorrt_llm.
    """
    streaming = get_input_scalar_by_name(request, 'streaming', batch_size,
                                         batch_index) or False
    return streaming


def get_lora_request_from_request(request, batch_size=1, batch_index=0):
    """Construct a LoRARequest from triton request inputs.

    Returns None if none of lora_id/lora_name/lora_path are provided. If
    any of those three is provided, all three are required. The optional
    lora_ckpt_source input selects the checkpoint format and defaults to
    "hf"; "nemo" is also accepted.

    Used in llmapi/tensorrt_llm.
    """
    lora_id = get_input_scalar_by_name(request, 'lora_id', batch_size,
                                       batch_index)
    lora_name = get_input_scalar_by_name(request, 'lora_name', batch_size,
                                         batch_index)
    lora_path = get_input_scalar_by_name(request, 'lora_path', batch_size,
                                         batch_index)
    if lora_id is None and lora_name is None and lora_path is None:
        return None
    if lora_id is None or lora_name is None or lora_path is None:
        raise pb_utils.TritonModelException(
            "lora_id, lora_name, and lora_path must all be provided together")
    lora_name = _decode_string_scalar(lora_name)
    lora_path = _decode_string_scalar(lora_path)
    lora_ckpt_source = get_input_scalar_by_name(request, 'lora_ckpt_source',
                                                batch_size, batch_index)
    if lora_ckpt_source is None:
        lora_ckpt_source = "hf"
    else:
        lora_ckpt_source = _decode_string_scalar(lora_ckpt_source)
        if lora_ckpt_source not in _LORA_CKPT_SOURCES:
            raise pb_utils.TritonModelException(
                f"lora_ckpt_source must be one of {_LORA_CKPT_SOURCES}, "
                f"got {lora_ckpt_source!r}")
    # Validate lora_path eagerly so a missing path raises the wrapper's
    # TritonModelException (matching the partial-input error path) instead
    # of letting LoRARequest.__post_init__ surface a raw ValueError.
    if not os.path.exists(lora_path):
        raise pb_utils.TritonModelException(
            f"lora_path does not exist on the Triton server's filesystem: "
            f"{lora_path}")
    # Deferred import: importing tensorrt_llm at module load time
    # initializes CUDA/MPI in the Triton Python backend process before the
    # engine subprocess is spawned. Other helpers follow the same pattern.
    from tensorrt_llm.executor.request import LoRARequest
    return LoRARequest(lora_name=lora_name,
                       lora_int_id=int(lora_id),
                       lora_path=lora_path,
                       lora_ckpt_source=lora_ckpt_source)


def get_input_scalar_by_name(request,
                             name,
                             expected_batch_size=1,
                             batch_index=0):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    tensor = tensor.as_numpy()

    if tensor.size != expected_batch_size:
        raise pb_utils.TritonModelException(
            f"Expected a scalar tensor for tensor {name}")

    return tensor.item(batch_index)


def get_input_tensor_by_name(request,
                             name,
                             expected_batch_size=None,
                             batch_index=None,
                             force_on_torch=False):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None

    if tensor.is_cpu() and not force_on_torch:
        tensor = tensor.as_numpy()
    else:
        tensor = from_dlpack(tensor.to_dlpack())

    if expected_batch_size is not None and tensor.shape[
            0] != expected_batch_size:
        raise pb_utils.TritonModelException(
            f"Expected batch size doesn't match batch size for tensor {name}. Expected {expected_batch_size} got {tensor.shape[0]}"
        )

    if batch_index is not None and expected_batch_size is not None and batch_index >= expected_batch_size:
        raise pb_utils.TritonModelException(
            f"Invalid batch index in get_input_tensor_by_name for {name}")

    if batch_index is not None:
        # Add leading 1 batch dimension
        if isinstance(tensor, np.ndarray):
            return np.expand_dims(tensor[batch_index], axis=0)
        elif isinstance(tensor, torch.Tensor):
            return torch.unsqueeze(tensor[batch_index], dim=0)
    else:
        return tensor


def read_parameter_as_type(value, name, pytype=str):
    if value == "":
        return None
    if value.startswith("${") and value.endswith("}"):
        return None
    if pytype is bool:
        return value.lower() in ["1", "true"]
    try:
        result = pytype(value)
        return result
    except:
        pb_utils.Logger.log_warning(
            f"Could not read parameter '{name}' with value '{value}', will use default."
        )
        return None


def get_parameter(model_config, name, pytype=str):
    if name not in model_config["parameters"]:
        return None
    return read_parameter_as_type(
        model_config["parameters"][name]['string_value'], name, pytype)
