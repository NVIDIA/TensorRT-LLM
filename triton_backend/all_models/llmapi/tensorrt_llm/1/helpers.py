import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack


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
    """
    Helper function to get trtllm.SamplingParams (LLMAPI) parameters from request
    Used in llmapi/tensorrt_llm
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
    """
    Helper function to get trtllm.SamplingParams (LLMAPI) parameters from request
    Used in llmapi/tensorrt_llm
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
    """
    Helper function to get streaming from request
    Used in llmapi/tensorrt_llm
    """
    streaming = get_input_scalar_by_name(request, 'streaming', batch_size,
                                         batch_index) or False
    return streaming


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
