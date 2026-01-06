# Adapted from: https://github.com/huggingface/flux-fast/blob/5027798d7f69a8e0e478df92f48663c40727f8ea/utils/pipeline_utils.py#L198
#
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

import gc
from functools import wraps

import torch
from torch.utils._pytree import tree_map_only


def cudagraph_wrapper(func):
    """
    Decorator to automatically handle CUDAGraph record/replay for the given function.

    This decorator caches CUDAGraphs based on the shapes of tensor arguments,
    providing automatic replay for performance optimization.

    Args:
        func: The function to be decorated

    Returns:
        The decorated function with CUDAGraph capabilities
    """
    _graphs = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if this is a bound method or if it will be called as a bound method
        is_bound_method = hasattr(func, "__self__") or hasattr(func, "im_self")

        # Additional check: see if we're getting called with more args than the function expects
        # This indicates the function is being called as a bound method but we didn't detect it
        expected_args = func.__code__.co_argcount
        if len(args) == expected_args and not is_bound_method:
            # Normal case - not a bound method
            args_for_key = args
            call_args = args
        elif len(args) == expected_args - 1 and "self" in func.__code__.co_varnames[:1]:
            is_bound_method = True
            args_for_key = args
            call_args = args
        else:
            # Default handling
            args_for_key = args
            call_args = args

        # Generate cache key based on tensor shapes (excluding self for bound methods)
        input_shapes = tuple(
            list(tuple(arg.shape) for arg in args_for_key if isinstance(arg, torch.Tensor))
            + list((k, tuple(kwargs[k].shape)) for k in sorted(kwargs.keys()) if isinstance(kwargs[k], torch.Tensor))
        )
        key = hash(input_shapes)

        if key in _graphs:
            # Use the cached wrapper if one exists. This will perform CUDAGraph replay
            wrapped, *_ = _graphs[key]
            outputs = wrapped(*call_args, **kwargs)
            return outputs

        # Record a new CUDAGraph and cache it for future use
        g = torch.cuda.CUDAGraph()
        in_args, in_kwargs = tree_map_only(torch.Tensor, lambda t: t.clone(), (call_args, kwargs))

        # note: warmup before capture
        for _ in range(2):
            # For bound methods that expect self as first parameter, we need to handle differently
            if hasattr(func, "__self__") and func.__code__.co_varnames[0] == "self":
                # This is a bound method, call it without passing self explicitly
                if len(in_args) == func.__code__.co_argcount:
                    # We have too many args, remove the first one (which would be self)
                    actual_args = in_args[1:]
                    func(*actual_args, **in_kwargs)
                else:
                    func(*in_args, **in_kwargs)
            else:
                func(*in_args, **in_kwargs)  # Stream warmup
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        with torch.cuda.graph(g):
            # Same logic for CUDAGraph recording
            if hasattr(func, "__self__") and func.__code__.co_varnames[0] == "self":
                if len(in_args) == func.__code__.co_argcount:
                    actual_args = in_args[1:]
                    out_tensors = func(*actual_args, **in_kwargs)
                else:
                    out_tensors = func(*in_args, **in_kwargs)
            else:
                out_tensors = func(*in_args, **in_kwargs)

        def replay_wrapper(*args, **kwargs):
            # Note that CUDAGraphs require inputs/outputs to be in fixed memory locations.
            # Inputs must be copied into the fixed input memory locations.
            def copy_inputs(dst, src):
                if isinstance(src, torch.Tensor):
                    dst.copy_(src)
                elif isinstance(src, (list, tuple)):
                    dst = [copy_inputs(a, b) for a, b in zip(dst, src)]
                else:
                    dst = src

            copy_inputs(in_args, args)

            for key in kwargs:
                copy_inputs(in_kwargs[key], kwargs[key])
            g.replay()

            # Clone outputs on the way out to disconnect them from the fixed output memory
            # locations. This allows for CUDAGraph reuse without accidentally overwriting memory
            def clone_outputs(out_tensors):
                if isinstance(out_tensors, torch.Tensor):
                    return out_tensors.clone()
                elif isinstance(out_tensors, (list, tuple)):
                    return [clone_outputs(o) for o in out_tensors]
                else:
                    return out_tensors

            return clone_outputs(out_tensors)

        # Cache function that does CUDAGraph replay
        _graphs[key] = (replay_wrapper, g, in_args, in_kwargs, out_tensors)
        return replay_wrapper(*call_args, **kwargs)

    return wrapper
