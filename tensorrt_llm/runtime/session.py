# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from math import prod
from typing import Any, Dict, List, Optional

# isort: off
import torch
import tensorrt as trt
# isort: on

from .._utils import torch_dtype_to_trt, trt_dtype_to_torch
from ..logger import logger


@contextlib.contextmanager
def _scoped_stream():
    '''Create a scoped cuda stream, and synchronize it when the context is destroyed
    '''
    #TODO: delete torch, use cuda native python bindings
    import torch
    stream = torch.cuda.current_stream()
    try:
        # return a handle, trt and other lib does not recognize torch.cuda.Stream
        yield stream.cuda_stream
    finally:
        stream.synchronize()


@dataclass
class TensorInfo:
    name: str
    dtype: trt.DataType
    shape: tuple

    # add more info like strides, formats if needed

    def numel(self):
        return prod(self.shape)

    def view(self, *shape):
        assert set(map(type, shape)) == {int}
        n_unknown = len(tuple(filter(lambda l: l < 0, shape)))
        new_shape = list(shape)
        if n_unknown == 0:
            assert prod(shape) == self.numel()
        elif n_unknown == 1:
            n_known_elements = prod(filter(lambda l: l >= 0, shape))
            for i, l in enumerate(new_shape):
                if l == -1:
                    assert self.numel() % n_known_elements == 0
                    new_shape[i] = self.numel() // n_known_elements
                    break
        else:
            raise ValueError('More than one dimensions need to be inferred!')
        return TensorInfo(self.name, self.dtype, tuple(new_shape))

    def __len__(self):
        return self.shape[0]

    def squeeze(self, dim=0):
        if self.shape[dim] != 1:
            raise ValueError(f"dim {dim} is {self.shape[dim]} instead of 1!")
        return TensorInfo(self.name, self.dtype,
                          self.shape[:dim] + self.shape[dim + 1:])


class Session(object):
    ''' Session is a managed TensorRT runtime.  '''

    def __init__(self, **kwargs):
        # use Session.from_serialized_engine to create a session
        pass

    def _init(self, engine_buffer=None):
        '''
        @brief: Setup TensorRT engines and context from a serialized engine file
        @param engine_buffer: a buffer holds the serialized TRT engine
        '''
        self._runtime = trt.Runtime(logger.trt_logger)
        if engine_buffer is not None:
            self._engine = self.runtime.deserialize_cuda_engine(engine_buffer)

        self._context = None
        if not self.engine.streamable_weights_size:
            self.__prepare_execution_contexts()
        return self

    def __prepare_execution_contexts(self):
        self._context = self.engine.create_execution_context()
        assert self._context is not None, "Failed to create an execution context!"
        with _scoped_stream() as stream:
            self._context.set_optimization_profile_async(0, stream)

    @staticmethod
    def from_serialized_engine(engine) -> Session:
        '''
        @brief: Create a session from a serialized engine
        @param engine: a serialized engine
        @return: a Session object
        '''
        session = Session()
        return session._init(engine)

    @staticmethod
    def from_engine(engine) -> Session:
        '''
        @brief: Create a session from an existing ICudaEngine engine
        @param engine: an ICudaEngine
        @return: a Session object
        '''
        session = Session()
        session.engine = engine
        return session._init()

    @property
    def runtime(self) -> trt.Runtime:
        return self._runtime

    @property
    def engine(self) -> trt.ICudaEngine:
        return self._engine

    @engine.setter
    def engine(self, engine: trt.ICudaEngine):
        self._engine = engine

    @property
    def context(self) -> trt.IExecutionContext:
        '''
        @brief: Get the default TensorRT execution context,
            use self.engine.create_execution_context() to create a new context if needed
        @return: one TensorRT execution context object
        '''
        return self._context

    @property
    def context_mem_size(self) -> int:
        return self.engine.device_memory_size_v2

    def _print_engine_info(self):
        '''print engine info for debug purpose, internal use only.
        '''
        refittable = self.engine.refittable
        num_layers = self.engine.num_layers
        device_memory_size = self.engine.device_memory_size_v2
        name = self.engine.name
        nb_profiles = self.engine.num_optimization_profiles
        logger.info(
            f"Engine:{name=:}, {refittable=:}, {num_layers=:}, {device_memory_size=:}, {nb_profiles=:}"
        )
        self._print_io_info()

    def _print_io_info(self):
        '''print engine i/o info for debug purpose, internal use only.
        '''

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            tformat = ";".join([
                self.engine.get_tensor_format_desc(name, p)
                for p in range(self.engine.num_optimization_profiles)
            ])
            logger.info(
                f"Tensor:{name=:}, {mode=:}, {shape=:}, {dtype=:}, {tformat=:}")

    def set_shapes(self,
                   tensor_dict: Dict[str, torch.Tensor],
                   context: Optional[trt.IExecutionContext] = None):
        if context is None:
            context = self.context

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                ok = context.set_input_shape(name, tensor_dict[name].shape)
                logger.debug(
                    f"setting input tensor {name} with shape {tensor_dict[name].shape}"
                )
                if not ok:
                    raise ValueError(
                        f"Couldn't assign {name} with shape {tensor_dict[name].shape}, "
                        f"engine supports [min, opt, max] = {self.engine.get_tensor_profile_shape(name, context.active_optimization_profile)}"
                    )

    def infer_shapes(
            self,
            inputs: List[TensorInfo],
            context: Optional[trt.IExecutionContext] = None
    ) -> List[TensorInfo]:
        '''
        @brief: Set input shapes to given context, and infer the output shapes from the given input shapes.
               This function should be called every time when the input shapes are changed before calling run().
               Or call the context.set_input_shape on all dynamic shaped input tensors manually.
        @param inputs: list of TensorInfo object, each item represents an input tensor
        @param context: TensorRT execution context, if None, use the default context
        @return: list of TensorInfo object, each item represents an output tensor, returns None if failed
        '''
        # set shape to the default context if context is not specified
        if context is None:
            context = self.context
        for i in inputs:
            if self.engine.get_tensor_mode(i.name) != trt.TensorIOMode.INPUT:
                raise ValueError(f"Tensor:{i.name} is not an input tensor")
            if self.engine.get_tensor_dtype(i.name) != i.dtype:
                raise ValueError(f"Tensor:{i.name} has wrong dtype")
            if not context.set_input_shape(i.name, i.shape):
                raise RuntimeError(
                    f"Could not set shape {i.shape} for tensor {i.name}. Please check the profile range for which your model was build."
                )

        outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = context.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
                outputs.append(TensorInfo(name, dtype, shape))
        return outputs

    def _set_weight_streaming(self, gpu_weights_percent):
        if not self.engine.streamable_weights_size:
            assert gpu_weights_percent == 1, "Engine built without weight streaming. Cannot set gpu_weights_percent to a value other than 1."
            return

        assert self.engine is not None

        self._context = None

        min = 0
        max = self.engine.streamable_weights_size
        budget = int(gpu_weights_percent * max)

        self.engine.weight_streaming_budget_v2 = budget
        assert self.engine.weight_streaming_budget_v2 == budget, "Failed to set weight streaming budget!"
        logger.info(
            f"Set gpu weights percent to {gpu_weights_percent}, which is {budget} bytes. Valid range: {min} bytes ~ {max} bytes."
        )

        try:
            self.__prepare_execution_contexts()
        except:
            free_mem = torch.cuda.mem_get_info()[0]
            if free_mem < budget:
                raise torch.cuda.OutOfMemoryError(
                    f"Out of Memory: Memory budget is {budget} bytes but only {free_mem} bytes are available on the GPU."
                )
            raise

    def run(self,
            inputs: Dict[str, Any],
            outputs: Dict[str, Any],
            stream,
            context=None) -> bool:
        '''
        @brief: Run the TensorRT engine with the given inputs and outputs
        @param inputs: dict of input tensors, key is tensor name, value is tensor pointer or torch tensor
        @param outputs: dict of output tensors, key is tensor name, value is tensor pointer or torch tensor
        @param stream: cuda stream to enqueue the TensorRT engine on
        @param context: TensorRT execution context, if None, use the default context
        @return: True if enqueue succeeded, note the enqueue is an async call,
            returning True does not mean the execution is finished
        '''
        # enqueue to the default context if context is not specified
        if context is None:
            context = self.context

        import torch
        for tensor_name in inputs:
            tensor = inputs[tensor_name]
            ptr = tensor.data_ptr() if isinstance(tensor,
                                                  torch.Tensor) else tensor
            context.set_tensor_address(tensor_name, ptr)
        for tensor_name in outputs:
            tensor = outputs[tensor_name]
            ptr = tensor.data_ptr() if isinstance(tensor,
                                                  torch.Tensor) else tensor
            context.set_tensor_address(tensor_name, ptr)
        ok = context.execute_async_v3(stream)
        return ok

    def _debug_run(self,
                   inputs: Dict[str, "torch.Tensor"],
                   context=None) -> Dict[str, "torch.Tensor"]:
        '''Run the engine enqueue with allocated output tensors, for debug purpose, since it is a sync call and slower than run
        '''
        import torch

        inputs_info = [
            TensorInfo(name, torch_dtype_to_trt(tensor.dtype), tensor.shape)
            for name, tensor in inputs.items()
        ]
        outputs_info = self.infer_shapes(inputs_info)
        outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device='cuda')
            for t in outputs_info
        }
        with _scoped_stream() as stream:
            self.run(inputs=inputs,
                     outputs=outputs,
                     stream=stream,
                     context=context)
        return outputs
