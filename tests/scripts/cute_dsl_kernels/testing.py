# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from functools import partial
from typing import Callable, Optional, Union

import cuda.bindings.driver as cuda_driver
import cuda.bindings.runtime as cuda_runtime
import cutlass.base_dsl.jit_executor
from cupti import cupti
from cutlass.cute.testing import JitArguments


class CuptiProfiler:
    """A class for managing CUPTI profiling measurements with start, stop, and duration methods.

    This class provides a clean interface for measuring CUDA kernel execution times
    using CUPTI (CUDA Profiling Tools Interface). It encapsulates the complexity
    of buffer management, callback registration, and activity tracking.

    Example usage:
        profiler = CuptiProfiler()
        profiler.start()
        # ... run your CUDA kernels ...
        profiler.stop()
        duration = profiler.get_duration()  # Returns total duration in milliseconds
    """

    def __init__(self, buffer_size: int = 8 * 1024 * 1024):
        """Initialize the CUPTI profiler.

        Args:
            buffer_size: Size of the CUPTI buffer in bytes (default: 8MB)
        """
        self.buffer_size = buffer_size
        self.timings = []
        self._is_active = False
        self._buffer_requested_callback = None
        self._buffer_completed_callback = None

    def _buffer_requested(self):
        """Internal callback for CUPTI buffer requests."""
        max_num_records = 0
        return self.buffer_size, max_num_records

    def _buffer_completed(self, activities: list):
        """Internal callback for processing completed CUPTI activities."""
        for activity in activities:
            start = activity.start if hasattr(activity, "start") else "nil"
            end = activity.end if hasattr(activity, "end") else "nil"
            duration = end - start if start != "nil" and end != "nil" else "nil"
            name = activity.name[:100] if hasattr(activity, "name") else "unknown"
            # Convert to milliseconds
            if duration != "nil":
                self.timings.append((name, duration / 1e6))
                # print(f"Activity: {name}, Duration: {duration / 1e6} ms")

    def start(self):
        """Start CUPTI profiling.

        Enables CUPTI activity tracking for concurrent kernels and registers
        the necessary callbacks for buffer management.

        Raises:
            ValueError: If CUPTI activity cannot be enabled
        """
        if self._is_active:
            raise RuntimeError("CUPTI profiler is already active")

        # Clear previous timings
        self.timings = []

        try:
            cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
        except cupti.cuptiError as e:
            raise ValueError(
                f"Error while enabling Activity Kind {cupti.ActivityKind.CONCURRENT_KERNEL.name}: {e}"
            )

        # Register callbacks
        self._buffer_requested_callback = self._buffer_requested
        self._buffer_completed_callback = partial(self._buffer_completed)

        cupti.activity_register_callbacks(
            self._buffer_requested_callback, self._buffer_completed_callback
        )

        self._is_active = True

    def stop(self):
        """Stop CUPTI profiling.

        Flushes all activities, disables CUPTI tracking, and finalizes the profiler.
        This method should be called after the kernels you want to measure have completed.
        """
        if not self._is_active:
            raise RuntimeError("CUPTI profiler is not active")

        # Flush all activities and cleanup
        cupti.activity_flush_all(0)
        cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
        cupti.finalize()

        self._is_active = False

    def get_duration(self) -> float:
        """Get the total duration of all measured activities in milliseconds.

        Returns:
            Total duration in milliseconds. Returns 0.0 if no activities were recorded.
        """
        return sum(timing[1] for timing in self.timings)


def _cuda_success(err: Union[tuple, cuda_runtime.cudaError_t, cuda_driver.CUresult], message: str):
    """Helper function to check CUDA API errors."""
    if isinstance(err, tuple):
        _cuda_success(err[0], message)
    elif isinstance(err, cuda_runtime.cudaError_t):
        error_message = cuda_runtime.cudaGetErrorString(err)[1].decode("utf-8")
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{message} : {error_message}")
    elif isinstance(err, cuda_driver.CUresult):
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            error_message = cuda_driver.cuGetErrorString(err)[1].decode("utf-8")
            raise RuntimeError(f"{message} : {error_message}")
    else:
        raise TypeError(f"{err} is an unexpected type : it should be a cudaError_t or CUresult")


def _does_kernel_use_stream(kernel: Callable, stream: cuda_driver.CUstream, *args, **kwargs):
    """Check if the kernel uses the provided non-default stream.

    It does this by capturing the stream and then checking if any kernels were launched.

    :param kernel: The kernel to check
    :type kernel: Callable
    :param stream: The stream to check
    :type stream: cuda_driver.CUstream
    :return: True if the kernel uses the stream, False otherwise
    :rtype: bool
    """
    assert int(stream) != int(cuda_driver.CUstream_flags.CU_STREAM_DEFAULT), (
        "Stream must be a non-default stream"
    )

    err = cuda_runtime.cudaStreamBeginCapture(
        stream, cuda_runtime.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
    )
    _cuda_success(err, "Error on stream capture")

    kernel(*args, **kwargs)

    err, graph = cuda_runtime.cudaStreamEndCapture(stream)
    _cuda_success(err, "Error on stream capture")

    # Get number of nodes in warmup graph to check it matches what is expected
    err, _, num_nodes = cuda_runtime.cudaGraphGetNodes(graph)
    _cuda_success(err, "Error on querying graph")
    return num_nodes > 0


def benchmark(
    callable: Callable,
    *,
    warmup_iterations: int = 10,
    iterations: int = 100,
    stream: Optional[cuda_driver.CUstream] = None,
    kernel_arguments: Optional[JitArguments] = None,
    workspace_generator: Optional[Callable[[], JitArguments]] = None,
    workspace_count: int = 1,
    use_cuda_graphs: bool = False,
    use_cupti: bool = False,
) -> float:
    """Benchmarks a callable function with the specified parameters.

    For example,
    .. code-block:: python

        from cutlass.cute.testing import benchmark

        @cute.jit
        def user_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda_driver.CUstream):
            # contents of the function
            pass

        time_us = benchmark(user_function, kernel_arguments=JitArguments(a, b, c, stream)
                            warmup_iterations=10, iterations=100
                            stream=stream)

    To prevent skewing results by repeately accessing the L2 cache, use the workspace_count and workspace_generator
    parameters to cycle through a number of different workspaces.

    .. code-block:: python

        from cutlass.cute.testing import benchmark


        @cute.jit
        def user_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
            # contents of the function
            pass


        def workspace_generator():
            # create a, b, and c
            return JitArguments(a, b, c)


        time_us = benchmark(
            user_function,
            workspace_generator=workspace_generator,
            workspace_count=10,
            warmup_iterations=10000,
            iterations=1000,
        )

    To benchmark you may always configure the function being profiled (callable), the warmup iterations, and
    the number of profiling iterations.

    Whenever the kernel being benchmarked runs in a non-default stream, the stream must be provided
    through the stream parameter.

    To use CUDA graphs, the callable must be a compiled @cute.jit annotated function.
    When using CUDA graphs, the kernel must be launched in a non-default stream.

    :param callable: The function to benchmark
    :type callable: Callable
    :param warmup_iterations: Number of warmup iterations, defaults to 10
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations, defaults to 100
    :type iterations: int, optional
    :param stream: Stream kernel is launched in, defaults to CUDA stream default
    :type stream: CUstream, None
    :param kernel_arguments: Kernel arguments to launch callable with, defaults to None
    :type kernel_arguments: JitArguments, None
    :param workspace_generator: Function that returns kernel arguments, defaults to None
    :type workspace_generator: Callable
    :param workspace_count: Number of workspaces (arguments) to loop through, looping through enough
        workspaces will keep the L2 cache cold
    :type workspace_count: int, optional
    :param use_cuda_graphs: Whether to use cuda graphs, defaults to False
    :type use_cuda_graphs: bool, optional

    :return: The benchmark time in microseconds
    :rtype: float
    """
    if stream is None:
        stream = cuda_driver.CUstream(cuda_driver.CUstream_flags.CU_STREAM_DEFAULT)

    if workspace_count < 1:
        raise ValueError("workspace_count must be at least 1")

    float("nan")
    if workspace_generator is None:
        # If no workspace generator is provided, we need a single workspace
        if workspace_count != 1:
            raise ValueError("Need a single workspace if not providing a generator")

        # If no workspace generator is provided, we need a kernel_argument
        if kernel_arguments is None:
            raise ValueError("Please pass a kernel argument if not providing a generator")

        def workspace_generator():
            return kernel_arguments

    workspaces = [workspace_generator() for _ in range(workspace_count)]

    for workspace in workspaces:
        if not isinstance(workspace, JitArguments):
            raise TypeError(
                "workspace_generator and/or kernel_arguments should use JitArguments type"
            )

    def _loop_and_call_kernel(iterations: int, workspace_index: int = 0):
        for _ in range(iterations):
            current_workspace = workspaces[workspace_index]
            callable(*current_workspace.args, **current_workspace.kwargs)
            workspace_index = (workspace_index + 1) % workspace_count
        return workspace_index

    # Create CUDA events for timing
    err, start_event = cuda_driver.cuEventCreate(cuda_driver.CUevent_flags.CU_EVENT_DEFAULT)
    _cuda_success(err, "Error on creating event")
    err, end_event = cuda_driver.cuEventCreate(cuda_driver.CUevent_flags.CU_EVENT_DEFAULT)
    _cuda_success(err, "Error on creating event")

    elapsed_time = float("nan")

    if use_cuda_graphs:
        # Check if the callable is a JitCompiledFunction or JitExecutor
        # These are functions that can be called to launch kernels
        compiled_types = (
            cutlass.base_dsl.jit_executor.JitCompiledFunction,
            cutlass.base_dsl.jit_executor.JitExecutor,
        )
        if not isinstance(callable, compiled_types):
            raise TypeError("Function must be precompiled to be used with CUDA Graphs")

        # Check if the stream is a non-default stream
        if int(stream) == int(cuda_driver.CUstream_flags.CU_STREAM_DEFAULT):
            raise ValueError(
                "Measuring with CUDA Graphs requires executing in a non-default stream"
            )

        workspace_index = 0

        # Capture warmup graph
        err = cuda_runtime.cudaStreamBeginCapture(
            stream, cuda_runtime.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        )
        _cuda_success(err, "Error on stream capture")

        workspace_index = _loop_and_call_kernel(warmup_iterations)
        err, gwarm = cuda_runtime.cudaStreamEndCapture(stream)
        _cuda_success(err, "Error on stream capture")

        # Get number of nodes in warmup graph to check it matches what is expected
        err, _, num_nodes = cuda_runtime.cudaGraphGetNodes(gwarm)
        _cuda_success(err, "Error on querying graph")
        # Assertion is >= since we may launch multiple kernels in one host function
        if num_nodes < warmup_iterations:
            raise ValueError(
                "CUDA stream passed to benchmark does not match the stream the kernel was launched in"
            )

        # Capture profiling graph
        err = cuda_runtime.cudaStreamBeginCapture(
            stream, cuda_runtime.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        )
        _cuda_success(err, "Error on stream capture")
        _loop_and_call_kernel(iterations, workspace_index)
        err, gprofile = cuda_runtime.cudaStreamEndCapture(stream)
        _cuda_success(err, "Error on stream capture")

        # Instantiate graphs
        err, gwarm = cuda_runtime.cudaGraphInstantiate(gwarm, 0)
        _cuda_success(err, "Error on graph instantiation")
        err, gprofile = cuda_runtime.cudaGraphInstantiate(gprofile, 0)
        _cuda_success(err, "Error on graph instantiation")

        # Launch warmup graph
        err = cuda_runtime.cudaGraphLaunch(gwarm, stream)
        _cuda_success(err, "Error on graph launch")

        # Record start time
        err = cuda_driver.cuEventRecord(start_event, stream)
        _cuda_success(err, "Error on recording event")

        # Launch profiling graph
        err = cuda_runtime.cudaGraphLaunch(gprofile, stream)
        _cuda_success(err, "Error on graph launch")

        # Record end time
        err = cuda_driver.cuEventRecord(end_event, stream)
        _cuda_success(err, "Error on recording event")
        err = cuda_driver.cuEventSynchronize(end_event)
        _cuda_success(err, "Error on synchronizing event")

        # Get elapsed time
        err, elapsed_time = cuda_driver.cuEventElapsedTime(start_event, end_event)
        _cuda_success(err, "Error on querying event")

        # Destroy graphs
        err = cuda_runtime.cudaGraphExecDestroy(gwarm)
        _cuda_success(err, "Error on destroying graph")
        err = cuda_runtime.cudaGraphExecDestroy(gprofile)
        _cuda_success(err, "Error on destroying graph")

    elif use_cupti:
        # Use the new CuptiProfiler class
        profiler = CuptiProfiler()

        # Warmup
        workspace_index = _loop_and_call_kernel(warmup_iterations)

        profiler.start()

        _loop_and_call_kernel(iterations, workspace_index)
        # Synchronize device
        err = cuda_runtime.cudaDeviceSynchronize()
        _cuda_success(err, "Error on synchronizing device")

        profiler.stop()
        elapsed_time = profiler.get_duration()

    else:
        if int(stream) != int(
            cuda_driver.CUstream_flags.CU_STREAM_DEFAULT
        ) and not _does_kernel_use_stream(
            callable, stream, *workspaces[0].args, **workspaces[0].kwargs
        ):
            raise ValueError(
                "CUDA stream passed to benchmark does not match the stream the kernel was launched in"
            )

        # Not using graphs
        # Warmup
        workspace_index = _loop_and_call_kernel(warmup_iterations)
        # Record start event
        err = cuda_driver.cuEventRecord(start_event, stream)
        _cuda_success(err, "Error on recording event")
        _loop_and_call_kernel(iterations, workspace_index)
        # Record end event
        err = cuda_driver.cuEventRecord(end_event, stream)
        _cuda_success(err, "Error on recording event")
        # Synchronize end event
        err = cuda_driver.cuEventSynchronize(end_event)
        _cuda_success(err, "Error on synchronizing event")
        err, elapsed_time = cuda_driver.cuEventElapsedTime(start_event, end_event)
        _cuda_success(err, "Error on querying event")

    # Destroy events
    err = cuda_driver.cuEventDestroy(start_event)
    _cuda_success(err, "Error on destroying event")
    err = cuda_driver.cuEventDestroy(end_event)
    _cuda_success(err, "Error on destroying event")

    return elapsed_time / iterations * 1e3
