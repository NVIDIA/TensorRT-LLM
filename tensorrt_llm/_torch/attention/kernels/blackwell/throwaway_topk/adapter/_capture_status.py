# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared capture, validation and compilation boundaries for MQA TopK."""

from __future__ import annotations
import functools


class _CaptureStatusError(RuntimeError):
    """A host boundary observed an invalidated CUDA capture."""


@functools.cache
def _capture_runtime():
    try:
        from cuda.bindings import runtime
    except ImportError:
        from cuda import cudart as runtime
    return runtime


def _check_capture_status(stream, step):
    runtime = _capture_runtime()
    result = runtime.cudaStreamGetCaptureInfo(stream.cuda_stream)
    error, status = result[:2]
    if (
        error != runtime.cudaError_t.cudaSuccess
        or status == runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusInvalidated
    ):
        raise _CaptureStatusError(
            f"CUDA graph capture invalid after {step}: cudaStreamGetCaptureInfo returned {error}, capture status {status}"
        )
