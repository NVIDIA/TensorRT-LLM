# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import functools
import os
import tempfile
from contextlib import contextmanager
from typing import Callable, Optional

try:
    import ray
except ImportError:
    import tensorrt_llm.executor.ray.stub as ray


def _configure_deep_gemm_cache(rank: int, gpu: int) -> None:
    if os.environ.get("TRTLLM_DEEP_GEMM_CACHE_PER_PROCESS", "1") == "0":
        return

    cache_dir = os.environ.get("TRTLLM_CACHE_DIR")
    if (cache_dir and os.environ.get("DG_JIT_CACHE_DIR") == os.path.join(
            os.path.expanduser(cache_dir), "deep_gemm")):
        from tensorrt_llm.logger import logger

        logger.warning_once(
            "TRTLLM_CACHE_DIR keeps DeepGEMM isolation enabled; set "
            "TRTLLM_DEEP_GEMM_CACHE_PER_PROCESS=0 for better cache "
            "reuse at the risk of concurrent writes.",
            key="deep_gemm_unified_cache_isolation")
        os.environ["DG_JIT_CACHE_DIR"] = os.path.join(
            os.environ["DG_JIT_CACHE_DIR"], f"deep_gemm_rank{rank}_gpu{gpu}")
    else:
        os.environ.setdefault(
            "DG_JIT_CACHE_DIR",
            os.path.join(tempfile.gettempdir(),
                         f"deep_gemm_rank{rank}_gpu{gpu}"))


@contextmanager
def unwrap_ray_errors():
    try:
        yield
    except ray.exceptions.RayTaskError as e:
        raise e.as_instanceof_cause() from e


def control_action_decorator(func: Optional[Callable] = None,
                             *,
                             drain: bool = True) -> Callable:
    """Wrap a method in the ``control_action`` context manager.

    Supports both bare and parameterized forms::

        @control_action_decorator                  # drain=True (default)
        def shutdown(self): ...

        @control_action_decorator(drain=False)     # non-draining variant
        def update_weights_via_ipc_zmq(self): ...
    """

    def decorator(f: Callable) -> Callable:

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            with self.engine.control_action(drain=drain):
                return f(self, *args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
