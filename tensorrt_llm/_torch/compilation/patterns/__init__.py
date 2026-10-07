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

from inspect import signature
from typing import Callable, List

import torch

MATCHER_SUBSYSTEM = "torch_compile"


def _make_pattern_example_inputs(
        search_fn: Callable[..., object]) -> List[torch.Tensor]:
    """Dummy inputs for register_replacement when using search_fn_pattern.

    Torch 2.13+ always builds initial_arg_info from example_inputs via
    _trace_args_for_initial_trace, even when search_fn_pattern is supplied.
    Empty [] raises IndexError. These tensors are only for signature
    flattening; matching still uses search_fn_pattern.
    """
    return [torch.empty(0) for _ in signature(search_fn).parameters]
