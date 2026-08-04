# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# import submodules that require registration process
from . import compile, custom_ops, export, models  # noqa: F401
from ._compat import TRTLLM_AVAILABLE

if TRTLLM_AVAILABLE:
    from . import shim  # noqa: F401

    # import AutoDeploy LLM and LlmArgs (require TRT-LLM base classes)
    from .llm import *
    from .llm_args import *

try:
    # This will overwrite the AutoModelForCausalLM.from_config to support modelopt quantization
    import modelopt
except ImportError:
    pass
