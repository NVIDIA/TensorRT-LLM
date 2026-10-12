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
"""Built-in per-model serving extensions.

Each submodule defines one :class:`~tensorrt_llm.serve.serving_extensions.ServingExtension`
subclass and registers it with ``register_serving_extension`` at import time.
Importing this package is what registers them; the registry in
``serving_extensions.py`` imports it on the first lookup, and
``openai_server.py`` imports it at startup so a broken extension module fails
the server start rather than the first request.

Submodules may import ``openai_protocol`` (request/response types), which in
turn imports the registry module. Nothing here may import ``openai_server``.
"""

from . import gpt_oss, kimi_k3

__all__ = ["gpt_oss", "kimi_k3"]
