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
"""FlashInfer API-logging and trace-template symbols that the pinned FlashInfer wheel predates.

The vendored kernels only use them as decorators and trace metadata; execution never depends on them.
"""


def flashinfer_experimental_api(func=None, *, trace=None, **_):
    """No-op stand-in for ``flashinfer.api_logging.flashinfer_experimental_api``."""
    if func is None:
        return lambda f: f
    return func


def __getattr__(name):
    """Trace dispatch templates (``flashinfer.trace.templates.attention``) are not used for execution."""
    return None
