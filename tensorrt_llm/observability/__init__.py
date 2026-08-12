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
"""Package-wide observability: logging and profiling.

Deliberately empty of re-exports.  ``logging`` is reached before ``torch`` is
prepared, while ``profiling`` imports ``torch`` at module scope; a re-export
here would make importing either one drag in the other.  Import the module you
need:

    from tensorrt_llm.observability.logging import logger
    from tensorrt_llm.observability import profiling
"""
