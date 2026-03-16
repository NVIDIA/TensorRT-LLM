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
"""
Conftest: Patch transformers version check to allow huggingface-hub > 1.0.

The installed huggingface-hub (1.7.1) is too new for the installed
transformers package, but the actual API surface used is compatible.
This patch must run before any tensorrt_llm import.
"""

import transformers.utils.versions

# Disable the strict version-range check that blocks huggingface-hub >= 1.0
transformers.utils.versions._compare_versions = lambda *a, **kw: None
