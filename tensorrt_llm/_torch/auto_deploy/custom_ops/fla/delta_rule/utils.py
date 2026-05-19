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
# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
import inspect
import os

import triton

FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"


supports_autotune_cache = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if supports_autotune_cache else {}
