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
"""specDSA (speculative DSA): fused decode sparse attention of the DeepSeek DSA models for Blackwell.

The indexer logits, the exact top-K selection, the mapping to compressed-KV rows and the sparse attention over the
selected rows run in one megakernel (few rows) or in two kernels (many rows). The top-K threshold is guessed from
sampled logits as in the GVR V2 top-K and checked by a count certificate; a row where the guess does not hold takes an
exact fallback.

The kernels are specialized for a spec (:mod:`.config`); DeepSeek-V4-Flash is the one with kernels today (``README.md``
lists the specs and how to add one). Import :mod:`.op` for the host side and :mod:`.config` for the specs; nothing is
re-exported here.
"""
