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
"""MoE-owned fused kernels.

Per-architecture kernels live under ``blackwell/`` and ``rubin/``, keyed below
that by kernel family, so a submodule may host several related operators
rather than exactly one. ``cutedsl_megamoe/`` keeps the MegaMoE kernels for
both architectures in a self-contained tree behind its own ``KernelClass``
entry points. Callers own the enablement checks and decide what to do when a
kernel does not apply.

Nothing is re-exported from this file on purpose: importing one kernel must
not drag in the others, several of which are arch-gated.
"""
