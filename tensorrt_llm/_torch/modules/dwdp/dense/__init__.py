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
"""Dense DWDP: row-sharded replicated dense weights, gathered two layers ahead of their use.

Every rank of the DWDP group keeps one row of each MoE decoder layer's attention blob ``A(l)`` and
MoE-dense blob ``D(l)``; the runtime gathers ``L(l+2) = A(l+2) + D(l+2)`` into small staging slots
on the expert DWDP copy stream.  Submodules:

* ``config`` -- ``TRTLLM_DWDP_DENSE*`` switches.
* ``types`` -- shared records, layout, slots, copy plans, CUDA seams, constants, errors.
* ``layout`` -- collector, checks, blob layout, checksums.
* ``repoint`` -- repointing module tensors into slots, rollback, release check, teardown.
* ``runtime`` -- ``DenseDWDP``: slot rings, events and the per-layer hooks.
* ``plan`` -- per-layer copy plans and their launcher.
* ``store`` -- the sharded row store (FABRIC and in-process loopback).
* ``watchdog`` -- the slot checksum watchdog of ``TRTLLM_DWDP_DENSE_DEBUG=2``.
* ``setup`` -- setup phases P1-P6 over five fixed collectives.
* ``gate`` -- the launch gate ``py_executor_creator`` runs before the model is built.

Import the submodules directly; this package imports none of them.
"""
