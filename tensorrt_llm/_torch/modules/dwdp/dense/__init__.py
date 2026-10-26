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

Every rank of the dense group keeps one row of each MoE decoder layer's attention blob ``A(l)``
and MoE-dense blob ``D(l)``; the runtime gathers ``L(l+2) = A(l+2) + D(l+2)`` into small staging
slots on a copy stream.  It runs in one of two modes (``types.DenseMode``): with expert DWDP
(attached) the dense group is the DWDP group and the copies share the expert DWDP copy stream;
standalone, on an attention-DP instance without expert DWDP, the dense group is the attention-DP
(tp) group and the copies run on the dense manager's own stream.  Submodules:

* ``config`` -- ``DenseConfig``, the settings setup and the runtime read.
* ``types`` -- shared records, layout, slots, copy plans, CUDA seams, constants, errors.
* ``layout`` -- collector, checks, blob layout, checksums.
* ``repoint`` -- repointing module tensors into slots, rollback, release check, teardown.
* ``runtime`` -- ``DenseDWDP``: slot rings, events and the per-layer hooks.
* ``plan`` -- per-layer copy plans and their launcher.
* ``store`` -- the sharded row store (FABRIC and in-process loopback).
* ``setup`` -- setup phases P1-P6 over six collectives.
* ``manager`` -- ``DenseDwdpManager``: owns the committed runtime in both modes, primes it (H0)
  and is what the model's hooks find through ``get_global_dense_dwdp_manager``; standalone it
  also builds the group and the transfer and runs setup.
* ``gate`` -- the launch gate ``py_executor_creator`` runs before the model is built; it picks
  the mode.

Import the submodules directly; this package imports none of them.
"""
