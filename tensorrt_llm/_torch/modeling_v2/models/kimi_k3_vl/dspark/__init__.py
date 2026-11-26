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
"""One-engine DSpark speculative decoding of the Kimi K3 family.

* ``shell.KimiK3SpecForCausalLM``: the ``DecoderModelForCausalLM`` base a K3 target derives
  from; it builds the drafter and the worker and runs the speculative step.
* ``drafter.K3DSparkDrafter``: the standalone MLA DSpark drafter.
* ``worker.K3DSparkWorker``: its ``SpecWorkerBase`` worker.

Importing the package imports none of them.
"""
