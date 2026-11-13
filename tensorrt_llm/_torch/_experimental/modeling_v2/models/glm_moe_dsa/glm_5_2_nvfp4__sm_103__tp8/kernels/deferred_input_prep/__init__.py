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
"""deferred_input_prep: inter-iteration decode input preparation in one launch.

``PrepStager`` stages the step's inputs and ``SamplerStoreWriter`` writes the speculative
sampler's store; ``compat`` describes the three parts. The engine has no step-input stager or
sampler store writer hook to declare them to, so no target serves ``deferred_input_prep`` or
``sampler_store`` and ``runtime.make_runtime_hooks`` declares neither. Entry points
(``dispatch.get_impl``):

* ``prep_stager``: the ``PrepStager`` class.
"""

from ...dispatch import register
from .sampler_store import SamplerStoreWriter
from .stager import PrepStager

register("deferred_input_prep", "prep_stager", PrepStager)

__all__ = ["PrepStager", "SamplerStoreWriter"]
