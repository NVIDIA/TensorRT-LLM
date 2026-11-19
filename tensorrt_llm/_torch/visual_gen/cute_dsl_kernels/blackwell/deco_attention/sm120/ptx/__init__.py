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
"""Inline PTX that the sm_120 family alone issues (RTX PRO 6000: warp-level MMAs), through the emitter of
deco_attention/common/ptx.py, which also holds the wrappers both families share. A wrapper is in the module of its
importers:

  shared.py    what both kernel packages import, or the preparation of Q and K (sm120/prep/)
  fp8pv.py     what the FP8-PV kernel package alone imports (sm120/fp8pv/)
  nvfp4pv.py   what the NVFP4-PV kernel package alone imports (sm120/nvfp4pv/)

Each wrapper emits exactly the instruction text it shows; whether a wrapper declares side effects is part of its
contract (it sets the side-effect marking of the inline assembly statement), and a docstring says "no side effects"
where the wrapper declares none. Two wrappers of one instruction differ in an operand or result type, in the text or
in the side effects, as their docstrings say. A module has the sections MMA and descriptors, barriers and waits,
conversions and packed arithmetic, memory access, each where it has wrappers of it."""
