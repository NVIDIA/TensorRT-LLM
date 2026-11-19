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
"""Preparation passes that the kernels of the sm_120 family share (the chain of a kernel launches them:
<kernel package>/prep.py, with the V pass of its PV format: <kernel package>/v_pass.py).

  basis.py    the basis launch: the eigensolver of the rank-32 compensation basis, and the CTA body in which the
              other CTAs run the V scan and the V pass of a kernel
  qk_pass.py  rotation, NVFP4 quantization and compensation channels of Q and K
  encode.py   the encode launch: UE4M3 scale factors of Q' and K', score scale"""
