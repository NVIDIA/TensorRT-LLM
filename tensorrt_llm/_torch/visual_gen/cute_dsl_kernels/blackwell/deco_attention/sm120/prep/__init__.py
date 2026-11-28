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
"""Preparation passes that the kernels of the sm_120 family share.

The chain of a kernel launches them (<kernel package>/prep.py), with the V pass of its PV format (v_pass.py there):
    basis.py    the basis launch: the eigensolver of the compensation basis, with a kernel's V scan and V pass
    qk_pass.py  rotation, NVFP4 quantization and compensation channels of Q and K
    encode.py   the encode launch: UE4M3 scale factors of Q' and K', score scale
    chain.py    the host side that the chains share: the compensation of a call, compile cache, launch sizes,
                scratch, the call
"""
