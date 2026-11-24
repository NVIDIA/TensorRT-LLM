# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTe DSL kernels for Kimi K3 KDA prefill on Blackwell GPUs.

Package-internal copy of the Kimi K3 KDA CuTe DSL kernels that the Rubin K3 runtime uses: the
prefill pipeline (``fused_k123``, ``akk_inverse``, ``k4_persistent``) and the multi-token verify
kernels (``kda_mtp_decode``, ``kda_mtp_decode_v2``, ``kda_mtp_decode_2cta``). The files are
copied unchanged; main's different KDA kernels under ``attention/kernels/blackwell/kimi_k3_kda``
are untouched. Nothing is imported here: the op modules ``kda/prefill_ops.py`` and
``kda/mtp_ops.py`` import these on first use.
"""
