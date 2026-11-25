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
"""sparse_mla_decode: sparse MLA generation attention (the sparse_attention contract).

Entry points registered with ``dispatch``:

* ``fmha_lib``: ``SparseMlaDecodeFmha``, an FMHA library class that the
  attention FMHA registry places ahead of the native libraries when ``sparse_mla_decode`` is
  selected.
* ``sparse_mla_decode``: the functional kernel wrapper.
"""

from ... import dispatch
from .sparse_mla import SparseMlaDecodeFmha, sparse_mla_decode

dispatch.register("sparse_mla_decode", "fmha_lib", SparseMlaDecodeFmha)
dispatch.register("sparse_mla_decode", "sparse_mla_decode", sparse_mla_decode)

__all__ = ["SparseMlaDecodeFmha", "sparse_mla_decode"]
