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
"""Attention modules for the PyTorch backend.

Holds the module-layer implementations -- standard attention, cross attention,
Multi-head Latent Attention, QK-norm attention and rotary embedding -- that
were previously mixed into the shared ``_torch/modules`` bucket. The backend
classes those modules dispatch to still live in ``_torch/attention_backend``.

``ATTENTION_DEVELOPER_GUIDE.md`` in this directory is the contract document for
attention metadata, backend families and KV-cache behaviour; read it before
changing anything here.

Nothing is re-exported from this file on purpose. The modules beside it pull in
torch, the attention backends and the distributed layer, and a re-export would
make the cheapest of them pay for the most expensive one at import time.
"""
