# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""megakernel: the persistent per-layer decode kernel for Rubin (``msoMegakernel.cu``) and its host code.

* ``schedule``: the host planner (unit lists, shared-memory placement, weight issue points), CPU only.
* ``library``: build / load / launch, the shared workspace, parameter blocks, stamps and timeline dumps.
* ``qkv``: the QKV stage of a decoder layer (weight repack, parameter fields, a stand-alone stage object).
* ``check``: float64 references and error statistics of check mode.
"""
