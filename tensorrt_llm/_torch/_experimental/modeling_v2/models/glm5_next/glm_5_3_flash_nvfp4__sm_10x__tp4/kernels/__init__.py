# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash kernel family of this target.

``csrc/`` holds the step program (``step.h`` / ``step.cpp``: one forward's decoder layers,
vocabulary-parallel LM head and MTP draft chain over the TensorRT-LLM runtime's pools and
metadata), its CUDA kernels (sm_100a / sm_103a: tcgen05 tensor cores, tensor memory, TMA) and
their pybind11 bindings; ``third_party/flashmla`` is the subset of FlashMLA's SM100 sparse prefill
they instantiate (MIT). ``_lib`` compiles them at first use; ``comm`` maps the collectives' peer
memory.
"""
