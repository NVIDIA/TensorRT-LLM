# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash NVFP4 / sm_10x / TP4: a step program in C++ and CUDA.

``csrc/`` holds the step program (``step.h`` / ``step.cpp``: one forward's decoder layers,
vocabulary-parallel LM head and MTP draft chain over the TensorRT-LLM runtime's pools and
metadata), its CUDA kernels (sm_100a / sm_103a: tcgen05 tensor cores, tensor memory, TMA) and
their torch bindings: the step program's structs are torch classes of the target's namespace,
and the kernels the Python side launches itself are its ops. FlashMLA's SM100 sparse prefill,
which the kernels instantiate, comes from the FlashMLA TensorRT-LLM pins. ``comm`` maps the
collectives' peer memory.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

native_ops = load_target_ops(__name__)  # torch.ops.mv2_glm5_next_glm_5_3_flash_nvfp4_sm_10x_tp4
native_classes = getattr(torch.classes, native_ops.name)
# the kernels' layout constants
layout = native_classes.Layout()
