# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of this target's native ops, built from its csrc/.

The loader imports this module right after the target is found in the family
library, so a fake for an op the library does not define fails the load. The ops
take device addresses as integers and write through them, so they return nothing.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)


@torch.library.register_fake(f"{_NS}::step_meta")
def _(*args):
    return None


@torch.library.register_fake(f"{_NS}::spec_sample_params")
def _(*args):
    return None


@torch.library.register_fake(f"{_NS}::spec_accept")
def _(*args):
    return None


@torch.library.register_fake(f"{_NS}::allreduce_lamport_init")
def _(*args):
    return None
