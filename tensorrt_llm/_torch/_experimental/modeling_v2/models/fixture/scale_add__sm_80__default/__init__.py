# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixture / sm_80 / default: the native-code fixture's only target.

Loading the package loads its native code: ``native_ops`` is
``torch.ops.mv2_fixture_scale_add_sm_80_default``, built from
``cpp/tensorrt_llm/modeling_v2/fixture/scaleAddSm80Default/``. A target with
native code starts this way, so that ``route`` importing the target is what
loads -- and checks -- its library. (Not ``ops``: that name is the target's
``ops/`` package, which would replace it on import.)
"""

from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

native_ops = load_target_ops(__name__)
