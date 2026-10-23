# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The native-code fixture: a family with no model behind it.

Its one target, ``scale_add__sm_80__default``, carries one native op so that CI
drives the whole native path -- the family library's build and packaging, the
loader's checks, the fake op, the GPU op -- on every GPU it has. No routing
module names it; tests import the target directly.

It is also the smallest complete example of a target with native code; its C++
half is ``cpp/tensorrt_llm/modeling_v2/fixture/`` and
``cpp/tensorrt_llm/modeling_v2/README.md`` walks through both.
"""
