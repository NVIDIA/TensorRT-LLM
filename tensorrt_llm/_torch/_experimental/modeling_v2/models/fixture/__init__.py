# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary infrastructure fixture for native code: a family with no model behind it.

Its one target, ``scale_add__sm_10x__default``, carries one native op so that CI
drives the whole native path -- the family library's build and packaging, the
loader's checks, the fake op, the GPU op -- on B300. No routing module names it;
tests import the target directly.

Remove the family, its C++ half ``cpp/tensorrt_llm/modeling_v2/fixture/`` and
its tests once a family with native code has GPU unit tests that run in CI.
Until then it is also the smallest complete example of a target with native
code, which ``cpp/tensorrt_llm/modeling_v2/README.md`` walks through.
"""
