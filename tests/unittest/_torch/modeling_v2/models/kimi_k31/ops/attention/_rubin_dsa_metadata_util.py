# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family DSA metadata tests' step objects."""

import inspect


def bare(cls, **attrs):
    """A ``cls`` metadata object without the engine's constructor. Read-only engine properties
    among ``attrs`` (step counts, ``seq_lens_cuda``, ...) become plain attributes of a subclass."""
    props = {
        k: v for k, v in attrs.items() if isinstance(inspect.getattr_static(cls, k, None), property)
    }
    if props:
        cls = type(f"_Step{cls.__name__}", (cls,), props)
    obj = object.__new__(cls)
    for key, value in attrs.items():
        if key not in props:
            setattr(obj, key, value)
    return obj
