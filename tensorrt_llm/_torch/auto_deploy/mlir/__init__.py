# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""MLIR-based transform infrastructure for AutoDeploy using xDSL.

This module provides an MLIR dialect and pattern-based fusion infrastructure
as an alternative to direct FX graph manipulation. It uses xDSL (a pure-Python
MLIR reimplementation) for dialect definition, pattern matching, and rewriting.

Usage requires ``pip install xdsl``. All imports are gated behind :data:`HAS_XDSL`.
"""

try:
    import xdsl  # noqa: F401
except ImportError:
    HAS_XDSL = False
else:
    # Workaround: xDSL's @irdl_op_definition enforces UPPERCASE ClassVar names,
    # but IRDLOperation's own 'assembly_format' and 'custom_directives' are lowercase.
    # This causes failures in pytest when TRT-LLM's C++ bindings are loaded first.
    # Patch the check to allow these known xDSL-internal fields.
    # Note: patch errors are intentionally not caught so incompatible xDSL versions
    # surface immediately rather than silently disabling MLIR support.
    import xdsl.utils.classvar as _cv

    _KNOWN_XDSL_CLASSVARS = frozenset({"assembly_format", "custom_directives"})
    _original_is_const_classvar = _cv.is_const_classvar

    def _patched_is_const_classvar(field_name, annotation, error_type):
        if field_name in _KNOWN_XDSL_CLASSVARS and _cv.is_classvar(annotation):
            return True
        return _original_is_const_classvar(field_name, annotation, error_type)

    _cv.is_const_classvar = _patched_is_const_classvar

    HAS_XDSL = True

__all__ = ["HAS_XDSL"]
