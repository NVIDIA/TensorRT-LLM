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
"""Names of the GLM-MoE-DSA torch ops and the layout of their C++ sources.

The model-specific kernels live in ``cpp/tensorrt_llm/kernels/glmMoeDsa/`` and
their torch ops in ``cpp/tensorrt_llm/thop/glmMoeDsa/``, one ``<stem>Op.cpp``
per kernel ``<stem>.cu``; the kernels build as one OBJECT library for the
SM100 family. Every op they register is
``trtllm::glm_moe_dsa_<what>``, and every such name the Python sources use is
registered there or by a ``torch.library.custom_op`` of the Python package
under a name the C++ sources leave free. The sources are read as text, so
the test needs neither torch nor the compiled extensions.
"""

import re
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[8]
_CPP = _ROOT / "cpp" / "tensorrt_llm"
_KERNELS = _CPP / "kernels" / "glmMoeDsa"
_THOP = _CPP / "thop" / "glmMoeDsa"
_PYTHON = _ROOT / "tensorrt_llm"

OP_PREFIX = "glm_moe_dsa_"

_REGISTRATION = re.compile(r"\bm\.(def|impl)\(\s*(\"([A-Za-z0-9_.]+)|[A-Z][A-Z0-9_]*)")
_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Z][A-Z0-9_]*)\s+\"([A-Za-z0-9_.]+)\"", re.M)
_PYTHON_OP = re.compile(
    r"\btorch\.library\.custom_op\(\s*[\"']trtllm::(%s[a-z0-9_]+)[\"']" % OP_PREFIX
)


def _registered(path):
    """(kind, op name) of every m.def / m.impl in ``path``.

    A name given by a macro is resolved through the file's ``#define``s (every
    definition counts); a macro without a string definition yields ``None``.
    """
    text = path.read_text(errors="replace")
    macros = {}
    for name, value in _DEFINE.findall(text):
        macros.setdefault(name, []).append(value)
    found = []
    for match in _REGISTRATION.finditer(text):
        kind = match.group(1)
        if match.group(3) is not None:
            found.append((kind, match.group(3)))
        else:
            found += [(kind, value) for value in macros.get(match.group(2), [None])]
    return found


def _thop_ops():
    ops = {}
    for path in sorted(_THOP.glob("*.cpp")):
        for kind, name in _registered(path):
            assert name is not None, f"{path.name}: m.{kind} names an op through an unknown macro"
            ops.setdefault(name, set()).add(kind)
    return ops


def _python_ops():
    """Every ``trtllm::glm_moe_dsa_*`` op a ``torch.library.custom_op`` of the package defines."""
    ops = set()
    for path in sorted(_PYTHON.rglob("*.py")):
        ops.update(_PYTHON_OP.findall(path.read_text(errors="replace")))
    return ops


def test_sources_sit_in_the_glm_moe_dsa_folders():
    kernels = sorted(p.stem for p in _KERNELS.glob("*.cu"))
    headers = sorted(p.stem for p in _KERNELS.glob("*.h"))
    ops = sorted(p.stem[: -len("Op")] for p in _THOP.glob("*Op.cpp"))
    assert kernels, "no kernel sources under kernels/glmMoeDsa"
    assert kernels == headers
    assert kernels == ops
    assert sorted(p.name for p in _THOP.glob("*.cpp")) == sorted(f"{s}Op.cpp" for s in ops)
    cmake = (_CPP / "thop" / "CMakeLists.txt").read_text()
    assert "add_th_common_family(th_common_glm_moe_dsa)" in cmake
    assert "add_subdirectory(glmMoeDsa)" in cmake
    assert "$<TARGET_OBJECTS:th_common_glm_moe_dsa>" in cmake
    assert "th_common_glm_moe_dsa" in (_THOP / "CMakeLists.txt").read_text()


def test_kernels_build_as_one_sm100_family_object_library():
    cmake = (_KERNELS / "CMakeLists.txt").read_text()
    listed = re.search(r"add_library\(\s*glm_moe_dsa_kernels_src\s+OBJECT\s+([^)]*)\)", cmake)
    assert listed is not None
    assert sorted(listed.group(1).split()) == sorted(p.name for p in _KERNELS.glob("*.cu"))
    assert "set_cuda_architectures(glm_moe_dsa_kernels_src 100f)" in cmake
    # The parent's globs leave the folder to its own target, which joins the
    # shared library's link line.
    parent = (_CPP / "kernels" / "CMakeLists.txt").read_text()
    assert "add_subdirectory(glmMoeDsa)" in parent
    assert 'list(FILTER SRC_CPP EXCLUDE REGEX "glmMoeDsa/.*")' in parent
    assert 'list(FILTER SRC_CU EXCLUDE REGEX "glmMoeDsa/.*")' in parent
    assert re.search(r"^\s*glm_moe_dsa_kernels_src$", (_CPP / "CMakeLists.txt").read_text(), re.M)


def test_every_op_defined_in_thop_glm_moe_dsa_has_the_model_prefix():
    ops = _thop_ops()
    assert ops, "no op registrations under thop/glmMoeDsa"
    assert sorted(n for n in ops if not n.startswith(OP_PREFIX)) == []
    # Every implemented op is also defined (the schema lives next to the kernel).
    assert sorted(n for n, kinds in ops.items() if "def" not in kinds) == []


def test_python_sources_use_only_registered_glm_moe_dsa_ops():
    used = re.compile(r"\b%s[a-z0-9_]+\b" % OP_PREFIX)
    thop, python = set(_thop_ops()), _python_ops()
    assert sorted(thop & python) == []
    registered = thop | python
    unknown = []
    for path in sorted(_PYTHON.rglob("*.py")):
        text = path.read_text(errors="replace")
        rel = str(path.relative_to(_ROOT))
        unknown += [(rel, m.group(0)) for m in used.finditer(text) if m.group(0) not in registered]
    assert unknown == []
