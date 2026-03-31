#!/usr/bin/env bash
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

# Wrapper for mypy invoked by the pre-commit type-check hook.
#
# When the compiled TensorRT-LLM bindings are present (bindings.*.so),
# performs a full type check including automatic type-stub installation.
# Otherwise, runs a lightweight check that tolerates missing compiled
# modules so that developers can type-check without building the wheel.
#
# Set MYPY_REQUIRE_BINDINGS=1 to fail when compiled bindings are missing
# (used by build_wheel.py to enforce the full check after compilation).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if python3 -c 'import tensorrt_llm.bindings'; then
    echo "Compiled bindings detected — running full mypy type check"
    exec mypy "$@"
else
    if [[ "${MYPY_REQUIRE_BINDINGS:-0}" -eq 1 ]]; then
        echo "ERROR: MYPY_REQUIRE_BINDINGS is set but no compiled bindings found" >&2
        exit 1
    fi
    echo "No compiled bindings — running lightweight mypy type check"
    # Without installed dependencies and/or compiled bindings every corresponding type
    # resolves to Any. The flags below suppress the strict-mode checks that cascade
    # from this:
    #   --ignore-missing-imports     → silences import errors for the .so modules
    #   --no-warn-return-any         → [no-any-return] on functions returning Any
    #   --no-warn-unused-ignores     → [unused-ignore] on "# type: ignore"
    #                                  comments that become unnecessary
    #   --allow-untyped-decorators   → [misc] on decorators whose type is Any
    #   --allow-subclassing-any      → [misc] on classes inheriting from Any
    exec mypy \
        --ignore-missing-imports \
        --no-warn-return-any \
        --no-warn-unused-ignores \
        --allow-untyped-decorators \
        --allow-subclassing-any \
        --no-install-types \
        --interactive \
        "$@"
fi
