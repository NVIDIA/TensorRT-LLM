#!/bin/bash
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

# Idempotently apply a patch to a FetchContent dependency.
#
# Usage: apply_patch.sh <dependency name> <patch file>
# Runs in the dependency's source directory (the PATCH_COMMAND working directory).
#
# The patch step re-runs on reconfigure, so it has to accept a tree that already carries the
# patch. When the tree carries neither the current patch nor a clean base (e.g. the patch file
# changed since it was applied) and the source directory is its own git checkout, the checkout is
# reset to its pinned commit and the patch is applied again.

set -u

name=$1
patch_file=$2

patch_quiet() {
    patch -p1 --batch --dry-run "$@" -i "${patch_file}" > /dev/null 2>&1
}

if patch_quiet --reverse --force; then
    exit 0
fi

if patch_quiet --forward; then
    exec patch -p1 --forward --batch --quiet -i "${patch_file}"
fi

# Only reset when the source directory is the root of its own repository: FetchContent sources
# live inside the build tree, which may itself be inside another git checkout.
toplevel=$(git rev-parse --show-toplevel 2> /dev/null)
if [[ -n "${toplevel}" && "${toplevel}" -ef "${PWD}" ]]; then
    echo "-- ${name}: source tree does not match ${patch_file##*/}; resetting it to $(git rev-parse --short HEAD) and re-applying"
    git reset --quiet --hard HEAD \
        && git clean --quiet -fd \
        && exec patch -p1 --forward --batch --quiet -i "${patch_file}"
fi

echo "Cannot apply ${patch_file} to ${name}: the source tree has conflicting changes." \
    "Remove ${PWD} and reconfigure." >&2
exit 1
