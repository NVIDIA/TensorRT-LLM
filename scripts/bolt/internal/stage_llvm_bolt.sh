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
#
# stage_llvm_bolt.sh - put the pinned llvm-bolt on PATH, downloading it if needed.
#
# No-op when llvm-bolt is already available, so callers can invoke it
# unconditionally. Prints the bin directory it staged (nothing when the toolchain
# was already present), and exports PATH for anything sourcing it.
#
# Usage:
#   . scripts/bolt/internal/stage_llvm_bolt.sh          # sourced: updates PATH
#   bash scripts/bolt/internal/stage_llvm_bolt.sh       # executed: prints bin dir
#
# Env:
#   BOLT_LLVM_STAGE_DIR  where to unpack (default ./.bolt-llvm)
#   GITHUB_MIRROR        mirror base in place of https://github.com, matching
#                        docker/common/install_ccache.sh and install_cmake.sh so
#                        this works inside an image build
#   LLVM_BOLT_VERSION    overrides the pin in llvm_bolt_version.sh

_bolt_stage_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_bolt_stage_llvm() {
    command -v llvm-bolt >/dev/null 2>&1 && return 0

    local dir="${BOLT_LLVM_STAGE_DIR:-$PWD/.bolt-llvm}"
    if [ -x "$dir/bin/llvm-bolt" ]; then
        export PATH="$dir/bin:$PATH"
        return 0
    fi

    . "$_bolt_stage_here/llvm_bolt_version.sh"

    local arch
    case "$(uname -m)" in
        aarch64) arch=ARM64 ;;
        x86_64)  arch=X64 ;;
        *) echo "[stage-llvm-bolt] unsupported arch $(uname -m)" >&2; return 1 ;;
    esac

    local base="${GITHUB_MIRROR:-https://github.com}"
    local tb="LLVM-${LLVM_BOLT_VERSION}-Linux-${arch}.tar.xz"
    local url="${base}/llvm/llvm-project/releases/download/llvmorg-${LLVM_BOLT_VERSION}/${tb}"

    # Extract into a staging dir and rename, so a concurrent or interrupted run
    # can never leave a half-populated toolchain that the -x check above would
    # then accept.
    local stage="${dir}.stage.$$"
    rm -rf "$stage"; mkdir -p "$stage"
    echo "[stage-llvm-bolt] staging llvm-bolt ${LLVM_BOLT_VERSION} -> $dir" >&2
    if ! curl -fSL --retry 10 --retry-all-errors --retry-delay 15 --connect-timeout 60 \
            -o "/tmp/${tb}.$$" "$url"; then
        rm -rf "$stage"; rm -f "/tmp/${tb}.$$"
        echo "[stage-llvm-bolt] download failed: $url" >&2
        return 1
    fi
    tar -xJf "/tmp/${tb}.$$" -C "$stage" --strip-components=1
    rm -f "/tmp/${tb}.$$"
    mv -T "$stage" "$dir" 2>/dev/null || rm -rf "$stage"

    [ -x "$dir/bin/llvm-bolt" ] || { echo "[stage-llvm-bolt] no llvm-bolt in $dir" >&2; return 1; }
    export PATH="$dir/bin:$PATH"
    echo "$dir/bin"
}

_bolt_stage_llvm
