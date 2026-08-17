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

# Enable authenticated access to github.com for this shell and everything it
# spawns: the `git clone`s in install_*.sh, `git submodule update` and the CMake
# FetchContent clones of the wheel build, `pip install git+https://github.com/...`
# and so on.
#
# GitHub meters anonymous requests per source IP and a whole CI pool shares very
# few of them, so unauthenticated clones start getting throttled once the
# pipeline is busy; requests carrying a token are metered per account instead.
#
# The token is optional -- without it everything falls back to anonymous access.
# It is read from a BuildKit secret mount (`docker buildx build --secret
# id=github_clone_token,env=GITHUB_CLONE_TOKEN`, wired up in docker/Makefile) or
# straight from GITHUB_CLONE_TOKEN in the environment.
#
# The rewrite is published through GIT_CONFIG_* environment variables on purpose:
# `git config --global` would write the token into $HOME/.gitconfig and bake it
# into the image layer. Env config needs git >= 2.31, which is checked up front
# (see git_env_config_supported below).
#
# Source this file, do not execute it:
#     . /opt/docker/common/github_auth.sh
# Sourcing it more than once (a RUN layer that already set it up calling an
# install script that sources it again) is a no-op -- the first one wins.

# git only reads GIT_CONFIG_COUNT / GIT_CONFIG_KEY_n / GIT_CONFIG_VALUE_n from
# the environment since 2.31 (2021-03). An older git silently ignores them, so
# an intended authenticated clone would quietly degrade to an anonymous one --
# detect that here and say so rather than letting it happen unnoticed.
git_env_config_supported() {
    local version major minor
    command -v git >/dev/null 2>&1 || return 1
    version=$(git --version 2>/dev/null | sed -n 's/^git version \([0-9][0-9.]*\).*/\1/p')
    [ -n "$version" ] || return 1
    major="${version%%.*}"
    minor="${version#*.}"
    minor="${minor%%.*}"
    case "$major$minor" in *[!0-9]*|"") return 1 ;; esac
    if [ "$major" -gt 2 ]; then
        return 0
    fi
    if [ "$major" -eq 2 ] && [ "$minor" -ge 31 ]; then
        return 0
    fi
    return 1
}

setup_github_auth() {
    # Keep the token out of the build log even under `set -x`.
    local restore_x=""
    case "$-" in *x*) restore_x=1 ;; esac
    set +x

    if [ -n "${GITHUB_AUTH_CONFIGURED:-}" ]; then
        if [ -n "$restore_x" ]; then set -x; fi
        return 0
    fi
    export GITHUB_AUTH_CONFIGURED=1

    if ! git_env_config_supported; then
        local git_desc
        git_desc=$(git --version 2>/dev/null) || git_desc="git not found"
        echo "[github_auth] ${git_desc}: no GIT_CONFIG_* environment support" \
             "(needs git >= 2.31), using anonymous github.com access."
        if [ -n "$restore_x" ]; then set -x; fi
        return 0
    fi

    local token="${GITHUB_CLONE_TOKEN:-}"
    local secret_file="${GITHUB_CLONE_TOKEN_FILE:-/run/secrets/github_clone_token}"
    if [ -z "$token" ] && [ -r "$secret_file" ]; then
        token=$(cat "$secret_file")
    fi

    if [ -n "$token" ]; then
        # Append rather than overwrite, in case the caller already carries its own
        # GIT_CONFIG_* entries. Per-repo mirrors (GITHUB_MIRROR, the GitLab mirrors
        # configured on CI agents) keep winning over this catch-all rewrite: git
        # resolves insteadOf by longest matching prefix.
        local idx="${GIT_CONFIG_COUNT:-0}"
        export "GIT_CONFIG_KEY_${idx}=url.https://x-access-token:${token}@github.com/.insteadOf"
        export "GIT_CONFIG_VALUE_${idx}=https://github.com/"
        export GIT_CONFIG_COUNT=$((idx + 1))
        echo "[github_auth] Using authenticated github.com access."
    else
        echo "[github_auth] No GitHub token available, using anonymous github.com access."
    fi

    if [ -n "$restore_x" ]; then set -x; fi
    return 0
}

setup_github_auth
