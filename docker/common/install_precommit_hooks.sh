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

# Pre-populate the pre-commit hook cache at image-build time.
#
# Why: CI runs `pre-commit run type-check` in the test container. pre-commit
# enumerates *every* remote repo in .pre-commit-config.yaml (it clones each one to
# read its hook manifest) before it runs the selected hook, so a single
# `git fetch origin --tags` against github.com can fail the whole stage on test
# nodes that lack reliable github access. Cloning the hooks here at build time
# (where github IS reachable -- see install_mooncake.sh / install_ucx.sh, which
# clone github directly) means the runtime store is warm and does zero network.
#
# The cache lives at $PRE_COMMIT_HOME (set as an ENV in the tritondevel stage so
# the runtime pre-commit reads the same store). Use a FIXED absolute path: hook
# virtualenvs bake absolute shebangs, so build-time and runtime paths must match.

set -ex

CONFIG_FILE="${1:?usage: install_precommit_hooks.sh <path-to-.pre-commit-config.yaml>}"
: "${PRE_COMMIT_HOME:=/opt/pre-commit-cache}"
export PRE_COMMIT_HOME

# pre-commit itself is (re)installed from requirements-dev.txt at test time; install
# it here only to populate the store. Keeping the version aligned with
# requirements-dev.txt avoids a store-format mismatch that would trigger a re-clone.
pip3 install pre-commit

# `pre-commit install-hooks` must run inside a git work tree. Seed a throwaway repo
# containing only the config; install-hooks does not need the target source files,
# only the config to clone each remote repo and build each hook environment.
SEED_DIR="$(mktemp -d)"
trap 'rm -rf "${SEED_DIR}"' EXIT
git init -q "${SEED_DIR}"
cp "${CONFIG_FILE}" "${SEED_DIR}/.pre-commit-config.yaml"
(
    cd "${SEED_DIR}"
    pre-commit install-hooks
)

# Sanity: the store should now hold cloned hook repos.
ls -la "${PRE_COMMIT_HOME}"
test -n "$(find "${PRE_COMMIT_HOME}" -maxdepth 1 -name 'repo*' -print -quit)" \
    || { echo "ERROR: pre-commit store has no cloned repos" >&2; exit 1; }
