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
# llvm_bolt_version.sh - the pinned llvm-bolt release, defined once.
#
# Every BOLT step has to run the SAME llvm-bolt. Instrumentation, merge, and
# apply exchange .fdata and .yaml profiles, and those formats are not guaranteed
# stable across releases -- so a version that drifts between steps does not fail
# the build, it silently produces a bad or empty profile. That failure mode is
# why this is a shared file rather than a comment asking five call sites to stay
# in sync.
#
# Sourced by:
#   scripts/bolt/internal/slurm_merge.sh          (merge job, cluster-side)
#   scripts/bolt/internal/perf_instrument_hook.sh (collect job, per node)
#   jenkins/Build.groovy                          (pre-merge consume, build pod)
#   jenkins/L0_Test.groovy                        (release wheel, build pod)
#   jenkins/BoltProfileGen.groovy                 (bootstrap, cluster frontend)
#
# The Groovy callers source this from the checkout they already have rather than
# interpolating a literal, so bumping the pin here is the whole change.
#
# Assignment is conditional so a one-off run can pin a different release from the
# environment without editing the file.
LLVM_BOLT_VERSION="${LLVM_BOLT_VERSION:-21.1.5}"
