#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# Temporary infrastructure fixture, not a model: one target with one op, so
# that CI drives the whole native path -- build, packaging, load,
# built_targets(), fake op, GPU op. Remove this family (with its Python package
# and tests) once a family with native code has GPU unit tests that run in CI.
# It is built for 100f only, the B300 its GPU test runs on, to keep the
# library it adds to every wheel small.
add_modeling_v2_family(fixture)

add_modeling_v2_target(
  fixture scale_add__sm_10x__default
  ARCHS 100f
  SOURCES kernels/scaleAdd.cu ops/scaleAddOp.cpp)
