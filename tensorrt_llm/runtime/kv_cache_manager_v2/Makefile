# ##################################################################################################
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
# ##################################################################################################

# Makefile for kv_cache_manager_v2
# Replaces build_mypyc.sh and rawref/build.sh

PYTHON ?= python3
RUNTIME_DIR ?= ..

.PHONY: all mypyc rawref clean clean_mypyc clean_rawref

# Default target
all: rawref mypyc

# Build the rawref C extension
rawref:
	cd rawref && $(PYTHON) setup.py build_ext --inplace

# Build the mypyc extension
# Must be run from the parent directory (runtime) as setup_mypyc.py expects
# kv_cache_manager_v2/ prefixes in module names.
mypyc:
	cd $(RUNTIME_DIR) && $(PYTHON) kv_cache_manager_v2/setup_mypyc.py build_ext --inplace

# Clean everything
clean: clean_rawref clean_mypyc

# Clean rawref build artifacts
clean_rawref:
	cd rawref && rm -rf build
	find rawref -name "*.so" -type f -delete

# Clean mypyc build artifacts
# Cleans build/ directory in runtime (parent) and .so files in this directory
clean_mypyc:
	rm -rf $(RUNTIME_DIR)/build
	find . -name "*.so" -type f ! -path "./rawref/*" -delete
