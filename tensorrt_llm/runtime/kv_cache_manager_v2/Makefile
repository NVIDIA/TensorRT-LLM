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
