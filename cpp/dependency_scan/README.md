# CPP Dependency Scanner

Scans TensorRT-LLM build artifacts (headers, libraries, binaries) and maps them to source dependencies.
A build artifact is any header file used in the build, and any linked static/dynamic library.

## Quick Start

```bash
# Run scanner (scans ../build by default)
python scan_build_artifacts.py

# Output: scan_output/known.yml, scan_output/unknown.yml
```

## Goals and Non-Goals

### Goals

This scanner is designed to:

1. **Map Build Artifacts to Dependencies**
   - Identify which source dependencies (container-origin, fetched, third-party) are used in the TensorRT-LLM C++ build
   - Use tools + developer-provided pattern data to map build artifacts to canonical packages.

2. **Achieve Complete Coverage**
   - Goal: 100% of build artifacts mapped to known dependencies.
   - Track unmapped artifacts in `unknown.yml` for iterative pattern refinement

4. **Enable Iterative Development**
   - Provide actionable output (`unknown.yml`) to guide pattern additions
   - Support YAML-based pattern definitions for easy maintenance
   - Validate patterns with schema checking

### Non-Goals

This scanner is **NOT** designed to:

1. Identify any source-integrated dependencies. A source-integrated dependency is any copy-pasted code directly from a third-party repository to the TensorRT-LLM codebase.
2. Identify pip-installed python runtime dependencies.
3. Be a one-size-fits-all solution catching all dependencies.
4. Enrich with license information for each dependency - or generate attributions.
5. Track transitive dependencies that are invisible to cmake.
6. provide Windows support.

## Usage

### Basic Usage

```bash
# Scan with default settings
python scan_build_artifacts.py

# Scan custom build directory
python scan_build_artifacts.py --build-dir /path/to/build

# Scan with custom output directory
python scan_build_artifacts.py --output-dir /path/to/output

# Validate YAML files
python scan_build_artifacts.py --validate
```

### Command-Line Arguments

- `--build-dir`: Build directory to scan (default: `../build/`)
- `--output-dir`: Output directory for reports (default: `scan_output/`)
- `--metadata-dir`: Metadata directory containing YAML files (default: `./metadata/`)
- `--validate`: Validate YAML files without running scanner

## Resolution Strategy

1. **dpkg-query**: System packages via Debian package manager
2. **YAML patterns**: Non-dpkg packages (CUDA, TensorRT, PyTorch, etc.)

## Output Format

### known.yml

```yaml
summary:
  total_artifacts: 6200
  mapped: 6200
  unmapped: 0
  coverage: 100.0%
  unique_dependencies: 48

dependencies:
  cuda-cudart:
    - /usr/local/cuda-12.9/include/cuda_runtime.h
    - /usr/local/cuda-12.9/include/cuda.h

  libc6:
    - /usr/include/stdio.h
    - -lpthread
    - -ldl

  pytorch:
    - /usr/local/lib/python3.12/dist-packages/torch/include/torch/torch.h
    - -ltorch
```

### unknown.yml

```yaml
summary:
  count: 0
  action_required: Add patterns to YAML files in metadata/ for these artifacts
artifacts: []
```

## Iterative Workflow

1. **Run scanner** on build directory
2. **Review** `scan_output/unknown.yml` for unmapped artifacts
3. **Add patterns** to `metadata/*.yml` files
4. **Re-run** to verify improved coverage
5. **Repeat** until all artifacts mapped

## Pattern Matching

### Strategy Priority (High → Low)

1. **Exact match**: `libcudart.so.12` → `cuda-cudart`
2. **Path alias**: `/build/pytorch/include/` → `pytorch`
3. **Generic inference**: `libfoobar.so` → `foobar`

### Adding Patterns

Edit existing or create new YAML file in `metadata/`:

```yaml
name: cutlass
description: CUDA Templates for Linear Algebra Subroutines

basename_matches:
  - libcutlass.a

linker_flags_matches:
  - -lcutlass

directory_matches:
  - cutlass              # Single: matches any /cutlass/ in path
  - 3rdparty/cutlass     # Multi: matches /3rdparty/cutlass/ sequence
```

#### Multi-Directory Patterns

Directory patterns support both single and multi-directory matching:

**Single Component:**
- `"pytorch"` matches any path containing `/pytorch/`
- Example: `/home/build/pytorch/include/torch.h` ✓

**Multi-Directory:**
- `"3rdparty/cutlass"` matches consecutive `/3rdparty/cutlass/` sequence
- `"foo/bar"` matches `/home/foo/bar/file.h` ✓
- `"foo/bar"` does NOT match `/home/foobar/file.h` ✗ (no substring matching)

**Matching Rules:**
- Exact component matching only (no substrings)
- `"oo/ba"` will NOT match `/foo/bar/`
- Rightmost match wins if pattern appears multiple times
- Leading/trailing slashes are ignored (`"/foo/bar/"` = `"foo/bar"`)

See `metadata/_template.yml` and `metadata/README.md` for details.

## YAML Dependencies

Each dependency file contains:

```yaml
name: pytorch
description: PyTorch machine learning framework
license: BSD-3-Clause
copyright: Copyright (c) PyTorch Contributors
homepage: https://pytorch.org/
source: container

basename_matches:
  - libtorch.so
  - libc10.so

linker_flags_matches:
  - -ltorch_python

directory_matches:
  - ATen
  - c10
  - caffe2
  - torch

aliases:
  - torch
```

Multiple dependencies can be grouped in list format (see `metadata/base.yml`, `metadata/cuda.yml`).

## Testing

```bash
cd tests
python -m pytest test_scan_build_artifacts.py -v
# Expected: 40 passed
```

## Troubleshooting

**Low dpkg coverage**
- Running on non-Debian system
- YAML dependencies will handle more as fallback, with concrete patterns.

**Many unknown artifacts**
1. Review `scan_output/unknown.yml`
2. Add patterns to `metadata/*.yml`
3. Run `--validate` to check syntax
4. Re-scan to verify

**Wrong mappings**
- Check pattern priorities in YAML files
- More specific patterns should be listed first
- Make sure the patterns are very specific, to avoid false positives, or interfering with other patterns.

**Slow performance**
- Use `--build-dir` to target specific subdirectories
- Reduce build artifacts scope

## Architecture

```
scan_build_artifacts.py (1,000 lines)
├── DpkgResolver - dpkg-query for system packages
├── ArtifactCollector - Parse D files, link files, wheels
├── PatternMatcher - 3-tier YAML pattern matching
└── OutputGenerator - Generate YAML reports
```

**Artifact Sources:**
- D files: CMake dependency files with headers. Dependency source header files.
- link.txt: Linker commands with libraries. Precompiled dependency artifacts.
- Wheels: Python binaries via readelf. Runtime dependency artifacts.

**Special Parsing Behaviors:**

1. **Malformed .d File Handling** (_parse_d_file method)
   - Some CMake-generated .d files contain paths with trailing colons
   - Example: `/usr/include/stdc-predef.h:` (should be `/usr/include/stdc-predef.h`)
   - Parser strips trailing colons to handle these malformed entries
   - Prevents duplicate artifacts and improves accuracy

2. **CMakeFiles Linker Artifact Extraction** (_parse_link_file method)
   - CMake generates special linker artifacts in CMakeFiles directories
   - Pattern: `/path/CMakeFiles/foo.dir/-Wl,-soname,libtest.so.1`
   - Parser extracts library name and converts to linker flag: `-ltest`
   - Enables proper dependency mapping for internal build artifacts

3. **3rdparty Submodule Resolution** (_parse_d_file method)
   - When D files contain relative paths with submodule directories that don't exist relative to the build directory, the scanner attempts to resolve them from the configured submodules directory
   - **Configuration**: Set via `THIRDPARTY_ROOT` constant in scan_build_artifacts.py (line 46)
   - **Default**: `TRTLLM_ROOT/3rdparty` (3 levels up from scanner location)
   - **Customization**: Edit the `THIRDPARTY_ROOT` constant if dependencies move (e.g., to `${CMAKE_BINARY_DIR}/_deps/`)
   - **Example**: `../../../../3rdparty/xgrammar/include/file.h` resolves to `{THIRDPARTY_ROOT}/xgrammar/include/file.h`

**Resolution Flow:**
1. Collect artifacts from build directory
2. Try dpkg-query resolution (PRIMARY)
3. Fall back to YAML patterns (FALLBACK)
4. Generate known.yml and unknown.yml reports

## Files

- `scan_build_artifacts.py` - Main scanner script
- `metadata/*.yml` - Dependency patterns (48 dependencies defined)
- `metadata/_template.yml` - Template for new dependencies
- `metadata/_schema.yml` - YAML validation schema
- `metadata/README.md` - Pattern documentation
- `tests/test_scan_build_artifacts.py` - Unit tests
- `BUG_FIX_SUMMARY.md` - Historical bug fix documentation
