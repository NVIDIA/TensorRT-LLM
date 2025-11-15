# CPP Dependency Scanner

Scans TensorRT-LLM build artifacts (headers, libraries, binaries) and maps them to source dependencies.
A build artifact is any header file used in the build, and any linked static/dynamic library.

## Quick Start

```bash
# Run scanner (scans ../build by default)
python scan_build_artifacts.py

# Output:
#   scan_output/known.yml         - Mapped artifacts
#   scan_output/unknown.yml       - Unmapped artifacts
#   scan_output/path_issues.yml   - Non-existent paths
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
2. **YAML patterns**: Non-dpkg packages (TensorRT, PyTorch, 3rdparty/ submodules, etc.)

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

### path_issues.yml

Reports artifacts whose resolved paths don't exist in the filesystem. This helps identify:
- Stale build artifacts from old builds
- Incorrectly resolved paths
- Optional headers that may not be present
- Temporary build files that were deleted

**Note:** Library artifacts are excluded from this report since they don't have meaningful path resolution metadata.

```yaml
summary:
  count: 1042
  total_artifacts: 12238
  percentage: 8.5%
  note: These header paths were resolved from .d files but do not exist in the filesystem (libraries excluded)

non_existent_paths:
- resolved_path: /usr/local/lib/python3.12/dist-packages/torch/include/ATen/ops/_cudnn_attention_backward.h
  type: header
  source: /home/.../trtGptModelInflightBatching.cpp.o.d
  d_file_path: /usr/local/lib/python3.12/dist-packages/torch/include/ATen/ops/_cudnn_attention_backward.h
```

**Field Descriptions:**
- `resolved_path`: The final canonicalized absolute path after resolution
- `type`: Artifact type (typically "header")
- `source`: The .d file where this path was found
- `d_file_path`: The original path as it appeared in the .d file (may be relative or absolute)

**Common Causes:**
- **Optional headers**: PyTorch/CUDA headers that don't exist in all installations (e.g., `_cudnn_attention_*`)
- **Old CUDA paths**: References to previous CUDA versions no longer installed (e.g., `cuda-13.0` when only `cuda-12.9` exists)
- **Build artifacts**: Temporary generated files deleted after build completion
- **Stale .d files**: Dependency files from previous builds with different directory structures

**Action:** Review the list and determine if these are expected (optional/temporary) or indicate path resolution issues.

## Iterative Workflow

1. **Run scanner** on build directory
2. **Review outputs**:
   - `scan_output/unknown.yml` - unmapped artifacts requiring pattern additions
   - `scan_output/path_issues.yml` - non-existent paths (may indicate stale builds or optional dependencies)
3. **Add patterns** to `metadata/*.yml` files for unknown artifacts
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

**High percentage in path_issues.yml**
- If >20%, likely indicates stale build artifacts - run a clean rebuild
- If <10%, likely optional/temporary headers - expected behavior
- Check for references to uninstalled CUDA versions

**Slow performance**
- Use `--build-dir` to target specific subdirectories
- Reduce build artifacts scope

## Architecture

```
scan_build_artifacts.py (1,300 lines)
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

3. **CMake .d File Path Resolution** (_parse_d_file method, lines 356-364)
   - **Critical Fix (October 2025)**: Changed context directory for path resolution
   - CMake generates .d files with paths relative to the **target's build directory** (where the Makefile for that target is located), **NOT** the top-level build directory
   - **Context Directory**: Parent directory of `CMakeFiles/` (e.g., `/build/tensorrt_llm/batch_manager/`)
   - **Example**: For .d file at `/build/tensorrt_llm/batch_manager/CMakeFiles/target.dir/file.cpp.o.d`:
     - **Context is**: `/build/tensorrt_llm/batch_manager/` (parent of CMakeFiles)
     - **NOT**: `/build/` (top-level build directory)
     - Relative path `../../../tensorrt_llm/...` resolves correctly from this context
   - **Before Fix**: Used `d_file.parent` (adjacent to CMakeFiles directory) - caused 49.9% path resolution errors
   - **After Fix**: Uses parent of CMakeFiles directory - reduced errors to 7.2%
   - **Path Existence Tracking**: Scanner marks each artifact with `path_exists` metadata and reports non-existent paths in `path_issues.yml`

   **Algorithm:**
   ```python
   d_file_parts = d_file.parts
   if 'CMakeFiles' in d_file_parts:
       cmake_idx = d_file_parts.index('CMakeFiles')
       context_dir = Path(*d_file_parts[:cmake_idx])  # Parent of CMakeFiles
   else:
       context_dir = self.build_dir  # Fallback
   ```

4. **3rdparty Submodule Resolution** (_parse_d_file method)
   - When D files contain relative paths with submodule directories that don't exist relative to the build directory, the scanner attempts to resolve them from the configured submodules directory
   - **Configuration**: Set via `THIRDPARTY_ROOT` constant in scan_build_artifacts.py
   - **Default**: `TRTLLM_ROOT/3rdparty` (3 levels up from scanner location)
   - **Customization**: Edit the `THIRDPARTY_ROOT` constant if dependencies move (e.g., to `${CMAKE_BINARY_DIR}/_deps/`)
   - **Example**: `../../../../3rdparty/xgrammar/include/file.h` resolves to `{THIRDPARTY_ROOT}/xgrammar/include/file.h`

**Resolution Flow:**
1. Collect artifacts from build directory
2. Try dpkg-query resolution (PRIMARY)
3. Fall back to YAML patterns (FALLBACK)
4. Generate known.yml, unknown.yml, and path_issues.yml reports

## Files

- `scan_build_artifacts.py` - Main scanner script
- `metadata/*.yml` - Dependency patterns
- `metadata/_template.yml` - Template for new dependencies
- `metadata/_schema.yml` - YAML validation schema
- `metadata/README.md` - Pattern documentation
- `tests/test_scan_build_artifacts.py` - Unit tests
