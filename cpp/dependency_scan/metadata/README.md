# Dependency Patterns

This directory contains pattern definitions for dependency detection in the TensorRT-LLM C++ dependency scanner.

## Quick Start

After running the scanner, check `scan_output/unknown.yml` for unmapped artifacts, then add patterns here.

## Structure

Each `.yml` file represents one or more dependencies:

```
metadata/
├── _template.yml       # Template for new dependencies
├── _schema.yml         # JSON schema for validation
├── base.yml           # Base system packages (list format)
├── cuda.yml           # CUDA packages (list format)
├── tensorrt-llm.yml   # Individual dependency
├── pytorch.yml
└── ... (23 total files)
```

## File Formats

### Individual Dependency File

Most dependencies use this format:

```yaml
# metadata/pytorch.yml

name: pytorch                      # Required: canonical name
description: PyTorch machine learning framework  # Required: min 10 chars

license: BSD-3-Clause              # Optional: SPDX identifier
copyright: Copyright (c) PyTorch Contributors    # Optional
homepage: https://pytorch.org/     # Optional: URL
source: container                  # Optional: how obtained (container, submodule, fetched)

basename_matches:                  # Exact basename matches
  - libtorch.so
  - libc10.so

linker_flags_matches:              # Linker flags (-l flags)
  - -ltorch_python

directory_matches:                 # Directory path patterns
  - ATen
  - c10
  - torch
```

### List Format (base.yml, cuda.yml)

System packages use a list format for compactness:

```yaml
# metadata/base.yml or cuda.yml

dependencies:
  - name: libc6
    description: GNU C Library
    source: container
    basename_matches:
      - libc.so.6
    linker_flags_matches:
      - -lc
      - -lpthread
      - -ldl
    directory_matches: []

  - name: libstdc++6
    description: GNU C++ Library
    source: container
    basename_matches:
      - libstdc++.so.6
    linker_flags_matches:
      - -lstdc++
    directory_matches: []
  # ... more system libraries
```

## Field Names Reference

**Current field names** (as of latest schema):
- `basename_matches` - Exact filename matches (not "patterns")
- `linker_flags_matches` - Linker flags (not "linker_flags")
- `directory_matches` - Path component patterns (not "path_components")

## Iterative Pattern Development

This section describes the recommended workflow for achieving high coverage through iterative pattern refinement.

### Workflow Steps

1. **Run the scanner** on your build directory:
   ```bash
   python scan_build_artifacts.py --build-dir /path/to/build
   ```

2. **Examine scan_output/unknown.yml** to identify unmapped artifacts:
   ```bash
   cat scan_output/unknown.yml
   ```

   Example output:
   ```yaml
   summary:
     count: 42
     action_required: Add patterns to YAML files in metadata/ for these artifacts

   artifacts:
     - /build/3rdparty/newlib/include/foo.h
     - /usr/local/cuda-13.0/include/cuda.h
     - libfoo.so
     - -lbar
   ```

3. **Analyze patterns** in unknown artifacts:
   - Group artifacts by logical dependency
   - Identify common directory paths
   - Note exact library names and linker flags
   - Look for version patterns (e.g., cuda-12.9, cuda-13.0)

4. **Add or update patterns** in metadata YAML files:
   - For new dependencies: Copy `_template.yml` and create new file
   - For existing dependencies: Update relevant YAML file
   - Use the most powerful matching strategy (see below)

5. **Validate your changes**:
   ```bash
   python scan_build_artifacts.py --validate
   ```

6. **Re-run scanner** to verify improvements:
   ```bash
   python scan_build_artifacts.py
   ```

7. **Check results**:
   ```bash
   # Check summary in scan_output/known.yml
   grep "coverage:" scan_output/known.yml

   # Check remaining unknowns
   grep "count:" scan_output/unknown.yml
   ```

8. **Repeat** steps 2-7 until `scan_output/unknown.yml` shows `count: 0`

### Achieving 100% Coverage

The goal is to reduce unknown artifacts to zero. Key strategies:

- **Start with directory_matches**: Most powerful pattern type (see below)
- **Use version-agnostic patterns**: Match across multiple versions (see next section)
- **Group related artifacts**: Single dependency file can match headers, libs, and linker flags
- **Test incrementally**: Add patterns for one dependency at a time
- **Validate frequently**: Catch syntax errors early with `--validate`

## Version-Agnostic Pattern Matching

For dependencies with multiple versions (e.g., CUDA 12.9, 13.0), use patterns that match all versions.

### Problem

Artifacts from different CUDA versions:
```
/usr/local/cuda-12.9/include/cuda.h
/usr/local/cuda-13.0/include/cuda.h
/usr/local/cuda/include/cuda.h
```

### Solution: Version-Agnostic Patterns

Use `directory_matches` with version-agnostic patterns:

```yaml
# metadata/cuda.yml
name: cuda-cudart
description: CUDA Runtime Library

directory_matches:
  - cuda-12.9      # Matches /cuda-12.9/ paths
  - cuda-13.0      # Matches /cuda-13.0/ paths
  - cuda           # Matches /cuda/ paths (generic fallback)
```

### When to Use This Approach

- **Multiple versions installed**: Different CUDA/TensorRT versions in same environment
- **Version symlinks**: Generic paths like `/usr/local/cuda/` alongside versioned ones
- **Forward compatibility**: Pattern works for future versions without updates
- **Container evolution**: Handles version changes between container builds

### Best Practices

1. **List specific versions first**: More specific patterns take priority
   ```yaml
   directory_matches:
     - cuda-12.9    # Specific version
     - cuda-13.0    # Specific version
     - cuda         # Generic fallback
   ```

2. **Use with basename_matches**: Combine with exact filename matching
   ```yaml
   basename_matches:
     - libcudart.so.12
     - libcudart.so.13

   directory_matches:
     - cuda-12.9
     - cuda-13.0
     - cuda
   ```

3. **Test across versions**: Verify patterns work with different installations

4. **Document version ranges**: Add comments for clarity
   ```yaml
   directory_matches:
     - cuda-12.9    # CUDA 12.9.x
     - cuda-13.0    # CUDA 13.0.x
     - cuda         # Generic (all versions)
   ```

## Adding Patterns

### When You See Unknown Artifacts

After running the scanner, check `scan_output/unknown.yml`:

```yaml
summary:
  count: 2
  action_required: Add patterns to YAML files in metadata/ for these artifacts

artifacts:
  - /build/3rdparty/newlib/include/foo.h
  - libfoo.so
```

### Option A: Add to Existing Dependency

If `libfoo.so` belongs to an existing dependency (e.g., `pytorch`):

1. Open `metadata/pytorch.yml`
2. Add to the `basename_matches` list:
   ```yaml
   basename_matches:
     - libtorch.so
     - libfoo.so      # ← Add here
   ```
3. Re-run scanner:
   ```bash
   python ../scan_build_artifacts.py
   ```

### Option B: Create New Dependency

If this is a new dependency:

1. Copy the template:
   ```bash
   cd metadata/
   cp _template.yml foo-library.yml
   ```

2. Edit the file:
   ```yaml
   name: foo-library
   description: Foo library for data processing
   source: submodule

   basename_matches:
     - libfoo.so
     - libfoo.a

   linker_flags_matches:
     - -lfoo

   directory_matches:
     - foo-library
   ```

3. Validate and re-run:
   ```bash
   python ../scan_build_artifacts.py --validate
   python ../scan_build_artifacts.py
   ```

## Pattern Matching Behavior

The scanner uses a **3-tier matching strategy**:

### 1. Exact Pattern Matching (HIGH confidence)
Matches exact filenames or linker flags:

**Basename matches:**
```yaml
basename_matches:
  - libcudart.so.12      # Matches only "libcudart.so.12" exactly
  - libcudart.so.12.0    # Matches only "libcudart.so.12.0" exactly
```

**Linker flags:**
```yaml
linker_flags_matches:
  - -lpthread    # Matches "-lpthread" in link.txt
  - -lcudart     # Matches "-lcudart"
```

### 2. Path Alias Matching (MEDIUM confidence)
Matches directory components in paths. **Now supports multi-directory patterns!**

**Single component:**
```yaml
directory_matches:
  - pytorch      # Matches any path containing /pytorch/
                 # Example: /build/pytorch/include/torch.h ✓
```

**Multi-directory (NEW):**
```yaml
directory_matches:
  - 3rdparty/cutlass           # Matches /3rdparty/cutlass/ sequence
  - external/NVIDIA/cutlass    # Matches full /external/NVIDIA/cutlass/ sequence
```

**Matching rules:**
- Exact component match only (no substring matching)
- `"foo/bar"` matches `/home/foo/bar/file.h` ✓
- `"foo/bar"` does NOT match `/home/foobar/file.h` ✗
- `"oo/ba"` does NOT match `/foo/bar/file.h` ✗
- Rightmost match wins if pattern appears multiple times in path

### 3. Generic Inference (LOW confidence)
Fallback: extracts library name from `-lfoo` → `foo`

### Pattern Matching Power Ranking

**Most Powerful → Least Powerful:**

1. **directory_matches** - Matches entire directories of headers/files
   - Example: `directory_matches: [pytorch]` matches 4,822+ PyTorch headers
   - Single pattern can cover hundreds or thousands of artifacts

2. **basename_matches** - Matches specific library files
   - Example: `basename_matches: [libtorch.so]` matches one library
   - Good for targeting specific libraries

3. **linker_flags_matches** - Matches linker flags in link.txt files
   - Example: `linker_flags_matches: [-ltorch]` matches one linker flag
   - Useful for libraries without headers in build

**Recommendation:** Start with `directory_matches` for maximum coverage with minimal patterns.

## Required Fields

Every dependency MUST have:

```yaml
name: my-dep        # Required: lowercase, hyphenated, + allowed
description: "..."  # Required: minimum 10 characters
```

At least one pattern section is required:
- `basename_matches` (exact filenames)
- `linker_flags_matches` (-l flags)
- `directory_matches` (path components)

## Optional Fields

Recommended for attribution/licensing:

```yaml
version: "1.0"                           # Optional: version string
license: "Apache-2.0"                    # Optional: SPDX identifier
copyright: "Copyright 2024 NVIDIA"       # Optional: copyright notice
homepage: "https://example.com"          # Optional: project URL
source: "submodule"                      # Optional: how obtained
```

Valid `source` values:
- `submodule` - Git submodules in 3rdparty/ directory
- `container` - Pre-installed in container image (e.g., PyTorch, CUDA)
- `fetched` - Downloaded from URL and built from source

## Multi-Directory Pattern Examples

### Example 1: Vendor Directory Boundaries

```yaml
# metadata/cutlass.yml
name: cutlass
description: CUDA Templates for Linear Algebra Subroutines
source: submodule

directory_matches:
  - cutlass              # Single: matches any /cutlass/ in path
  - 3rdparty/cutlass     # Multi: matches /3rdparty/cutlass/ sequence
  - external/NVIDIA/cutlass  # Multi: matches full sequence
```

**Why multi-directory?** Prevents false positives:
- `"cutlass"` alone might match `/other-project/cutlass/` (unwanted)
- `"3rdparty/cutlass"` is more specific and safer

### Example 2: Nested Dependencies

```yaml
# metadata/dlpack.yml
name: dlpack
description: Deep Learning Pack
source: submodule

directory_matches:
  - 3rdparty/xgrammar/3rdparty/dlpack  # Nested submodule path
```

Matches `/build/3rdparty/xgrammar/3rdparty/dlpack/include/dlpack.h`

## Finding Which File to Edit

Search by library name:

```bash
cd metadata/
grep -r "libtorch.so" .
# Output: ./pytorch.yml:  - libtorch.so
```

Search by dependency name:

```bash
grep "^name: pytorch" *.yml
# Output: pytorch.yml:name: pytorch
```

List all dependencies:

```bash
grep "^name:" *.yml | sort
```

Search in list format files (base.yml, cuda.yml):

```bash
grep -A 5 "name: libc6" base.yml
```

## Validation

### Manual Validation

After adding patterns, validate the YAML structure:

```bash
python ../scan_build_artifacts.py --validate
```

Expected output:

```
================================================================================
YAML Validation
================================================================================

✓ base.yml:libc6
✓ base.yml:libstdc++6
✓ cuda.yml:cuda-cudart-dev
✓ pytorch.yml
✓ tensorrt-llm.yml
...

================================================================================
Results: 25/25 valid, 0/25 invalid
================================================================================
```

### Re-run Scanner

After adding patterns, re-run the scanner:

```bash
python ../scan_build_artifacts.py
```

Check `scan_output/unknown.yml` - should have fewer (or zero) artifacts:

```yaml
summary:
  count: 0  # Improved from previous run!
  coverage: 100.0%

artifacts: []
```

### Schema Validation

The `_schema.yml` file defines validation rules:
- Required fields: `name`, `description`
- Field types (string, array, etc.)
- Field patterns (e.g., linker flags must start with `-l`)
- Minimum lengths
- Unique items in arrays

## Common Mistakes

### 1. Using Old Field Names

```yaml
patterns: [...]           # ❌ Wrong (old name)
basename_matches: [...]   # ✓ Correct

linker_flags: [...]       # ❌ Wrong (old name)
linker_flags_matches: [...] # ✓ Correct

path_components: [...]    # ❌ Wrong (old name)
directory_matches: [...]  # ✓ Correct
```

### 2. Missing Required Fields

```yaml
name: my-dep        # ✓ Required
description: "..."  # ✓ Required (min 10 chars)
```

### 3. Empty Pattern Sections

```yaml
basename_matches: []        # ❌ Need at least one pattern section
linker_flags_matches: []
directory_matches: []
```

Must have at least one of: `basename_matches`, `linker_flags_matches`, or `directory_matches`

### 4. Wrong Linker Flag Format

```yaml
linker_flags_matches:
  - pthread         # ❌ Wrong
  - -lpthread       # ✓ Correct (must start with -l)
```

### 5. Substring Matching in directory_matches

```yaml
directory_matches:
  - oo/ba           # ❌ Won't match /foo/bar/ (no substring matching)
  - foo/bar         # ✓ Correct (exact component match)
```

### 6. Invalid source Field

```yaml
source: apt           # ❌ Wrong (old enum value)
source: container     # ✓ Correct (new enum)
source: pip           # ❌ Wrong (old enum value)
source: submodule     # ✓ Correct (new enum)
```

### 7. Duplicate Patterns Across Files

Scanner will warn if same pattern appears in multiple files:

```
Warning: Duplicate pattern 'libfoo.so' found in bar.yml
(previously mapped to 'foo', now 'bar')
```

Last loaded file wins (alphabetical order). Remove duplicates.

### 8. Invalid Name Format

```yaml
name: MyDep         # ❌ Wrong (uppercase)
name: my_dep        # ❌ Wrong (underscore)
name: my-dep        # ✓ Correct (lowercase, hyphenated)
name: cuda-12       # ✓ Correct (numbers ok)
name: libstdc++6    # ✓ Correct (+ allowed)
```

## Troubleshooting

### Issue: Unknown artifacts not resolving after adding pattern

**Cause**: Pattern doesn't match artifact path.

**Solution**:
1. Check exact artifact path in `scan_output/unknown.yml`
2. Use correct field names: `basename_matches`, not `patterns`
3. For directories, use `directory_matches`
4. Check for typos in pattern

Example:
```yaml
# If unknown.yml shows:
artifacts:
  - /build/pytorch/lib/libtorch.so.2.0

# Add exact match:
basename_matches:
  - libtorch.so.2.0

# OR use directory matching:
directory_matches:
  - pytorch
```

### Issue: Multi-directory pattern not working

**Cause**: Substring matching expectations.

**Solution**:
- Multi-directory patterns require **exact component matches**
- `"oo/ba"` will NOT match `/foo/bar/`
- Use full component names: `"foo/bar"`

Example:
```yaml
directory_matches:
  - vendor/cutlass      # ✓ Matches /vendor/cutlass/
  - cutlass             # ✓ Also works (single component)
  - end/cutlass         # ❌ Won't match /vendor/cutlass/ (no substring matching)
```

### Issue: dpkg.yml not found

**Cause**: File was renamed to `base.yml`.

**Solution**:
```bash
# Old (incorrect)
grep "pattern" metadata/dpkg.yml

# New (correct)
grep "pattern" metadata/base.yml
```

### Issue: Validation fails with schema error

**Cause**: YAML structure doesn't match schema.

**Solution**:
1. Compare with `_template.yml`
2. Ensure required fields present (`name`, `description`)
3. Check linker flags start with `-l`
4. Use correct field names: `basename_matches`, `linker_flags_matches`, `directory_matches`

Example error:
```
❌ foo.yml: 'description' is too short (minimum 10 characters)
```

Fix:
```yaml
description: "Foo library for data processing"  # At least 10 chars
```

### Issue: Coverage decreased after changes

**Cause**: Removed or moved patterns incorrectly.

**Solution**:
1. Check git diff to see what changed
2. Re-add removed patterns
3. Run validation to ensure no syntax errors

```bash
git diff metadata/
python ../scan_build_artifacts.py --validate
```

## Best Practices

1. **One dependency per file** (except base.yml/cuda.yml for system libs)
2. **Use descriptive names**: `cuda-cudart-12` not `cudart12`
3. **Use multi-directory patterns** for vendored dependencies to avoid false positives
4. **Add metadata** (license, copyright, homepage) for attribution
5. **Validate after changes**: `python ../scan_build_artifacts.py --validate`
6. **Test coverage**: Re-run scanner after adding patterns
7. **Use correct field names**: `basename_matches`, not `patterns`
8. **Keep base.yml for system libraries only** (resolved via dpkg-query)
9. **Use `source: container`** for pre-installed packages (PyTorch, CUDA)
10. **Use `source: submodule`** for 3rdparty/ git submodules
11. **Start with directory_matches**: Most powerful pattern type for coverage
12. **Use version-agnostic patterns**: Match multiple versions with single pattern

## Resolution Strategy

The scanner uses a **two-tier resolution strategy**:

### PRIMARY: dpkg-query
- System-installed packages
- High confidence
- Handles all CUDA, system libraries automatically

### FALLBACK: YAML Patterns
Only used when dpkg-query doesn't know about the artifact:
1. Exact basename match → High confidence
2. Exact linker flag match → High confidence
3. Directory alias match → Medium confidence
4. Generic library inference → Low confidence

**Key insight**: Most CUDA and system packages are resolved via dpkg-query (PRIMARY), not YAML patterns. This is why `cuda.yml` and `base.yml` are sparse - they only contain fallback patterns for artifacts dpkg doesn't know about.

## Example: Complete Dependency File

```yaml
# metadata/cutlass.yml

name: cutlass
description: CUDA Templates for Linear Algebra Subroutines

version: "3.5.0"
license: BSD-3-Clause
copyright: Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES
homepage: https://github.com/NVIDIA/cutlass
source: submodule

basename_matches:
  - libcutlass.a

linker_flags_matches:
  - -lcutlass

directory_matches:
  - cutlass
  - 3rdparty/cutlass           # Multi-directory: prevents false positives
  - external/NVIDIA/cutlass    # Multi-directory: vendor-specific path
```

## Schema Reference

See `_schema.yml` for full JSON schema definition.

Key constraints:
- `name`: Required, string, pattern `^[a-z0-9-+]+$`, min length 1
- `description`: Required, string, min length 10
- `version`: Optional, string, min length 1
- `basename_matches`: Optional, array of strings, unique items
- `linker_flags_matches`: Optional, array of strings matching `^-l`, unique items
- `directory_matches`: Optional, array of strings, unique items (supports multi-directory)
- `source`: Optional, enum (submodule/container/fetched)

At least one of `basename_matches`, `linker_flags_matches`, or `directory_matches` required.

## Support

For issues or questions:
- Review `_schema.yml` for validation rules
- See `_template.yml` for new dependency template
- Run `python ../scan_build_artifacts.py --help` for CLI options
- Check scanner source code: `scan_build_artifacts.py` (PatternMatcher class, lines 620-926)
- Review output files: `scan_output/known.yml` and `scan_output/unknown.yml`
- See main README: `../README.md` for architecture and workflow details
