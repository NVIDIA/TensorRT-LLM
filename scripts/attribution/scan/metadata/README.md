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
├── cuda.yml            # CUDA packages (container-installed)
├── pytorch.yml         # PyTorch (container-installed)
├── tensorrt.yml        # TensorRT (container-installed)
├── cpython.yml         # Python interpreter (container-installed)
├── nvshmem.yml         # NVSHMEM (fetched)
└── nixl.yml            # NIXL (container-installed)
```

**Note:** Most dependencies are resolved automatically via:
- **dpkg-query**: System packages (libc, libzmq, openmpi, etc.)
- **fetch_content.json**: FetchContent dependencies (cutlass, xgrammar, etc.)
- **Vendor inference**: Vendored code in `third-party/`, `_deps/`, etc.

YAML patterns are only needed for container-installed packages that aren't dpkg-managed.

## File Formats

### Individual Dependency File

Most dependencies use this format:

```yaml
# metadata/pytorch.yml

name: pytorch                      # Required: canonical name
description: PyTorch machine learning framework  # Required: min 10 chars
source: container                  # Optional: how obtained (container, submodule, fetched)

basename_matches:                  # Exact basename matches
  - libtorch.so
  - libc10.so

directory_matches:                 # Directory path patterns
  - ATen
  - c10
  - torch
```

### List Format (cuda.yml)

For related packages, a list format can be used:

```yaml
# metadata/cuda.yml

dependencies:
- name: cuda
  description: CUDA Runtime API libraries, headers, and device code linking
  source: container
  basename_matches:
    - libcudart_static.a
  directory_matches:
    - cuda-*
    - cuda
    - cooperative_groups
    - cub
    - thrust
```

## Field Names Reference

**Current field names** (as of latest schema):
- `basename_matches` - Exact filename matches
- `directory_matches` - Path component patterns

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

   directory_matches:
     - foo-library
   ```

3. Validate and re-run:
   ```bash
   python ../scan_build_artifacts.py --validate
   python ../scan_build_artifacts.py
   ```

## Pattern Matching Behavior

The scanner uses a **2-tier matching strategy**:

### 1. Exact Basename Matching (HIGH confidence)
Matches exact filenames:

```yaml
basename_matches:
  - libcudart.so.12      # Matches only "libcudart.so.12" exactly
  - libcudart.so.12.0    # Matches only "libcudart.so.12.0" exactly
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

### Pattern Matching Power Ranking

**Most Powerful → Least Powerful:**

1. **directory_matches** - Matches entire directories of headers/files
   - Example: `directory_matches: [pytorch]` matches 4,822+ PyTorch headers
   - Single pattern can cover hundreds or thousands of artifacts

2. **basename_matches** - Matches specific library files
   - Example: `basename_matches: [libtorch.so]` matches one library
   - Good for targeting specific libraries

**Recommendation:** Start with `directory_matches` for maximum coverage with minimal patterns.

## Required Fields

Every dependency MUST have:

```yaml
name: my-dep        # Required: lowercase, hyphenated, + allowed
description: "..."  # Required: minimum 10 characters
```

At least one pattern section is required:
- `basename_matches` (exact filenames)
- `directory_matches` (path components)

## Optional Fields

```yaml
version: "1.0"                           # Optional: version string
source: "container"                      # Optional: how obtained
```

Valid `source` values:
- `submodule` - Git submodules in 3rdparty/ directory
- `container` - Pre-installed in container image (e.g., PyTorch, CUDA)
- `fetched` - Downloaded from URL and built from source

## Multi-Directory Pattern Examples

### Example 1: Single vs Multi-Directory Patterns

```yaml
# Example: CUDA patterns
name: cuda
description: CUDA Runtime API libraries
source: container

directory_matches:
  - cuda              # Single: matches any /cuda/ in path
  - cuda-*            # Glob: matches /cuda-12/, /cuda-13/, etc.
  - cooperative_groups  # Specific CUDA component
```

**Why multi-directory?** Prevents false positives:
- `"cuda"` alone might match unrelated paths
- Specific component names like `cooperative_groups` are safer

### Example 2: Container-Installed Framework

```yaml
# Example: PyTorch patterns
name: pytorch
description: PyTorch machine learning framework
source: container

directory_matches:
  - ATen              # PyTorch tensor library
  - c10               # Core library
  - torch             # Main framework
```

These patterns match container-installed PyTorch at paths like `/usr/local/lib/python/torch/`

**Note:** Most vendored/fetched dependencies are handled automatically by `fetch_content.json` and vendor inference. YAML patterns are only needed for container-installed packages.

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

Search in list format files (cuda.yml):

```bash
grep -A 5 "name: cuda" cuda.yml
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

✓ cuda.yml:cuda
✓ pytorch.yml
✓ tensorrt.yml
✓ cpython.yml
✓ nvshmem.yml
✓ nixl.yml

================================================================================
Results: 6/6 valid, 0/6 invalid
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
directory_matches: []
```

Must have at least one of: `basename_matches` or `directory_matches`.

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

### Issue: Validation fails with schema error

**Cause**: YAML structure doesn't match schema.

**Solution**:
1. Compare with `_template.yml`
2. Ensure required fields present (`name`, `description`)
3. Use correct field names: `basename_matches`, `directory_matches`

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

1. **Only add YAML patterns when necessary** - Most deps are handled by dpkg, fetch_content, or vendor inference
2. **Use `source: container`** for container-installed packages (PyTorch, CUDA, TensorRT)
3. **Use descriptive names**: `cuda` not `cudart`
4. **Start with directory_matches**: Most powerful pattern type for coverage
5. **Use version-agnostic patterns**: Match multiple versions with single pattern (e.g., `cuda-*`)
6. **Validate after changes**: `python ../scan_build_artifacts.py --validate`
7. **Test coverage**: Re-run scanner after adding patterns
8. **Use correct field names**: `basename_matches`, `directory_matches`

## Resolution Strategy

The scanner uses a **two-tier resolution strategy**:

### PRIMARY: dpkg-query
- System-installed packages
- High confidence
- Handles all CUDA, system libraries automatically

### FALLBACK: Pattern Matching
Only used when dpkg-query doesn't know about the artifact:
1. **fetch_content.json** → Automatic `_deps/<name>-src` aliases
2. **Vendor inference** → Paths with `third-party/`, `_deps/`, `3rdparty/`, etc.
3. **YAML patterns** → Container-installed packages
4. **Generic library inference** → Low confidence fallback

**Key insight**: Most dependencies are resolved automatically. YAML patterns are only needed for container-installed packages that aren't dpkg-managed (PyTorch, CUDA, TensorRT, etc.).

## Example: Complete Dependency File

```yaml
# metadata/pytorch.yml

name: pytorch
description: PyTorch machine learning framework
source: container

basename_matches:
  - libc10.so
  - libc10_cuda.so
  - libtorch.so
  - libtorch_python.so

directory_matches:
  - ATen
  - c10
  - caffe2
  - torch
```

## Schema Reference

See `_schema.yml` for full JSON schema definition.

Key constraints:
- `name`: Required, string, pattern `^[a-z0-9-+]+$`, min length 1
- `description`: Required, string, min length 10
- `version`: Optional, string, min length 1
- `basename_matches`: Optional, array of strings, unique items
- `directory_matches`: Optional, array of strings, unique items (supports multi-directory)
- `source`: Optional, enum (submodule/container/fetched)

At least one of `basename_matches` or `directory_matches` required.

## Support

For issues or questions:
- Review `_schema.yml` for validation rules
- See `_template.yml` for new dependency template
- Run `python ../scan_build_artifacts.py --help` for CLI options
- Check scanner source code: `scan_build_artifacts.py` (PatternMatcher class, lines 620-926)
- Review output files: `scan_output/known.yml` and `scan_output/unknown.yml`
- See main README: `../README.md` for architecture and workflow details
