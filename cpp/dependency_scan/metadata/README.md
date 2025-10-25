# Dependency Patterns

This directory contains pattern definitions for dependency detection in the TensorRT-LLM C++ dependency scanner.

## Quick Start

After running the scanner, check `scan_output/unknown.yml` for unmapped artifacts, then add patterns here.

## Structure

Each `.yml` file represents one dependency:

```
metadata/
├── _template.yml       # Template for new dependencies
├── _schema.yml         # JSON schema for validation
├── dpkg.yml           # System libraries (list format)
├── tensorrt-llm.yml   # Individual dependency
├── pytorch.yml
├── cuda-cudart-12.yml
└── ... (62+ dependencies)
```

## File Formats

### Individual Dependency File

Most dependencies use this format:

```yaml
# metadata/tensorrt-llm.yml

name: tensorrt-llm                    # Required: canonical name
version: "1.2.0"                      # Required: version
description: TensorRT-LLM libraries   # Required: description

license: Apache-2.0                   # Optional: license
copyright: "Copyright 2024 NVIDIA"    # Optional: copyright
homepage: "https://..."               # Optional: URL
source: "built-from-source"           # Optional: how obtained

patterns:                             # Artifact patterns (exact or substring)
  - libth_common.so
  - libpg_utils.so

linker_flags:                         # Linker flags (-l flags)
  - -ltensorrt_llm

path_components:                      # Path matching
  - tensorrt_llm

aliases:                              # Other names
  - trtllm
```

### dpkg.yml (List Format)

System libraries use a list format for compactness:

```yaml
# metadata/dpkg.yml

dependencies:
  - name: libc6
    version: "2.35"
    description: GNU C Library
    patterns:
      - libc.so.6
    linker_flags:
      - -lc
      - -lpthread
      - -ldl
      - -lm
      - -lrt
    path_components: []
    aliases: []

  - name: libstdc++6
    version: "13.0"
    description: GNU C++ Library
    patterns:
      - libstdc++.so.6
    linker_flags:
      - -lstdc++
    path_components: []
    aliases: []
  # ... more system libraries
```

## Adding Patterns

### When You See Unknown Artifacts

After running the scanner, check `scan_output/unknown.yml`:

```yaml
summary:
  count: 2
  action_required: "Add patterns to YAML files in metadata/ for these artifacts"

artifacts:
  - /build/3rdparty/newlib/include/foo.h
  - libfoo.so
```

### Option A: Add to Existing Dependency

If `libfoo.so` belongs to an existing dependency (e.g., `pytorch`):

1. Open `metadata/pytorch.yml`
2. Add to the `patterns` list:
   ```yaml
   patterns:
     - libtorch.so
     - libfoo.so      # ← Add here
   ```
3. Re-run scanner:
   ```bash
   python ../scan_build_artifacts.py --build-dir /path --output-dir validation_output/
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
   version: "1.0"
   description: Foo library for data processing

   patterns:
     - libfoo.so
   ```

3. Validate and re-run:
   ```bash
   python ../scan_build_artifacts.py --validate
   python ../scan_build_artifacts.py --build-dir /path --output-dir validation_output/
   ```

## Pattern Matching Behavior

The scanner tries matches in this order:

1. **Exact match on basename**: `libfoo.so` == `libfoo.so`
2. **Exact match on full path**: `/path/to/libfoo.so` == `/path/to/libfoo.so`
3. **Substring match**: `foo.cpython` in `foo.cpython-312-x86_64.so`

**No need to specify match type** - the scanner tries both automatically!

### Examples

**Exact match** (`patterns`):
```yaml
patterns:
  - libcudart.so.12      # Matches only "libcudart.so.12"
  - libcudart.so.12.0    # Matches only "libcudart.so.12.0"
```

**Substring match** (`patterns`):
```yaml
patterns:
  - deep_ep_cpp  # Matches "deep_ep_cpp_tllm.cpython-312-x86_64-linux-gnu.so"
```

**Linker flags** (`linker_flags`):
```yaml
linker_flags:
  - -lpthread    # Matches "-lpthread" in link.txt
  - -lcudart     # Matches "-lcudart"
```

**Path components** (`path_components`):
```yaml
path_components:
  - pytorch      # Matches "/build/pytorch/include/torch.h"
  - cuda-12      # Matches "/usr/local/cuda-12/include/cuda.h"
```

## Required Fields

Every dependency MUST have:

```yaml
name: my-dep        # Required: lowercase, hyphenated
version: "1.0"      # Required: version string
description: "..."  # Required: minimum 10 characters
```

At least one pattern section is required:
- `patterns` (artifact filenames)
- `linker_flags` (-l flags)
- `path_components` (directory names)

## Optional Fields

Recommended for attribution/licensing:

```yaml
license: "Apache-2.0"                    # SPDX identifier
copyright: "Copyright 2024 NVIDIA"       # Copyright notice
homepage: "https://example.com"          # Project URL
source: "apt"                            # How obtained
```

Valid `source` values:
- `apt` - Installed via apt/dpkg
- `pip` - Installed via pip
- `built-from-source` - Compiled from source
- `bundled` - Bundled with project
- `download` - Downloaded binary
- `other` - Other method

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
Dependencies directory: ./dependencies

✓ cuda-cudart-12.yml
✓ pytorch.yml
✓ tensorrt-llm.yml
...
✓ dpkg.yml:libc6
✓ dpkg.yml:libstdc++6
...

================================================================================
Results: 73/73 valid, 0/73 invalid
================================================================================
```

### Re-run Scanner

After adding patterns, re-run the scanner:

```bash
python ../scan_build_artifacts.py --build-dir /path --output-dir validation_output/
```

Check `validation_output/unknown.yml` - should have fewer (or zero) artifacts:

```yaml
summary:
  count: 0  # Improved from previous run!
  action_required: "Add patterns to YAML files in metadata/ for these artifacts"

artifacts: []
```

### Schema Validation

The `_schema.yml` file defines validation rules:
- Required fields
- Field types (string, array, etc.)
- Field patterns (e.g., linker flags must start with `-l`)
- Minimum lengths
- Unique items in arrays

Validation requires `jsonschema`:
```bash
pip install jsonschema
```

## Common Mistakes

### 1. Missing Required Fields

```yaml
name: my-dep        # ✓ Required
version: "1.0"      # ✓ Required
description: "..."  # ✓ Required (min 10 chars)
```

### 2. Empty Pattern Sections

```yaml
patterns: []        # ❌ Need at least one pattern section
linker_flags: []
path_components: []
```

Must have at least one of: `patterns`, `linker_flags`, or `path_components`

### 3. Wrong Linker Flag Format

```yaml
linker_flags:
  - pthread         # ❌ Wrong
  - -lpthread       # ✓ Correct (must start with -l)
```

### 4. Duplicate Patterns Across Files

Scanner will warn if same pattern appears in multiple files:

```
Warning: Duplicate pattern 'libfoo.so' found in bar.yml
(previously mapped to 'foo', now 'bar')
```

Last loaded file wins (alphabetical order). Remove duplicates.

### 5. Invalid Name Format

```yaml
name: MyDep         # ❌ Wrong (uppercase)
name: my_dep        # ❌ Wrong (underscore)
name: my-dep        # ✓ Correct (lowercase, hyphenated)
name: cuda-12       # ✓ Correct (numbers ok)
name: libstdc++6    # ✓ Correct (+ allowed)
```

### 6. Missing Description

```yaml
description: "Test"              # ❌ Too short (min 10 chars)
description: "Test library"      # ✓ Correct (10+ chars)
```

## Troubleshooting

### Issue: Unknown artifacts not resolving after adding pattern

**Cause**: Pattern doesn't match artifact path.

**Solution**:
1. Check exact artifact path in `scan_output/unknown.yml`
2. Try substring pattern if exact doesn't work
3. Use path_components for directory-based matching
4. Check for typos in pattern

Example:
```yaml
# If unknown.yml shows:
artifacts:
  - /build/pytorch/lib/libtorch.so.2.0

# Try adding:
patterns:
  - libtorch.so.2.0      # Exact match
  # OR
  - libtorch.so          # Substring match
```

### Issue: Validation fails with schema error

**Cause**: YAML structure doesn't match schema.

**Solution**:
1. Compare with `_template.yml`
2. Ensure required fields present
3. Check linker flags start with `-l`
4. Verify arrays have unique items

Example error:
```
❌ foo.yml: 'description' is too short (minimum 10 characters)
```

Fix:
```yaml
description: "Foo library for data processing"  # At least 10 chars
```

### Issue: Can't find which file contains pattern

**Cause**: Pattern in dpkg.yml or nested in list.

**Solution**:
```bash
# Search all files including dpkg.yml
grep -r "pattern-name" .

# Search dpkg.yml specifically
grep -A 5 "pattern-name" dpkg.yml
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

## Advanced Usage

### Wildcards and Regex

Patterns are exact or substring matches - **no wildcards or regex supported**.

Instead of:
```yaml
patterns:
  - libfoo.*.so      # ❌ Not supported
```

Use multiple entries:
```yaml
patterns:
  - libfoo.so
  - libfoo.so.1
  - libfoo.so.2
```

Or use path_components:
```yaml
path_components:
  - foo-library      # Matches any file in foo-library/ directory
```

### Bundled Binaries with Version Suffixes

For Python extensions with version suffixes:

```yaml
patterns:
  - my_module.cpython   # Matches my_module.cpython-312-x86_64-linux-gnu.so
```

Substring match handles version variations automatically.

### Aliases for Package Name Variations

If dpkg returns different package names:

```yaml
name: cuda-cudart-dev
aliases:
  - cuda-cudart-12-dev
  - cuda-cudart-13-dev
```

### Multiple Versions

Create separate files for major versions:

```
metadata/
├── cuda-cudart-11.yml
├── cuda-cudart-12.yml
└── cuda-cudart-13.yml
```

Each with version-specific patterns:

```yaml
# cuda-cudart-12.yml
patterns:
  - libcudart.so.12
  - libcudart.so.12.0
path_components:
  - cuda-12
```

## Best Practices

1. **One dependency per file** (except dpkg.yml for system libs)
2. **Use descriptive names**: `cuda-cudart-12` not `cudart12`
3. **Include version in name** for versioned dependencies
4. **Add metadata** (license, copyright, homepage) for attribution
5. **Validate after changes**: `python ../scan_build_artifacts.py --validate`
6. **Test coverage**: Re-run scanner after adding patterns
7. **Document rationale**: Add comments for non-obvious patterns
8. **Keep dpkg.yml for system libraries only**
9. **Use substring patterns sparingly** (prefer exact matches)
10. **Commit changes atomically** (one dependency per commit)

## Examples

### Example 1: Adding CUDA Library Pattern

Unknown artifact in `scan_output/unknown.yml`:
```yaml
artifacts:
  - /usr/local/cuda-12.0/lib64/libcublasLt.so.12
```

Add to existing `cuda-cublas-12.yml`:
```yaml
name: cuda-cublas-12
version: "12.0"
description: NVIDIA CUDA Basic Linear Algebra Subroutines library version 12

patterns:
  - libcublas.so.12
  - libcublasLt.so.12      # ← Add here

linker_flags:
  - -lcublas
  - -lcublasLt             # ← Add here

path_components:
  - cuda-12

aliases: []
```

### Example 2: Creating New Dependency

Unknown artifact in `scan_output/unknown.yml`:
```yaml
artifacts:
  - /build/3rdparty/nlohmann_json/include/json.hpp
```

Create `nlohmann-json.yml`:
```yaml
name: nlohmann-json
version: "3.11.2"
description: JSON for Modern C++ header-only library

license: MIT
copyright: "Copyright (c) 2013-2022 Niels Lohmann"
homepage: "https://github.com/nlohmann/json"
source: bundled

patterns: []              # Header-only, no library files

linker_flags: []

path_components:
  - nlohmann_json         # Matches path component
  - json                  # Also matches "json" directory

aliases:
  - nlohmann-json-dev
```

Re-run scanner:
```bash
python ../scan_build_artifacts.py
cat scan_output/unknown.yml  # Should be empty or reduced
```

### Example 3: Bundled Python Extension

Unknown artifact in `scan_output/unknown.yml`:
```yaml
artifacts:
  - tensorrt_llm/libs/executor_worker.cpython-312-x86_64-linux-gnu.so
```

Create `tensorrt-llm.yml`:
```yaml
name: tensorrt-llm
version: "1.2.0"
description: TensorRT-LLM inference optimization libraries

source: built-from-source

patterns:
  - executor_worker.cpython    # Substring match for versioned extensions
  - libth_common.so

linker_flags: []

path_components:
  - tensorrt_llm

aliases:
  - trtllm
```

### Example 4: System Library in dpkg.yml

Unknown artifact in `scan_output/unknown.yml`:
```yaml
artifacts:
  - -lgomp
```

Add to `dpkg.yml`:
```yaml
dependencies:
  # ... existing entries ...

  - name: libgomp1
    version: "13.0"
    description: GCC OpenMP runtime library
    patterns:
      - libgomp.so.1
    linker_flags:
      - -lgomp          # ← Add here
    path_components: []
    aliases: []
```

## Integration with Scanner

### Scanner Resolution Order

1. **dpkg-query** (PRIMARY)
   - System-installed packages
   - High confidence

2. **YAML patterns** (FALLBACK)
   - Exact pattern match → High confidence
   - Substring pattern match → High confidence
   - Path alias match → Medium confidence
   - Generic library inference → Low confidence

### Output Files

The scanner generates simplified YAML output files:

- **`scan_output/known.yml`**: Successfully mapped artifacts (paths only, no metadata)
- **`scan_output/unknown.yml`**: Unmapped artifacts needing patterns (simple list)

**Known artifacts example:**
```yaml
summary:
  total_artifacts: 6198
  mapped: 6198
  unmapped: 0
  coverage: "100.0%"
  unique_dependencies: 45

dependencies:
  cuda-cudart:
    - /usr/local/cuda-12.9/include/cuda_runtime.h
    - /usr/local/cuda-12.9/include/cuda.h

  libc6:
    - /usr/include/stdio.h
    - -lpthread
```

**Unknown artifacts example:**
```yaml
summary:
  count: 2
  action_required: "Add patterns to YAML files in metadata/ for these artifacts"

artifacts:
  - /build/unknown/foo.h
  - libmystery.so
```

**Key benefits of YAML output:**
- Human-readable format
- Smaller file sizes vs JSON
- Easy to scan and parse
- Aligned with metadata/*.yml format consistency

### Workflow Integration

1. **Initial scan**:
   ```bash
   python ../scan_build_artifacts.py
   ```

2. **Check unknown artifacts**:
   ```bash
   cat scan_output/unknown.yml
   ```

3. **Add patterns** to metadata/ files

4. **Validate changes**:
   ```bash
   python ../scan_build_artifacts.py --validate
   ```

5. **Re-scan to verify**:
   ```bash
   python ../scan_build_artifacts.py
   cat scan_output/unknown.yml  # Should be reduced/empty
   ```

6. **Review known artifacts**:
   ```bash
   cat scan_output/known.yml  # Check coverage improved
   ```

## Schema Reference

See `_schema.yml` for full JSON schema definition.

Key constraints:
- `name`: Required, string, pattern `^[a-z0-9-+]+$`, min length 1
- `version`: Required, string, min length 1
- `description`: Required, string, min length 10
- `patterns`: Optional, array of strings, unique items
- `linker_flags`: Optional, array of strings matching `^-l`, unique items
- `path_components`: Optional, array of strings, unique items
- `aliases`: Optional, array of strings, unique items
- `source`: Optional, enum (apt/pip/built-from-source/bundled/download/other)

At least one of `patterns`, `linker_flags`, or `path_components` required.

## Migration from patterns.json

If migrating from old `patterns.json` format, see `MIGRATION_GUIDE.md` for detailed instructions.

Quick migration:
```bash
python migrate_to_dependencies.py
```

## Support

For issues or questions:
- Review `_schema.yml` for validation rules
- See `_template.yml` for new dependency template
- Check `MIGRATION_GUIDE.md` for migration from patterns.json
- Run `python ../scan_build_artifacts.py --help` for CLI options
- Check scanner source code: `scan_build_artifacts.py` (PatternMatcher class)
- Review output files: `scan_output/known.yml` and `scan_output/unknown.yml`
