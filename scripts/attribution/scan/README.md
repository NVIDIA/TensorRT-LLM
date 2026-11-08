# Build Input Scanning

This directory contains modules for scanning build artifacts and mapping them to dependencies.

See [../README.md](../README.md) for an overview of the attribution system.

## Modules

- **`identify_build_inputs.py`** - Collects build artifacts from CMake `.d` files (headers and libraries)
- **`map_dependencies.py`** - Maps file paths to dependencies via dpkg-query and YAML patterns
- **`scan_build_artifacts.py`** - Debugging tool for generating detailed coverage reports

## YAML Pattern Matching

For files not resolvable via dpkg-query, the system uses YAML patterns in `metadata/`.

### Pattern Types

```yaml
name: example-lib
description: Example library

# Match by filename
basename_matches:
  - libexample.so
  - libexample.a

# Match by directory in path
directory_matches:
  - example/include    # Matches /path/to/example/include/file.h
  - 3rdparty/example   # Matches /path/3rdparty/example/file.h

# Extract version from path (optional)
version_patterns:
  - example-(\d+\.\d+)  # Captures version from /example-1.2/include/
```

### Pattern Priority

1. **Exact basename match**: `libcudart.so.12` → `cuda-cudart`
2. **Directory match**: `/ATen/include/` → `pytorch`

### Adding New Patterns

1. Copy `metadata/_template.yml` to a new file
2. Define patterns for the dependency
3. Validate with `python scan_build_artifacts.py --validate`

See `metadata/README.md` for detailed pattern documentation.

## Debugging Tool

The `scan_build_artifacts.py` script generates detailed reports for debugging:

```bash
# Generate coverage reports
python scan_build_artifacts.py --build-dir ../../../cpp/build

# Output:
#   scan_output/known.yml         - Mapped artifacts by dependency
#   scan_output/unknown.yml       - Unmapped artifacts
#   scan_output/path_issues.yml   - Non-existent paths

# Validate YAML patterns
python scan_build_artifacts.py --validate
```

### Report Formats

**known.yml** - Successfully mapped artifacts:
```yaml
summary:
  total_artifacts: 6200
  mapped: 6200
  coverage: 100.0%
dependencies:
  cuda-cudart:
    version: "12.9"
    artifacts:
      - /usr/local/cuda-12.9/include/cuda_runtime.h
```

**unknown.yml** - Artifacts needing patterns:
```yaml
summary:
  count: 5
  action_required: Add patterns to metadata/*.yml
artifacts:
  - /some/unknown/header.h
```

**path_issues.yml** - Non-existent paths (stale builds):
```yaml
summary:
  count: 100
  percentage: 1.5%
non_existent_paths:
  - resolved_path: /usr/local/cuda-13.0/include/cuda.h
    source: /build/CMakeFiles/target.dir/file.cpp.o.d
```
