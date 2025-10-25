# CPP Dependency Scanner

Scans TensorRT-LLM build artifacts (headers, libraries, binaries) and maps them to source dependencies.

## Quick Start

```bash
# Run scanner (scans ../build by default)
python scan_build_artifacts.py

# Output: scan_output/known.yml, scan_output/unknown.yml
```

## Usage

```bash
# Custom build directory
python scan_build_artifacts.py --build-dir /path/to/build

# Custom output directory
python scan_build_artifacts.py --output-dir reports/

# Validate YAML files
python scan_build_artifacts.py --validate
```

## Resolution Strategy

1. **dpkg-query**: System packages via Debian package manager
2. **YAML patterns**: Non-dpkg packages (CUDA, TensorRT, PyTorch, etc.)

## Output Format

### known.yml

```yaml
summary:
  total_artifacts: 6198
  mapped: 6198
  unmapped: 0
  coverage: "100.0%"

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
  count: 36
  action_required: "Add patterns to YAML files in metadata/"

artifacts:
  - /build/3rdparty/newlib/include/foo.h
  - /build/unknown/libmystery.so
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
name: newlib
version: "4.0"
description: Newlib C library for embedded systems

patterns:
  - libnewlib.so

linker_flags:
  - -lnewlib

path_components:
  - newlib
  - 3rdparty/newlib
```

See `metadata/_template.yml` and `metadata/README.md` for details.

## YAML Dependencies

Each dependency file contains:

```yaml
name: pytorch
version: "2.0"
description: PyTorch machine learning framework
license: BSD-3-Clause
copyright: Copyright (c) PyTorch Contributors
homepage: https://pytorch.org/
source: pip

patterns:
  - libtorch.so
  - libc10.so

linker_flags:
  - -ltorch
  - -lc10

path_components:
  - pytorch
  - torch

aliases:
  - torch
```

Multiple dependencies can be grouped in list format (see `metadata/dpkg.yml`, `metadata/cuda.yml`).

## Testing

```bash
cd tests
python -m pytest test_scan_build_artifacts.py -v
# Expected: 34 passed
```

## Troubleshooting

**Low dpkg coverage**
- Running on non-Debian system
- YAML dependencies will handle more as fallback

**Many unknown artifacts**
1. Review `scan_output/unknown.yml`
2. Add patterns to `metadata/*.yml`
3. Run `--validate` to check syntax
4. Re-scan to verify

**Wrong mappings**
- Check pattern priorities in YAML files
- More specific patterns should be listed first

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
- D files: CMake dependency files with headers
- link.txt: Linker commands with libraries
- Wheels: Python binaries via readelf

**Resolution Flow:**
1. Collect artifacts from build directory
2. Try dpkg-query resolution (PRIMARY)
3. Fall back to YAML patterns (FALLBACK)
4. Generate known.yml and unknown.yml reports

## Files

- `scan_build_artifacts.py` - Main scanner script
- `metadata/*.yml` - Dependency patterns (8 dependencies defined)
- `metadata/_template.yml` - Template for new dependencies
- `metadata/_schema.yml` - YAML validation schema
- `metadata/README.md` - Pattern documentation
- `tests/test_scan_build_artifacts.py` - Unit tests

## Requirements

Python 3.8+ with stdlib only. No external dependencies required.

## License

Same as TensorRT-LLM parent project.
