# TensorRT-LLM Attribution System

This directory contains the tooling for generating license attributions for TensorRT-LLM wheels.

## Overview

The attribution system tracks C++ dependencies used in the TensorRT-LLM build and generates:
- **ATTRIBUTIONS.md** - Human-readable license attributions for the wheel
- **SBOM.json** - Machine-readable Software Bill of Materials (CycloneDX format)

## Quick Start

Attribution generation is integrated into `scripts/build_wheel.py` and runs automatically during wheel builds. For manual runs:

```bash
# Run the attribution workflow
python scripts/attribute.py --build-dir cpp/build

# Output (on success):
#   cpp/build/attribution/ATTRIBUTIONS.md
#   cpp/build/attribution/SBOM.json

# Output (when data is missing):
#   cpp/build/attribution/import_payload.json  - Dependencies to add
#   cpp/build/attribution/file_mappings.json   - File mappings to register
#   cpp/build/attribution/README.txt           - Instructions for completion
```

## Architecture

```
scripts/
├── attribute.py              # Main workflow orchestrator
└── attribution/
    ├── scan/                 # Build input scanning
    │   ├── identify_build_inputs.py   # Collects headers/libraries from build
    │   ├── map_dependencies.py        # Maps files to dependency names
    │   └── metadata/                  # YAML patterns for non-system packages
    ├── sbom/                 # Database controller and CLI
    │   └── src/trtllm_sbom/  # 'trtllm-sbom' CLI implementation
    └── data/                 # Attribution database
        ├── cas/              # Content-addressable storage for license data
        ├── dependency_metadata.yml
        └── files_to_dependency.yml
```

## Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  C++ Build      │     │  Scan & Map      │     │  Generate       │
│  (cpp/build/)   │────▶│  Dependencies    │────▶│  Attributions   │
│  .d files       │     │                  │     │                 │
│  (headers &     │     │  identify_build_ │     │  trtllm-sbom    │
│   libraries)    │     │  inputs.py       │     │  generate       │
└─────────────────┘     │  map_deps.py     │     └─────────────────┘
                        └──────────────────┘              │
                                 │                        │
                                 ▼                        ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  Attribution DB  │     │  Output Files   │
                        │  (data/)         │────▶│  ATTRIBUTIONS.md│
                        │                  │     │  SBOM.json      │
                        └──────────────────┘     └─────────────────┘
```

## Components

### `scripts/attribute.py`

The main workflow script that:
1. Collects input files from build directory and detects actively used dependency versions
2. Checks attribution database coverage via `trtllm-sbom generate --dry-run`
   - If all files covered: generates an SBOM and an attributions file, then exits successfully
3. Maps unrecognized files to dependencies and creates database payload templates
4. Attempts automatic import of dependency payloads
   - If successful: retries generation
5. Provides manual instructions if automatic steps fail

### `scan/` - Build Input Scanning

Modules for identifying build dependencies:
- **`identify_build_inputs.py`** - Parses CMake `.d` files to collect headers and libraries
- **`map_dependencies.py`** - Maps file paths to dependency names using various heuristics
- **`metadata/`** - YAML pattern files for ad-hoc dependency installations (TensorRT, PyTorch, etc.)

See [scan/README.md](scan/README.md) for pattern syntax and debugging tools.

### `sbom/` - Database Controller and Low-Level CLI

The `trtllm-sbom` CLI manages the attribution database:
- **`import`** - Add license/copyright information for a dependency
- **`map-files`** - Associate files with dependencies
- **`generate`** - Create ATTRIBUTIONS.md and SBOM.json from build input files
- **`validate`** - Check database integrity
- **`list`** - Show registered dependencies

See [sbom/README.md](sbom/README.md) for CLI reference and storage schema.

### `data/` - Attribution Database

Content-addressable storage for license texts and metadata:
- `cas/` - License and copyright texts (SHA-256 addressed)
- `dependency_metadata.yml` - Dependency info (license, copyright, etc.)
- `files_to_dependency.yml` - File path to dependency mappings

Per-file copyright notices are extracted on-the-fly during attribution file generation from the actual build input files.

## Common Workflows

### When Attribution Generation Fails

If `attribute.py` reports missing data:

1. Review the generated files in `cpp/build/attribution/`:
   - `import_payload.json` - Dependencies needing license info
   - `file_mappings.json` - Files needing dependency association
   - `README.txt` - Detailed instructions

2. For each dependency in `import_payload.json`:
   - Locate the license file and copyright notice
   - Fill in the `license` and `copyright` fields

3. Import the completed data:
   ```bash
   trtllm-sbom import -j cpp/build/attribution/import_payload.json
   trtllm-sbom map-files -j cpp/build/attribution/file_mappings.json
   ```

4. Re-run `attribute.py` to verify success

### Adding a New Dependency

New dependencies should be automatically detected in most cases, such that no manual action is required; however, `map_dependencies.py` sometimes isn't able to identify new dependencies. If you run into mapping issues, you can create a pattern file in `scan/metadata/`to give `map_dependencies.py` some hints, before following the above "When Attribution Generation Fails" section to manually map files from the new dependency.

### Debugging Mapping Issues

Use the standalone scanner for detailed file mapping reports:

```bash
cd scripts/attribution/scan
python scan_build_artifacts.py --build-dir ../../../cpp/build

# Review:
#   scan_output/unknown.yml    - Unmapped files needing patterns
#   scan_output/known.yml      - Successfully mapped files
```

## Version Detection

The system automatically detects dependency versions from a few sources:
- dpkg/rpm/python package versions
- file paths (e.g., `/usr/local/cuda-12.9/` → CUDA 12.9)
- contents of version header files

## CI Integration

The attribution workflow runs automatically in `scripts/build_wheel.py`:
- On success: `ATTRIBUTIONS.md` is copied to the wheel package
- On failure: Build continues with hard-coded attribution files (with warning)
