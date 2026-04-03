# trtllm-sbom CLI

Command-line tool for managing the TensorRT-LLM attribution database.

See [../README.md](../README.md) for an overview of the attribution system.

## Installation

```bash
pip install -e scripts/attribution/sbom
```

## Commands

### import - Add Dependencies

Add a new dependency with license information:

```bash
trtllm-sbom import -d "dependency" -v "version" \
  -l path/to/license.txt \
  -c path/to/copyright.txt \
  -a path/to/attribution.txt \
  -s "https://github.com/org/repo/tree/v1.0.0"
```

Options:
- `-d, --dependency` - Dependency name (required)
- `-v, --version` - Version string (required)
- `-l, --license` - Path to license file (required)
- `-c, --copyright` - Path to copyright file (optional if license includes copyright)
- `-a, --attribution` - Path to attribution file (optional)
- `-s, --source` - Source code URL (required)
- `--overwrite` - Update existing entries

Using JSON input:

```bash
trtllm-sbom import -j payload.json
```

```json
[
  {
    "dependency": "example",
    "version": "1.0.0",
    "license": "path/to/LICENSE",
    "copyright": "path/to/COPYRIGHT",
    "source": "https://github.com/org/repo/tree/v1.0.0"
  }
]
```

### map-files - Associate Files with Dependencies

Map files to a dependency:

```bash
trtllm-sbom map-files -d "dependency" -v "version" file1.h file2.h
```

Using JSON input:

```bash
trtllm-sbom map-files -j mappings.json
```

```json
[
  {
    "dependency": "example",
    "version": "1.0.0",
    "files": ["path/to/file1.h", "path/to/file2.h"]
  }
]
```

### generate - Create Attributions and SBOM

Generate output files from a list of input files:

```bash
trtllm-sbom generate -a ATTRIBUTIONS.md -s SBOM.json file1.h file2.h ...
```

Options:
- `-a, --attributions` - Output path for ATTRIBUTIONS.md
- `-s, --sbom` - Output path for SBOM.json
- `--dry-run` - Check coverage without generating output
- `--missing-files FILE` - Write unmapped files to JSON
- `--active-versions FILE` - Filter to specific dependency versions

Dry-run to check coverage:

```bash
trtllm-sbom generate --dry-run --missing-files missing.json file1.h file2.h
```

Filter by active versions:

```bash
trtllm-sbom generate --active-versions versions.json -a attr.md -s sbom.json files...
```

`versions.json` format:
```json
{
  "cuda": ["12.9"],
  "pytorch": ["2.5.0"]
}
```

### validate - Check Database Integrity

```bash
trtllm-sbom validate
```

Checks:
- CAS checksums are valid
- All CAS references exist
- Every dependency has a license
- Every file maps to exactly one dependency
- Copyright exists if license lacks copyright notice

### list - Show Registered Dependencies

```bash
trtllm-sbom list
trtllm-sbom list --format json
```

## Storage Schema

Data is stored in `scripts/attribution/data/`:

```
data/
├── cas/                      # Content-addressable storage
│   └── ab/cd/abcd1234...     # License/copyright texts (SHA-256)
├── dependency_metadata.yml   # Dependency info
└── files_to_dependency.yml   # File → dependency mappings
```

Per-file copyright notices are extracted on-the-fly during `generate` from the actual build input files.

### dependency_metadata.yml

```yaml
tensorrt/10.14.0:
  license: abcd1234...        # CAS address of license text
  copyright: efgh5678...      # CAS address (optional)
  attribution: ijkl9012...    # CAS address (optional)
  source: "https://..."       # Source code URL
```

### files_to_dependency.yml

```yaml
tensorrt/10.14.0:
  - 21f7e91129b871c5040081    # File checksum
  - 34d7a91129b871c5040081
```

## Testing

```bash
cd scripts/attribution/sbom
python -m pytest tests/ -v
```
