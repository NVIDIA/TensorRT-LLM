This is a template file used by scripts/attribute.py to generate README.txt
when attribution data is missing. Edit this file to update the instructions
shown to users who need to manually complete attribution data.

--- TEMPLATE START ---
================================================================================
ATTRIBUTION DATA REQUIRED
================================================================================

Some dependencies are missing attribution data. This file explains how to
complete the data in 'import_payload.json' before running the attribution workflow.

The following files are available in this directory:
  - import_payload.json: Attribution data to import for dependencies
  - file_mappings.json: File-to-dependency mappings

================================================================================
QUICK START
================================================================================

1. Edit 'file_mappings.json' to map all files to their dependencies
2. Edit 'import_payload.json' to fill in missing fields (see FIELD REFERENCE below)
3. Run: trtllm-sbom import -j import_payload.json
4. Run: trtllm-sbom map-files -j file_mappings.json
5. Re-run the build to verify all attribution data is complete

================================================================================
UNDERSTANDING file_mappings.json
================================================================================

The file_mappings.json file maps input files to their dependencies:

{
  "dependency": "cutlass",
  "version": "v4.3.0",
  "files": [
    "/path/to/file1.h",
    "/path/to/file2.h"
  ]
}

This file is auto-generated and usually requires no manual editing. However,
you may need to edit it if:

1. FILES ARE MAPPED TO the "unknown" DEPENDENCY
   If you see an entry with "dependency": "unknown", these files could not be
   automatically identified. You need to:
   - Identify which dependency each file belongs to
   - Move the files to the correct dependency entry (or create a new one)
   - Add the new dependency to import_payload.json if it doesn't exist

2. FILES ARE MAPPED TO THE WRONG DEPENDENCY
   If files are incorrectly attributed, move them to the correct entry.

3. A NEW DEPENDENCY NEEDS TO BE ADDED
   If you add a new entry to file_mappings.json, you must also add a
   corresponding entry to import_payload.json with the license information.

================================================================================
UNDERSTANDING import_payload.json
================================================================================

Each entry in import_payload.json represents a dependency that needs attribution:

{
  "dependency": "example-lib",     // Name of the dependency
  "version": "1.2.3",              // Version used in the build
  "license": "/path/to/LICENSE",   // REQUIRED: Path to license file
  "copyright": "",                 // OPTIONAL: Path to copyright/notice file
  "attribution": "",               // OPTIONAL: Path to third-party notices
  "source": ""                     // OPTIONAL: URL to source code
}

The script has auto-populated fields where possible. You need to fill in
any empty fields that are marked REQUIRED, and optionally fill in others.

================================================================================
FIELD REFERENCE (import_payload.json)
================================================================================

LICENSE (required)
------------------
Path to the file containing the full license text (e.g., LICENSE, LICENSE.txt,
COPYING). This is the legal license under which the code is distributed.

If auto-detection found the wrong file or no file, locate the correct license
in the dependency's source code or download it from the project's repository.

COPYRIGHT (optional)
--------------------
Path to a file containing copyright notices, author information, or contributor
lists. Common files: NOTICE, AUTHORS, CONTRIBUTORS, COPYRIGHT.

This field is OPTIONAL if the license file already contains copyright notices
(which is common - most LICENSE files include "Copyright (c) YEAR Author").
Only fill this in if there's a SEPARATE file with additional copyright info.

ATTRIBUTION (optional)
----------------------
Path to a file listing third-party dependencies vendored within this dependency.
Common files: THIRD-PARTY-NOTICES, NOTICE (when it lists vendored deps).

Most dependencies don't need this field. Only fill it in if the dependency
bundles other libraries and has a separate file documenting them.

SOURCE (optional)
-----------------
A permalink URL to the exact version of the source code used. Format:

  For GitHub:  https://github.com/org/repo/tree/<tag-or-commit>
  For GitLab:  https://gitlab.com/org/repo/-/tree/<tag-or-commit>

Examples:
  https://github.com/NVIDIA/cutlass/tree/v4.3.0
  https://github.com/nlohmann/json/tree/v3.12.0

================================================================================
DEPENDENCY CATEGORIES
================================================================================

The import_payload.json contains several types of dependencies:

1. FETCHCONTENT DEPENDENCIES (e.g., cutlass, json, nanobind)
   - Usually auto-populated with license and source
   - If empty, the files are in: cpp/build/_deps/<name>-src/

2. VENDORED DEPENDENCIES (e.g., hedley, picojson, fmt)
   - Code embedded inside another dependency (e.g., hedley is inside json)
   - License/source must be found in the vendored location or upstream project
   - Example: hedley's license is at:
     cpp/build/_deps/json-src/include/nlohmann/thirdparty/hedley/hedley_undef.hpp
     (contains license in header comments, or find it at github.com/nemequ/hedley)

3. SYSTEM PACKAGES (e.g., gcc, glibc, nccl, openmpi)
   - Installed via package manager (RPM/apt)
   - License may be in /usr/share/licenses/<package>/ or /usr/share/doc/<package>/
   - Source URLs are typically the upstream project (e.g., gnu.org for gcc/glibc)

4. SPECIAL CASES
   - tensorrt-llm: This project, no license is needed
   - tensorrt: NVIDIA proprietary, contact @NVIDIA/trt-llm-oss-compliance for proper attribution

================================================================================
COMMON ISSUES
================================================================================

"License file not found"
  → The dependency's license file has a non-standard name or location.
    Search the source directory for files containing "license" or "copying".

"Multiple licenses detected"
  → The auto-detection found multiple license files.
    Manually specify the correct path to the dependency's license, or
    concatenate them into one file if necessary.
    If a detected license is for a vendored dependency, add it to the
    "attribution" field.

"Files mapped to unknown dependency"
  → The auto-detection could not identify the dependency for some files.
    1. Look at the file paths to identify the dependency
    2. Check if a YAML pattern file exists in scripts/attribution/scan/metadata/
    3. Either add a new YAML pattern file, or manually edit file_mappings.json
       to move the files to the correct dependency
