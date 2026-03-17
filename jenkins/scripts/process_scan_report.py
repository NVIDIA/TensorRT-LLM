# Copyright 2026, NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Diff container scan reports (vulns + licenses) between release and base
# images for each platform (amd64, arm64).
#
# Usage:
#   python3 process_scan_report.py --scan-report-dir <dir> --output-dir <dir>
#
# Expected input layout:
#   <scan-report-dir>/
#     release_amd64_amd64/vulns.json  licenses.json
#     release_arm64_arm64/vulns.json  licenses.json
#     base_amd64_amd64/vulns.json     licenses.json
#     base_arm64_arm64/vulns.json     licenses.json
#
# Output per platform:
#   <output-dir>/vuln_diff_<platform>.json
#   <output-dir>/license_diff_<platform>.json

import argparse
import json
import os

HIGH_SEVERITY = frozenset({"Critical", "High"})

NON_PERMISSIVE_LICENSE_PREFIXES = ("GPL", "LGPL", "GCC GPL", "AGPL", "SSPL", "CPAL", "EUPL")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def is_non_permissive(licenses):
    return any(lic.startswith(NON_PERMISSIVE_LICENSE_PREFIXES) for lic in licenses)


def dedup_vulns(vulns):
    """Deduplicate by (vuln, package_name, package_version), merging package_paths."""
    seen = {}
    for v in vulns:
        key = (v["vuln"], v["package_name"], v["package_version"])
        if key not in seen:
            seen[key] = dict(v)
            seen[key]["package_paths"] = [v["package_path"]]
        else:
            if v["package_path"] not in seen[key]["package_paths"]:
                seen[key]["package_paths"].append(v["package_path"])
    return seen


def dedup_licenses(contents):
    """Deduplicate by (package, version, type), merging license lists."""
    seen = {}
    for e in contents:
        key = (e["package"], e["version"], e["type"])
        if key not in seen:
            seen[key] = dict(e)
            seen[key]["licenses"] = sorted(set(e["licenses"]))
        else:
            seen[key]["licenses"] = sorted(set(seen[key]["licenses"]) | set(e["licenses"]))
    return seen


def diff_vulns(release_path, base_path):
    release_data = load_json(release_path)
    base_data = load_json(base_path)

    release_vulns = dedup_vulns(release_data["vulnerabilities"])
    base_vulns = dedup_vulns(base_data["vulnerabilities"])

    added = [
        v
        for k, v in release_vulns.items()
        if k not in base_vulns and v.get("severity") in HIGH_SEVERITY
    ]
    removed = [v for k, v in base_vulns.items() if k not in release_vulns]

    added.sort(key=lambda v: (v["severity"], v["package_name"], v["vuln"]))
    removed.sort(key=lambda v: (v["severity"], v["package_name"], v["vuln"]))

    def _by_severity(vulns):
        counts = {}
        for v in vulns:
            counts[v["severity"]] = counts.get(v["severity"], 0) + 1
        return counts

    return {
        "release_image": release_data["image_tag"],
        "base_image": base_data["image_tag"],
        "summary": {
            "added_count": len(added),
            "removed_count": len(removed),
            "added_by_severity": _by_severity(added),
            "removed_by_severity": _by_severity(removed),
        },
        "added_in_release": added,
        "removed_from_base": removed,
    }


def diff_licenses(release_path, base_path):
    release_data = load_json(release_path)
    base_data = load_json(base_path)

    release_pkgs = dedup_licenses(release_data["contents"])
    base_pkgs = dedup_licenses(base_data["contents"])

    added = [
        v
        for k, v in release_pkgs.items()
        if k not in base_pkgs and is_non_permissive(v["licenses"])
    ]
    removed = [v for k, v in base_pkgs.items() if k not in release_pkgs]

    changed = []
    for k in release_pkgs:
        if k in base_pkgs:
            r_licenses = release_pkgs[k]["licenses"]
            b_licenses = base_pkgs[k]["licenses"]
            if r_licenses != b_licenses:
                changed.append(
                    {
                        "package": k[0],
                        "version": k[1],
                        "type": k[2],
                        "base_licenses": b_licenses,
                        "release_licenses": r_licenses,
                    }
                )

    added.sort(key=lambda e: (e["package"], e["version"]))
    removed.sort(key=lambda e: (e["package"], e["version"]))
    changed.sort(key=lambda e: (e["package"], e["version"]))

    return {
        "release_image": release_data["image_tag"],
        "base_image": base_data["image_tag"],
        "summary": {
            "added_count": len(added),
            "removed_count": len(removed),
            "license_changes_count": len(changed),
        },
        "added_in_release": added,
        "removed_from_base": removed,
        "license_changes": changed,
    }


def process_platform(scan_dir, platform, output_dir):
    release_dir = os.path.join(scan_dir, f"release_{platform}_{platform}")
    base_dir = os.path.join(scan_dir, f"base_{platform}_{platform}")

    for d in [release_dir, base_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Expected scan directory not found: {d}")

    os.makedirs(output_dir, exist_ok=True)

    vuln_diff = diff_vulns(
        os.path.join(release_dir, "vulns.json"),
        os.path.join(base_dir, "vulns.json"),
    )
    license_diff = diff_licenses(
        os.path.join(release_dir, "licenses.json"),
        os.path.join(base_dir, "licenses.json"),
    )

    vuln_out = os.path.join(output_dir, f"vuln_diff_{platform}.json")
    license_out = os.path.join(output_dir, f"license_diff_{platform}.json")

    with open(vuln_out, "w") as f:
        json.dump(vuln_diff, f, indent=2)
    with open(license_out, "w") as f:
        json.dump(license_diff, f, indent=2)

    print(
        f"[{platform}] Vulnerabilities : "
        f"+{vuln_diff['summary']['added_count']} added, "
        f"-{vuln_diff['summary']['removed_count']} removed  "
        f"| by severity (added): {vuln_diff['summary']['added_by_severity']}"
    )
    print(
        f"[{platform}] Licenses        : "
        f"+{license_diff['summary']['added_count']} pkgs added, "
        f"-{license_diff['summary']['removed_count']} removed, "
        f"{license_diff['summary']['license_changes_count']} license changes"
    )
    print(f"[{platform}] Output          : {vuln_out}, {license_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Diff container scan reports between release and base images."
    )
    parser.add_argument(
        "--scan-report-dir",
        required=True,
        help="Directory containing per-image scan subdirectories",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write diff JSON files")
    args = parser.parse_args()

    for platform in ["amd64", "arm64"]:
        process_platform(args.scan_report_dir, platform, args.output_dir)


if __name__ == "__main__":
    main()
