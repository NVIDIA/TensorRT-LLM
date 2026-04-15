#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Deep-clean stale packages from site-packages before pip install.

pip installs dependency packages before uninstalling the old version of
the target package.  For packages like nvidia-cutlass-dsl whose deps
(nvidia-cutlass-dsl-libs-base) write into the same directory tree, the
uninstall step removes files that deps just installed, leaving the
package broken.  See the CUTLASS DSL quick-start docs for background.

Run this script before ``pip install`` to uninstall cleanly first.

Usage:
    # Clean known problematic packages, then install
    python3 scripts/clean_site_packages.py
    pip install -r requirements.txt

    # Clean specific packages
    python3 scripts/clean_site_packages.py nvidia-cutlass-dsl
    python3 scripts/clean_site_packages.py nvidia-cutlass-dsl flashinfer

    # Dry-run (show what would be removed)
    python3 scripts/clean_site_packages.py --dry-run

    # Clean all packages whose installed version differs from requirements.txt
    python3 scripts/clean_site_packages.py --from-requirements requirements.txt
"""

import argparse
import glob
import os
import re
import shutil
import site
import subprocess
import sys

# Packages known to break during pip in-place upgrades due to shared
# namespace directories between the meta-wheel and its deps.
KNOWN_PROBLEMATIC = [
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-cu13",
]


def get_site_packages_dir():
    """Return the primary site-packages directory."""
    dirs = site.getsitepackages()
    for d in dirs:
        if "site-packages" in d and os.path.isdir(d):
            return d
    return dirs[0] if dirs else None


def get_installed_version(package_name):
    """Return installed version of a package, or None if not installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def parse_requirements(req_file):
    """Parse requirements.txt and return {package_name: version_spec}."""
    packages = {}
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            match = re.match(r"^([a-zA-Z0-9_-]+)\s*==\s*([^\s;#]+)", line)
            if match:
                packages[match.group(1).lower()] = match.group(2)
    return packages


def find_package_dirs(site_dir, package_name):
    """Find all directories/files in site-packages related to a package."""
    normalized = package_name.replace("-", "_")
    pattern = os.path.join(site_dir, f"{normalized}*")
    return glob.glob(pattern)


def clean_package(site_dir, package_name, dry_run=False):
    """Uninstall a package via pip, then remove any leftover files."""
    installed = get_installed_version(package_name)
    if installed:
        print(f"  pip uninstall {package_name} (version {installed})")
        if not dry_run:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", package_name],
                capture_output=True,
            )

    leftover = find_package_dirs(site_dir, package_name)
    for path in leftover:
        if dry_run:
            print(f"  would remove: {path}")
        else:
            print(f"  removing: {path}")
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    if not installed and not leftover:
        print(f"  {package_name}: not installed, nothing to clean")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Deep-clean stale packages from site-packages")
    parser.add_argument(
        "packages", nargs="*", help="Package names to clean (default: known problematic packages)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be removed without deleting"
    )
    parser.add_argument(
        "--from-requirements",
        metavar="FILE",
        help="Clean packages whose installed version differs from the file",
    )
    args = parser.parse_args()

    site_dir = get_site_packages_dir()
    if not site_dir:
        print("ERROR: Could not find site-packages directory", file=sys.stderr)
        sys.exit(1)
    print(f"site-packages: {site_dir}")

    packages_to_clean = []

    if args.from_requirements:
        required = parse_requirements(args.from_requirements)
        for name, version in required.items():
            installed = get_installed_version(name)
            if installed and installed != version:
                print(f"  {name}: installed={installed}, required={version} -> will clean")
                packages_to_clean.append(name)
        if not packages_to_clean:
            print("All pinned packages match installed versions, nothing to clean.")
            return
    elif args.packages:
        packages_to_clean = args.packages
    else:
        packages_to_clean = KNOWN_PROBLEMATIC

    if args.dry_run:
        print("DRY RUN — no files will be deleted\n")

    cleaned = 0
    for pkg in packages_to_clean:
        print(f"\nCleaning: {pkg}")
        if clean_package(site_dir, pkg, dry_run=args.dry_run):
            cleaned += 1

    print(f"\nDone. Cleaned {cleaned} package(s).")


if __name__ == "__main__":
    main()
