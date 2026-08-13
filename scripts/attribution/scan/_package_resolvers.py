import os
import re
import subprocess
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version
from typing import Any, Dict, List, Optional, Set, Tuple

from _constants import LICENSE_PATTERNS, NOTICE_ATTRIBUTION_PATTERNS


def _find_matching_files(directory: str, patterns: Set[str]) -> List[str]:
    """Find files in directory matching patterns (case-insensitive, non-recursive)."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and any(fnmatch(f.lower(), p) for p in patterns)
    )


def _normalize_package_name(package: str) -> str:
    """Normalize package names for consistent dependency tracking."""
    cuda_related = (
        "cuda",
        "cublas",
        "curand",
        "cusolver",
        "cusparse",
        "cufft",
        "nvjitlink",
        "nvptxcompiler",
        "culibos",
    )

    # Remove CUDA version suffixes (e.g., cuda-cccl-12-9 → cuda-cccl, nvptxcompiler-13-1 → nvptxcompiler)
    match = re.match(r"^(.+?)-(\d+)-(\d+)$", package)
    if match and any(x in match.group(1) for x in cuda_related):
        package = match.group(1)

    # Strip other common suffixes and prefixes
    for suffix in ("-devel", "-dev"):
        if package.endswith(suffix):
            package = package[: -len(suffix)]
            break
    if package.startswith("lib"):
        package = package[3:]

    # CUDA-related packages -> cuda
    if package.startswith("cuda-") or package in cuda_related:
        return "cuda"
    elif package.startswith("glibc-"):
        return "glibc"
    elif package in {"kernel-headers", "linux-headers", "linux-libc"}:
        return "linux"
    elif package in {"ibverbs"}:
        return "rdma-core"
    elif package in {"zmq3"}:
        return "zeromq"
    elif package.startswith("gcc-toolset-"):
        return "gcc"

    # Ubuntu-specific: versioned GCC packages (gcc-13, g++-13, libgcc-13) -> gcc
    elif re.match(r"^(gcc|g\+\+)-(\d+)$", package):
        return "gcc"

    # Ubuntu-specific: versioned Python packages (python3.12, python3.10) -> python
    elif re.match(r"^python(\d+\.\d+)$", package):
        return "cpython"

    # Ubuntu-specific: libstdc++ -> stdc++-N -> gcc
    elif re.match(r"^stdc\+\+(-\d+)?$", package):
        return "gcc"

    # Ubuntu-specific: libc6 becomes c6 after stripping lib prefix -> glibc
    elif re.match(r"^c\d+$", package):
        return "glibc"

    return package


def _normalize_version(package: str, version: str) -> str:
    """Normalize version strings for specific packages."""
    # CUDA: shorten to first two parts (e.g., 13.1.80-1 → 13.1)
    if package == "cuda" and version:
        match = re.match(r"^(\d+\.\d+)", version)
        if match:
            return match.group(1)
    return version


def _normalize_package(raw_package: str, version: str) -> Tuple[str, str, str]:
    """Normalize the name and version strings for specific packages."""
    normalized_pkg = _normalize_package_name(raw_package)
    return normalized_pkg, _normalize_version(normalized_pkg, version), raw_package


class Resolver(ABC):
    """Base class for dependency resolvers."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    @abstractmethod
    def get_package(self, file_path: str) -> Optional[Tuple[str, str, str]]:
        """Return (dependency_name, version, raw_package) or None."""

    def get_license_path(self, key: str) -> str:
        """Return license path(s) for a package/file. Override in subclasses."""
        return ""


class DpkgResolver(Resolver):
    """Resolves files using dpkg-query (Debian/Ubuntu)."""

    # Patterns that indicate a file is NOT from a dpkg-managed package
    # These files should be skipped to avoid expensive dpkg-query calls
    SKIP_PATTERNS = (
        "/_deps/",  # CMake FetchContent dependencies
        "/site-packages/",  # Python packages
        "/dist-packages/",  # Python packages (system)
        "/3rdparty/",  # Vendored third-party code
        "/third_party/",  # Vendored third-party code
        "/external/",  # External dependencies
        "/build/",  # Build artifacts (within project)
        "/opt/",  # Optional packages (usually not dpkg-managed)
    )

    def __init__(self):
        super().__init__()
        # Track which packages we've already cached all files for
        self._cached_packages: Set[str] = set()
        # Negative cache: directories known to NOT contain dpkg-managed files
        self._non_dpkg_dirs: Set[str] = set()

    def _cache_all_package_files(self, raw_package: str, result: Tuple[str, str, str]) -> None:
        """Cache all files belonging to a package for fast future lookups."""
        if raw_package in self._cached_packages:
            return

        try:
            # Get all files belonging to this package
            proc = subprocess.run(
                ["dpkg-query", "-L", raw_package],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                for line in proc.stdout.strip().split("\n"):
                    file_path = line.strip()
                    if file_path and file_path not in self._cache:
                        self._cache[file_path] = result
                self._cached_packages.add(raw_package)
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

    def get_package(self, file_path: str) -> Optional[Tuple[str, str, str]]:
        if file_path in self._cache:
            return self._cache[file_path]

        # Skip files that are clearly not from system packages
        # This avoids expensive dpkg-query subprocess calls
        if any(pattern in file_path for pattern in self.SKIP_PATTERNS):
            self._cache[file_path] = None
            return None

        # Check negative directory cache - if we know this directory has no dpkg files
        dir_path = os.path.dirname(file_path)
        if dir_path in self._non_dpkg_dirs:
            self._cache[file_path] = None
            return None

        result = None
        try:
            proc = subprocess.run(
                ["dpkg-query", "-S", file_path], capture_output=True, text=True, timeout=5
            )
            if proc.returncode == 0 and ":" in proc.stdout:
                raw_package = proc.stdout.split(":", 1)[0].split(":")[0]
                version = ""
                ver_proc = subprocess.run(
                    ["dpkg-query", "-W", "-f=${Version}", raw_package],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if ver_proc.returncode == 0:
                    version = ver_proc.stdout.strip()
                result = _normalize_package(raw_package, version)
                # Cache ALL files belonging to this package for fast future lookups
                self._cache_all_package_files(raw_package, result)
            else:
                # File not found in any dpkg package - cache the directory as non-dpkg
                self._non_dpkg_dirs.add(dir_path)
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        self._cache[file_path] = result
        return result

    # Package-specific license preambles (e.g., for dual-licensed packages)
    LICENSE_PREAMBLES: Dict[str, str] = {
        "libibverbs-dev": (
            "NVIDIA actively chooses the OpenIB.org BSD (MIT variant) license "
            "to apply to files with this copyright and license.\n\n"
        ),
    }

    def get_license_path(self, raw_package: str) -> str:
        cache_key = f"license:{raw_package}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = ""
        copyright_path = f"/usr/share/doc/{raw_package}/copyright"
        if os.path.exists(copyright_path):
            try:
                with open(copyright_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                refs = set(re.findall(r"/usr/share/common-licenses/([A-Za-z0-9._-]+)", content))
                if refs:
                    combined = f"/tmp/dpkg_license_{raw_package}.txt"
                    with open(combined, "w", encoding="utf-8") as out:
                        # Add package-specific preamble if defined
                        if raw_package in self.LICENSE_PREAMBLES:
                            out.write(self.LICENSE_PREAMBLES[raw_package])
                        out.write(content)
                        for name in sorted(refs):
                            # Strip trailing punctuation (e.g., "LGPL." -> "LGPL")
                            name = name.rstrip(".,;:!?")
                            path = f"/usr/share/common-licenses/{name}"
                            # Resolve symlinks (e.g., GPL -> GPL-3)
                            resolved_path = os.path.realpath(path)
                            if os.path.isfile(resolved_path):
                                out.write(f"\n\n{'=' * 60}\nContents of {path}:\n{'=' * 60}\n\n")
                                with open(
                                    resolved_path, "r", encoding="utf-8", errors="replace"
                                ) as lf:
                                    out.write(lf.read())
                    result = combined
                else:
                    result = copyright_path
            except (IOError, OSError):
                pass

        self._cache[cache_key] = result
        return result


class RpmResolver(Resolver):
    """Resolves files using rpm (RHEL/CentOS)."""

    def get_package(self, file_path: str) -> Optional[Tuple[str, str, str]]:
        if file_path in self._cache:
            return self._cache[file_path]

        result = None
        if os.path.isabs(file_path) and os.path.exists(file_path):
            try:
                proc = subprocess.run(
                    ["rpm", "-qf", "--qf", "%{NAME}\t%{VERSION}-%{RELEASE}", file_path],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0:
                    parts = proc.stdout.strip().split("\t")
                    if len(parts) >= 2:
                        result = _normalize_package(parts[0], parts[1])
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass

        self._cache[file_path] = result
        return result

    def get_license_path(self, raw_package: str) -> str:
        cache_key = f"license:{raw_package}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        packages = [raw_package]
        for suffix in ("-devel", "-dev"):
            if raw_package.endswith(suffix):
                packages.append(raw_package[: -len(suffix)])
                break

        paths = []
        for pkg in packages:
            for base in [f"/usr/share/licenses/{pkg}", f"/usr/share/doc/{pkg}"]:
                paths.extend(_find_matching_files(base, LICENSE_PATTERNS))

        result = " OR ".join(sorted(set(paths)))
        self._cache[cache_key] = result
        return result


class PythonPackageResolver(Resolver):
    """Resolves files in Python site-packages/dist-packages."""

    # Packages where NOTICE files should go to attribution instead of license
    NOTICE_TO_ATTRIBUTION = {"torch"}
    PATTERNS = [re.compile(r"/(site|dist)-packages/([^/]+)/")]

    def get_package(self, file_path: str) -> Optional[Tuple[str, str, str]]:
        if file_path in self._cache:
            return self._cache[file_path]

        result = None
        for pattern in self.PATTERNS:
            match = pattern.search(file_path)
            if match:
                pkg = re.sub(r"[-.].*\.(dist-info|egg-info)$", "", match.group(2))
                version = ""
                for name in [pkg.replace("_", "-"), pkg]:
                    try:
                        version = get_package_version(name)
                        break
                    except PackageNotFoundError:
                        pass
                result = (pkg, version, pkg)
                break

        self._cache[file_path] = result
        return result

    def _get_all_license_files(self, file_path: str, license_patterns: Set[str]) -> List[str]:
        """Get all license-related files for a Python package."""
        paths = []
        for pattern in self.PATTERNS:
            match = pattern.search(file_path)
            if match:
                pkg_name = match.group(2)
                base_path = file_path[: match.end() - 1]
                site_dir = file_path[: file_path.find("/", match.start() + 1)]

                paths.extend(_find_matching_files(base_path, license_patterns))

                if os.path.isdir(site_dir):
                    for entry in os.listdir(site_dir):
                        if entry.endswith(".dist-info") and any(
                            entry.startswith(f"{n}-")
                            for n in [
                                pkg_name,
                                pkg_name.replace("-", "_"),
                                pkg_name.replace("_", "-"),
                            ]
                        ):
                            dist_path = os.path.join(site_dir, entry)
                            lic_dir = os.path.join(dist_path, "licenses")
                            if os.path.isdir(lic_dir):
                                paths.extend(_find_matching_files(lic_dir, license_patterns))
                            paths.extend(_find_matching_files(dist_path, license_patterns))
                            break
                break
        return paths

    def get_license_path(self, file_path: str, dependency: str = "") -> str:
        patterns = LICENSE_PATTERNS
        # For specific packages, exclude NOTICE files from license (they go to attribution)
        if dependency in self.NOTICE_TO_ATTRIBUTION:
            patterns = patterns - NOTICE_ATTRIBUTION_PATTERNS
        paths = self._get_all_license_files(file_path, patterns)
        return " OR ".join(paths)

    def get_attribution_path(self, file_path: str, dependency: str = "") -> str:
        """Get attribution files (e.g., NOTICE) for packages that need them separate."""
        if dependency not in self.NOTICE_TO_ATTRIBUTION:
            return ""
        paths = self._get_all_license_files(file_path, NOTICE_ATTRIBUTION_PATTERNS)
        return " OR ".join(paths)
