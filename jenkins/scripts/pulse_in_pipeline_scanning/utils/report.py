from utils.es import get_latest_license_preapproved_container_deps

from .common import load_json

HIGH_SEVERITY = frozenset({"Critical", "High"})


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
    """Deduplicate by (package, type), merging license lists."""
    seen = {}
    for e in contents:
        key = (e["package"], e["type"])
        if key not in seen:
            seen[key] = dict(e)
            seen[key]["licenses"] = sorted(set(e["licenses"]))
        else:
            seen[key]["licenses"] = sorted(set(seen[key]["licenses"]) | set(e["licenses"]))
    return seen


def get_vulns(release_path):
    release_data = load_json(release_path)

    release_vulns = dedup_vulns(release_data["vulnerabilities"])
    introduced_vulns = [v for _, v in release_vulns.items() if v.get("severity") in HIGH_SEVERITY]

    return introduced_vulns


def get_preapproved_deps_map(scan_type):
    preapproved_deps = get_latest_license_preapproved_container_deps(scan_type)
    map_preapproved_deps = {}
    for item in preapproved_deps:
        # Key by (name, type) so approvals are type-specific.
        # Entries without s_package_type (approved before this field existed) use None
        # and act as a wildcard: is_preapproved() treats (name, None) as matching any type.
        key = (item["s_package_name"], item.get("s_package_type"))
        map_preapproved_deps[key] = True

    return map_preapproved_deps


def is_preapproved(map_preapproved_deps, package_name, package_type):
    """Return True if (package_name, package_type) is covered by the preapproval map.

    Checks exact (name, type) match first, then falls back to (name, None) for
    legacy approvals recorded before the type field was added.
    """
    return (package_name, package_type) in map_preapproved_deps or (
        package_name,
        None,
    ) in map_preapproved_deps


def diff_licenses(scan_type, release_path, base_path=None):
    map_preapproved_deps = get_preapproved_deps_map(scan_type)

    release_data = load_json(release_path)
    release_pkgs = dedup_licenses(release_data["contents"])

    base_pkgs = {}
    if base_path:
        base_data = load_json(base_path)
        base_pkgs = dedup_licenses(base_data["contents"])

    introduced_licenses = [
        v
        for k, v in release_pkgs.items()
        if (k not in base_pkgs) and not is_preapproved(map_preapproved_deps, k[0], k[1])
    ]

    introduced_licenses.sort(key=lambda e: (e["package"], e["version"]))

    return introduced_licenses
