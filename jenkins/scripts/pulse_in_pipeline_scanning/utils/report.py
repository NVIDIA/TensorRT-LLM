from utils.es import get_latest_license_preapproved_container_deps

from .common import is_non_permissive, load_json

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
    """Deduplicate by (package, version, type), merging license lists."""
    seen = {}
    for e in contents:
        key = e["package"]
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


def diff_licenses(release_path, base_path):
    release_data = load_json(release_path)
    base_data = load_json(base_path)
    preapproved_deps = get_latest_license_preapproved_container_deps()
    map_preapproved_deps = {}
    for item in preapproved_deps:
        map_preapproved_deps[item["s_package_name"]] = True

    release_pkgs = dedup_licenses(release_data["contents"])
    base_pkgs = dedup_licenses(base_data["contents"])

    introduced_licenses = [
        v
        for k, v in release_pkgs.items()
        if (
            (k not in base_pkgs)
            and (k not in map_preapproved_deps)
            and is_non_permissive(v["licenses"])
        )
    ]

    introduced_licenses.sort(key=lambda e: (e["package"], e["version"]))

    return introduced_licenses
