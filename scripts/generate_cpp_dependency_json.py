"""Generate an SPDX SBOM of third-party source URLs used in the CMake build.

This script produces a record of third-party dependencies exactly as consumed
during the build. Each dependency is mirrored to the GitLab OSS components group
(https://gitlab.com/nvidia/tensorrt-llm/oss-components), and the resulting
third-party-sources.json (an SPDX 2.3 JSON document) is copied into the
container image so that source references are distributed alongside build
artifacts, satisfying open-source license obligations in a traceable and
auditable way, and so the document can be ingested directly by Black Duck.
"""

import argparse
import json
import logging
import os
import pathlib
import re
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


GITLAB_OSS_GROUP = "nvidia/tensorrt-llm/oss-components"
GITLAB_API_BASE = "https://gitlab.com/api/v4"

REPO_URL_OVERWRITE = {"deep_ep_download": "https://github.com/deepseek-ai/DeepEP"}

_FETCH_CONTENT_JSON = pathlib.Path(__file__).parent.parent / "3rdparty" / "fetch_content.json"

_NOASSERTION = "NOASSERTION"


def _load_fetch_content_index() -> dict[str, dict]:
    """Return a name->entry mapping from fetch_content.json."""
    data = json.loads(_FETCH_CONTENT_JSON.read_text())
    return {dep["name"]: dep for dep in data.get("dependencies", [])}


def get_source_info(
    deps_dir: pathlib.Path,
    package_name: str,
    fetch_content_index: dict[str, dict] | None = None,
) -> dict[str, str]:
    """Return {'url': ..., 'tag': ...} for package_name.

    Read directly from fetch_content.json.
    """
    index = fetch_content_index or _load_fetch_content_index()
    dep = index.get(package_name, {})
    repo = dep.get("git_repository", "").replace("${github_base_url}", "https://github.com")
    return {"url": repo, "tag": dep.get("git_tag", "")}


def check_oss_components(package_name: str) -> tuple[str, int] | None:
    """Return (web_url, project_id) if package_name exists in oss-components, else None."""
    project_path = urllib.parse.quote(f"{GITLAB_OSS_GROUP}/{package_name}", safe="")
    url = f"{GITLAB_API_BASE}/projects/{project_path}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            return data["web_url"], data["id"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def commit_exists_in_project(project_id: int, ref: str) -> bool:
    """Return True if ref (tag or commit SHA) exists in the GitLab project."""
    encoded_ref = urllib.parse.quote(ref, safe="")
    url = f"{GITLAB_API_BASE}/projects/{project_id}/repository/commits/{encoded_ref}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as resp:
            resp.read()
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def get_namespace_id() -> int:
    """Return the numeric namespace ID for GITLAB_OSS_GROUP."""
    group_path = urllib.parse.quote(GITLAB_OSS_GROUP, safe="")
    url = f"{GITLAB_API_BASE}/groups/{group_path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
        return data["id"]


def create_oss_component(
    package_name: str, namespace_id: int, upstream_url: str
) -> tuple[str, int]:
    """Create a new project under oss-components with a pull mirror and return (web_url, project_id)."""
    payload = json.dumps(
        {
            "name": package_name,
            "namespace_id": namespace_id,
            "visibility": "public",
            "import_url": upstream_url,
            "mirror": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{GITLAB_API_BASE}/projects",
        data=payload,
        headers={"PRIVATE-TOKEN": TOKEN, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
        return data["web_url"], data["id"]


def wait_for_mirror(project_id: int, poll_interval: int = 10, timeout: int = 300) -> None:
    """Poll until the project mirror import finishes; return False if it times out."""
    url = f"{GITLAB_API_BASE}/projects/{project_id}"
    req = urllib.request.Request(url)
    deadline = time.monotonic() + timeout
    while True:
        with urllib.request.urlopen(req) as resp:
            status = json.loads(resp.read()).get("import_status")
        if status == "finished":
            return
        if status == "failed":
            raise RuntimeError(f"Mirror import failed for project {project_id}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Mirror import for project {project_id} still {status!r} after {timeout}s"
            )
        logger.info("Mirror import status: %s, retrying in %ds...", status, poll_interval)
        time.sleep(poll_interval)


def _github_purl(upstream_url: str, tag: str) -> str | None:
    """Return a GitHub purl (for Black Duck component matching) or None if not applicable."""
    if not upstream_url or "github.com" not in upstream_url:
        return None
    path = upstream_url.split("github.com/", 1)[-1].rstrip("/").removesuffix(".git")
    parts = path.split("/")
    if len(parts) != 2:
        return None
    owner, repo = parts
    return f"pkg:github/{owner}/{repo}@{tag}" if tag else f"pkg:github/{owner}/{repo}"


def _spdx_package(name: str, tag: str, upstream_url: str, mirror_url: str) -> dict:
    spdx_id = "SPDXRef-Package-" + re.sub(r"[^a-zA-Z0-9.-]", "-", name)
    download_location = f"git+{mirror_url}@{tag}" if tag else (mirror_url or _NOASSERTION)
    package = {
        "SPDXID": spdx_id,
        "name": name,
        "versionInfo": tag or _NOASSERTION,
        "downloadLocation": download_location,
        "filesAnalyzed": False,
        "licenseConcluded": _NOASSERTION,
        "licenseDeclared": _NOASSERTION,
        "copyrightText": _NOASSERTION,
    }
    purl = _github_purl(upstream_url, tag)
    if purl:
        package["externalRefs"] = [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": purl,
            }
        ]
    return package


def build_spdx_document(third_party_packages: list[dict]) -> dict:
    """Build an SPDX 2.3 JSON document describing the resolved third-party packages."""
    root_spdx_id = "SPDXRef-Package-tensorrt-llm-cpp-deps"
    packages = [
        {
            "SPDXID": root_spdx_id,
            "name": "tensorrt-llm-cpp-third-party-dependencies",
            "versionInfo": _NOASSERTION,
            "downloadLocation": _NOASSERTION,
            "filesAnalyzed": False,
            "licenseConcluded": _NOASSERTION,
            "licenseDeclared": _NOASSERTION,
            "copyrightText": _NOASSERTION,
        }
    ]
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": root_spdx_id,
        }
    ]
    for pkg in third_party_packages:
        spdx_package = _spdx_package(
            pkg["name"], pkg["tag"], pkg["upstream_url"], pkg["mirror_url"]
        )
        packages.append(spdx_package)
        relationships.append(
            {
                "spdxElementId": root_spdx_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": spdx_package["SPDXID"],
            }
        )

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "tensorrt-llm-cpp-third-party-sources",
        "documentNamespace": (
            f"https://github.com/NVIDIA/TensorRT-LLM/spdxdocs/tensorrt-llm-cpp-deps-{uuid.uuid4()}"
        ),
        "creationInfo": {
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "creators": ["Tool: generate_cpp_dependency_json.py"],
        },
        "packages": packages,
        "relationships": relationships,
    }


def main():
    global TOKEN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deps-dir",
        type=pathlib.Path,
        required=True,
        help="Path to the third party dependencies directory, e.g. ${CMAKE_BINARY_DIR}/_deps",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Path to the output directory where third party sources will be copied",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITLAB_TOKEN"),
        help="GitLab private token (defaults to $GITLAB_TOKEN env var)",
    )

    args = parser.parse_args()
    TOKEN = args.token

    src_dirs = list(sorted(args.deps_dir.glob("*-src")))
    if not src_dirs:
        raise ValueError(f"No source directories found in {args.deps_dir}")

    namespace_id: int | None = None
    third_party_packages = []

    for src_dir in src_dirs:
        package_name = src_dir.name.removesuffix("-src")
        source_info = get_source_info(args.deps_dir, package_name)
        if package_name in REPO_URL_OVERWRITE:
            source_info["url"] = REPO_URL_OVERWRITE[package_name]
        logger.info(
            "%s -> upstream url=%s tag=%s", package_name, source_info["url"], source_info["tag"]
        )
        result = check_oss_components(package_name)
        if result is not None:
            oss_url, project_id = result
            logger.info("%s -> found in oss-components: %s", package_name, oss_url)
        elif not TOKEN:
            logger.warning(
                "%s -> NOT found in oss-components and no GITLAB_TOKEN provided; "
                "skipping mirror creation, keeping upstream url",
                package_name,
            )
            third_party_packages.append(
                {
                    "name": package_name,
                    "tag": source_info["tag"],
                    "upstream_url": source_info["url"],
                    "mirror_url": source_info["url"],
                }
            )
            continue
        else:
            logger.info("%s -> NOT found in oss-components, creating repo", package_name)
            if namespace_id is None:
                namespace_id = get_namespace_id()
            oss_url, project_id = create_oss_component(
                package_name, namespace_id, source_info["url"]
            )
            logger.info("%s -> created: %s", package_name, oss_url)
            logger.info("%s -> waiting for mirror import to finish", package_name)
            wait_for_mirror(project_id)
            logger.info("%s -> mirror import finished", package_name)

        tag = source_info["tag"]
        if tag and commit_exists_in_project(project_id, tag):
            logger.info("%s -> ref %r confirmed in oss-components, updating url", package_name, tag)
            third_party_packages.append(
                {
                    "name": package_name,
                    "tag": tag,
                    "upstream_url": source_info["url"],
                    "mirror_url": oss_url,
                }
            )
        else:
            logger.warning(
                "%s -> ref %r not found in oss-components repo, keeping upstream url",
                package_name,
                tag,
            )

    spdx_document = build_spdx_document(third_party_packages)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "third-party-sources.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spdx_document, f, indent=2)
        f.write("\n")
    logger.info("Wrote SPDX SBOM to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
