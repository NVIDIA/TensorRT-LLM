"""Generate a JSON manifest of third-party source URLs used in the CMake build.

This script produces a record of third-party dependencies exactly as consumed
during the build. Each dependency is mirrored to the GitLab OSS components group
(https://gitlab.com/nvidia/tensorrt-llm/oss-components), and the resulting
third-party-sources.json is copied into the container image so that source
references are distributed alongside build artifacts, satisfying open-source
license obligations in a traceable and auditable way.
"""

import argparse
import json
import logging
import os
import pathlib
import time
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


GITLAB_OSS_GROUP = "nvidia/tensorrt-llm/oss-components"
GITLAB_API_BASE = "https://gitlab.com/api/v4"

REPO_URL_OVERWRITE = {"deep_ep_download": "https://github.com/deepseek-ai/DeepEP"}

_FETCH_CONTENT_JSON = pathlib.Path(__file__).parent.parent / "3rdparty" / "fetch_content.json"


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
    fetch_content = json.loads(_FETCH_CONTENT_JSON.read_text())
    dep_source_index = {dep["name"]: dep for dep in fetch_content.get("dependencies", [])}
    third_party_source_list = []

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
            if package_name not in dep_source_index:
                logger.warning("%s -> not found in fetch_content.json, skipping", package_name)
                continue
            third_party_source_list.append(
                {**dep_source_index[package_name], "git_repository": oss_url}
            )
        else:
            logger.warning(
                "%s -> ref %r not found in oss-components repo, keeping upstream url",
                package_name,
                tag,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "third-party-sources.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(third_party_source_list, f, indent=2)
        f.write("\n")
    logger.info("Wrote oss fetch_content to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
