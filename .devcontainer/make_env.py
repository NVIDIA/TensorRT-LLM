#!/usr/bin/env python3

import json
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

JENKINS_PROPS_PATH = Path("jenkins/current_image_tags.properties")
DEV_CONTAINER_ENV_PATH = Path(".devcontainer/devcontainer.env")
DEV_CONTAINER_USER_ENV_PATH = Path(".devcontainer/devcontainer.env.user")
DOT_ENV_PATH = Path(".devcontainer/.env")
COMPOSE_OVERRIDE_PATH = Path(".devcontainer/docker-compose.override.yml")
COMPOSE_OVERRIDE_EXAMPLE_PATH = Path(
    ".devcontainer/docker-compose.override-example.yml")

HOME_DIR_VAR = "HOME_DIR"
SOURCE_DIR_VAR = "SOURCE_DIR"
DEV_CONTAINER_IMAGE_VAR = "DEV_CONTAINER_IMAGE"
BUILD_LOCAL_VAR = "BUILD_LOCAL"
JENKINS_IMAGE_VAR = "LLM_DOCKER_IMAGE"
LOCAL_HF_HOME_VAR = "LOCAL_HF_HOME"

LOGGER = logging.getLogger("make_env")


def _load_env(env_files: List[Path]) -> Dict[str, str]:
    """Evaluate files using 'sh' and return resulting environment."""
    with TemporaryDirectory("trtllm_make_env") as temp_dir:
        json_path = Path(temp_dir) / 'env.json'
        subprocess.run(
            ("(echo set -a && cat " +
             " ".join(shlex.quote(str(env_file)) for env_file in env_files) +
             " && echo && echo exec /usr/bin/env python3 -c \"'import json; import os; print(json.dumps(dict(os.environ)))'\""
             + f") | sh > {json_path}"),
            shell=True,
            check=True,
        )
        with open(json_path, "r") as f:
            env = json.load(f)
    return env


def _detect_rootless() -> bool:
    proc = subprocess.run("./docker/detect_rootless.sh",
                          capture_output=True,
                          check=True,
                          shell=True)
    return bool(int(proc.stdout.decode("utf-8").strip()))


def _handle_rootless(env_inout: Dict[str, str]):
    is_rootless = _detect_rootless()
    if is_rootless:
        LOGGER.info("Docker Rootless Mode detected.")
        if HOME_DIR_VAR not in env_inout:
            raise ValueError(
                "Docker Rootless Mode requires setting HOME_DIR in devcontainer.env.user"
            )
        if SOURCE_DIR_VAR not in env_inout:
            raise ValueError(
                "Docker Rootless Mode requires setting SOURCE_DIR in devcontainer.env.user"
            )

        # Handle HF_HOME
        if "HF_HOME" in os.environ and "HF_HOME" in env_inout:
            raise ValueError(
                "Docker Rootless Mode requires either not setting HF_HOME at all or overriding it in devcontainer.env.user"
            )
        if env_inout[LOCAL_HF_HOME_VAR].startswith(env_inout["HOME"]):
            env_inout[LOCAL_HF_HOME_VAR] = env_inout[LOCAL_HF_HOME_VAR].replace(
                env_inout["HOME"], env_inout[HOME_DIR_VAR], 1)
    else:
        env_inout[HOME_DIR_VAR] = env_inout["HOME"]
        env_inout[SOURCE_DIR_VAR] = os.getcwd()


def _select_prebuilt_image(env: Dict[str, str]) -> Optional[str]:
    # Jenkins image
    candidate_images: List[str] = [env[JENKINS_IMAGE_VAR]]

    # NGC images
    proc = subprocess.run(
        r"git tag --sort=creatordate --merged=HEAD | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+' | sed -E 's/^v(.*)$/\1/' | tac",
        shell=True,
        capture_output=True,
        check=True,
    )
    for git_tag in proc.stdout.splitlines():
        git_tag = git_tag.strip()
        candidate_images.append(f"nvcr.io/nvidia/tensorrt-llm/devel:{git_tag}")

    # Check image availability
    for candidate_image in candidate_images:
        LOGGER.info(f"Trying image {candidate_image}")

        try:
            subprocess.run(
                f"docker run --rm -it --pull=missing --entrypoint=/bin/true {shlex.quote(candidate_image)}",
                check=True,
                shell=True)
        except subprocess.CalledProcessError:
            continue

        LOGGER.info(f"Using image {candidate_image}")
        return candidate_image

    LOGGER.info("No pre-built image found!")
    return None


def _build_local_image() -> str:
    LOGGER.info("Building container image locally")

    with TemporaryDirectory("trtllm_make_env") as temp_dir:
        log_path = Path(temp_dir) / "build.log"
        subprocess.run(
            f"make -C docker devel_build | tee {shlex.quote(str(log_path))}",
            check=True,
            shell=True,
        )
        with open(log_path) as f:
            build_log = f.read()

    # Handle escaped and actual line breaks
    build_log_lines = re.sub(r"\\\n", " ", build_log).splitlines()
    for build_log_line in build_log_lines:
        tokens = shlex.split(build_log_line)
        if tokens[:3] != ["docker", "buildx", "build"]:
            continue
        token = None
        while tokens and not (token := tokens.pop(0)).startswith("--tag"):
            pass
        if token is None:
            continue
        if token.startswith("--arg="):
            token = token.removeprefix("--arg=")
        else:
            if not tokens:
                continue
            token = tokens.pop(0)
        return token  # this is the image URI
    raise RuntimeError(
        f"Could not parse --tag argument from build log: {build_log}")


def _ensure_compose_override():
    if not COMPOSE_OVERRIDE_PATH.exists():
        LOGGER.info(
            f"Creating initial {COMPOSE_OVERRIDE_PATH} from {COMPOSE_OVERRIDE_EXAMPLE_PATH}"
        )
        COMPOSE_OVERRIDE_PATH.write_bytes(
            COMPOSE_OVERRIDE_EXAMPLE_PATH.read_bytes())


def _update_dot_env(env: Dict[str, str]):
    LOGGER.info(f"Updating {DOT_ENV_PATH}")

    output_lines = [
        "# NOTE: This file is generated by make_env.py, modify devcontainer.env.user instead of this file.\n",
        "\n",
    ]

    for env_key, env_value in env.items():
        if os.environ.get(env_key) == env_value:
            # Only storing differences w.r.t. base env
            continue
        output_lines.append(f"{env_key}=\"{shlex.quote(env_value)}\"\n")

    with open(DOT_ENV_PATH, "w") as f:
        f.writelines(output_lines)


def main():
    env_files = [
        JENKINS_PROPS_PATH,
        DEV_CONTAINER_ENV_PATH,
    ]

    if DEV_CONTAINER_USER_ENV_PATH.exists():
        env_files.append(DEV_CONTAINER_USER_ENV_PATH)

    env = _load_env(env_files)
    _handle_rootless(env_inout=env)

    # Determine container image to use
    image_uri = env.get(DEV_CONTAINER_IMAGE_VAR)
    if image_uri:
        LOGGER.info(f"Using user-provided container image: {image_uri}")
    else:
        build_local = bool(int(
            env[BUILD_LOCAL_VAR].strip())) if BUILD_LOCAL_VAR in env else None
        image_uri = None
        if not build_local:
            image_uri = _select_prebuilt_image(env)
        if image_uri is None:
            if build_local is False:
                raise RuntimeError(
                    "No suitable container image found and local build disabled."
                )
            image_uri = _build_local_image()
            LOGGER.info(f"Using locally built container image: {image_uri}")
        env[DEV_CONTAINER_IMAGE_VAR] = image_uri

    _ensure_compose_override()

    _update_dot_env(env)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception as e:
        LOGGER.error(f"{e.__class__.__name__}: {e}")
        sys.exit(-1)
