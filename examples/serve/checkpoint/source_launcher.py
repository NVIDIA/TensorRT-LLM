# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Start one isolated warm source from a locally available, operator-pinned S3 image."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import deployment
from docker_adapter import LABEL, LAUNCHER, PORT, Adapter
from engine_client import EngineClient, _remaining, generate_tokens

_KEY = "TRTLLM_RUNTIME_CONTROL_API_KEY"
_SERVED_MODEL = "resident-tp1"
_IMAGE = re.compile(r"(?:[a-z0-9][a-z0-9./:_-]*@)?sha256:[a-f0-9]{64}")
_IMAGE_INFO = (
    '{"id":{{json .Id}},"os":{{json .Os}},"arch":{{json .Architecture}},'
    '"volumes":{{json (index .Config "Volumes")}}}'
)
_ENV = (
    "SNAPSHOT_CONTROL_DIR=/snapshot-control",
    "HF_HUB_OFFLINE=1",
    "TORCHINDUCTOR_COMPILE_THREADS=1",
    "OMPI_MCA_pml=ob1",
    "OMPI_MCA_btl=tcp,self",
    "OMPI_MCA_btl_tcp_if_include=lo",
    "OMPI_MCA_oob_tcp_if_include=lo",
    "PRTE_MCA_oob_tcp_if_include=lo",
    "OMPI_ALLOW_RUN_AS_ROOT=1",
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
    "TLLM_WORKER_USE_SINGLE_PROCESS=0",
    "TRTLLM_NO_USAGE_STATS=1",
    "TLLM_CUDA_GRAPH_STRICT_BUFFERS=1",
    "TLLM_NCCL_SYMMETRIC_ZERO_COPY=0",
    "TRTLLM_PROXY_BATCH_REQUESTS=0",
    "TRTLLM_GPU_KEEPALIVE=0",
)


def launch(
    config: deployment.Deployment,
    template: str,
    image: str,
    key: str,
    budget: float,
    *,
    probe_out: Path | None = None,
) -> str:
    """Return the ready source's full Docker ID. Never retry create/start or adopt an old source."""
    if not re.fullmatch("[0-9a-f]{32}", template) or not _IMAGE.fullmatch(image):
        raise ValueError("Expected a new 32-hex template and a digest-pinned local workload image")
    if not key.strip() or any(c in key for c in "\x00\n\r"):
        raise ValueError(f"Expected a nonempty single-line {_KEY}")
    if not math.isfinite(budget) or not 0 < budget <= 3600:
        raise ValueError("Budget must be positive and at most 3600 seconds")
    if probe_out is not None:
        if os.path.lexists(probe_out):
            raise FileExistsError(probe_out)
        if not probe_out.parent.is_dir():
            raise NotADirectoryError(probe_out.parent)
    deadline = time.monotonic() + budget
    deployment.check_health(config, deadline=deadline)
    host = Adapter(config, template, template, deadline)
    info = json.loads(host.docker("image", "inspect", "--format", _IMAGE_INFO, image))
    if (
        not re.fullmatch("sha256:[0-9a-f]{64}", info["id"])
        or ("@" not in image and info["id"] != image)
        or (info["os"], info["arch"]) != ("linux", "amd64")
        or info["volumes"]
    ):
        raise ValueError("Expected local linux/amd64 image with no declared volumes")
    if os.path.lexists(host.journal) or os.path.lexists(
        config.root / "checkpoints/artifacts" / template
    ):
        raise FileExistsError("Template already has checkpoint state")
    # This also fences a retry after a lost Docker reply. Leave it empty for bind_source.
    host.control.mkdir(mode=0o700)
    cid = (
        host.docker(
            "create",
            "--pull=never",
            "--name",
            f"warp-source-{template}",
            "--label",
            f"{LABEL}={template}",
            "--network=none",
            "--ipc=private",
            "--restart=no",
            "--no-healthcheck",
            "--shm-size=1g",
            "--cpus=16",
            "--memory=128g",
            "--memory-swap=128g",
            "--gpus",
            f"device={config.gpu}",
            "--mount",
            f"type=bind,src={host.control},dst=/snapshot-control",
            "--mount",
            f"type=bind,src={config.root}/interpose,dst=/tmp/snapshot-cuda,readonly",
            "--env",
            f"NVIDIA_VISIBLE_DEVICES={config.gpu}",
            *(arg for setting in _ENV for arg in ("--env", setting)),
            "--env-file",
            "/dev/stdin",
            "--entrypoint",
            LAUNCHER,
            info["id"],
            "--library",
            "/tmp/snapshot-cuda/libcuinterpose.so",
            "--",
            "trtllm-serve",
            "/model",
            "--served_model_name",
            _SERVED_MODEL,
            "--config",
            "/etc/trtllm/resident-tp1.yaml",
            "--enable_checkpoint_mode",
            "--num_serve_frontends",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            data=f"{_KEY}={key}\n".encode(),
        )
        .strip()
        .decode("ascii")
    )
    if not re.fullmatch("[0-9a-f]{64}", cid):
        raise RuntimeError("No full source ID; reconcile the named container before retrying")
    host.docker("start", cid)
    engine = EngineClient(PORT, key, container=cid)
    while True:
        if host.profile(cid, source=True)["image"] != info["id"]:
            raise ValueError("Source image differs from the inspected pin")
        try:
            engine._exchange("/ready", min(deadline, time.monotonic() + 5))
            break
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            time.sleep(min(1, _remaining(deadline)))
    host.control_binding(cid)
    engine._status("running", deadline)
    if probe_out is not None:
        prompt = [17]
        expected = generate_tokens(cid, PORT, _SERVED_MODEL, prompt, 4, _remaining(deadline))
        _remaining(deadline)
        deployment._write(probe_out, {"prompt_token_ids": prompt, "expected_token_ids": expected})
    return cid


def main() -> None:
    """Launch a fixed-profile source and print only its immutable ID on success."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--image", required=True, help="Local S3 workload image pinned by SHA-256")
    parser.add_argument(
        "--probe-out", type=Path, help="Create a new deterministic reference JSON file"
    )
    parser.add_argument("--budget", type=float, default=600, help="Total startup budget in seconds")
    args = parser.parse_args()
    print(
        launch(
            deployment.load_deployment(args.deployment),
            args.template,
            args.image,
            os.environ.get(_KEY, ""),
            args.budget,
            probe_out=args.probe_out,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f"Source launch failed: {exc}")
