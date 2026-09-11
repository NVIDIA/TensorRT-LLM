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
"""Real-GPU ModelExpress donor/receiver qualification tests."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn
from urllib.parse import urlparse

import pytest
import torch
from defs.trt_test_alternative import cleanup_process_tree, popen
from packaging.version import InvalidVersion, Version

from tensorrt_llm._torch.weight_sharing import ArtifactIdentity

_WORKER = Path(__file__).with_name("mx_e2e_worker.py")
_WEIGHT_SUFFIXES = frozenset({".bin", ".ckpt", ".gguf", ".pt", ".pth", ".safetensors"})
_RECEIVER_FAILURE_MARKERS = (
    "falling back to disk",
    "partial fallback",
    "size mismatch",
    "still missing",
    "mx p2p transfer failed",
    "source sourceidentity incompatible",
    "sourceidentity mismatch",
    "invalid sourceidentity",
)
_RDMA_TRANSFER_PATTERN = re.compile(
    r"\[Worker\s+(\d+)\].*?RDMA transfer complete:\s+"
    r"(\d+)\s+tensors,\s+([0-9.]+)\s+GB",
    re.IGNORECASE,
)
_DONOR_PROCESS_FAILURE_MARKERS = (
    b"Segfault encountered",
    b"Primary job terminated normally, but",
    b"process returned a non-zero exit code",
)
_DONOR_PROCESS_FAILURE_OVERLAP = max(len(marker) for marker in _DONOR_PROCESS_FAILURE_MARKERS) - 1
_MINIMUM_MODELEXPRESS_VERSION = Version("0.5.1")
_MODELEXPRESS_VERSION_PREFIX = "MODELEXPRESS_VERSION="
_MX_PREFLIGHT_SCRIPT = """
import importlib.metadata as metadata
import importlib.util


def module_exists(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


assert module_exists("modelexpress.engines.trtllm"), "no TRT-LLM adapter is available"
from modelexpress.nixl_transfer import is_nixl_available

assert is_nixl_available(), "NIXL is unavailable"
print(f"MODELEXPRESS_VERSION={metadata.version('modelexpress')}")
"""


@dataclass(frozen=True)
class MxE2ECase:
    """One real-model qualification row for the shared MX harness."""

    model_env: str
    default_model_subdir: str
    repository_cache_prefix: str
    tp_size: int


_MX_CASES = (
    pytest.param(
        MxE2ECase(
            model_env="TRTLLM_MX_LLAMA_MODEL",
            default_model_subdir="llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
            repository_cache_prefix="models--trtllm-mx-e2e--llama-tp1",
            tp_size=1,
        ),
        id="llama-bf16-tp1",
        marks=pytest.mark.skip_less_device(2),
    ),
    pytest.param(
        MxE2ECase(
            model_env="TRTLLM_MX_LLAMA_MODEL",
            default_model_subdir="llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
            repository_cache_prefix="models--trtllm-mx-e2e--llama-tp2",
            tp_size=2,
        ),
        id="llama-bf16-tp2",
        marks=pytest.mark.skip_less_device(4),
    ),
    pytest.param(
        MxE2ECase(
            model_env="TRTLLM_MX_QWEN2_MODEL",
            default_model_subdir="Qwen2-7B-Instruct",
            repository_cache_prefix="models--trtllm-mx-e2e--qwen2-tp1",
            tp_size=1,
        ),
        id="qwen2-bf16-tp1",
        marks=pytest.mark.skip_less_device(2),
    ),
    pytest.param(
        MxE2ECase(
            model_env="TRTLLM_MX_QWEN2_MODEL",
            default_model_subdir="Qwen2-7B-Instruct",
            repository_cache_prefix="models--trtllm-mx-e2e--qwen2-tp2",
            tp_size=2,
        ),
        id="qwen2-bf16-tp2",
        marks=pytest.mark.skip_less_device(4),
    ),
    pytest.param(
        MxE2ECase(
            model_env="TRTLLM_MX_QWEN3_MODEL",
            default_model_subdir="Qwen3/Qwen3-8B",
            repository_cache_prefix="models--trtllm-mx-e2e--qwen3-tp1",
            tp_size=1,
        ),
        id="qwen3-bf16-tp1",
        marks=pytest.mark.skip_less_device(2),
    ),
    pytest.param(
        MxE2ECase(
            model_env="TRTLLM_MX_QWEN3_MODEL",
            default_model_subdir="Qwen3/Qwen3-8B",
            repository_cache_prefix="models--trtllm-mx-e2e--qwen3-tp2",
            tp_size=2,
        ),
        id="qwen3-bf16-tp2",
        marks=pytest.mark.skip_less_device(4),
    ),
    pytest.param(
        MxE2ECase(
            model_env="TRTLLM_MX_MISTRAL_MODEL",
            default_model_subdir="Mistral-7B-Instruct-v0.3",
            repository_cache_prefix="models--trtllm-mx-e2e--mistral-tp1",
            tp_size=1,
        ),
        id="mistral-bf16-tp1",
        marks=pytest.mark.skip_less_device(2),
    ),
    pytest.param(
        MxE2ECase(
            model_env="TRTLLM_MX_MISTRAL_MODEL",
            default_model_subdir="Mistral-7B-Instruct-v0.3",
            repository_cache_prefix="models--trtllm-mx-e2e--mistral-tp2",
            tp_size=2,
        ),
        id="mistral-bf16-tp2",
        marks=pytest.mark.skip_less_device(4),
    ),
)


def _qualification_required() -> bool:
    return os.environ.get("TRTLLM_MX_E2E_REQUIRED") == "1"


def _skip_or_fail(message: str) -> NoReturn:
    if _qualification_required():
        pytest.fail(message)
    pytest.skip(message)


def _resolve_model_path(case: MxE2ECase) -> Path:
    configured_path = os.environ.get(case.model_env)
    if configured_path:
        model_path = Path(configured_path).expanduser()
    else:
        configured_root = os.environ.get("LLM_MODELS_ROOT")
        if configured_root:
            models_root = Path(configured_root).expanduser()
        else:
            models_root = Path("/home/scratch.trt_llm_data_ci/llm-models")
            if not models_root.exists():
                models_root = Path("/scratch.trt_llm_data/llm-models")
        model_path = models_root / case.default_model_subdir

    if not model_path.is_dir():
        _skip_or_fail(f"MX E2E model directory does not exist: {model_path}. Set {case.model_env}.")
    return model_path.absolute()


def _require_mx_environment(required_gpus: int) -> tuple[str, tuple[str, ...]]:
    mx_url = os.environ.get("MODEL_EXPRESS_URL")
    if not mx_url:
        _skip_or_fail("MODEL_EXPRESS_URL must point to an isolated ModelExpress test service")
    assert mx_url is not None

    parsed_url = urlparse(mx_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
        _skip_or_fail(f"MODEL_EXPRESS_URL is invalid: {mx_url!r}")
    try:
        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    except ValueError as error:
        _skip_or_fail(f"MODEL_EXPRESS_URL is invalid: {mx_url!r}: {error}")
    deadline = time.monotonic() + 30
    while True:
        try:
            with socket.create_connection((parsed_url.hostname, port), timeout=1):
                break
        except OSError as error:
            if time.monotonic() >= deadline:
                _skip_or_fail(f"ModelExpress service {mx_url!r} is unreachable: {error}")
            time.sleep(1)

    try:
        preflight = subprocess.run(
            [sys.executable, "-c", _MX_PREFLIGHT_SCRIPT],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        _skip_or_fail(f"ModelExpress/NIXL preflight timed out after {error.timeout} seconds")
    if preflight.returncode != 0:
        detail = (preflight.stdout + preflight.stderr).strip()
        _skip_or_fail(f"ModelExpress/NIXL preflight failed: {detail}")
    version_lines = [
        line
        for line in preflight.stdout.splitlines()
        if line.startswith(_MODELEXPRESS_VERSION_PREFIX)
    ]
    if len(version_lines) != 1:
        _skip_or_fail(f"ModelExpress preflight did not report its version: {preflight.stdout!r}")
    modelexpress_version = version_lines[0].removeprefix(_MODELEXPRESS_VERSION_PREFIX)
    try:
        parsed_modelexpress_version = Version(modelexpress_version)
    except InvalidVersion:
        _skip_or_fail(f"ModelExpress client reported invalid version {modelexpress_version!r}")
    if parsed_modelexpress_version < _MINIMUM_MODELEXPRESS_VERSION:
        _skip_or_fail(
            f"Unsupported ModelExpress client version {modelexpress_version!r}; "
            f"requires >= {_MINIMUM_MODELEXPRESS_VERSION}"
        )
    print(f"MX E2E client preflight passed: ModelExpress {modelexpress_version}")

    configured_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured_devices and configured_devices.strip().lower() != "all":
        gpu_ids = tuple(
            device.strip() for device in configured_devices.split(",") if device.strip()
        )
    else:
        gpu_ids = tuple(str(index) for index in range(torch.cuda.device_count()))
    if len(gpu_ids) < required_gpus:
        _skip_or_fail(f"MX E2E requires {required_gpus} GPUs, but only {len(gpu_ids)} are visible")
    return mx_url, gpu_ids[:required_gpus]


def _checkpoint_files(model_path: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in model_path.rglob("*") if path.is_file()))


def _build_canonical_snapshot(case: MxE2ECase, model_path: Path, run_dir: Path) -> Path:
    files = _checkpoint_files(model_path)
    if not files:
        pytest.fail(f"MX E2E model directory is empty: {model_path}")
    if not any(path.suffix.lower() in _WEIGHT_SUFFIXES for path in files):
        pytest.fail(f"MX E2E donor checkpoint contains no recognized weight files: {model_path}")

    source_identity = ArtifactIdentity.from_checkpoint(model_path)
    run_id = uuid.uuid4().hex[:12]
    repository_cache_name = f"{case.repository_cache_prefix}-{run_id}"
    snapshot = (
        run_dir / "donor-hf-cache" / repository_cache_name / "snapshots" / source_identity.digest
    )
    for path in files:
        destination = snapshot / path.relative_to(model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(path.resolve())
    return snapshot


def _build_metadata_only_snapshot(donor_snapshot: Path, run_dir: Path) -> Path:
    repository_cache_name = donor_snapshot.parents[1].name
    revision = donor_snapshot.name
    receiver_snapshot = (
        run_dir / "receiver-hf-cache" / repository_cache_name / "snapshots" / revision
    )

    for path in _checkpoint_files(donor_snapshot):
        if path.suffix.lower() in _WEIGHT_SUFFIXES:
            continue
        destination = receiver_snapshot / path.relative_to(donor_snapshot)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination, follow_symlinks=True)

    if not (receiver_snapshot / "config.json").is_file():
        pytest.fail("The metadata-only MX receiver snapshot has no config.json")
    receiver_weights = tuple(
        path
        for path in _checkpoint_files(receiver_snapshot)
        if path.suffix.lower() in _WEIGHT_SUFFIXES
    )
    if receiver_weights:
        pytest.fail(f"The MX receiver snapshot unexpectedly contains weights: {receiver_weights}")
    return receiver_snapshot


def _worker_command(
    *,
    role: str,
    model_path: Path,
    tp_size: int,
    output_path: Path,
    mx_url: str | None = None,
    ready_file: Path | None = None,
    stop_file: Path | None = None,
    max_serve_seconds: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(_WORKER),
        "--role",
        role,
        "--model",
        str(model_path),
        "--tp-size",
        str(tp_size),
        "--output",
        str(output_path),
    ]
    if mx_url is not None:
        command.extend(("--mx-url", mx_url))
    if ready_file is not None:
        command.extend(("--ready-file", str(ready_file)))
    if stop_file is not None:
        command.extend(("--stop-file", str(stop_file)))
    if max_serve_seconds is not None:
        command.extend(("--max-serve-seconds", str(max_serve_seconds)))
    return command


def _worker_environment(gpu_ids: tuple[str, ...], transfer_log_dir: Path) -> dict[str, str]:
    transfer_log_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MX_TRANSFER_LOG_DIR": str(transfer_log_dir),
            # NIXL's wheel bundles UCX. Keep OpenMPI's UCX-capable components
            # from loading the container UCX stack into the same process.
            "OMPI_MCA_pml": "ob1",
            "OMPI_MCA_osc": "pt2pt",
            "OMPI_MCA_btl": "self,vader,tcp",
            "OMPI_MCA_coll": "^hcoll,ucc",
            "PYTHONUNBUFFERED": "1",
            "TLLM_LOG_LEVEL": "INFO",
        }
    )
    return environment


def _log_tail(log_path: Path, lines: int = 120) -> str:
    if not log_path.exists():
        return "<log was not created>"
    return "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _run_worker(
    command: list[str], environment: dict[str, str], log_path: Path, timeout_s: int
) -> None:
    try:
        with (
            log_path.open("w", encoding="utf-8") as log_file,
            popen(
                command,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                suppress_output_info=True,
            ) as process,
        ):
            returncode = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        pytest.fail(f"MX E2E worker timed out: {' '.join(command)}\n{_log_tail(log_path)}")
    if returncode != 0:
        pytest.fail(
            f"MX E2E worker exited with status {returncode}: "
            f"{' '.join(command)}\n{_log_tail(log_path)}"
        )


def _wait_for_donor(
    process: subprocess.Popen[bytes], ready_file: Path, log_path: Path, timeout_s: int
) -> None:
    deadline = time.monotonic() + timeout_s
    log_offset = 0
    failure_window = b""
    while not ready_file.exists():
        returncode = process.poll()
        if returncode is not None:
            pytest.fail(
                f"MX donor exited before publication with status {returncode}\n{_log_tail(log_path)}"
            )
        if log_path.exists():
            with log_path.open("rb") as log_file:
                log_file.seek(log_offset)
                failure_window += log_file.read()
                log_offset = log_file.tell()
        if any(marker in failure_window for marker in _DONOR_PROCESS_FAILURE_MARKERS):
            pytest.fail(f"MX donor worker failed before publication\n{_log_tail(log_path)}")
        failure_window = failure_window[-_DONOR_PROCESS_FAILURE_OVERLAP:]
        if time.monotonic() >= deadline:
            pytest.fail(f"MX donor did not become ready within {timeout_s}s\n{_log_tail(log_path)}")
        time.sleep(0.5)


def _stop_donor(process: subprocess.Popen[bytes], stop_file: Path) -> int:
    if process.poll() is not None:
        assert process.returncode is not None
        return process.returncode
    stop_file.write_text("stop\n", encoding="utf-8")
    try:
        return process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        cleanup_process_tree(process, has_session=True)
        return process.wait(timeout=30)


def _load_tokens(output_path: Path) -> list[list[int]]:
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    token_ids = payload.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids or not all(token_ids):
        pytest.fail(f"MX E2E worker produced invalid token IDs: {payload}")
    return token_ids


def _assert_transfer_evidence(
    case: MxE2ECase,
    receiver_log_path: Path,
) -> None:
    receiver_log = receiver_log_path.read_text(encoding="utf-8", errors="replace")
    expected_ranks = set(range(case.tp_size))
    transfers_by_rank: dict[int, list[tuple[int, float]]] = {}
    for rank, tensor_count, size_gb in _RDMA_TRANSFER_PATTERN.findall(receiver_log):
        transfers_by_rank.setdefault(int(rank), []).append((int(tensor_count), float(size_gb)))

    assert set(transfers_by_rank) == expected_ranks, (
        f"Expected RDMA transfer completion for ranks {expected_ranks}, "
        f"got {set(transfers_by_rank)}\nReceiver log:\n{receiver_log}"
    )
    receiver_log_lower = receiver_log.lower()

    for marker in _RECEIVER_FAILURE_MARKERS:
        assert marker not in receiver_log_lower, (
            f"MX receiver log contains failure marker {marker!r}"
        )

    for rank in sorted(expected_ranks):
        transfers = transfers_by_rank[rank]
        assert len(transfers) == 1, (
            f"Expected one RDMA transfer completion for rank {rank}, got {transfers}"
        )
        tensor_count, size_gb = transfers[0]
        assert tensor_count > 0 and size_gb > 0, (
            f"MX receiver rank {rank} reported an empty transfer: "
            f"{tensor_count} tensors, {size_gb} GB"
        )


@pytest.mark.parametrize("case", _MX_CASES)
def test_mx_donor_receiver(case: MxE2ECase, tmp_path: Path) -> None:
    """Compare HF, MX donor, and no-weight-shards MX receiver outputs."""
    required_gpus = case.tp_size * 2
    mx_url, gpu_ids = _require_mx_environment(required_gpus)
    model_path = _resolve_model_path(case)
    timeout_s = int(os.environ.get("TRTLLM_MX_E2E_TIMEOUT_S", "1200"))
    # NIXL binds base_port + device_id. The donor and receiver share one CI
    # network namespace, so give their TP ranks adjacent, non-overlapping ranges.
    metadata_port = int(os.environ.get("MX_METADATA_PORT", "5555"))

    donor_snapshot = _build_canonical_snapshot(case, model_path, tmp_path)
    receiver_snapshot = _build_metadata_only_snapshot(donor_snapshot, tmp_path)
    donor_identity = ArtifactIdentity.from_checkpoint(donor_snapshot)
    receiver_identity = ArtifactIdentity.from_checkpoint(receiver_snapshot)
    assert donor_identity.scheme == "hf_snapshot_revision"
    assert receiver_identity == donor_identity

    donor_gpu_ids = gpu_ids[: case.tp_size]
    receiver_gpu_ids = gpu_ids[case.tp_size :]
    baseline_output = tmp_path / "baseline.json"
    donor_output = tmp_path / "donor.json"
    receiver_output = tmp_path / "receiver.json"
    baseline_log = tmp_path / "baseline.log"
    donor_log = tmp_path / "donor.log"
    receiver_log = tmp_path / "receiver.log"
    donor_ready = tmp_path / "donor.ready"
    donor_stop = tmp_path / "donor.stop"

    # Leave donor GPUs untouched in case MPI worker teardown trails the baseline process.
    _run_worker(
        _worker_command(
            role="baseline",
            model_path=donor_snapshot,
            tp_size=case.tp_size,
            output_path=baseline_output,
        ),
        _worker_environment(receiver_gpu_ids, tmp_path / "baseline-transfer-logs"),
        baseline_log,
        timeout_s,
    )

    donor_environment = _worker_environment(donor_gpu_ids, tmp_path / "donor-transfer-logs")
    donor_environment["MX_METADATA_PORT"] = str(metadata_port)
    donor_returncode = None
    with (
        donor_log.open("w", encoding="utf-8") as donor_log_file,
        popen(
            _worker_command(
                role="donor",
                model_path=donor_snapshot,
                tp_size=case.tp_size,
                output_path=donor_output,
                mx_url=mx_url,
                ready_file=donor_ready,
                stop_file=donor_stop,
                max_serve_seconds=timeout_s + 120,
            ),
            env=donor_environment,
            stdout=donor_log_file,
            stderr=subprocess.STDOUT,
            suppress_output_info=True,
        ) as donor_process,
    ):
        try:
            _wait_for_donor(donor_process, donor_ready, donor_log, timeout_s)
            receiver_environment = _worker_environment(
                receiver_gpu_ids, tmp_path / "receiver-transfer-logs"
            )
            receiver_environment["MX_METADATA_PORT"] = str(metadata_port + case.tp_size)
            _run_worker(
                _worker_command(
                    role="receiver",
                    model_path=receiver_snapshot,
                    tp_size=case.tp_size,
                    output_path=receiver_output,
                    mx_url=mx_url,
                ),
                receiver_environment,
                receiver_log,
                timeout_s,
            )
        finally:
            donor_returncode = _stop_donor(donor_process, donor_stop)

    assert donor_returncode == 0, (
        f"MX donor exited with status {donor_returncode}\n{_log_tail(donor_log)}"
    )
    donor_payload = json.loads(donor_output.read_text(encoding="utf-8"))
    assert donor_payload["server_query_timeout_s"] == 0
    baseline_tokens = _load_tokens(baseline_output)
    assert _load_tokens(donor_output) == baseline_tokens
    assert _load_tokens(receiver_output) == baseline_tokens
    _assert_transfer_evidence(
        case,
        receiver_log,
    )
