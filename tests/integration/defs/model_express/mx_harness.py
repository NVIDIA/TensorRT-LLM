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
"""Shared pytest-side helpers for the ModelExpress donor/receiver qualification harness.

The harness runs three roles of `mx_e2e_worker.py` on disjoint GPU sets: an HF
baseline, a live MX donor that publishes its weights, and an MX receiver that
starts from a metadata-only snapshot and can only succeed through a real P2P
transfer. Test modules keep their case tables and orchestration; everything
reusable lives here so that additional qualification tests (for example the
post-merge accuracy canaries) share one definition of snapshots, worker
launching, transfer evidence, and weight-manifest comparison.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn
from urllib.parse import urlparse

import pytest
import torch
from packaging.version import InvalidVersion, Version

from defs.model_express.mx_evidence import (
    DONOR_PROCESS_FAILURE_MARKERS,
    DONOR_PROCESS_FAILURE_OVERLAP,
    check_receiver_transfer_logs,
    summarize_transfer_logs,
    transfer_logs_by_rank,
)
from defs.trt_test_alternative import cleanup_process_tree, popen
from tensorrt_llm._torch.weight_sharing import (
    WEIGHT_MANIFEST_FILE_PATTERN,
    ArtifactIdentity,
    WeightManifest,
    compare_weight_manifests,
    load_weight_manifest,
)

WORKER_PATH = Path(__file__).with_name("mx_e2e_worker.py")
WEIGHT_SUFFIXES = frozenset({".bin", ".ckpt", ".gguf", ".pt", ".pth", ".safetensors"})
MINIMUM_MODELEXPRESS_VERSION = Version("0.4.1")
_MODELEXPRESS_VERSION_PREFIX = "MODELEXPRESS_VERSION="
MX_PREFLIGHT_SCRIPT = """
import importlib.metadata as metadata
import importlib.util


def module_exists(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


assert module_exists("modelexpress.trtllm_live_transfer") or module_exists(
    "modelexpress.engines.trtllm"
), "no TRT-LLM adapter is available"
from modelexpress.nixl_transfer import is_nixl_available

assert is_nixl_available(), "NIXL is unavailable"
print(f"MODELEXPRESS_VERSION={metadata.version('modelexpress')}")
"""

ROLES = ("baseline", "donor", "receiver")
MX_ROLES = ("donor", "receiver")
# The behavioral probe: every role must generate at least this many prompts and
# exactly this many greedy tokens per prompt (see `mx_e2e_worker.py`).
MIN_PROBE_PROMPTS = 8
MIN_PROBE_NEW_TOKENS = 32
# Transfer-tier manifests are compared on parameters only: the receiver's
# `cache_derived_state()` runs after the P2P boundary, so derived buffers are
# not part of the transferred contract there. Buffers are enforced at the
# final tier. Flip to `("param", "buffer")` once real runs show buffers are
# also byte-identical at the boundary.
TRANSFER_TIER_KINDS = ("param",)
FINAL_TIER_PAIRS = (("baseline", "donor"), ("baseline", "receiver"), ("donor", "receiver"))

ManifestKey = tuple[str, str, int]


@dataclass(frozen=True, kw_only=True)
class MxE2ECase:
    """One real-model qualification row for the shared MX harness.

    `final_manifest_exempt_patterns` lists `fnmatch` patterns of tensor names
    whose final-tier digest may legitimately differ between roles (never a
    numeric tolerance, never at the transfer tier). Every pattern needs a code
    comment explaining why; the qualified BF16 dense families use none.
    """

    model_env: str
    default_model_subdir: str
    repository_cache_prefix: str
    tp_size: int
    final_manifest_exempt_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class MxRunLayout:
    """Where one test run keeps its worker outputs, logs, and manifests."""

    root: Path

    def output(self, role: str) -> Path:
        return self.root / f"{role}.json"

    def log(self, role: str) -> Path:
        return self.root / f"{role}.log"

    def transfer_logs(self, role: str) -> Path:
        return self.root / f"{role}-transfer-logs"

    @property
    def donor_ready(self) -> Path:
        return self.root / "donor.ready"

    @property
    def donor_stop(self) -> Path:
        return self.root / "donor.stop"

    @property
    def manifest_dir(self) -> Path:
        return self.root / "weight-manifests"

    @property
    def timing_path(self) -> Path:
        return self.root / "timing.json"


# --------------------------------------------------------------------------- #
# Prerequisites
# --------------------------------------------------------------------------- #


def qualification_required() -> bool:
    return os.environ.get("TRTLLM_MX_E2E_REQUIRED") == "1"


def skip_or_fail(message: str) -> NoReturn:
    if qualification_required():
        pytest.fail(message)
    pytest.skip(message)


def resolve_model_path(case: MxE2ECase) -> Path:
    configured = os.environ.get(case.model_env)
    if configured:
        model_path = Path(configured)
    else:
        models_root_env = os.environ.get("LLM_MODELS_ROOT")
        if models_root_env:
            models_root = Path(models_root_env)
        else:
            models_root = Path("/home/scratch.trt_llm_data_ci/llm-models")
            if not models_root.exists():
                models_root = Path("/scratch.trt_llm_data/llm-models")
        model_path = models_root / case.default_model_subdir

    if not model_path.is_dir():
        skip_or_fail(f"MX E2E model directory does not exist: {model_path}. Set {case.model_env}.")
    return model_path.absolute()


def require_mx_environment(required_gpus: int) -> tuple[str, tuple[str, ...]]:
    mx_url = os.environ.get("MODEL_EXPRESS_URL")
    if not mx_url:
        skip_or_fail("MODEL_EXPRESS_URL must point to an isolated ModelExpress test service")
    assert mx_url is not None

    parsed_url = urlparse(mx_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
        skip_or_fail(f"MODEL_EXPRESS_URL is invalid: {mx_url!r}")
    try:
        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    except ValueError as error:
        skip_or_fail(f"MODEL_EXPRESS_URL is invalid: {mx_url!r}: {error}")
    deadline = time.monotonic() + 30
    while True:
        try:
            with socket.create_connection((parsed_url.hostname, port), timeout=1):
                break
        except OSError as error:
            if time.monotonic() >= deadline:
                skip_or_fail(f"ModelExpress service {mx_url!r} is unreachable: {error}")
            time.sleep(1)

    try:
        preflight = subprocess.run(
            [sys.executable, "-c", MX_PREFLIGHT_SCRIPT],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        skip_or_fail(f"ModelExpress/NIXL preflight timed out after {error.timeout} seconds")
    if preflight.returncode != 0:
        detail = (preflight.stdout + preflight.stderr).strip()
        skip_or_fail(f"ModelExpress/NIXL preflight failed: {detail}")
    version_lines = [
        line
        for line in preflight.stdout.splitlines()
        if line.startswith(_MODELEXPRESS_VERSION_PREFIX)
    ]
    if len(version_lines) != 1:
        skip_or_fail(f"ModelExpress preflight did not report its version: {preflight.stdout!r}")
    modelexpress_version = version_lines[0].removeprefix(_MODELEXPRESS_VERSION_PREFIX)
    try:
        parsed_modelexpress_version = Version(modelexpress_version)
    except InvalidVersion:
        skip_or_fail(f"ModelExpress client reported invalid version {modelexpress_version!r}")
    if parsed_modelexpress_version < MINIMUM_MODELEXPRESS_VERSION:
        skip_or_fail(
            f"Unsupported ModelExpress client version {modelexpress_version!r}; "
            f"requires >= {MINIMUM_MODELEXPRESS_VERSION}"
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
        skip_or_fail(f"MX E2E requires {required_gpus} GPUs, but only {len(gpu_ids)} are visible")
    return mx_url, gpu_ids[:required_gpus]


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #


def checkpoint_files(model_path: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in model_path.rglob("*") if path.is_file()))


def build_canonical_snapshot(case: MxE2ECase, model_path: Path, run_dir: Path) -> Path:
    """Symlink the checkpoint into an HF-cache-shaped snapshot keyed by its `ArtifactIdentity`."""
    files = checkpoint_files(model_path)
    if not files:
        pytest.fail(f"MX E2E model directory is empty: {model_path}")
    if not any(path.suffix.lower() in WEIGHT_SUFFIXES for path in files):
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


def build_metadata_only_snapshot(donor_snapshot: Path, run_dir: Path) -> Path:
    """Copy every non-weight file of the donor snapshot; the receiver must transfer the rest."""
    repository_cache_name = donor_snapshot.parents[1].name
    revision = donor_snapshot.name
    receiver_snapshot = (
        run_dir / "receiver-hf-cache" / repository_cache_name / "snapshots" / revision
    )

    for path in checkpoint_files(donor_snapshot):
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            continue
        destination = receiver_snapshot / path.relative_to(donor_snapshot)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination, follow_symlinks=True)

    if not (receiver_snapshot / "config.json").is_file():
        pytest.fail("The metadata-only MX receiver snapshot has no config.json")
    receiver_weights = tuple(
        path
        for path in checkpoint_files(receiver_snapshot)
        if path.suffix.lower() in WEIGHT_SUFFIXES
    )
    if receiver_weights:
        pytest.fail(f"The MX receiver snapshot unexpectedly contains weights: {receiver_weights}")
    return receiver_snapshot


# --------------------------------------------------------------------------- #
# Worker processes
# --------------------------------------------------------------------------- #


def worker_command(
    *,
    role: str,
    model_path: Path,
    tp_size: int,
    output_path: Path,
    mx_url: str | None = None,
    ready_file: Path | None = None,
    stop_file: Path | None = None,
    max_serve_seconds: int | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    command = [
        sys.executable,
        str(WORKER_PATH),
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
    command.extend(extra_args)
    return command


def worker_environment(
    gpu_ids: tuple[str, ...], transfer_log_dir: Path, manifest_dir: Path | None = None
) -> dict[str, str]:
    """Environment for a worker subprocess (set before any MPI/UCX import happens).

    `manifest_dir` enables the per-rank weight manifests; the worker forwards it,
    together with its role, to the executor rank processes through
    `LLM(env_overrides=...)`.
    """
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
    if manifest_dir is not None:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        environment["MX_WEIGHT_MANIFEST_DIR"] = str(manifest_dir)
    else:
        environment.pop("MX_WEIGHT_MANIFEST_DIR", None)
    return environment


def log_tail(log_path: Path, lines: int = 120) -> str:
    if not log_path.exists():
        return "<log was not created>"
    return "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def run_worker(
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
        pytest.fail(f"MX E2E worker timed out: {' '.join(command)}\n{log_tail(log_path)}")
    if returncode != 0:
        pytest.fail(
            f"MX E2E worker exited with status {returncode}: "
            f"{' '.join(command)}\n{log_tail(log_path)}"
        )


def wait_for_donor(
    process: subprocess.Popen[bytes], ready_file: Path, log_path: Path, timeout_s: int
) -> None:
    deadline = time.monotonic() + timeout_s
    log_offset = 0
    failure_window = b""
    while not ready_file.exists():
        returncode = process.poll()
        if returncode is not None:
            pytest.fail(
                f"MX donor exited before publication with status {returncode}\n{log_tail(log_path)}"
            )
        if log_path.exists():
            with log_path.open("rb") as log_file:
                log_file.seek(log_offset)
                failure_window += log_file.read()
                log_offset = log_file.tell()
        if any(marker in failure_window for marker in DONOR_PROCESS_FAILURE_MARKERS):
            pytest.fail(f"MX donor worker failed before publication\n{log_tail(log_path)}")
        failure_window = failure_window[-DONOR_PROCESS_FAILURE_OVERLAP:]
        if time.monotonic() >= deadline:
            pytest.fail(f"MX donor did not become ready within {timeout_s}s\n{log_tail(log_path)}")
        time.sleep(0.5)


def stop_donor(process: subprocess.Popen[bytes], stop_file: Path) -> int:
    if process.poll() is not None:
        assert process.returncode is not None
        return process.returncode
    stop_file.write_text("stop\n", encoding="utf-8")
    try:
        return process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        cleanup_process_tree(process, has_session=True)
        return process.wait(timeout=30)


@dataclass
class DonorHandle:
    """The running donor process; `returncode` is filled in when the session ends."""

    process: subprocess.Popen[bytes]
    returncode: int | None = None


@contextmanager
def donor_session(
    *,
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    ready_file: Path,
    stop_file: Path,
    timeout_s: int,
) -> Iterator[DonorHandle]:
    """Start the donor, wait until it has published, and always stop it afterwards."""
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
        handle = DonorHandle(process)
        try:
            wait_for_donor(process, ready_file, log_path, timeout_s)
            yield handle
        finally:
            handle.returncode = stop_donor(process, stop_file)


# --------------------------------------------------------------------------- #
# Payloads and transfer evidence
# --------------------------------------------------------------------------- #


def load_payload(output_path: Path) -> dict[str, object]:
    if not output_path.is_file():
        pytest.fail(f"MX E2E worker wrote no payload: {output_path}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        pytest.fail(f"MX E2E worker produced an invalid payload: {payload!r}")
    return payload


def assert_probe_payload(
    payload: Mapping[str, object],
    *,
    role: str,
    min_prompts: int = MIN_PROBE_PROMPTS,
    min_new_tokens: int = MIN_PROBE_NEW_TOKENS,
) -> list[list[int]]:
    """Validate the widened behavioral probe of one role and return its token IDs."""
    token_ids = payload.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids or not all(token_ids):
        pytest.fail(f"MX E2E {role} produced invalid token IDs: {payload}")
    prompt_count = payload.get("prompt_count")
    max_new_tokens = payload.get("max_new_tokens")
    assert prompt_count == len(token_ids) >= min_prompts, (
        f"MX E2E {role} generated {len(token_ids)} sequences for {prompt_count} prompts; "
        f"expected at least {min_prompts}"
    )
    assert isinstance(max_new_tokens, int) and max_new_tokens >= min_new_tokens, (
        f"MX E2E {role} reports max_new_tokens={max_new_tokens}; expected at least {min_new_tokens}"
    )
    short = [len(ids) for ids in token_ids if len(ids) != max_new_tokens]
    assert not short, (
        f"MX E2E {role} produced sequences of lengths {short}; every sequence must have exactly "
        f"{max_new_tokens} tokens (end_id=-1)"
    )
    return token_ids


def assert_transfer_evidence(
    case: MxE2ECase,
    receiver_log_path: Path,
    receiver_transfer_log_dir: Path,
    receiver_payload: Mapping[str, object] | None = None,
) -> None:
    """Authoritative post-check that the receiver's weights arrived through MX P2P."""
    receiver_log = receiver_log_path.read_text(encoding="utf-8", errors="replace")
    try:
        transfer_logs = transfer_logs_by_rank(receiver_transfer_log_dir)
    except ValueError as error:
        pytest.fail(f"{error}\nReceiver log:\n{receiver_log}")
    problems = check_receiver_transfer_logs(transfer_logs, case.tp_size, extra_text=receiver_log)
    if problems:
        pytest.fail(
            "MX receiver transfer evidence is incomplete:\n- "
            + "\n- ".join(problems)
            + f"\nReceiver log tail:\n{log_tail(receiver_log_path)}"
        )
    if receiver_payload is not None and receiver_payload.get("transfer_evidence") is not None:
        expected = {
            str(rank): summary.to_dict()
            for rank, summary in sorted(summarize_transfer_logs(receiver_transfer_log_dir).items())
        }
        assert receiver_payload["transfer_evidence"] == expected, (
            "The receiver's self-reported transfer evidence differs from the harness summary: "
            f"{receiver_payload['transfer_evidence']} vs {expected}"
        )


# --------------------------------------------------------------------------- #
# Weight manifests
# --------------------------------------------------------------------------- #


def collect_weight_manifests(
    manifest_dir: Path, case: MxE2ECase
) -> dict[ManifestKey, WeightManifest]:
    """Load every manifest of a run and require the exact expected (family, role, rank) set."""
    if not manifest_dir.is_dir():
        pytest.fail(f"No weight manifests were written: {manifest_dir} does not exist")
    manifests: dict[ManifestKey, WeightManifest] = {}
    for path in sorted(manifest_dir.iterdir()):
        match = WEIGHT_MANIFEST_FILE_PATTERN.match(path.name) if path.is_file() else None
        if match is None:
            pytest.fail(f"Unexpected entry in the weight manifest directory: {path}")
        key: ManifestKey = (match["family"], match["role"], int(match["rank"]))
        manifest = load_weight_manifest(path)
        context = manifest.context
        assert (context.get("family"), context.get("role"), context.get("rank")) == key, (
            f"Weight manifest {path.name} carries context {context} that differs from its name"
        )
        if key[0] == "final":
            assert context.get("world_size") == case.tp_size, (
                f"Weight manifest {path.name} reports world_size={context.get('world_size')}; "
                f"expected {case.tp_size}"
            )
        manifests[key] = manifest

    expected_keys = {("final", role, rank) for role in ROLES for rank in range(case.tp_size)}
    expected_keys |= {("transfer", role, rank) for role in MX_ROLES for rank in range(case.tp_size)}
    missing = sorted(expected_keys - manifests.keys())
    unexpected = sorted(manifests.keys() - expected_keys)
    if missing or unexpected:
        pytest.fail(
            "Weight manifest set differs from the expected (family, role, rank) keys: "
            f"absent={missing}, extra={unexpected}. An HF baseline never reaches an MX transfer "
            "boundary, so a transfer manifest for it means the roles are misconfigured."
        )
    return manifests


def _assert_manifests_equal(
    expected: WeightManifest,
    actual: WeightManifest,
    expected_label: str,
    actual_label: str,
    *,
    kinds: Sequence[str] = ("param", "buffer"),
    exempt_patterns: Sequence[str] = (),
) -> None:
    diff = compare_weight_manifests(expected, actual, kinds=kinds, exempt_patterns=exempt_patterns)
    if diff.exempted_digest_diffs:
        print(
            f"MX E2E manifest: exempted digest differences {expected_label} vs {actual_label}: "
            f"{list(diff.exempted_digest_diffs)}"
        )
    if not diff.is_empty:
        pytest.fail(diff.describe(expected_label, actual_label))


def assert_weight_manifests(
    case: MxE2ECase, manifests: Mapping[ManifestKey, WeightManifest]
) -> None:
    """Enforce the transfer and final manifest tiers of one run.

    Transfer tier: donor-at-publish and receiver-at-receive parameters are
    byte-identical, never exempted. Final tier: baseline, donor, and receiver
    are pairwise byte-identical after finalization, honoring only the row's
    `final_manifest_exempt_patterns`.
    """

    def context(family: str, role: str, rank: int) -> Mapping[str, object]:
        return manifests[(family, role, rank)].context

    for rank in range(case.tp_size):
        assert context("final", "baseline", rank).get("checkpoint_format") == "HF"
        for role in MX_ROLES:
            assert context("final", role, rank).get("checkpoint_format") == "MX"
        assert context("final", "donor", rank).get("weights_preloaded") is False, (
            "The donor must load from disk, not from a preloaded MX source"
        )
        assert context("final", "receiver", rank).get("weights_preloaded") is True, (
            "The receiver must report preloaded (P2P) weights"
        )
        assert context("transfer", "donor", rank).get("boundary") == "donor_publish"
        assert context("transfer", "receiver", rank).get("boundary") == "receiver_p2p_success"

        _assert_manifests_equal(
            manifests[("transfer", "donor", rank)],
            manifests[("transfer", "receiver", rank)],
            f"donor@publish rank{rank}",
            f"receiver@receive rank{rank}",
            kinds=TRANSFER_TIER_KINDS,
        )
        for expected_role, actual_role in FINAL_TIER_PAIRS:
            _assert_manifests_equal(
                manifests[("final", expected_role, rank)],
                manifests[("final", actual_role, rank)],
                f"{expected_role}@final rank{rank}",
                f"{actual_role}@final rank{rank}",
                exempt_patterns=case.final_manifest_exempt_patterns,
            )


# --------------------------------------------------------------------------- #
# Artifacts and timing
# --------------------------------------------------------------------------- #


def report_timings(
    case: MxE2ECase,
    payloads: Mapping[str, Mapping[str, object]],
    manifests: Mapping[ManifestKey, WeightManifest],
    layout: MxRunLayout,
) -> list[dict[str, object]]:
    """Print one `MX E2E timing` line per role and rank and persist them as JSON."""
    rows: list[dict[str, object]] = []
    for role in ROLES:
        payload = payloads[role]
        for rank in range(case.tp_size):
            final = manifests.get(("final", role, rank))
            transfer = manifests.get(("transfer", role, rank))
            row: dict[str, object] = {
                "role": role,
                "rank": rank,
                "load_seconds": payload.get("load_seconds"),
                "generate_seconds": payload.get("generate_seconds"),
                "manifest_final_seconds": final.context.get("build_seconds") if final else None,
                "manifest_transfer_seconds": (
                    transfer.context.get("build_seconds") if transfer else None
                ),
                "bytes_hashed": final.context.get("bytes_hashed") if final else None,
                "entry_count": len(final.entries) if final else None,
            }
            rows.append(row)
            print("MX E2E timing " + " ".join(f"{key}={value}" for key, value in row.items()))
    layout.timing_path.write_text(
        json.dumps({"tp_size": case.tp_size, "rows": rows}, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def archive_run_artifacts(
    output_dir: str | Path | None, case_id: str, layout: MxRunLayout
) -> Path | None:
    """Copy payloads, logs, manifests, and timing into the CI-archived output directory."""
    if not output_dir:
        return None
    destination = Path(output_dir) / "model_express" / case_id
    destination.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.json", "*.log"):
        for path in layout.root.glob(pattern):
            if path.is_file():
                shutil.copy2(path, destination / path.name)
    if layout.manifest_dir.is_dir():
        manifest_destination = destination / layout.manifest_dir.name
        manifest_destination.mkdir(exist_ok=True)
        for path in layout.manifest_dir.glob("*.json"):
            shutil.copy2(path, manifest_destination / path.name)
    for role in ROLES:
        transfer_logs = layout.transfer_logs(role)
        if transfer_logs.is_dir():
            shutil.copytree(transfer_logs, destination / transfer_logs.name, dirs_exist_ok=True)
    print(f"MX E2E artifacts archived to {destination}")
    return destination


__all__ = [
    "FINAL_TIER_PAIRS",
    "MIN_PROBE_NEW_TOKENS",
    "MIN_PROBE_PROMPTS",
    "MINIMUM_MODELEXPRESS_VERSION",
    "MX_PREFLIGHT_SCRIPT",
    "MX_ROLES",
    "ROLES",
    "TRANSFER_TIER_KINDS",
    "WEIGHT_SUFFIXES",
    "WORKER_PATH",
    "DonorHandle",
    "ManifestKey",
    "MxE2ECase",
    "MxRunLayout",
    "archive_run_artifacts",
    "assert_probe_payload",
    "assert_transfer_evidence",
    "assert_weight_manifests",
    "build_canonical_snapshot",
    "build_metadata_only_snapshot",
    "checkpoint_files",
    "collect_weight_manifests",
    "donor_session",
    "load_payload",
    "log_tail",
    "qualification_required",
    "report_timings",
    "require_mx_environment",
    "resolve_model_path",
    "run_worker",
    "skip_or_fail",
    "stop_donor",
    "wait_for_donor",
    "worker_command",
    "worker_environment",
]
