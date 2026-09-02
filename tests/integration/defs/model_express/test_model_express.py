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

import os
from pathlib import Path

import pytest

from defs.model_express.mx_harness import (
    ROLES,
    MxE2ECase,
    MxRunLayout,
    archive_run_artifacts,
    assert_probe_payload,
    assert_transfer_evidence,
    assert_weight_manifests,
    build_canonical_snapshot,
    build_metadata_only_snapshot,
    collect_weight_manifests,
    donor_session,
    load_payload,
    log_tail,
    report_timings,
    require_mx_environment,
    resolve_model_path,
    run_worker,
    worker_command,
    worker_environment,
)
from tensorrt_llm._torch.weight_sharing import ArtifactIdentity

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


@pytest.mark.parametrize("case", _MX_CASES)
def test_mx_donor_receiver(case: MxE2ECase, tmp_path: Path, output_dir, request) -> None:
    """Compare HF, MX donor, and no-weight-shards MX receiver outputs and weight manifests.

    Three checks stack up: exact greedy token equality across the roles (the
    behavioral probe), per-rank transfer evidence proving the receiver's bytes
    arrived through P2P, and byte-exact weight manifests at the transfer
    boundary and after finalization. Artifacts are archived under `output_dir`
    when pytest runs with `--output-dir` (always in CI), including on failure.
    """
    required_gpus = case.tp_size * 2
    mx_url, gpu_ids = require_mx_environment(required_gpus)
    model_path = resolve_model_path(case)
    timeout_s = int(os.environ.get("TRTLLM_MX_E2E_TIMEOUT_S", "1200"))
    layout = MxRunLayout(tmp_path)

    donor_snapshot = build_canonical_snapshot(case, model_path, tmp_path)
    receiver_snapshot = build_metadata_only_snapshot(donor_snapshot, tmp_path)
    donor_identity = ArtifactIdentity.from_checkpoint(donor_snapshot)
    receiver_identity = ArtifactIdentity.from_checkpoint(receiver_snapshot)
    assert donor_identity.scheme == "hf_snapshot_revision"
    assert receiver_identity == donor_identity

    donor_gpu_ids = gpu_ids[: case.tp_size]
    receiver_gpu_ids = gpu_ids[case.tp_size :]

    try:
        # Leave donor GPUs untouched in case MPI worker teardown trails the baseline process.
        run_worker(
            worker_command(
                role="baseline",
                model_path=donor_snapshot,
                tp_size=case.tp_size,
                output_path=layout.output("baseline"),
            ),
            worker_environment(
                receiver_gpu_ids, layout.transfer_logs("baseline"), layout.manifest_dir
            ),
            layout.log("baseline"),
            timeout_s,
        )

        with donor_session(
            command=worker_command(
                role="donor",
                model_path=donor_snapshot,
                tp_size=case.tp_size,
                output_path=layout.output("donor"),
                mx_url=mx_url,
                ready_file=layout.donor_ready,
                stop_file=layout.donor_stop,
                max_serve_seconds=timeout_s + 120,
            ),
            environment=worker_environment(
                donor_gpu_ids, layout.transfer_logs("donor"), layout.manifest_dir
            ),
            log_path=layout.log("donor"),
            ready_file=layout.donor_ready,
            stop_file=layout.donor_stop,
            timeout_s=timeout_s,
        ) as donor:
            run_worker(
                worker_command(
                    role="receiver",
                    model_path=receiver_snapshot,
                    tp_size=case.tp_size,
                    output_path=layout.output("receiver"),
                    mx_url=mx_url,
                ),
                worker_environment(
                    receiver_gpu_ids, layout.transfer_logs("receiver"), layout.manifest_dir
                ),
                layout.log("receiver"),
                timeout_s,
            )

        assert donor.returncode == 0, (
            f"MX donor exited with status {donor.returncode}\n{log_tail(layout.log('donor'))}"
        )
        payloads = {role: load_payload(layout.output(role)) for role in ROLES}
        assert payloads["donor"]["server_query_timeout_s"] == 0
        tokens = {role: assert_probe_payload(payloads[role], role=role) for role in ROLES}
        assert (
            payloads["baseline"]["prompt_lengths"]
            == payloads["donor"]["prompt_lengths"]
            == payloads["receiver"]["prompt_lengths"]
        )
        assert tokens["donor"] == tokens["baseline"], "MX donor tokens differ from the HF baseline"
        assert tokens["receiver"] == tokens["baseline"], (
            "MX receiver tokens differ from the HF baseline"
        )
        assert_transfer_evidence(
            case, layout.log("receiver"), layout.transfer_logs("receiver"), payloads["receiver"]
        )
        manifests = collect_weight_manifests(layout.manifest_dir, case)
        assert_weight_manifests(case, manifests)
        report_timings(case, payloads, manifests, layout)
    finally:
        archive_run_artifacts(output_dir, request.node.callspec.id, layout)
