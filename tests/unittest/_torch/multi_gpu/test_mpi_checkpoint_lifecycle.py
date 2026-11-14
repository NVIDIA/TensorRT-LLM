# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native resident lifecycle roundtrip, not a Snapshot process-restore qualification."""

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.executor.checkpoint import (
    AbortCheckpointRequest,
    ActivateRequest,
    CheckpointContext,
    CheckpointLifecycle,
    GenerationValidation,
    HostRestoreAttestation,
    PrepareCaptureRequest,
    ValidateRuntimeRequest,
    checkpoint_probe_sampling_params,
)
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.serve.checkpoint import ExecutorCheckpointBackend


@pytest.mark.gpu2
@pytest.mark.timeout(600, method="thread")
@pytest.mark.parametrize(
    ("tp", "stop_at"),
    [(tp, stop_at) for tp in (1, 2) for stop_at in ("capture_safe", "runtime_ready", "serving")]
    + [(2, "abort_after_validation")],
)
def test_mpi_resident_checkpoint_lifecycle(tp: int, stop_at: str, monkeypatch) -> None:
    """Exercise real rank drain, private graph replay and explicit activation.

    Args:
        tp: Fixed tensor-parallel cohort size.
        stop_at: Boundary at which context exit must shut down the whole cohort.
        monkeypatch: Test-owned environment restoration fixture.
    """
    environment = {
        "TRTLLM_ENABLE_CHECKPOINT_LIFECYCLE": "1",
        "TLLM_NCCL_SYMMETRIC_ZERO_COPY": "0",
        "TRTLLM_PROXY_BATCH_REQUESTS": "0",
        "TRTLLM_GPU_KEEPALIVE": "0",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    with LLM(
        model=str(llm_models_root() / "Qwen3/Qwen3-0.6B"),
        tensor_parallel_size=tp,
        allreduce_strategy="NCCL",
        max_batch_size=4,
        max_seq_len=512,
        kv_cache_config=KvCacheConfig(
            use_kv_cache_manager_v2=True,
            enable_block_reuse=False,
            host_cache_size=0,
            max_tokens=2048,
        ),
        env_overrides=environment,
    ) as llm:
        prompt = [1, 2, 3, 4]
        sampling = checkpoint_probe_sampling_params(8)
        reference = list(llm.generate([prompt], sampling)[0].outputs[0].token_ids)
        lifecycle = CheckpointLifecycle(ExecutorCheckpointBackend(llm))
        initial = lifecycle.get_lifecycle_status()
        common = dict(
            protocol_version=1,
            template_id="native-test",
            profile_digest=initial.profile.profile_digest,
            cohort_digest=initial.profile.cohort_digest,
            time_budget_seconds=60,
        )
        source = CheckpointContext(
            **common,
            operation_id="prepare",
            session_kind="source",
            session_id=initial.source_session_id,
            expected_state_version=0,
        )
        prepared = lifecycle.prepare_capture(PrepareCaptureRequest(context=source))
        assert prepared.status == "completed", prepared.error
        assert len(prepared.rank_results) == tp
        assert not lifecycle.get_lifecycle_status().readiness.admission_open
        if stop_at == "capture_safe":
            return
        restored = CheckpointContext(
            **common,
            operation_id="validate",
            session_kind="restore",
            session_id="native-destination",
            expected_state_version=prepared.state_version,
        )
        validated = lifecycle.validate_runtime(
            ValidateRuntimeRequest(
                context=restored,
                source_session_id=initial.source_session_id,
                validation=GenerationValidation(
                    prompt_token_ids=prompt, expected_token_ids=reference
                ),
            )
        )
        assert validated.status == "completed", validated.error
        assert all("warmed_graph_replayed" in item.checks for item in validated.rank_results)
        assert not lifecycle.get_lifecycle_status().readiness.admission_open
        if stop_at == "runtime_ready":
            return
        if stop_at == "abort_after_validation":
            abort_context = restored.model_copy(
                update={"operation_id": "abort", "expected_state_version": validated.state_version}
            )
            aborted = lifecycle.abort_operation(
                AbortCheckpointRequest(context=abort_context, reason="native abort test")
            )
            assert aborted.status == "completed", aborted.error
            assert {item.rank for item in aborted.rank_results} == set(range(tp))
            assert all("progression_stopped" in item.checks for item in aborted.rank_results)
            status = lifecycle.get_lifecycle_status()
            assert status.stage == "aborted"
            assert not status.readiness.admission_open
            assert not status.readiness.serving_ready
            with pytest.raises(
                RuntimeError, match="Cannot enqueue requests while executor admission is failed"
            ):
                llm.generate([prompt], sampling)
            return
        active = restored.model_copy(
            update={"operation_id": "activate", "expected_state_version": validated.state_version}
        )
        result = lifecycle.activate(
            ActivateRequest(
                context=active,
                readiness_id=validated.readiness_id,
                host_restore=HostRestoreAttestation(
                    result_id="simulated-host-notification",
                    status="succeeded",
                    session_id=active.session_id,
                    template_id=active.template_id,
                    profile_digest=active.profile_digest,
                    cohort_digest=active.cohort_digest,
                ),
            )
        )
        assert result.status == "completed", result.error
        assert lifecycle.get_lifecycle_status().readiness.serving_ready
        assert list(llm.generate([prompt], sampling)[0].outputs[0].token_ids) == reference
