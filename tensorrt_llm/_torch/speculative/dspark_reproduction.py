# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark adapter for explicit offline collection and ordinary automatic policy."""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from mpi4py import MPI

from tensorrt_llm._torch.pyexecutor.dspark_observation_runtime import observation_factory
from tensorrt_llm._torch.speculative.dspark_cost_profiler import _collective, response_rows
from tensorrt_llm._utils import mpi_comm
from tensorrt_llm.executor.worker import GenerationExecutorWorker
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, TorchLlmArgs
from tensorrt_llm.tools.dspark_cost_export import (
    source_origin_from_args,
    source_origin_input_metadata,
)
from tensorrt_llm.tools.dspark_observations import validate_responses
from tensorrt_llm.tools.verify_costs import (
    canonical_sha256,
    create_private_output_directory,
    isolated_control_communicator,
    parse_json,
    sha256_file,
    write_new_json,
)

_POLICY_FIELDS = (
    "enable_confidence_scheduling",
    "enable_fused_confidence_scheduler",
    "validate_confidence_artifact_freshness",
    "confidence_sts_path",
    "confidence_sps_table_path",
    "confidence_sps_live_fingerprint_path",
    "confidence_admission_receipt_path",
    "confidence_admission_receipt_sha256",
)


class ObservationWorker(GenerationExecutorWorker):
    def __init__(self, *, collect_calibration=False, **kwargs):
        self._collect_calibration = collect_calibration
        if kwargs.get("rpc_addr") is not None:
            raise ValueError("Offline collection/benchmark cannot expose serving RPC")
        super().__init__(**kwargs)

    def _get_pytorch_executor_factory(self):
        return observation_factory(collect_calibration=self._collect_calibration)


def _prepare(args, world):
    if os.environ.get("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"):
        raise ValueError("Collection and benchmarking require real target acceptance")
    if (
        args.physical_k < 1
        or args.batch_size < 1
        or args.output_tokens < 1
        or args.warmup_cohorts < 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        raise ValueError("Positive K/BS/output/full budget and nonnegative warmups required")
    if world > 1 and MPI.Query_thread() < MPI.THREAD_MULTIPLE:
        raise ValueError("Concurrent ordinary worker/control MPI requires THREAD_MULTIPLE")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("speculative_config"), dict):
        raise ValueError("Explicit public DSpark speculative_config required")
    spec = config["speculative_config"]
    if spec.get("max_draft_len") != args.physical_k:
        raise ValueError("CLI physical K differs from actual configuration")
    numerical = copy.deepcopy(config)
    for field in _POLICY_FIELDS:
        numerical["speculative_config"].pop(field, None)
    if args.mode == "dynamic":
        if args.physical_k == 1 or args.cost_table is None:
            raise ValueError("Dynamic chain needs K>1 and a structurally valid measured cost table")
        spec["enable_confidence_scheduling"] = True
        spec["confidence_sps_table_path"] = str(args.cost_table.resolve(strict=True))
        if args.calibration is not None:
            spec["confidence_sts_path"] = str(args.calibration.resolve(strict=True))
    else:
        for field in _POLICY_FIELDS:
            spec.pop(field, None)
        spec["enable_confidence_scheduling"] = False
        if args.cost_table is not None or args.calibration is not None:
            raise ValueError("Static/calibration collection does not consume dynamic artifacts")
    llm_args = TorchLlmArgs(**{**config, "model": str(args.model), "backend": "pytorch"})
    mapping = llm_args.parallel_config.to_mapping()
    factor = world if mapping.enable_attention_dp else 1
    if (
        not isinstance(llm_args.speculative_config, DSparkDecodingConfig)
        or not llm_args.speculative_config.draft_is_embedded_in_target
        or mapping.tp_size != world
        or mapping.pp_size != 1
        or mapping.cp_size != 1
        or args.batch_size % factor
        or llm_args.max_batch_size < args.batch_size // factor
        or llm_args.cuda_graph_config is None
        or not llm_args.cuda_graph_config.enable_padding
    ):
        raise ValueError(
            "K/topology/BS capacity and padded decode graphs must match the request cohort"
        )
    if (
        llm_args.guided_decoding_backend is not None
        or llm_args.batched_logits_processor is not None
        or llm_args.lora_config is not None
        or llm_args.cache_transceiver_config is not None
        or llm_args.kv_connector_config is not None
        or llm_args.enable_in_graph_sampling
        or llm_args.encode_only
        or llm_args.enable_resource_governor
    ):
        raise ValueError(
            "This fixed-cohort driver supports ordinary aggregate greedy generation only"
        )
    if args.mode == "collect" and llm_args.disable_overlap_scheduler:
        raise ValueError(
            "This source-paired collector currently uses the original overlap device-input seam"
        )
    data = args.inputs.read_bytes()
    inputs = [parse_json(line) for line in data.splitlines() if line.strip()][: args.batch_size]
    if len(inputs) != args.batch_size or any(
        not isinstance(row, dict)
        or not isinstance(row.get("prompt_token_ids"), list)
        or not row["prompt_token_ids"]
        or any(type(token) is not int or token < 0 for token in row["prompt_token_ids"])
        for row in inputs
    ):
        raise ValueError("One nonempty integer-token prompt per selected request required")
    for index, row in enumerate(inputs):
        row["input_id"] = index
        if len(row["prompt_token_ids"]) + args.output_tokens > llm_args.max_seq_len:
            raise ValueError("Fixed output would exceed original max_seq_len")
    origin = source_origin_from_args(args)
    contract = {
        "physical_k": args.physical_k,
        "batch_size": args.batch_size,
        "output_tokens": args.output_tokens,
        "inputs_sha256": canonical_sha256(inputs),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "numerical_configuration_sha256": canonical_sha256(numerical),
        "warmup_cohorts": args.warmup_cohorts,
        "sampling": {"temperature": 0.0, "top_k": 1, "ignore_eos": True, "beam_width": 1},
    }
    return llm_args, inputs, contract, origin


def _observed_identity(worker, args, origin):
    # Finite source/native/runtime provenance. HEAD is only recorded base;
    # observed module bytes include this overlay and are never relabeled old.
    modules = (
        "tensorrt_llm._torch.speculative.dspark",
        "tensorrt_llm._torch.speculative.dspark_verify",
        "tensorrt_llm._torch.speculative.dspark_reproduction",
        "tensorrt_llm._torch.pyexecutor.model_engine",
        "tensorrt_llm._torch.pyexecutor.py_executor",
        "tensorrt_llm._torch.pyexecutor.cuda_graph_runner",
        "tensorrt_llm._torch.pyexecutor.dspark_observation_runtime",
        "tensorrt_llm._torch.speculative.spec_sampler_base",
        "tensorrt_llm.tools.dspark_observations",
        "tensorrt_llm.llmapi.llm_args",
    )
    loaded = {name: sha256_file(Path(importlib.import_module(name).__file__)) for name in modules}
    if source_origin_from_args(args) != origin:
        raise ValueError("Observed source changed during model construction")
    for name, digest in loaded.items():
        path = args.source_root / (name.replace(".", "/") + ".py")
        if sha256_file(path) != digest:
            raise ValueError("Actual loaded Python bytes differ from recorded source origin")
    native_paths = set(torch.ops.loaded_libraries)
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if (
            name.startswith(("tensorrt_llm.", "torch._C"))
            and path
            and Path(path).suffix in (".so", ".pyd")
        ):
            native_paths.add(path)
    native = {str(Path(path).resolve()): sha256_file(Path(path)) for path in sorted(native_paths)}
    if not native:
        raise ValueError("Actual native runtime identity missing")
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    mapping = worker.engine.model_engine.mapping
    return {
        **origin,
        "source_origin_observation": source_origin_input_metadata(
            getattr(args, "recorded_source_origin", None),
            getattr(args, "recorded_source_origin_sha256", None),
        ),
        "loaded_source_sha256": loaded,
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "tensorrt_llm": importlib.metadata.version("tensorrt_llm"),
        },
        "local_native_libraries": native,
        "local_device": {"name": device.name, "major": device.major, "minor": device.minor},
        "topology": {
            "tp": mapping.tp_size,
            "ep": mapping.moe_ep_size,
            "attention_dp": mapping.enable_attention_dp,
            "pp": mapping.pp_size,
        },
        "artifact_sha256": {
            name: sha256_file(path)
            for name, path in (("cost", args.cost_table), ("calibration", args.calibration))
            if path
        },
    }


def run_native(args):
    with isolated_control_communicator(mpi_comm()) as control:
        return _run_native(args, control)


def _run_native(args, control):
    rank, world = control.Get_rank(), control.Get_size()
    deadline = time.monotonic() + args.timeout
    llm_args, inputs, contract, origin = _collective(
        control, "prepare", lambda: _prepare(args, world)
    )
    if len(set(control.allgather(canonical_sha256((contract, origin))))) != 1:
        raise ValueError("Ranks disagree on request/configuration/source inputs")
    _collective(
        control,
        "fresh outputs",
        lambda: create_private_output_directory(args.output) if rank == 0 else None,
    )
    worker, record, payload = None, None, None
    succeeded, shutdown_completed = False, False
    try:

        def construct():
            nonlocal worker
            worker = ObservationWorker(
                collect_calibration=args.mode == "collect",
                engine=None,
                hf_model_dir=args.model,
                tokenizer=None,
                llm_args=llm_args,
                is_llm_executor=True,
            )
            return worker

        _collective(control, "construct", construct)
        identity = _collective(
            control, "observed identity", lambda: _observed_identity(worker, args, origin)
        )
        identity["devices"] = control.allgather(identity.pop("local_device"))
        identity["rank_native_libraries"] = control.allgather(
            identity.pop("local_native_libraries")
        )
        # Only measured runtime/source fields determine matched identity;
        # artifact origin metadata is preserved separately, not rewritten.
        contract["observed_runtime_sha256"] = canonical_sha256(
            {key: value for key, value in identity.items() if key != "artifact_sha256"}
        )
        for cohort in range(args.warmup_cohorts + 1):
            _collective(
                control,
                "reset prior cohort routing",
                lambda: worker.engine.observation_receipt(reset=True, control=control),
            )
            control.Barrier()
            started = time.monotonic()
            responses, error = None, None
            if rank == 0:
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Full original startup/model budget exhausted")
                    responses = response_rows(worker, inputs, args.output_tokens, remaining)
                except Exception as exception:
                    error = type(exception).__name__ + ": " + str(exception)
            error, responses, elapsed = control.bcast(
                (error, responses, time.monotonic() - started), root=0
            )
            if error:
                raise RuntimeError(error)
            _collective(
                control,
                "responses",
                lambda: validate_responses(responses, inputs, args.output_tokens),
            )
            receipt = _collective(
                control,
                "original drain",
                lambda: worker.engine.observation_receipt(control=control),
            )
            if cohort == args.warmup_cohorts:
                routing = control.allgather(receipt["routing"])
                combined = {
                    name: sum(row[name] for row in routing)
                    for name in ("native_replays", "compact_replays")
                }
                record = {
                    "schema": "dspark-public-run-v1",
                    "mode": args.mode,
                    "calibration_collection": args.mode == "collect",
                    "inputs": inputs,
                    "responses": responses,
                    "elapsed_seconds": elapsed,
                    "comparison_contract": contract,
                    "identity": identity,
                    "resolved_confidence_enabled": receipt["resolved_confidence_enabled"],
                    "runtime_confidence_enabled": receipt["runtime_confidence_enabled"],
                    "routing": combined,
                    "rank_graph_keys": control.allgather(receipt["graph_keys"]),
                    "task_accuracy_qualified": False,
                    "production_admission_qualified": False,
                }
                if args.mode == "collect":
                    record["rank_drain"] = control.allgather(receipt["drain"])
        if args.mode == "collect":
            payload = _collective(
                control, "paired full-K payload", worker.engine._native_pairs.payload
            )
            payload["meta"]["identity"] = identity
        succeeded = True
    finally:
        failure = sys.exc_info()[1]
        try:
            _collective(control, "original shutdown", lambda: worker.shutdown() if worker else None)
            shutdown_completed = worker is not None
        finally:
            if record is not None:
                record["run_complete"] = succeeded and shutdown_completed
                record["model_shutdown_completed"] = shutdown_completed
                _collective(
                    control,
                    "preserve record",
                    lambda: write_new_json(args.output / f"rank-{rank}-run.json", record),
                )
            if succeeded and shutdown_completed and payload is not None:
                payload["meta"]["model_shutdown_completed"] = True

                def preserve_pairs():
                    with (args.output / f"rank-{rank}-pairs.pt").open("xb") as handle:
                        torch.save(payload, handle)

                _collective(control, "preserve pairs", preserve_pairs)
            if not succeeded or not shutdown_completed:
                _collective(
                    control,
                    "preserve failure",
                    lambda: write_new_json(
                        args.output / f"rank-{rank}-failure.json",
                        {
                            "schema": "dspark-public-failure-v1",
                            "run_complete": False,
                            "failure_type": type(failure).__name__
                            if failure
                            else "ShutdownFailure",
                            "model_shutdown_completed": shutdown_completed,
                        },
                    ),
                )
