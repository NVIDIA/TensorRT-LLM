# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark chain adapter and explicit offline driver; no serving admission."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable

import torch
import yaml
from mpi4py import MPI

from tensorrt_llm._torch.pyexecutor.dspark_request_profile import create_profile_executor
from tensorrt_llm._torch.speculative.dspark_cost_profile import DSparkCostProfileConfig
from tensorrt_llm._utils import mpi_comm
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.worker import GenerationExecutorWorker
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, TorchLlmArgs
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.tools.dspark_cost_export import (
    observed_engine_fingerprint,
    source_origin_from_args,
    source_origin_input_metadata,
)
from tensorrt_llm.tools.verify_costs import (
    canonical_sha256,
    create_private_output_directory,
    isolated_control_communicator,
    parse_json,
    sha256_file,
    write_new_json,
)

__all__ = ["run_profile"]


class OwnedInputWorker(GenerationExecutorWorker):
    """Construct the owned factory inside each original worker, not its proxy."""

    def __init__(
        self,
        *,
        profile: DSparkCostProfileConfig,
        **kwargs,
    ) -> None:
        self._profile = profile
        if kwargs.get("rpc_addr") is not None:
            raise ValueError("Offline profiling cannot expose a serving RPC endpoint")
        super().__init__(**kwargs)

    def _get_pytorch_executor_factory(self) -> Callable:
        # Retain every original worker setup operation and its ordering.
        return partial(create_profile_executor, profile=self._profile)


def response_rows(worker, inputs, output_length, timeout):
    """Use original request conversion/enqueue/futures, with real native IDs."""
    deadline = time.monotonic() + timeout
    futures = []
    for row in inputs:
        if len(row["prompt_token_ids"]) + output_length > worker.max_seq_len:
            raise ValueError("Original max_seq_len would shorten the fixed output length")
        sampling = SamplingParams(
            max_tokens=output_length,
            temperature=0.0,
            top_k=1,
            n=1,
            best_of=1,
            ignore_eos=True,
            detokenize=False,
            exclude_input_from_output=True,
        )
        request = GenerationRequest(
            prompt_token_ids=list(row["prompt_token_ids"]),
            sampling_params=sampling,
            streaming=False,
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("Original total child request budget exhausted before enqueue")
        future = worker.submit(request)
        native_id = worker._client_id_to_request_id[request.id]
        futures.append((row, request.id, native_id, future))
    results = []
    for row, client_id, native_id, future in futures:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Original total child request budget exhausted before response")
        result = future.result(timeout=remaining)
        outputs = result.outputs
        if (
            len(outputs) != 1
            or outputs[0].finish_reason != "length"
            or len(outputs[0].token_ids) != output_length
            or any(type(token) is not int or token < 0 for token in outputs[0].token_ids)
        ):
            raise RuntimeError("Original response did not complete the fixed greedy output")
        results.append(
            {
                "input_id": row["input_id"],
                "client_id": client_id,
                "native_request_id": native_id,
                "prompt_token_ids": row["prompt_token_ids"],
                "output_token_ids": list(outputs[0].token_ids),
                "finish_reason": outputs[0].finish_reason,
            }
        )
    return results


def _collective(control, label, callback):
    error, result = None, None
    try:
        result = callback()
    except Exception as exception:
        error = type(exception).__name__ + ": " + str(exception)
    errors = control.allgather(error)
    if any(error is not None for error in errors):
        raise RuntimeError(label + ": " + repr(errors))
    return result


def _identity(worker, args, profile, expected_files):
    """Record live installed sources/runtime and adapter-defined execution layout."""
    modules = (
        "tensorrt_llm._torch.pyexecutor.model_engine",
        "tensorrt_llm._torch.pyexecutor._util",
        "tensorrt_llm._torch.pyexecutor.py_executor_creator",
        "tensorrt_llm._torch.pyexecutor.cuda_graph_runner",
        "tensorrt_llm._torch.pyexecutor.py_executor",
        "tensorrt_llm.executor.base_worker",
        "tensorrt_llm.executor.worker",
        "tensorrt_llm.llmapi.llm_args",
        "tensorrt_llm._torch.speculative.dspark",
        "tensorrt_llm._torch.pyexecutor.dspark_profile_engine",
        "tensorrt_llm._torch.pyexecutor.dspark_request_profile",
        "tensorrt_llm._torch.speculative.dspark_cost_profile",
        "tensorrt_llm._torch.speculative.dspark_cost_profiler",
        "tensorrt_llm.tools.verify_costs",
    )
    if profile.measurement_scope == "whole-step":
        modules += ("tensorrt_llm.tools.dspark_cost_export",)
    sources = {}
    for name in modules:
        module = importlib.import_module(name)
        sources[name] = sha256_file(Path(module.__file__))
    native_paths = set(torch.ops.loaded_libraries)
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("tensorrt_llm.", "torch._C")):
            path = getattr(module, "__file__", None)
            if path is not None and Path(path).suffix in (".so", ".pyd"):
                native_paths.add(path)
    native = {str(Path(path).resolve()): sha256_file(Path(path)) for path in sorted(native_paths)}
    if not native:
        raise RuntimeError("Loaded native library identity is missing")
    package = importlib.metadata.metadata("tensorrt_llm")
    source_urls = [
        url for url in package.get_all("Project-URL", []) if url.startswith("Source Commit,")
    ]
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    versions = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "tensorrt_llm": importlib.metadata.version("tensorrt_llm"),
    }
    actual_files = {
        "inputs_sha256": sha256_file(args.inputs),
        "model_config_sha256": sha256_file(args.model / "config.json"),
    }
    if profile.measurement_scope == "whole-step":
        actual_files["recorded_source_origin"] = source_origin_from_args(args)
        actual_files["source_origin_observation"] = source_origin_input_metadata(
            getattr(args, "recorded_source_origin", None),
            getattr(args, "recorded_source_origin_sha256", None),
        )
    if actual_files != expected_files:
        raise ValueError("Profile inputs or model configuration changed after validation")
    mapping = worker.engine.model_engine.mapping
    model = worker.engine.model_engine.model
    attention_classes = sorted(
        {
            type(module).__module__ + "." + type(module).__qualname__
            for module in model.modules()
            if type(module).__module__.startswith("tensorrt_llm._torch.attention")
        }
    )
    if profile.measurement_scope == "whole-step":
        # Retain observed model/module overlays as well as the adapter roster.
        # Native library bytes are already recorded independently above.
        for instance in model.modules():
            name = type(instance).__module__
            if name.startswith("tensorrt_llm.") and name not in sources:
                module = importlib.import_module(name)
                path = getattr(module, "__file__", None)
                if path is not None and Path(path).suffix == ".py":
                    sources[name] = sha256_file(Path(path))
    return {
        "identity_scope": "DIAGNOSTIC_PARTIAL_LOADED_RUNTIME",
        "production_live_fingerprint_qualified": False,
        "loaded_source_sha256": sources,
        "package_source_commit_urls": source_urls,
        "runtime": versions,
        "local_native_libraries": native,
        "local_device": {
            "name": device.name,
            "major": device.major,
            "minor": device.minor,
            "total_memory": device.total_memory,
        },
        "topology": {
            "tp": mapping.tp_size,
            "pp": mapping.pp_size,
            "cp": mapping.cp_size,
            "ep": mapping.moe_ep_size,
            "attention_dp": mapping.enable_attention_dp,
        },
        "runtime_configuration": {
            "max_batch_size": worker.llm_args.max_batch_size,
            "max_seq_len": worker.max_seq_len,
            "max_num_tokens": worker.llm_args.max_num_tokens,
            "disable_overlap_scheduler": worker.llm_args.disable_overlap_scheduler,
            "enable_iter_perf_stats": worker.llm_args.enable_iter_perf_stats,
            "iter_perf_stats_interval": worker.llm_args.iter_perf_stats_interval,
            "cuda_graph_config": worker.llm_args.cuda_graph_config.model_dump(mode="json"),
            "kv_cache_config": worker.llm_args.kv_cache_config.model_dump(mode="json"),
            "configured_attention_backend": str(worker.llm_args.attn_backend),
        },
        **actual_files,
        "profile": profile.model_dump(mode="json"),
        "generations_per_shape": args.generations,
        "sampling": {
            "output_tokens": args.output_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "ignore_eos": True,
            "beam_width": 1,
        },
        "declared_shape_ids": [canonical_sha256(_shape(cell, profile)) for cell in profile.cells],
        "observed_attention_module_classes": attention_classes,
        "kernel_dispatch_qualification": False,
    }


def _shape(cell, profile) -> dict:
    """DSpark-specific shape identity; never reduce different attention layouts."""
    return {
        "workload": "dspark_chain",
        "physical_k": profile.physical_k,
        "graph_batch_size": cell.graph_batch_size,
        "verifier_budget": cell.verifier_budget,
        "attention_layout": "uniform" if cell.verifier_budget == 0 else "packed_chain_rows",
        "captured_head": profile.record_confidence_head,
    }


def _raw_generation(generation, cell, repetition, identity, rank, world):
    """The DSpark adapter decodes lifecycle records; the shared reducer does not."""
    samples = {row["sample_index"]: row for row in generation["observation"]["samples"]}
    output = []
    for join in generation["lifecycle_joins"]:
        index = join["sample_index"]
        if index is None:
            continue
        sample = samples.pop(index)
        scope = identity["profile"].get("measurement_scope", "graph-replay")
        duration = (
            join["host_step_time_ms"]
            if scope == "whole-step"
            else sample["replay_entry_elapsed_ms"]
        )
        output.append(
            {
                "sample_index": index,
                "executor_iteration": join["executor_iteration"],
                "elapsed_ms": duration,
                "graph_replay_diagnostic_ms": sample["replay_entry_elapsed_ms"],
                "eligible": all(join["sampler_update_applied"]),
                "request_ids": sample["request_ids"],
                "graph_key": sample["graph_key"],
            }
        )
    if samples:
        raise RuntimeError("Unjoined graph replay samples")
    graph_keys = {canonical_sha256(row["graph_key"]) for row in output}
    if len(graph_keys) != 1:
        raise RuntimeError("Prescribed cell observed zero or incompatible graph replay keys")
    profile = DSparkCostProfileConfig(**identity["profile"])
    shape = _shape(cell, profile)
    return {
        "schema": "verify-cost-raw-v1",
        "rank": rank,
        "world_size": world,
        "generation": repetition,
        "generation_closed": True,
        "model_shutdown_completed": False,
        "identity": identity,
        "timing_scope": (
            "original-executor-loop-hostStepTimeMS"
            if profile.measurement_scope == "whole-step"
            else "original-graph-replay-entry-including-static-copies"
        ),
        "declared_shape_id": canonical_sha256(shape),
        "execution_shape": {**shape, "observed_graph_key": output[0]["graph_key"]},
        "samples": output,
        "lifecycle": generation,
        "policy_overhead_measured": False,
        "production_admission_qualified": False,
    }


def _prepare(args, world):
    """Validate rank-local inputs before any peer enters collective construction."""
    if world > 1 and MPI.Query_thread() < MPI.THREAD_MULTIPLE:
        raise ValueError("Concurrent runtime/profile MPI requires THREAD_MULTIPLE")
    if (
        args.generations < 1
        or args.output_tokens < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        raise ValueError("Positive generations, output length and total timeout required")
    grid = parse_json(args.grid.read_bytes())
    if not isinstance(grid, dict):
        raise ValueError("The profile grid must be a JSON object")
    if "physical_k" in grid and grid["physical_k"] != args.physical_k:
        raise ValueError("CLI physical K and grid differ")
    cost_scope = getattr(args, "cost_scope", "graph-replay")
    if "measurement_scope" in grid and grid["measurement_scope"] != cost_scope:
        raise ValueError("CLI measurement scope and grid differ")
    profile = DSparkCostProfileConfig(
        **{**grid, "physical_k": args.physical_k, "measurement_scope": cost_scope}
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The runtime configuration must be a YAML mapping")
    llm_args = TorchLlmArgs(**{**config, "model": str(args.model), "backend": "pytorch"})
    if profile.measurement_scope == "whole-step" and (
        not llm_args.enable_iter_perf_stats or getattr(args, "source_root", None) is None
    ):
        raise ValueError("Whole-step collection requires enable_iter_perf_stats and --source-root")
    spec = llm_args.speculative_config
    mapping = llm_args.parallel_config.to_mapping()
    if profile.measurement_scope == "whole-step" and world > 1 and not mapping.enable_attention_dp:
        raise ValueError(
            "Schema-v2 whole-step export currently requires attention-DP fanout or one rank"
        )
    if (
        not isinstance(spec, DSparkDecodingConfig)
        or spec.max_draft_len != profile.physical_k
        or mapping.tp_size != world
        or mapping.pp_size != 1
        or mapping.cp_size != 1
    ):
        raise ValueError("Actual MPI topology/model physical K must match the profile")
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
        raise ValueError("The offline driver requires aggregate greedy generation without adapters")
    input_bytes = args.inputs.read_bytes()
    inputs = [parse_json(line) for line in input_bytes.splitlines() if line.strip()]
    factor = world if mapping.enable_attention_dp else 1
    maximum = max(cell.graph_batch_size for cell in profile.cells) * factor
    if len(inputs) < maximum or any(
        not isinstance(row, dict)
        or not isinstance(row.get("prompt_token_ids"), list)
        or not row["prompt_token_ids"]
        or any(type(token) is not int or token < 0 for token in row["prompt_token_ids"])
        for row in inputs
    ):
        raise ValueError("Enough nonempty original integer-token prompts are required")
    for index, row in enumerate(inputs):
        row["input_id"] = index
    files = {
        "inputs_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "model_config_sha256": sha256_file(args.model / "config.json"),
    }
    if profile.measurement_scope == "whole-step":
        files["recorded_source_origin"] = source_origin_from_args(args)
        files["source_origin_observation"] = source_origin_input_metadata(
            getattr(args, "recorded_source_origin", None),
            getattr(args, "recorded_source_origin_sha256", None),
        )
    fingerprint = canonical_sha256(
        {
            "config": config,
            "profile": profile.model_dump(mode="json"),
            "inputs": inputs,
            "generations": args.generations,
            "output_tokens": args.output_tokens,
            "timeout": args.timeout,
            "output_directory": str(args.output.resolve()),
            **files,
        }
    )
    return profile, llm_args, inputs, factor, fingerprint, files


def run_profile(args) -> None:
    """Use original worker requests and state updates for a prescribed offline grid."""
    # Original request-count/payload broadcasts stay on mpi_comm().  Main-thread
    # profile collectives need a separate context before any worker can start.
    with isolated_control_communicator(mpi_comm()) as control:
        _run_profile(args, control)


def _run_profile(args, control) -> None:
    """Keep the isolated control context alive through startup and all closure."""
    rank, world = control.Get_rank(), control.Get_size()
    profile, llm_args, inputs, factor, fingerprint, expected_files = _collective(
        control, "prepare", lambda: _prepare(args, world)
    )
    fingerprints = control.allgather(fingerprint)
    if len(set(fingerprints)) != 1:
        raise ValueError("Ranks disagree on inputs/configuration/grid")
    _collective(
        control,
        "output directory",
        lambda: create_private_output_directory(args.output) if rank == 0 else None,
    )
    worker = None
    records = []
    succeeded = False
    shutdown_completed = False
    try:

        def construct():
            nonlocal worker
            # Preserve a successful local owner even if a peer reports failure
            # at the following collective error-agreement boundary.
            worker = OwnedInputWorker(
                profile=profile,
                engine=None,
                hf_model_dir=args.model,
                tokenizer=None,
                llm_args=llm_args,
                is_llm_executor=True,
            )
            return worker

        _collective(control, "construct", construct)
        identity = _collective(
            control, "identity", lambda: _identity(worker, args, profile, expected_files)
        )
        identity["devices"] = control.allgather(identity.pop("local_device"))
        identity["rank_native_libraries"] = control.allgather(
            identity.pop("local_native_libraries")
        )
        if profile.measurement_scope == "whole-step":
            identity["recorded_engine_fingerprint"] = _collective(
                control,
                "recorded cost engine metadata",
                lambda: observed_engine_fingerprint(
                    identity,
                    source_root=args.source_root,
                    world_size=world,
                    recorded_origin=getattr(args, "recorded_source_origin", None),
                    recorded_origin_sha256=getattr(args, "recorded_source_origin_sha256", None),
                ),
            )
        if len(set(control.allgather(canonical_sha256(identity)))) != 1:
            raise ValueError("Actual rank identities differ")
        deadline = time.monotonic() + args.timeout
        generation_id = 0
        for cell in profile.cells:
            selected = inputs[: factor * cell.graph_batch_size]
            for repetition in range(args.generations):
                _collective(
                    control,
                    "begin",
                    lambda: worker.engine.begin_profile_generation(generation_id, cell),
                )
                responses, error = None, None
                if rank == 0:
                    try:
                        responses = response_rows(
                            worker,
                            selected,
                            args.output_tokens,
                            max(0, deadline - time.monotonic()),
                        )
                    except Exception as exception:
                        error = type(exception).__name__ + ": " + str(exception)
                error, responses = control.bcast((error, responses), root=0)
                if error is not None:
                    raise RuntimeError(error)
                generation = _collective(
                    control,
                    "drain",
                    lambda: worker.engine.end_profile_generation(control=control),
                )

                def join_generation():
                    by_id = {row["native_request_id"]: row for row in responses}
                    if len(by_id) != len(responses):
                        raise RuntimeError("Duplicate native response IDs")
                    for final in generation["original_final_requests"]:
                        response = by_id[final["request_id"]]
                        if final["tokens"] != (
                            response["prompt_token_ids"] + response["output_token_ids"]
                        ):
                            raise RuntimeError("Final native request tokens and response differ")
                    return _raw_generation(generation, cell, repetition, identity, rank, world)

                record = _collective(control, "generation join", join_generation)
                record["responses"] = responses
                records.append(record)
                generation_id += 1
        succeeded = True
    finally:
        original_failure = sys.exc_info()[1]
        try:
            # Every rank enters the same control phase, including a rank whose
            # constructor failed. Do not manufacture a model-shutdown receipt
            # for a worker that was never successfully constructed.
            _collective(
                control, "shutdown", lambda: worker.shutdown() if worker is not None else None
            )
            shutdown_completed = worker is not None
        finally:
            for index, record in enumerate(records):
                record["model_shutdown_completed"] = shutdown_completed
                record["run_complete"] = succeeded and shutdown_completed
                _collective(
                    control,
                    "preserve raw",
                    lambda: write_new_json(
                        args.output / f"rank-{rank}-record-{index}.json", record
                    ),
                )
            if not succeeded or not shutdown_completed:
                _collective(
                    control,
                    "preserve failure",
                    lambda: write_new_json(
                        args.output / f"rank-{rank}-failure.json",
                        {
                            "schema": "verify-cost-run-failure-v1",
                            "run_complete": False,
                            "failure_type": (
                                type(original_failure).__name__
                                if original_failure
                                else "ShutdownFailure"
                            ),
                            "model_shutdown_completed": shutdown_completed,
                        },
                    ),
                )
    if not succeeded:
        raise RuntimeError("Offline generation did not complete")
