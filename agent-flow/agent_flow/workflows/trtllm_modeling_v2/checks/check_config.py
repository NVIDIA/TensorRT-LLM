# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the flow reads off a deployment's native options without importing TensorRT-LLM.

A deployment states its engine's options as ``trtllm-serve --config`` takes them
(``spec/deployments.py``), so nothing here merges, overlays or rewrites them: the options
a session's server is started with are the spec's own. What is left is reading: the
decode CUDA-graph batch sizes an engine of those options captures
(:func:`options_capture`, a planning preview for the survey).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Mapping, Sequence

from ..spec import deployments as deployments_mod

#: The capture limit ``CudaGraphConfig`` uses when neither ``batch_sizes`` nor
#: ``max_batch_size`` is set (``BaseCudaGraphConfig.validate_base_cuda_graph_config``).
LLM_API_CAPTURE_MAX_BATCH_SIZE = 128


def llm_api_batch_sizes(
    batch_sizes: Sequence[int] | None = None,
    max_batch_size: int = 0,
    enable_padding: bool = False,
) -> tuple[int, ...]:
    """The decode batch sizes ``CudaGraphConfig(...)`` resolves to.

    The LLM API's own rule, restated so the orchestrator never imports
    TensorRT-LLM: ``BaseCudaGraphConfig.validate_base_cuda_graph_config`` and
    ``BaseCudaGraphConfig._generate_cuda_graph_batch_sizes`` in
    ``tensorrt_llm/llmapi/llm_args.py``. Raises ``ValueError`` where it does:
    on a ``max_batch_size`` beside explicit ``batch_sizes`` that is not their
    maximum. The engine's later filter by each launch's own limits
    (``filter_cuda_graph_batch_sizes``) is not part of this list.
    """
    if batch_sizes:
        sizes = sorted(int(b) for b in batch_sizes)
        if max_batch_size and int(max_batch_size) != sizes[-1]:
            raise ValueError(
                f"cuda_graph_config.max_batch_size {max_batch_size} is not "
                f"max(cuda_graph_config.batch_sizes) {sizes[-1]}; the LLM API refuses the pair"
            )
        return tuple(sizes)
    limit = int(max_batch_size) or LLM_API_CAPTURE_MAX_BATCH_SIZE
    if enable_padding:
        generated = [1, 2, 4] + [i * 8 for i in range(1, 17)]
        while generated[-1] + 64 <= limit:
            generated.append(generated[-1] + 64)
    else:
        generated = list(range(1, 32)) + [32, 64, 128]
        generated += [2**i for i in range(8, math.ceil(math.log(limit, 2)))]
    sizes = sorted(size for size in generated if size <= limit)
    if not sizes or sizes[-1] != limit:
        sizes.append(limit)
    return tuple(sizes)


def format_batches(sizes: Sequence[int]) -> str:
    """``1-32, 64, 128``: a batch list with each run of three or more consecutive sizes joined."""
    parts: list[str] = []
    run: list[int] = []
    for size in [*sizes, None]:
        if run and size == run[-1] + 1:
            run.append(size)
            continue
        if len(run) >= 3:
            parts.append(f"{run[0]}-{run[-1]}")
        else:
            parts.extend(str(s) for s in run)
        run = [size] if size is not None else []
    return ", ".join(parts)


@dataclass(frozen=True)
class CaptureList:
    """The decode CUDA-graph batch sizes one engine captures, and whether it pads to them.

    Two lists are equal when they capture the same sizes with the same
    padding; ``source`` says where one was read and takes no part in that.
    """

    batch_sizes: tuple[int, ...]
    enable_padding: bool
    source: str = dataclass_field(default="", compare=False)

    def describe(self) -> str:
        padded = "padded" if self.enable_padding else "unpadded"
        return f"{len(self.batch_sizes)} batches [{format_batches(self.batch_sizes)}], {padded}"


#: What a user who sets no ``cuda_graph_config`` gets from the LLM API.
LLM_API_DEFAULT_CAPTURE = CaptureList(
    llm_api_batch_sizes(), False, source="the LLM API default, `CudaGraphConfig()`"
)


def options_capture(
    options: Mapping[str, Any], source: str = "the options file"
) -> CaptureList | None:
    """What an engine built from LLM-API ``options`` captures; ``None`` when it captures nothing.

    Options that do not name ``cuda_graph_config`` get ``CudaGraphConfig()``,
    the field's default in ``TorchLlmArgs``; ``cuda_graph_config: null``
    turns CUDA graphs off. A ``speculative_config.draft_len_schedule`` turns
    padding on and adds its batch thresholds to the list, as ``TorchLlmArgs``'s
    speculative-config validator does (``DecodeCudaGraphConfig._merge_schedule_keys``).
    """
    if "cuda_graph_config" not in options:
        config: Any = {}
        where = f"{LLM_API_DEFAULT_CAPTURE.source}: {source} sets no `cuda_graph_config`"
    else:
        config = options["cuda_graph_config"]
        where = f"`cuda_graph_config` in {source}"
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError(f"`cuda_graph_config` in {source} is not a mapping: {config!r}")
    padding = bool(config.get("enable_padding", False))
    sizes = llm_api_batch_sizes(
        config.get("batch_sizes"), int(config.get("max_batch_size") or 0), padding
    )
    speculative = options.get("speculative_config")
    schedule = speculative.get("draft_len_schedule") if isinstance(speculative, Mapping) else None
    if isinstance(schedule, Mapping) and schedule:
        padding = True
        sizes = tuple(sorted({*sizes, *(int(k) for k in schedule if int(k) <= sizes[-1])}))
        where += " with `speculative_config.draft_len_schedule`"
    return CaptureList(sizes, padding, source=where)


@dataclass(frozen=True)
class LaunchCapture:
    """A planning preview of one deployment's capture list, not actual capture evidence."""

    name: str
    #: The deployment it is a server of; ``name`` is the deployment's too for its one
    #: server, and ``<deployment>.<role>`` for a role of a disaggregated one.
    unit: str
    options: str
    resolved: CaptureList | None


def launch_captures(task_data: Mapping[str, Any]) -> list[LaunchCapture]:
    """Each deployment's capture list, one per role of a disaggregated one.

    Raises ``ValueError`` on a ``cuda_graph_config`` the LLM API refuses.
    """
    found = []
    for deployment in deployments_mod.resolve(task_data):
        for role, server in deployment.servers.items():
            name = (
                deployment.name if role == deployments_mod.SERVER else f"{deployment.name}.{role}"
            )
            source = server.options_path or f"the options of `{name}`"
            try:
                resolved = options_capture(server.options, source)
            except ValueError as exc:
                raise ValueError(f"`{name}`: {exc}") from exc
            found.append(LaunchCapture(name, deployment.name, source, resolved))
    return found


__all__ = [
    "LLM_API_CAPTURE_MAX_BATCH_SIZE",
    "LLM_API_DEFAULT_CAPTURE",
    "CaptureList",
    "LaunchCapture",
    "format_batches",
    "launch_captures",
    "llm_api_batch_sizes",
    "options_capture",
]
