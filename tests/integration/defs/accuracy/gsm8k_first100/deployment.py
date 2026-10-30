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
"""The served deployment: explicit trigger, staged inputs, recipe and hardware identity.

Inputs are staged by the runner below ``LLM_MODELS_ROOT`` and named by three relative paths:
the target checkpoint (``TRTLLM_TEST_TARGET_RELATIVE_PATH``), the DFlash2 drafter checkpoint
(``TRTLLM_TEST_DRAFTER_RELATIVE_PATH``) and the FP8 KV-cache scale table
(``TRTLLM_TEST_KV_SCALES_RELATIVE_PATH``). The dataset is read from
``<LLM_MODELS_ROOT>/datasets/openai/gsm8k``. Before any launch every input is checked and
digested: the checkpoints' metadata files by content, their weight files by name and size,
and the scale table's ``target`` and ``draft`` sections against the checkpoints' layer counts.
The curated recipe's drafter placeholder is replaced in a per-run copy; nothing else in the
recipe is changed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

import yaml

from . import protocol

GUARD_ENV = "TRTLLM_RUN_NVFP4_SERVING_GSM8K100"
MODE_ENV = "TRTLLM_NVFP4_SERVING_GSM8K100_MODE"
TARGET_ENV = "TRTLLM_TEST_TARGET_RELATIVE_PATH"
DRAFTER_ENV = "TRTLLM_TEST_DRAFTER_RELATIVE_PATH"
KV_SCALES_ENV = "TRTLLM_TEST_KV_SCALES_RELATIVE_PATH"
DATASET_RELATIVE_DIR = "datasets/openai/gsm8k"

RECIPE_RELATIVE_PATH = "examples/configs/curated/beam-nvfp4-dflash2-fp8kv-aggregated.yaml"
DRAFTER_PLACEHOLDER = "<DRAFTER_CHECKPOINT>"
REASONING_PARSER = "beam_harmony_text"
#: The server environment the recipe documents; the scale table path is added per run.
SERVER_ENV = {"TRTLLM_SKIP_KV_CACHE_ESTIMATION": "1"}
KV_SCALES_SERVER_ENV = "TRTLLM_MODELING_V2_KV_CACHE_SCALES"

REQUIRED_GPU_COUNT = 4
REQUIRED_GPU_NAME = "GB300"
REQUIRED_COMPUTE_CAPABILITY = "10.3"

TARGET_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)
TARGET_OPTIONAL_FILES = ("generation_config.json", "hf_quant_config.json")
DRAFTER_FILES = ("config.json",)
DRAFTER_OPTIONAL_FILES = ("model.safetensors.index.json",)

_PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*>")
_GPU_FIELDS = ("index", "name", "uuid", "compute_cap", "memory.total", "driver_version")


def explicit_trigger(environ: Mapping[str, str]) -> bool:
    """True only when ``TRTLLM_RUN_NVFP4_SERVING_GSM8K100=1``; unset, empty or 0 is off."""
    value = environ.get(GUARD_ENV, "")
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{GUARD_ENV} must be 1 to run the qualification or unset/0, got {value!r}")


def run_mode(environ: Mapping[str, str]) -> str:
    """``measurement`` (default) or ``regression``."""
    mode = environ.get(MODE_ENV, "") or "measurement"
    if mode not in ("measurement", "regression"):
        raise ValueError(f"{MODE_ENV} must be measurement or regression, got {mode!r}")
    return mode


@dataclass(frozen=True)
class Inputs:
    """Resolved locations of the staged inputs; ``relative`` keeps the configured names."""

    target: str
    drafter: str
    kv_scales: str
    dataset_dir: str
    relative: dict[str, str]


def _relative(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "")
    if not value:
        raise ValueError(f"{name} must name the input relative to LLM_MODELS_ROOT")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a path below LLM_MODELS_ROOT, got {value!r}")
    return path.as_posix()


def resolve_inputs(environ: Mapping[str, str], models_root: str) -> Inputs:
    """The staged target, drafter, scale table and dataset directory below ``models_root``."""
    relative = {name: _relative(environ, name) for name in (TARGET_ENV, DRAFTER_ENV, KV_SCALES_ENV)}
    relative["dataset"] = DATASET_RELATIVE_DIR
    paths = {key: os.path.join(models_root, value) for key, value in relative.items()}
    for key in (TARGET_ENV, DRAFTER_ENV, "dataset"):
        if not os.path.isdir(paths[key]):
            raise FileNotFoundError(f"staged input {relative[key]!r} ({key}) is not a directory")
    if not os.path.isfile(paths[KV_SCALES_ENV]):
        raise FileNotFoundError(f"staged scale table {relative[KV_SCALES_ENV]!r} is not a file")
    return Inputs(
        target=paths[TARGET_ENV],
        drafter=paths[DRAFTER_ENV],
        kv_scales=paths[KV_SCALES_ENV],
        dataset_dir=paths["dataset"],
        relative=relative,
    )


def _weight_files(directory: str) -> list[str]:
    index_path = os.path.join(directory, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path} has no weight_map")
        return sorted(set(weight_map.values()))
    return ["model.safetensors"]


def describe_checkpoint(
    directory: str, required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Digests of a checkpoint's metadata files and the name/size inventory of its weights.

    Every ``required`` file and every weight file named by the safetensors index (or the
    single ``model.safetensors``) must exist. Weight contents are not hashed.
    """
    files = {}
    for name in required + optional:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            if name in required:
                raise FileNotFoundError(f"checkpoint file {name} is missing")
            continue
        files[name] = {"sha256": protocol.sha256_file(path), "bytes": os.path.getsize(path)}
    inventory = []
    for name in _weight_files(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"weight file {name} named by the checkpoint is missing")
        inventory.append([name, os.path.getsize(path)])
    with open(os.path.join(directory, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    layers = config.get("num_hidden_layers")
    return {
        "files": files,
        "weights": {
            "count": len(inventory),
            "bytes": sum(size for _, size in inventory),
            "inventory_sha256": protocol.sha256_text(protocol.canonical_json(inventory)),
        },
        "num_hidden_layers": layers if type(layers) is int else None,
    }


def _check_scale_section(table: Mapping[str, Any], section: str, layers: Optional[int]) -> str:
    entry = table.get(section)
    if not isinstance(entry, dict):
        raise ValueError(f"the scale table has no {section!r} section")
    count, scales = entry.get("layers"), entry.get("scales")
    if type(count) is not int or count < 1 or not isinstance(scales, dict):
        raise ValueError(f"scale section {section!r} needs an integer 'layers' and a 'scales' map")
    if sorted(scales) != sorted(str(i) for i in range(count)):
        raise ValueError(f"scale section {section!r} must name layers 0..{count - 1}")
    for key, value in scales.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"scale section {section!r} layer {key} is not finite and > 0")
    if layers is not None and count != layers:
        raise ValueError(
            f"scale section {section!r} covers {count} layers; its checkpoint has {layers}"
        )
    return protocol.sha256_text(protocol.canonical_json(entry))


def describe_kv_scales(
    path: str, target_layers: Optional[int], draft_layers: Optional[int]
) -> dict[str, Any]:
    """Digest of the scale table and of its ``target`` and ``draft`` sections."""
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    if not isinstance(table, dict):
        raise ValueError("the scale table must be a JSON object")
    return {
        "sha256": protocol.sha256_file(path),
        "target_sha256": _check_scale_section(table, "target", target_layers),
        "draft_sha256": _check_scale_section(table, "draft", draft_layers),
    }


def _placeholders(value: Any, where: str = "") -> list[str]:
    if isinstance(value, dict):
        return [hit for key, item in value.items() for hit in _placeholders(item, f"{where}.{key}")]
    if isinstance(value, list):
        return [hit for i, item in enumerate(value) for hit in _placeholders(item, f"{where}[{i}]")]
    if isinstance(value, str) and _PLACEHOLDER.search(value):
        return [f"{where.lstrip('.')}={value}"]
    return []


def materialize_recipe(template: str, drafter_dir: str) -> tuple[str, dict[str, Any]]:
    """The recipe text with its drafter placeholder replaced, and its identity and settings.

    The recipe must be the aggregated FP8-KV DFlash2 deployment this test qualifies: a
    ``speculative_config.speculative_model`` holding the placeholder (and no other
    placeholder), an FP8 KV cache, the reply channel parser, a positive ``max_batch_size``
    (the client's concurrency) and a ``max_seq_len`` above the completion allowance.
    """
    with open(template, "rb") as f:
        raw = f.read()
    try:
        config = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError(f"{template} is not valid YAML: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"{template} is not a YAML mapping")
    spec = config.get("speculative_config")
    if not isinstance(spec, dict) or spec.get("speculative_model") != DRAFTER_PLACEHOLDER:
        raise ValueError(
            f"the recipe's speculative_config.speculative_model must be {DRAFTER_PLACEHOLDER}"
        )
    spec["speculative_model"] = drafter_dir
    unresolved = _placeholders(config)
    if unresolved:
        raise ValueError(f"the recipe has unresolved placeholders: {unresolved}")
    batch, seq = config.get("max_batch_size"), config.get("max_seq_len")
    if type(batch) is not int or batch < 1:
        raise ValueError(f"the recipe must set a positive integer max_batch_size, got {batch!r}")
    if type(seq) is not int or seq <= protocol.MAX_TOKENS:
        raise ValueError(
            f"the recipe's max_seq_len {seq!r} cannot hold a prompt plus the "
            f"{protocol.MAX_TOKENS}-token allowance"
        )
    if config.get("reasoning_parser") != REASONING_PARSER:
        raise ValueError(f"the recipe must select reasoning_parser {REASONING_PARSER}")
    kv_cache = config.get("kv_cache_config")
    if not isinstance(kv_cache, dict) or kv_cache.get("dtype") != "fp8":
        raise ValueError("the recipe must select an FP8 KV cache (kv_cache_config.dtype: fp8)")
    text = yaml.safe_dump(config, sort_keys=False)
    settings = {key: value for key, value in config.items() if key != "speculative_config"}
    settings["speculative_config"] = {
        key: value for key, value in spec.items() if key != "speculative_model"
    }
    return text, {
        "template_sha256": hashlib.sha256(raw).hexdigest(),
        "materialized_sha256": protocol.sha256_text(text),
        "max_batch_size": batch,
        "max_seq_len": seq,
        "settings": settings,
    }


def serve_command(target_dir: str, config_path: str, addr_path: str) -> list[str]:
    """The documented aggregated launch; the kernel picks the port and reports it."""
    return [
        "trtllm-serve",
        target_dir,
        "--config",
        config_path,
        "--chat_template",
        os.path.join(target_dir, "chat_template.jinja"),
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--report_addr",
        addr_path,
    ]


def server_environment(base: Mapping[str, str], kv_scales: str) -> dict[str, str]:
    """The launch environment: the caller's, the recipe's settings and no network fetches."""
    env = dict(base)
    env.update(SERVER_ENV)
    env[KV_SCALES_SERVER_ENV] = kv_scales
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def query_gpus(
    environ: Mapping[str, str], run: Callable[..., Any] = subprocess.run
) -> list[dict[str, str]]:
    """The GPUs a launched server would use, in ``CUDA_VISIBLE_DEVICES`` order."""
    result = run(
        ["nvidia-smi", f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed with exit code {result.returncode}: {result.stderr}")
    gpus = [
        dict(zip(_GPU_FIELDS, (part.strip() for part in line.split(","))))
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return gpus
    selected = []
    for token in (part.strip() for part in visible.split(",")):
        if not token:
            continue
        match = next((gpu for gpu in gpus if token in (gpu["index"], gpu["uuid"])), None)
        if match is None:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES entry {token!r} names no GPU")
        selected.append(match)
    return selected


def check_hardware(gpus: list[Mapping[str, str]]) -> list[str]:
    """Reasons the visible GPUs cannot host the four-GB300 (SM103) recipe; empty if they can."""
    problems = []
    if len(gpus) < REQUIRED_GPU_COUNT:
        problems.append(f"{len(gpus)} GPU(s) visible; the recipe needs {REQUIRED_GPU_COUNT}")
    for gpu in gpus[:REQUIRED_GPU_COUNT]:
        if REQUIRED_GPU_NAME not in gpu.get("name", ""):
            problems.append(f"GPU {gpu.get('index')} is {gpu.get('name')!r}, not a GB300")
        if gpu.get("compute_cap") != REQUIRED_COMPUTE_CAPABILITY:
            problems.append(
                f"GPU {gpu.get('index')} has compute capability {gpu.get('compute_cap')!r}, "
                f"not {REQUIRED_COMPUTE_CAPABILITY} (SM103)"
            )
    return problems


@contextlib.contextmanager
def gpu_memory_poller(csv_path: Path, gpu_ids: list[str]) -> Iterator[subprocess.Popen]:
    """Poll per-GPU used memory once a second into ``csv_path`` until the context ends."""
    with open(csv_path, "w", encoding="utf-8") as out:
        poller = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,uuid,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-lms",
                "1000",
                "-i",
                ",".join(gpu_ids),
            ],
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        try:
            yield poller
        finally:
            poller.terminate()
            try:
                poller.wait(timeout=10)
            except subprocess.TimeoutExpired:
                poller.kill()
                poller.wait()
