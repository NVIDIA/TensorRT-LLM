# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""TRT-LLM usage telemetry collection and reporting.

Collects anonymous usage data (system info, GPU config, model architecture)
and sends it to NVIDIA's NvTelemetry/GXT service. Runs in a background
daemon thread, never blocks or crashes the main process.

Adapted from PR #11299 (usage lib POC), with:
- GXT Event Protocol v1.6 envelope (NvTelemetry-compliant)
- Architecture-class-only model sanitization
- DO_NOT_TRACK industry-standard env var support
- First-launch console notification

Environment variables:
    TRTLLM_NO_USAGE_STATS: Set to "1" to disable telemetry.
    TELEMETRY_DISABLED: Set to "true" or "1" to disable telemetry.
    DO_NOT_TRACK: Set to "1" to disable telemetry (industry standard).
    TRTLLM_USAGE_STATS_SERVER: Override the GXT endpoint URL.
    TRTLLM_USAGE_HEARTBEAT_INTERVAL: Heartbeat interval in seconds (default 600).
    TRTLLM_USAGE_FORCE_ENABLED: Set to "1" to force-enable telemetry even in
        CI/test environments (e.g., for staging deployments run via CI).

CI/Test auto-detection:
    Telemetry is automatically disabled when running in CI environments or
    test frameworks to ensure only real deployment data is collected. Detected
    via well-known environment variables set by CI systems (CI, GITHUB_ACTIONS,
    JENKINS_URL, etc.) and test runners (PYTEST_CURRENT_TEST). Override with
    TRTLLM_USAGE_FORCE_ENABLED=1 if needed.
"""

import json
import logging
import os
import platform
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from tensorrt_llm.usage import schema

logger = logging.getLogger("tensorrt_llm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DISAGG_ROLE_ENV = "TRTLLM_DISAGG_ROLE"
_DISAGG_DEPLOYMENT_ID_ENV = "TRTLLM_DISAGG_DEPLOYMENT_ID"
_DEFAULT_ENDPOINT = "https://events.gfe.nvidia.com/v1.1/events/json"
_HTTP_TIMEOUT = 2.0
_MAX_HEARTBEATS = 1000


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that rejects all redirects (SSRF protection).

    build_opener() auto-adds HTTPRedirectHandler unless a *subclass* is
    provided.  By passing this handler, the default is replaced and any
    3xx response raises HTTPError instead of being followed.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


try:
    _OPT_OUT_FILE: Optional[Path] = Path.home() / ".config" / "trtllm" / "do_not_track"
except (RuntimeError, KeyError):
    # Path.home() fails when HOME is unset and passwd lookup fails
    # (e.g. minimal containers).  Degrade gracefully — the file-based
    # opt-out simply becomes unavailable; env-var opt-out still works.
    _OPT_OUT_FILE = None

# ---------------------------------------------------------------------------
# CI / Test environment detection
# ---------------------------------------------------------------------------

# Well-known environment variables set by CI systems.
# If any of these are set (to any non-empty value), telemetry is auto-disabled.
_CI_ENV_VARS = (
    "CI",  # GitHub Actions, GitLab CI, Travis CI, generic
    "GITHUB_ACTIONS",  # GitHub Actions
    "JENKINS_URL",  # Jenkins
    "GITLAB_CI",  # GitLab CI
    "BUILDKITE",  # Buildkite
    "CIRCLECI",  # CircleCI
    "TRAVIS",  # Travis CI
    "TF_BUILD",  # Azure DevOps Pipelines
    "TEAMCITY_VERSION",  # TeamCity
    "CODEBUILD_BUILD_ID",  # AWS CodeBuild
)

# Well-known environment variables set by test frameworks.
_TEST_ENV_VARS = (
    "PYTEST_CURRENT_TEST",  # Set by pytest during test execution
)


def _is_ci_or_test_environment() -> bool:
    """Detect if we are running inside a CI pipeline or test framework.

    Returns True if any well-known CI or test environment variable is set
    to a non-empty value. This ensures telemetry only fires in real
    deployment scenarios -- not during development, testing, or CI runs.

    Neither vLLM nor NeMo DataDesigner implement CI/test auto-detection;
    they rely on CI engineers remembering to set opt-out env vars, which
    is fragile. By detecting CI/test environments automatically, we
    avoid polluting telemetry data with non-deployment noise.

    Users who genuinely want telemetry from CI (e.g., staging deployments)
    can override this by setting TRTLLM_USAGE_FORCE_ENABLED=1.
    """
    # Allow force-enable override for CI-based deployments
    if os.environ.get("TRTLLM_USAGE_FORCE_ENABLED", "0") == "1":
        return False

    for var in _CI_ENV_VARS:
        if os.environ.get(var):
            return True
    for var in _TEST_ENV_VARS:
        if os.environ.get(var):
            return True
    return False


def _get_stats_server() -> str:
    """Read endpoint URL at call time so env changes after import take effect.

    Validates overrides: HTTPS required, domain must be *.nvidia.com.
    Invalid overrides fall back to the default endpoint.
    """
    override = os.environ.get("TRTLLM_USAGE_STATS_SERVER")
    if override is None:
        return _DEFAULT_ENDPOINT

    try:
        parsed = urllib.parse.urlparse(override)
        if parsed.scheme != "https":
            logger.warning(
                "TRTLLM_USAGE_STATS_SERVER must use HTTPS; "
                "ignoring override and using default endpoint."
            )
            return _DEFAULT_ENDPOINT
        host = (parsed.hostname or "").lower()
        if not (host == "nvidia.com" or host.endswith(".nvidia.com")):
            logger.warning(
                "TRTLLM_USAGE_STATS_SERVER must be an *.nvidia.com domain; "
                "ignoring override and using default endpoint."
            )
            return _DEFAULT_ENDPOINT
    except Exception:
        logger.warning(
            "TRTLLM_USAGE_STATS_SERVER is not a valid URL; "
            "ignoring override and using default endpoint."
        )
        return _DEFAULT_ENDPOINT

    logger.info(f"Telemetry endpoint overridden: {override}")
    return override


def _get_heartbeat_interval() -> int:
    """Read heartbeat interval at call time, with safe fallback on bad values."""
    try:
        val = int(os.environ.get("TRTLLM_USAGE_HEARTBEAT_INTERVAL", "600"))
        return val if val > 0 else 600
    except ValueError:
        return 600


# ---------------------------------------------------------------------------
# Notification (shown once per process)
# ---------------------------------------------------------------------------

_NOTIFICATION_SHOWN = threading.Event()
_USAGE_NOTICE = (
    "TRT-LLM collects anonymous usage data to help improve the product. "
    "This data cannot be traced back to any individual user. "
    "No user-identifying information, persistent identifiers, or prompts "
    "are collected. To disable, set TRTLLM_NO_USAGE_STATS=1, "
    "TELEMETRY_DISABLED=true, or pass "
    "TelemetryConfig(disabled=True). "
    "See https://github.com/NVIDIA/TensorRT-LLM for details."
)


def _show_usage_notification():
    """Show a one-time usage notification via logger (thread-safe)."""
    if not _NOTIFICATION_SHOWN.is_set():
        _NOTIFICATION_SHOWN.set()
        logger.info(_USAGE_NOTICE)


# ---------------------------------------------------------------------------
# Opt-out check
# ---------------------------------------------------------------------------


def is_usage_stats_enabled(telemetry_disabled: bool = False) -> bool:
    """Check whether usage stats collection is enabled.

    Returns False if any of these conditions are met:
    - telemetry_disabled=True (programmatic opt-out via LLM API or CLI)
    - TRTLLM_NO_USAGE_STATS=1
    - TELEMETRY_DISABLED=true/1 (case-insensitive)
    - DO_NOT_TRACK=1 (industry standard: https://consoledonottrack.com/)
    - File ~/.config/trtllm/do_not_track exists
    - Running in a CI pipeline or test framework (auto-detected)
      Override with TRTLLM_USAGE_FORCE_ENABLED=1 if needed.
    """
    if telemetry_disabled:
        return False
    if os.environ.get("TRTLLM_NO_USAGE_STATS", "0") == "1":
        return False
    if os.environ.get("TELEMETRY_DISABLED", "").lower() in ("1", "true"):
        return False
    if os.environ.get("DO_NOT_TRACK", "0") == "1":
        return False
    if _OPT_OUT_FILE is not None and _OPT_OUT_FILE.exists():
        return False
    if _is_ci_or_test_environment():
        logger.debug(
            "Telemetry auto-disabled: CI/test environment detected. "
            "Set TRTLLM_USAGE_FORCE_ENABLED=1 to override."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


def _get_trtllm_version() -> str:
    """Get TRT-LLM package version, or 'unknown' if not installed."""
    try:
        import tensorrt_llm

        return getattr(tensorrt_llm, "__version__", "unknown")
    except (ImportError, AttributeError):
        return "unknown"


# ---------------------------------------------------------------------------
# System info collection (from PR #11299)
# ---------------------------------------------------------------------------


def _collect_system_info() -> Dict[str, Any]:
    """Collect platform, Python version, CPU info."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def _collect_gpu_info() -> Dict[str, Any]:
    """Collect GPU info via torch.cuda. Returns empty dict if unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
            "cuda_version": torch.version.cuda or "unknown",
        }
    except (ImportError, RuntimeError, AttributeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Model info extraction (sanitized -- architecture class name only)
# ---------------------------------------------------------------------------


def _extract_architecture_class_name(pretrained_config: Any) -> Optional[str]:
    """Extract the architecture class name from a pretrained model config.

    Handles three config formats:

    1. **HF PretrainedConfig** (from ``transformers.PretrainedConfig``):
       Has ``.architectures`` — a *list* of strings, e.g. ``["LlamaForCausalLM"]``.
       This is the standard format when loading from a HuggingFace model dir.

    2. [DEPRECATED] **TRT-LLM PretrainedConfig** (from ``tensorrt_llm.models.modeling_utils``):
       Has ``.architecture`` — a *singular string*, e.g. ``"LlamaForCausalLM"``.
       This is the format used in TRT-LLM checkpoint ``config.json`` files
       (``_ModelFormatKind.TLLM_CKPT``).

    3. [DEPRECATED] **Engine config loaded by HF** (``transformers.PretrainedConfig.from_pretrained``
       reading a TRT-LLM engine dir):
       The engine ``config.json`` has top-level keys ``pretrained_config`` (dict)
       and ``build_config`` (dict). HF's loader puts these as attributes on a
       generic ``PretrainedConfig`` object. The architecture string is at
       ``pretrained_config["architecture"]``.
    """
    if pretrained_config is None:
        return None
    try:
        # Case 1: HF PretrainedConfig — .architectures (plural list)
        architectures = getattr(pretrained_config, "architectures", None)
        if architectures and isinstance(architectures, (list, tuple)) and len(architectures) > 0:
            return str(architectures[0])

        # Case 2: TRT-LLM PretrainedConfig / TLLM_CKPT — .architecture (singular str)
        architecture = getattr(pretrained_config, "architecture", None)
        if architecture and isinstance(architecture, str):
            return architecture

        # Case 3: HF from_pretrained on engine dir — nested pretrained_config dict
        nested_config = getattr(pretrained_config, "pretrained_config", None)
        if isinstance(nested_config, dict) and "architecture" in nested_config:
            return str(nested_config["architecture"])

        # Last resort: config class name (e.g. "LlamaConfig")
        return type(pretrained_config).__name__
    except (AttributeError, TypeError, KeyError, IndexError):
        return None


# ---------------------------------------------------------------------------
# TRT-LLM config extraction
# ---------------------------------------------------------------------------


def _extract_trtllm_config(llm_args: Any) -> Dict[str, Any]:
    """Extract TRT-LLM configuration from LlmArgs.

    Args:
        llm_args: The args object from BaseLLM (TrtLlmArgs, TorchLlmArgs, etc.)

    Returns:
        Dict of config values, with None for unavailable fields.
    """
    if llm_args is None:
        return {}

    config = {}
    try:
        # Backend detection
        backend = getattr(llm_args, "backend", None)
        if backend is not None:
            config["backend"] = str(backend)
        else:
            # Infer backend from args class when not explicitly set
            cls_name = type(llm_args).__name__
            if "TrtLlm" in cls_name:
                config["backend"] = "tensorrt"

        # Parallelism
        parallel_config = getattr(llm_args, "parallel_config", None)
        if parallel_config is not None:
            config["tensor_parallel_size"] = getattr(parallel_config, "tp_size", None)
            config["pipeline_parallel_size"] = getattr(parallel_config, "pp_size", None)
            config["context_parallel_size"] = getattr(parallel_config, "cp_size", None)
            moe_ep = getattr(parallel_config, "moe_ep_size", None)
            if moe_ep is not None:
                # Map -1 (auto/unset) to 0 for telemetry; PositiveInt schema.
                config["moe_expert_parallel_size"] = max(moe_ep, 0)
            moe_tp = getattr(parallel_config, "moe_tp_size", None)
            if moe_tp is not None:
                config["moe_tensor_parallel_size"] = max(moe_tp, 0)

        # dtype
        dtype = getattr(llm_args, "dtype", None)
        if dtype is not None:
            config["dtype"] = str(dtype)

        # Quantization
        quant_config = getattr(llm_args, "quant_config", None)
        if quant_config is not None:
            quant_algo = getattr(quant_config, "quant_algo", None)
            if quant_algo is not None:
                config["quantization_algo"] = str(quant_algo)

        # KV cache dtype
        kv_cache_config = getattr(llm_args, "kv_cache_config", None)
        if kv_cache_config is not None:
            kv_dtype = getattr(kv_cache_config, "dtype", None)
            if kv_dtype is not None:
                config["kv_cache_dtype"] = str(kv_dtype)

    except (AttributeError, TypeError):
        pass  # fail-silent

    return config


# ---------------------------------------------------------------------------
# Feature flag collection
# ---------------------------------------------------------------------------

# Keys and defaults for the features JSON blob. All keys are always present
# in the output to simplify downstream analytics (no ambiguity between
# "feature disabled" and "field missing because old client version").
_FEATURES_DEFAULTS = {
    "lora": False,
    "speculative_decoding": False,
    "prefix_caching": False,
    "cuda_graphs": False,
    "chunked_context": False,
    "data_parallel_size": 1,
    "checkpoint_format": "HF",
    "load_format": "AUTO",
}


def _feature_enum_or_str(value: Any, default: str) -> str:
    """Convert low-cardinality config enum/string values for telemetry."""
    if value is None:
        return default
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(value, str) and value:
        return value
    return default


def _collect_features(llm_args: Any) -> str:
    """Collect feature flags from llm_args and return as compact JSON string.

    Inspects the LlmArgs object for enabled features (LoRA, speculative
    decoding, prefix caching, CUDA graphs, chunked context, data parallelism).
    Returns a JSON-serialized dict with snake_case keys. All keys are always
    present with safe defaults, even if extraction fails.

    The output is a string suitable for the ``featuresJson`` field in the
    GXT event schema (``stringVariableLength``).

    Args:
        llm_args: The args object from BaseLLM (TrtLlmArgs, TorchLlmArgs, etc.)
                  May be None.

    Returns:
        Compact JSON string, e.g. '{"lora":false,"speculative_decoding":false,...}'
    """
    features = dict(_FEATURES_DEFAULTS)

    if llm_args is None:
        return json.dumps(features, separators=(",", ":"))

    try:
        # LoRA: enabled if enable_lora flag is True OR lora_config is provided.
        # On PyTorch backend, enable_lora is ignored when lora_config is set,
        # so checking both catches all cases.
        enable_lora = getattr(llm_args, "enable_lora", False) or False
        lora_config = getattr(llm_args, "lora_config", None)
        features["lora"] = bool(enable_lora or lora_config is not None)

        # Speculative decoding: enabled if speculative_config is not None.
        spec_config = getattr(llm_args, "speculative_config", None)
        features["speculative_decoding"] = spec_config is not None

        # Prefix caching (KV block reuse): kv_cache_config.enable_block_reuse.
        # kv_cache_config has a default_factory (never None in practice), but
        # we guard defensively since llm_args may be a mock or partial object.
        kv_cache_config = getattr(llm_args, "kv_cache_config", None)
        if kv_cache_config is not None:
            block_reuse = getattr(kv_cache_config, "enable_block_reuse", None)
            if block_reuse is not None:
                features["prefix_caching"] = bool(block_reuse)

        # CUDA graphs: two different config paths depending on backend.
        # PyTorch backend: cuda_graph_config (TorchLlmArgs only).
        #   None = disabled; CudaGraphConfig() = enabled (default).
        # TRT backend: extended_runtime_perf_knob_config.cuda_graph_mode (TrtLlmArgs only).
        cuda_graph_config = getattr(llm_args, "cuda_graph_config", None)
        ext_config = getattr(llm_args, "extended_runtime_perf_knob_config", None)
        if cuda_graph_config is not None:
            # PyTorch path: presence of config object means enabled
            features["cuda_graphs"] = True
        elif ext_config is not None:
            # TRT path: explicit cuda_graph_mode flag
            features["cuda_graphs"] = bool(getattr(ext_config, "cuda_graph_mode", False))

        # Chunked context / chunked prefill: defined on BaseLlmArgs.
        features["chunked_context"] = bool(getattr(llm_args, "enable_chunked_prefill", False))

        # Checkpoint/load axes: low-cardinality, non-sensitive config values
        # used to distinguish HF/AUTO baseline, MX-only, GMS-only, and MX+GMS
        # compositions. Never include model names, paths, or server URLs here.
        features["checkpoint_format"] = _feature_enum_or_str(
            getattr(llm_args, "checkpoint_format", None), "HF"
        )
        features["load_format"] = _feature_enum_or_str(
            getattr(llm_args, "load_format", None), "AUTO"
        )

        # Data parallel size: derived from parallel_config.
        # dp_size = tp_size if enable_attention_dp else 1 (no dp_size field exists).
        parallel_config = getattr(llm_args, "parallel_config", None)
        if parallel_config is not None:
            enable_adp = getattr(parallel_config, "enable_attention_dp", False)
            if enable_adp:
                tp_size = getattr(parallel_config, "tp_size", 1) or 1
                features["data_parallel_size"] = int(tp_size)

    except Exception:
        pass  # fail-silent: return whatever we collected so far

    return json.dumps(features, separators=(",", ":"))


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


def _send_to_gxt(payload: dict) -> None:
    """Send a GXT payload via HTTP POST. Fail-silent.

    Uses urllib (stdlib) with 2s timeout and no redirects (SSRF protection).
    """
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _get_stats_server(),
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        # SSRF protection: use a custom opener that does NOT follow redirects.
        # build_opener auto-adds HTTPRedirectHandler unless a subclass is
        # provided, so we pass a handler that rejects all redirects.
        opener = urllib.request.build_opener(
            urllib.request.HTTPHandler,
            urllib.request.HTTPSHandler,
            _NoRedirectHandler,
        )
        opener.open(req, timeout=_HTTP_TIMEOUT)
    except (urllib.error.URLError, OSError, ValueError, TypeError):
        pass  # fail-silent: network errors, timeouts, etc.


# ---------------------------------------------------------------------------
# Background reporter (daemon thread)
# ---------------------------------------------------------------------------


def _clamp_str(value: str, max_len: int) -> str:
    """Truncate a string to max_len if it exceeds the limit."""
    return value[:max_len] if len(value) > max_len else value


def _background_reporter(
    llm_args: Any,
    pretrained_config: Any,
    usage_context: str = "",
) -> None:
    """Background thread entry point. Sends initial report + heartbeats.

    This function is the target of the daemon thread spawned by report_usage().
    It is wrapped in try/except at every level to ensure fail-silent behavior.
    """
    try:
        session_id = uuid.uuid4().hex
        trtllm_version = _get_trtllm_version()

        # --- Collect initial data ---
        system_info = _collect_system_info()
        gpu_info = _collect_gpu_info()
        arch_class_name = _extract_architecture_class_name(pretrained_config)
        trtllm_config = _extract_trtllm_config(llm_args)
        features_json = _collect_features(llm_args)

        # Disaggregated serving metadata (set by serve.py orchestrator)
        disagg_role = os.environ.get(_DISAGG_ROLE_ENV, "")
        deployment_id = os.environ.get(_DISAGG_DEPLOYMENT_ID_ENV, "")

        # --- Build initial report event ---
        # All fields are required by the SMS schema. Use empty string / 0
        # as sentinel values when actual data is unavailable (e.g., no GPU).
        # String values are clamped to schema limits (ShortString=128,
        # LongString=256) to prevent ValidationError from real-world data
        # exceeding the Pydantic field constraints.
        _S = schema._SHORT_STR  # ShortString maxLength
        _L = schema._LONG_STR  # LongString maxLength
        initial_event = schema.TrtllmInitialReport(
            trtllmVersion=_clamp_str(trtllm_version or "", _S),
            # System info
            platform=_clamp_str(system_info.get("platform") or "", _L),
            pythonVersion=_clamp_str(system_info.get("python_version") or "", _S),
            cpuArchitecture=_clamp_str(system_info.get("cpu_architecture") or "", _S),
            cpuCount=system_info.get("cpu_count") or 0,
            # GPU info
            gpuCount=gpu_info.get("gpu_count") or 0,
            gpuName=_clamp_str(gpu_info.get("gpu_name") or "", _L),
            gpuMemoryMB=gpu_info.get("gpu_memory_mb") or 0,
            cudaVersion=_clamp_str(gpu_info.get("cuda_version") or "", _S),
            # Model
            architectureClassName=_clamp_str(arch_class_name or "", _L),
            # Config
            backend=_clamp_str(trtllm_config.get("backend") or "", _S),
            tensorParallelSize=trtllm_config.get("tensor_parallel_size") or 1,
            pipelineParallelSize=trtllm_config.get("pipeline_parallel_size") or 1,
            contextParallelSize=trtllm_config.get("context_parallel_size") or 1,
            moeExpertParallelSize=trtllm_config.get("moe_expert_parallel_size", 0),
            moeTensorParallelSize=trtllm_config.get("moe_tensor_parallel_size", 0),
            dtype=_clamp_str(trtllm_config.get("dtype") or "", _S),
            quantizationAlgo=_clamp_str(trtllm_config.get("quantization_algo") or "", _S),
            kvCacheDtype=_clamp_str(trtllm_config.get("kv_cache_dtype") or "", _S),
            # Ingress point
            ingressPoint=_clamp_str(usage_context or "", _S),
            # Feature flags
            featuresJson=features_json,
            # Disaggregated serving
            disaggRole=_clamp_str(disagg_role, _S),
            deploymentId=_clamp_str(deployment_id, _S),
        )

        # --- Send initial report ---
        payload = schema.build_gxt_payload(
            event=initial_event,
            session_id=session_id,
            trtllm_version=trtllm_version,
        )
        _send_to_gxt(payload)

        # --- Heartbeat loop ---
        heartbeat_interval = _get_heartbeat_interval()
        for seq in range(_MAX_HEARTBEATS):
            if _REPORTER_STOP.wait(timeout=heartbeat_interval):
                return  # stop requested

            try:
                heartbeat_event = schema.TrtllmHeartbeat(seq=seq)
                heartbeat_payload = schema.build_gxt_payload(
                    event=heartbeat_event,
                    session_id=session_id,
                    trtllm_version=trtllm_version,
                )
                _send_to_gxt(heartbeat_payload)
            except (urllib.error.URLError, OSError, ValueError, TypeError):
                pass  # fail-silent on individual heartbeat

    except Exception:
        pass  # fail-silent: entire background reporter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_REPORTER_STARTED = False
_REPORTER_LOCK = threading.Lock()
_REPORTER_STOP = threading.Event()  # signal heartbeat loop to exit


def report_usage(
    llm_args: Any = None,
    pretrained_config: Any = None,
    telemetry_config: Any = None,
) -> None:
    """Start background usage telemetry reporting.

    Call this once after model initialization. It spawns a daemon thread
    that sends an initial report and periodic heartbeats. Subsequent calls
    are no-ops (only one reporter thread per process).

    This function is fail-silent -- it will never raise an exception or
    block the calling thread.

    Args:
        llm_args: The LlmArgs object from BaseLLM (for config extraction).
        pretrained_config: The pretrained model config (for architecture name).
        telemetry_config: TelemetryConfig object (opt-out + usage context).
    """
    global _REPORTER_STARTED
    try:
        # Extract fields from TelemetryConfig (defensive -- may be None or wrong type)
        disabled = False
        usage_context = ""
        if telemetry_config is not None:
            disabled = getattr(telemetry_config, "disabled", False)
            ctx = getattr(telemetry_config, "usage_context", None)
            if ctx is not None:
                usage_context = ctx.value if hasattr(ctx, "value") else str(ctx)

        if not is_usage_stats_enabled(telemetry_disabled=disabled):
            return

        # Only rank 0 in a TP group should report (matches vLLM behavior).
        # NOTE: This import is intentionally deferred (not top-level) because
        # usage_lib.py must be importable without the full TRT-LLM stack —
        # test conftest stubs out tensorrt_llm. The try/except ensures
        # lightweight installs and test environments aren't broken.
        try:
            from tensorrt_llm._utils import mpi_rank  # noqa: E402 — deferred by design

            if mpi_rank() != 0:
                return
        except Exception:
            pass  # fail-silent: if we can't determine rank, proceed

        with _REPORTER_LOCK:
            if _REPORTER_STARTED:
                return
            _REPORTER_STARTED = True

        _show_usage_notification()

        thread = threading.Thread(
            target=_background_reporter,
            args=(llm_args, pretrained_config, usage_context),
            daemon=True,
            name="trtllm-usage-stats",
        )
        thread.start()

    except Exception:
        with _REPORTER_LOCK:
            _REPORTER_STARTED = False
