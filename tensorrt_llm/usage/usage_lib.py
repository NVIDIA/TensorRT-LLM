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
daemon thread and never crashes the main process. Terminal delivery may wait
for at most 0.5 seconds during an instrumented shutdown boundary.

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

import atexit
import json
import logging
import os
import platform
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

from tensorrt_llm.usage import schema
from tensorrt_llm.usage.config import UsageContext
from tensorrt_llm.usage.llmapi_config import _failure_llm_api_config_payloads
from tensorrt_llm.usage.llmapi_config import (
    collect_llm_api_config_payloads as _collect_llm_api_config_payloads,
)

logger = logging.getLogger("tensorrt_llm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DISAGG_ROLE_ENV = "TRTLLM_DISAGG_ROLE"
_DISAGG_DEPLOYMENT_ID_ENV = "TRTLLM_DISAGG_DEPLOYMENT_ID"
_DEFAULT_ENDPOINT = "https://events.gfe.nvidia.com/v1.1/events/json"
_HTTP_TIMEOUT = 2.0
_MAX_HEARTBEATS = 1000
_TERMINAL_FLUSH_TIMEOUT = 0.5
_ALLOWED_USAGE_CONTEXTS = frozenset(context.value for context in UsageContext)
_T = TypeVar("_T")


@dataclass(frozen=True)
class TerminalOutcome:
    """One typed process-boundary or previously observed terminal outcome."""

    termination_kind: schema.TerminationKind
    component: Optional[schema.TerminationComponent] = None
    reporting_source: schema.ReportingSource = "self"
    exit_code_known: Optional[bool] = None
    exit_code: int = 0
    signal_number: int = 0

    def with_observation(self, observation: Optional["TerminalOutcome"]) -> "TerminalOutcome":
        """Prefer an earlier causal observation over boundary classification."""
        if observation is None:
            return self
        if observation.exit_code_known is None:
            exit_code_known = self.exit_code_known
            exit_code = self.exit_code
        else:
            exit_code_known = observation.exit_code_known
            exit_code = observation.exit_code
        return TerminalOutcome(
            termination_kind=observation.termination_kind,
            component=observation.component or self.component,
            reporting_source=observation.reporting_source,
            exit_code_known=exit_code_known,
            exit_code=exit_code,
            signal_number=observation.signal_number,
        )


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
       This is the format used in TRT-LLM checkpoint ``config.json`` files.

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

        # Case 2: TRT-LLM PretrainedConfig — .architecture (singular str)
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
    # TODO: Consolidate with llmApiConfigJson, which now captures a near-superset
    # of these columns (backend, quant_config.quant_algo, MoE parallel sizes).
    # Blocker is downstream SMS/dashboard migration: dropping them is a breaking
    # wire-schema change, not a code-only refactor.
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
    # TODO: Deduplicate featuresJson with llmApiConfigJson once remaining
    # derived-only flags have explicit safe fields in LLM API config telemetry.

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
        session = _get_session()
        session_id = session.session_id if session is not None else uuid.uuid4().hex
        trtllm_version = session.trtllm_version if session is not None else _get_trtllm_version()
        # --- Collect initial data ---
        system_info = _collect_system_info()
        gpu_info = _collect_gpu_info()
        arch_class_name = _extract_architecture_class_name(pretrained_config)
        trtllm_config = _extract_trtllm_config(llm_args)
        features_json = _collect_features(llm_args)
        try:
            llm_api_config_json, llm_api_config_meta_json = _collect_llm_api_config_payloads(
                llm_args
            )
        except Exception:
            # Double net: reporter must not die if collector call breaks.
            # Intentionally broad -- this is the daemon-thread fail-silent guard;
            # telemetry must never crash user inference, so anything the narrowed
            # collector net lets propagate is contained here. Shared failure
            # payload keeps metadata shape in one place.
            args_class = type(llm_args).__name__ if llm_args is not None else ""
            llm_api_config_json, llm_api_config_meta_json = _failure_llm_api_config_payloads(
                args_class=args_class
            )

        # Use the latest process counters after the collection work above.
        event_snapshot = _event_snapshot(usage_context)

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
            # Reserved for TRTLLM-411. This change does not populate hashes.
            architectureClassHash="",
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
            # Feature flags
            featuresJson=features_json,
            llmApiConfigJson=llm_api_config_json,
            llmApiConfigMetaJson=llm_api_config_meta_json,
            **_session_event_fields(event_snapshot),
        )

        # --- Send initial report ---
        payload = schema.build_gxt_payload(
            event=initial_event,
            session_id=session_id,
            trtllm_version=trtllm_version,
        )
        if session is not None and not session.claim_initial():
            return
        _send_to_gxt(payload)

        # --- Heartbeat loop ---
        heartbeat_interval = _get_heartbeat_interval()
        for seq in range(_MAX_HEARTBEATS):
            if _REPORTER_STOP.wait(timeout=heartbeat_interval):
                return  # stop requested

            try:
                event_snapshot = _event_snapshot(usage_context)
                heartbeat_event = schema.TrtllmHeartbeat(
                    seq=seq,
                    **_session_event_fields(event_snapshot),
                )
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
    finally:
        _finish_background_reporter()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_REPORTER_STARTED = False
_REPORTER_ACTIVE = False
_REPORTER_LOCK = threading.Lock()
_REPORTER_STOP = threading.Event()  # signal heartbeat loop to exit
_PENDING_TERMINAL: Optional[tuple[dict, threading.Event]] = None
_PROCESS_PID = os.getpid()
_PROCESS_EXIT_HOOK_REGISTERED = False

_DISTRIBUTED_SIZE_ENV_VARS = (
    "WORLD_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
)


class _TelemetrySession:
    """Process-local identity, correlation, counters, and terminal state."""

    def __init__(
        self,
        usage_context: str,
        component: schema.TerminationComponent,
        lifecycle_phase: schema.LifecyclePhase,
    ) -> None:
        self.owner_pid = os.getpid()
        self.session_id = uuid.uuid4().hex
        self.trtllm_version = _get_trtllm_version()
        self.usage_context = usage_context
        self.disagg_role = ""
        self.deployment_id = ""
        self.component = component
        self.lifecycle_phase = lifecycle_phase
        self.observed_signal = 0
        self.observed_outcome: Optional[TerminalOutcome] = None

        self.llm_initialization_attempts = 0
        self.llm_instances_created = 0
        self.active_llm_instances = 0
        self.max_concurrent_llm_instances = 0
        self.llm_initialization_failures = 0

        self.disabled = False
        self.initial_reported = False
        self.terminal_reported = False
        self.lock = threading.Lock()
        self.refresh_metadata()

    @staticmethod
    def _increment(value: int) -> int:
        return min(value + 1, schema._UINT32_MAX)

    def _refresh_metadata_unlocked(self) -> None:
        if not self.disagg_role:
            self.disagg_role = os.environ.get(_DISAGG_ROLE_ENV, "")
        if not self.deployment_id:
            self.deployment_id = os.environ.get(_DISAGG_DEPLOYMENT_ID_ENV, "")

        normalized_role = self.disagg_role.lower()
        if self.component == "server" and (
            normalized_role.startswith("ctx")
            or normalized_role.startswith("gen")
            or normalized_role in ("context", "generation")
        ):
            self.component = "disagg_worker"

    def refresh_metadata(self) -> None:
        """Promote correlation fields that become known after early startup."""
        with self.lock:
            self._refresh_metadata_unlocked()

    def configure(
        self,
        *,
        usage_context: str = "",
        component: Optional[schema.TerminationComponent] = None,
        lifecycle_phase: Optional[schema.LifecyclePhase] = None,
    ) -> None:
        """Promote process metadata without replacing authoritative values."""
        with self.lock:
            if self.disabled:
                return
            if self.usage_context in ("", "unknown") and usage_context not in (
                "",
                "unknown",
            ):
                self.usage_context = usage_context
            if self.component == "unknown" and component is not None:
                self.component = component
            if lifecycle_phase is not None:
                self.lifecycle_phase = lifecycle_phase
            self._refresh_metadata_unlocked()

    def record_llm_initialization_attempt(self) -> bool:
        """Increment the attempt counter and enter model initialization."""
        with self.lock:
            if self.disabled or self.terminal_reported:
                return False
            self.llm_initialization_attempts = self._increment(self.llm_initialization_attempts)
            self.lifecycle_phase = "model_initialization"
            if self.component == "unknown":
                self.component = "llm"
            self._refresh_metadata_unlocked()
            return True

    def record_llm_initialization_failure(self) -> None:
        """Increment the initialization failure counter."""
        with self.lock:
            if not self.disabled and not self.terminal_reported:
                self.llm_initialization_failures = self._increment(self.llm_initialization_failures)

    def record_llm_initialized(self) -> bool:
        """Record a successfully constructed and active LLM object."""
        with self.lock:
            if self.disabled or self.terminal_reported:
                return False
            self.llm_instances_created = self._increment(self.llm_instances_created)
            self.active_llm_instances = self._increment(self.active_llm_instances)
            self.max_concurrent_llm_instances = max(
                self.max_concurrent_llm_instances,
                self.active_llm_instances,
            )
            self.lifecycle_phase = "serving"
            return True

    def record_llm_shutdown(self) -> None:
        """Decrement the active gauge without allowing it to become negative."""
        with self.lock:
            if not self.disabled and self.active_llm_instances > 0:
                self.active_llm_instances -= 1

    def set_lifecycle_phase(self, lifecycle_phase: schema.LifecyclePhase) -> None:
        """Update the best-known process lifecycle phase."""
        with self.lock:
            if not self.disabled and not self.terminal_reported:
                self.lifecycle_phase = lifecycle_phase

    def set_usage_context(self, usage_context: str) -> None:
        """Set an authoritative allowlisted process ingress point."""
        if usage_context not in _ALLOWED_USAGE_CONTEXTS:
            return
        with self.lock:
            if not self.disabled and not self.terminal_reported:
                self.usage_context = usage_context

    def record_observed_signal(self, signal_number: int) -> None:
        """Remember a handled signal for the authoritative outer boundary."""
        if signal_number <= 0 or signal_number > schema._UINT32_MAX:
            return
        with self.lock:
            if not self.disabled and not self.terminal_reported:
                self.observed_signal = signal_number

    def record_termination_observation(self, outcome: TerminalOutcome) -> None:
        """Remember a causal classification until the process actually exits."""
        with self.lock:
            if self.disabled or self.terminal_reported or self.observed_outcome is not None:
                return
            self.observed_outcome = outcome

    def get_termination_observation(self) -> Optional[TerminalOutcome]:
        """Return the first causal terminal observation, if any."""
        with self.lock:
            return self.observed_outcome

    def _snapshot_unlocked(self) -> dict[str, Any]:
        self._refresh_metadata_unlocked()
        return {
            "ingressPoint": _clamp_str(self.usage_context, schema._SHORT_STR),
            "disaggRole": _clamp_str(self.disagg_role, schema._SHORT_STR),
            "deploymentId": _clamp_str(self.deployment_id, schema._SHORT_STR),
            "llmInitializationAttempts": self.llm_initialization_attempts,
            "llmInstancesCreated": self.llm_instances_created,
            "activeLlmInstances": self.active_llm_instances,
            "maxConcurrentLlmInstances": self.max_concurrent_llm_instances,
            "llmInitializationFailures": self.llm_initialization_failures,
            "lifecyclePhase": self.lifecycle_phase,
            "component": self.component,
            "observedSignal": self.observed_signal,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return an atomic metadata and counter snapshot."""
        with self.lock:
            return self._snapshot_unlocked()

    def claim_terminal(
        self, outcome: TerminalOutcome
    ) -> Optional[tuple[dict[str, Any], TerminalOutcome]]:
        """Atomically merge causal context and claim the terminal slot."""
        with self.lock:
            if self.disabled or self.terminal_reported:
                return None
            outcome = outcome.with_observation(self.observed_outcome)
            self.terminal_reported = True
            return self._snapshot_unlocked(), outcome

    def claim_initial(self) -> bool:
        """Claim the success-only initial report before network delivery."""
        with self.lock:
            if self.disabled or self.initial_reported or self.terminal_reported:
                return False
            self.initial_reported = True
            return True

    def disable(self) -> None:
        """Prevent a stale reference from emitting after process opt-out."""
        with self.lock:
            self.disabled = True


_SESSION: Optional[_TelemetrySession] = None
_SESSION_LOCK = threading.Lock()
_SESSION_DISABLED = False


def _ensure_process_state() -> None:
    """Reset inherited process-local state after ``fork()``."""
    global _NOTIFICATION_SHOWN
    global _PENDING_TERMINAL
    global _PROCESS_PID
    global _REPORTER_ACTIVE
    global _REPORTER_LOCK
    global _REPORTER_STARTED
    global _REPORTER_STOP
    global _SESSION
    global _SESSION_LOCK

    current_pid = os.getpid()
    if current_pid == _PROCESS_PID:
        return

    _PROCESS_PID = current_pid
    _SESSION = None
    _SESSION_LOCK = threading.Lock()
    _REPORTER_STARTED = False
    _REPORTER_ACTIVE = False
    _REPORTER_LOCK = threading.Lock()
    _REPORTER_STOP = threading.Event()
    _PENDING_TERMINAL = None
    _NOTIFICATION_SHOWN = threading.Event()


def _get_session() -> Optional[_TelemetrySession]:
    """Return this process's telemetry session, if one exists."""
    _ensure_process_state()
    with _SESSION_LOCK:
        session = _SESSION
    if session is not None and session.owner_pid != os.getpid():
        return None
    return session


def _deactivate_usage_session() -> None:
    """Stop process telemetry after any authoritative opt-out decision."""
    global _SESSION
    global _SESSION_DISABLED
    with _SESSION_LOCK:
        session = _SESSION
        _SESSION = None
        _SESSION_DISABLED = True
    if session is not None:
        session.disable()
    _REPORTER_STOP.set()


def _empty_event_snapshot(usage_context: str = "") -> dict[str, Any]:
    """Return the schema defaults used by isolated reporter tests."""
    return {
        "ingressPoint": _clamp_str(usage_context, schema._SHORT_STR),
        "disaggRole": _clamp_str(os.environ.get(_DISAGG_ROLE_ENV, ""), schema._SHORT_STR),
        "deploymentId": _clamp_str(
            os.environ.get(_DISAGG_DEPLOYMENT_ID_ENV, ""), schema._SHORT_STR
        ),
        "llmInitializationAttempts": 0,
        "llmInstancesCreated": 0,
        "activeLlmInstances": 0,
        "maxConcurrentLlmInstances": 0,
        "llmInitializationFailures": 0,
        "lifecyclePhase": "unknown",
        "component": "unknown",
        "observedSignal": 0,
    }


def _event_snapshot(usage_context: str = "") -> dict[str, Any]:
    """Return the current session snapshot or schema defaults."""
    session = _get_session()
    return session.snapshot() if session is not None else _empty_event_snapshot(usage_context)


def _session_event_fields(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Select correlation and lifecycle counters shared by all events."""
    return {
        "ingressPoint": snapshot["ingressPoint"],
        "disaggRole": snapshot["disaggRole"],
        "deploymentId": snapshot["deploymentId"],
        "llmInitializationAttempts": snapshot["llmInitializationAttempts"],
        "llmInstancesCreated": snapshot["llmInstancesCreated"],
        "activeLlmInstances": snapshot["activeLlmInstances"],
        "maxConcurrentLlmInstances": snapshot["maxConcurrentLlmInstances"],
        "llmInitializationFailures": snapshot["llmInitializationFailures"],
    }


def _validated_usage_context(value: Any) -> str:
    """Return a bounded ingress category or the unset sentinel."""
    if isinstance(value, UsageContext):
        return value.value
    if isinstance(value, str) and value in _ALLOWED_USAGE_CONTEXTS:
        return value
    return ""


def _telemetry_settings(
    telemetry_config: Any,
    default_usage_context: str = "",
) -> tuple[bool, str]:
    """Extract opt-out and ingress values from a telemetry config."""
    disabled = False
    usage_context = ""
    if telemetry_config is not None:
        if isinstance(telemetry_config, dict):
            # Raw dictionaries have not passed TelemetryConfig validation. An
            # explicit opt-out is authoritative; every other value fails closed
            # for early failure reporting.
            disabled_value = telemetry_config.get("disabled")
            usage_context_value = telemetry_config.get("usage_context")
            if disabled_value is not True:
                disabled_value = None
        else:
            missing = object()
            disabled_value = getattr(telemetry_config, "disabled", missing)
            usage_context_value = getattr(telemetry_config, "usage_context", None)

        if isinstance(disabled_value, bool):
            disabled = disabled_value
        else:
            disabled = True

        usage_context = _validated_usage_context(usage_context_value)

    if usage_context in ("", "unknown"):
        usage_context = _validated_usage_context(default_usage_context)
    return disabled, usage_context


def _is_reporting_rank() -> bool:
    """Return whether this process is the telemetry-reporting MPI rank."""
    try:
        from tensorrt_llm._utils import mpi_rank  # noqa: E402 — deferred by design

        return mpi_rank() == 0
    except Exception:
        for name in _DISTRIBUTED_SIZE_ENV_VARS:
            try:
                if int(os.environ.get(name, "1")) > 1:
                    return False
            except ValueError:
                return False
        return True


def _report_process_exit() -> None:
    """Best-effort fallback for direct ``LLM()`` interpreter exits."""
    try:
        session = _get_session()
        if session is None or not is_usage_stats_enabled():
            return

        snapshot = session.snapshot()
        signal_number = snapshot["observedSignal"]
        outcome = TerminalOutcome(
            exit_code_known=False,
            exit_code=0,
            signal_number=signal_number,
            termination_kind="signal" if signal_number else "unknown",
        )
        report_exit(
            outcome,
            lifecycle_phase=None,
        )
    except Exception:
        pass


def start_usage_session(
    telemetry_config: Any = None,
    *,
    default_usage_context: str = "",
    component: Optional[schema.TerminationComponent] = None,
    lifecycle_phase: Optional[schema.LifecyclePhase] = None,
) -> bool:
    """Create the process-local telemetry session without sending data.

    Returns whether telemetry is active for the process. Repeated calls reuse
    the existing session and may promote an unknown ingress point to a known
    one.
    """
    global _PROCESS_EXIT_HOOK_REGISTERED
    global _SESSION
    try:
        _ensure_process_state()
        if _SESSION_DISABLED:
            return False
        if isinstance(telemetry_config, dict) and telemetry_config.get("disabled") is not True:
            # Defer until Pydantic has validated this user-supplied config.
            return False
        disabled, usage_context = _telemetry_settings(
            telemetry_config,
            default_usage_context,
        )
        if not is_usage_stats_enabled(telemetry_disabled=disabled):
            _deactivate_usage_session()
            return False
        with _SESSION_LOCK:
            if _SESSION_DISABLED:
                return False
            if _SESSION is None:
                _REPORTER_STOP.clear()
                _SESSION = _TelemetrySession(
                    usage_context=usage_context,
                    component=component or "unknown",
                    lifecycle_phase=lifecycle_phase or "unknown",
                )
                if not _PROCESS_EXIT_HOOK_REGISTERED:
                    atexit.register(_report_process_exit)
                    _PROCESS_EXIT_HOOK_REGISTERED = True
            session = _SESSION

        session.configure(
            usage_context=usage_context,
            component=component,
            lifecycle_phase=lifecycle_phase,
        )
        return True
    except Exception:
        return False


def _session_call(
    operation: Callable[[_TelemetrySession], _T],
    default: _T,
) -> _T:
    """Call an existing process session without affecting the application."""
    try:
        session = _get_session()
        if session is None:
            return default
        return operation(session)
    except Exception:
        return default


def record_llm_initialization_attempt(
    telemetry_config: Any = None,
    *,
    default_usage_context: str = "llm_class",
) -> bool:
    """Start local tracking and record entry into an LLM constructor."""
    if not start_usage_session(
        telemetry_config,
        default_usage_context=default_usage_context,
        component="llm",
        lifecycle_phase="model_initialization",
    ):
        return False
    return _session_call(
        lambda session: session.record_llm_initialization_attempt(),
        False,
    )


def record_llm_initialization_failure() -> None:
    """Record a handled Python exception from an LLM constructor."""
    _session_call(lambda session: session.record_llm_initialization_failure(), None)


def record_llm_initialized() -> bool:
    """Record one successfully constructed LLM object."""
    return _session_call(lambda session: session.record_llm_initialized(), False)


def record_llm_shutdown() -> None:
    """Mark one successfully tracked LLM object inactive."""
    _session_call(lambda session: session.record_llm_shutdown(), None)


def set_lifecycle_phase(lifecycle_phase: schema.LifecyclePhase) -> None:
    """Update the best-known phase for an outer process boundary."""
    _session_call(lambda session: session.set_lifecycle_phase(lifecycle_phase), None)


def set_usage_context(usage_context: str) -> None:
    """Set a known process ingress after command resolution."""
    _session_call(lambda session: session.set_usage_context(usage_context), None)


def record_observed_signal(signal_number: int) -> None:
    """Remember a handled signal without reporting an exit prematurely."""
    _session_call(lambda session: session.record_observed_signal(signal_number), None)


def record_termination_observation(outcome: TerminalOutcome) -> None:
    """Remember a terminal cause for a surviving authoritative boundary."""
    _session_call(lambda session: session.record_termination_observation(outcome), None)


def get_termination_observation() -> Optional[TerminalOutcome]:
    """Return a pending terminal classification, if one was observed."""
    return _session_call(lambda session: session.get_termination_observation(), None)


def get_observed_signal() -> int:
    """Return the signal observed by the current process, if any."""
    return _session_call(
        lambda session: int(session.snapshot()["observedSignal"]),
        0,
    )


def _send_terminal_event(payload: dict, completion: threading.Event) -> None:
    """Attempt terminal delivery and always release bounded waiters."""
    try:
        _send_to_gxt(payload)
    finally:
        completion.set()


def _finish_background_reporter() -> None:
    """Deactivate the reporter and flush a terminal payload queued to it."""
    global _PENDING_TERMINAL
    global _REPORTER_ACTIVE
    pending = None
    try:
        with _REPORTER_LOCK:
            _REPORTER_ACTIVE = False
            pending = _PENDING_TERMINAL
            _PENDING_TERMINAL = None
        if pending is not None:
            _send_terminal_event(*pending)
    except Exception:
        if pending is not None:
            pending[1].set()


def report_exit(
    outcome: TerminalOutcome,
    *,
    lifecycle_phase: Optional[schema.LifecyclePhase] = None,
    telemetry_config: Any = None,
    default_usage_context: str = "",
) -> bool:
    """Attempt one correlated terminal event with a bounded delivery wait.

    The first valid caller claims the terminal slot. Later calls are no-ops,
    even when shutdown and error paths race. The return value reports whether
    this call claimed the slot, not whether network delivery succeeded.
    """
    claimed = False
    try:
        disabled, _ = _telemetry_settings(
            telemetry_config,
            default_usage_context,
        )
        if not is_usage_stats_enabled(telemetry_disabled=disabled):
            _deactivate_usage_session()
            return False
        session = _get_session()
        if session is None:
            return False
        # Rank selection may change during startup when disaggregated serving
        # splits the world communicator into context and generation groups.
        # Keep setup local on every process and decide only at emission time so
        # each group's eventual rank 0 can report without duplicate rank events.
        if not _is_reporting_rank():
            return False

        terminal = session.claim_terminal(outcome)
        if terminal is None:
            return False
        snapshot, outcome = terminal
        claimed = True
        _show_usage_notification()

        exit_code_known = outcome.exit_code_known is True
        exit_code = outcome.exit_code
        signal_number = outcome.signal_number
        if not exit_code_known:
            exit_code = 0
        elif not isinstance(exit_code, int) or not 0 <= exit_code <= schema._UINT32_MAX:
            exit_code_known = False
            exit_code = 0
        if not isinstance(signal_number, int) or not 0 <= signal_number <= schema._UINT32_MAX:
            signal_number = 0

        event = schema.TrtllmExitReport(
            exitCodeKnown=exit_code_known,
            exitCode=exit_code,
            signalNumber=signal_number,
            terminationKind=outcome.termination_kind,
            lifecyclePhase=lifecycle_phase or snapshot["lifecyclePhase"],
            component=outcome.component or snapshot["component"],
            reportingSource=outcome.reporting_source,
            **_session_event_fields(snapshot),
        )
        payload = schema.build_gxt_payload(
            event=event,
            session_id=session.session_id,
            trtllm_version=session.trtllm_version,
        )

        completion = threading.Event()
        queued_to_reporter = False
        global _PENDING_TERMINAL
        with _REPORTER_LOCK:
            if _REPORTER_ACTIVE:
                _PENDING_TERMINAL = (payload, completion)
                queued_to_reporter = True

        _REPORTER_STOP.set()
        if queued_to_reporter:
            completion.wait(timeout=_TERMINAL_FLUSH_TIMEOUT)
            return True

        thread = threading.Thread(
            target=_send_terminal_event,
            args=(payload, completion),
            daemon=True,
            name="trtllm-usage-terminal",
        )
        thread.start()
        completion.wait(timeout=_TERMINAL_FLUSH_TIMEOUT)
        return True
    except Exception:
        return claimed


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
    global _PENDING_TERMINAL
    global _REPORTER_ACTIVE
    global _REPORTER_STARTED
    try:
        _, usage_context = _telemetry_settings(telemetry_config)
        if not start_usage_session(telemetry_config):
            return
        # See report_exit(): the authoritative communicator may not exist at
        # early session creation time, so rank gating belongs at emission.
        if not _is_reporting_rank():
            return

        with _REPORTER_LOCK:
            if _REPORTER_STARTED:
                return
            _REPORTER_STARTED = True
            _REPORTER_ACTIVE = True

        _show_usage_notification()

        thread = threading.Thread(
            target=_background_reporter,
            args=(llm_args, pretrained_config, usage_context),
            daemon=True,
            name="trtllm-usage-stats",
        )
        thread.start()

    except Exception:
        pending = None
        with _REPORTER_LOCK:
            _REPORTER_STARTED = False
            _REPORTER_ACTIVE = False
            pending = _PENDING_TERMINAL
            _PENDING_TERMINAL = None
        if pending is not None:
            pending[1].set()
